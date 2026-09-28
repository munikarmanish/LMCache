#!/usr/bin/env bash
# launch_node.sh — Bring up ONE node of the 2-node cross-node KV-cache
# test: an LMCache MP server (with a CXL or NIXL-peer L2 adapter) plus a
# native vLLM instance wired to it over tcp://localhost:5555.
#
# Each node serves its own OpenAI-compatible vLLM endpoint. A separate
# lightweight router (see router.py / launch_router.sh) sits in front of
# the two nodes for routing-strategy experiments; it is NOT launched here.
#
# Replaces the separate launch_mp_server.sh + launch_vllm.sh: this script
#   1. starts the MP server (output -> terminal AND log file),
#   2. waits for the MP server's HTTP healthcheck,
#   3. starts vLLM (output -> log file only),
#   4. waits for vLLM's /health,
#   5. prints a summary of the running server config.
#
# Usage:
#   ./launch_node.sh <node_id 0|1> <mode cxl|nixl>
#
# Examples:
#   ./launch_node.sh 0 cxl      # node0, CXL adapter
#   ./launch_node.sh 1 nixl     # node1, NIXL-peer (RDMA) adapter
#
#   # A different model, tensor-parallel over both GPUs:
#   MODEL=Qwen/Qwen3-32B TP=2 ./launch_node.sh 0 cxl
#
#   # Two single-GPU instances on one node, sharing one MP server and
#   # therefore one CXL pool. Start the first normally, then:
#   GPUS=0 ./launch_node.sh 0 cxl
#   GPUS=1 VLLM_PORT=8011 SKIP_MP_SERVER=1 ./launch_node.sh 0 cxl
#
# Model / parallelism env: MODEL, TP, DTYPE, GPUS, GPU_MEM_UTIL.
# L1_SIZE_GB sets the MP server's L1 (DRAM) capacity (default 64).
# Ports (override to co-locate instances): VLLM_PORT, LMC_ZMQ_PORT,
# LMC_HTTP_PORT. SKIP_MP_SERVER=1 attaches to an already-running MP
# server instead of starting one.
#
# NOTE: node_id indexes a row in the shared CXL lock table, so it stays
# 1:1 with the physical node. Co-located instances share one MP server
# rather than taking separate node_ids.
#
# Env toggles:
#   GPUDIRECT=1   (nixl only) pull remote hits straight into the GPU staging
#                 buffer, skipping the L1/DRAM landing and the H2D bounce.
#                 Off by default. Set it on BOTH nodes. Requires
#                 GPUDirect-capable RDMA; see
#                 docs/design/v1/distributed/l2_adapters/nixl_peer_gpudirect.md
#     GPUDIRECT=1 ./launch_node.sh 0 nixl
#
#   LMC_PROFILE=1 emit the per-request stage breakdown (PROFILE /
#                 PROFILE-L2LK log lines: l2lk, l1rsv, l2load, pf_wait,
#                 ret_h2d, ret_scat ...). Off by default — it adds a CUDA
#                 stream sync per retrieve, so it perturbs the TTFT it
#                 measures. Use it to attribute a change, not to benchmark.
#     LMC_PROFILE=1 ./launch_node.sh 1 nixl
#
# Peers are NOT taken from the CLI: the two-node topology is hard-coded
# from the known IPs of node0 (192.168.128.75) and node1 (192.168.128.76).
#
# Requires: jq, and the lmcache venv on PATH (lmcache + vllm CLIs).

set -euo pipefail

# ---------------------------------------------------------------------------
# Activate the lmcache virtualenv (puts the lmcache + vllm CLIs on PATH).
# ---------------------------------------------------------------------------
VENV="$HOME/.virtualenvs/lmcache"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: lmcache venv not found at $VENV" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---------------------------------------------------------------------------
# Static 2-node topology (NOT configurable via CLI).
# ---------------------------------------------------------------------------
NODE0_HOST=192.168.128.75
NODE1_HOST=192.168.128.76

# ---------------------------------------------------------------------------
# Ports / model (same on both nodes — each binds its own).
# ---------------------------------------------------------------------------
LMC_ZMQ_PORT="${LMC_ZMQ_PORT:-5555}"    # MP server ZMQ (vLLM connector talks here)
LMC_HTTP_PORT="${LMC_HTTP_PORT:-8090}"  # MP server HTTP (healthcheck, /lookup_hits)
VLLM_PORT="${VLLM_PORT:-8010}"          # vLLM OpenAI API (/health, /v1/..., /metrics)
NIXL_GPU_INIT_PORT="${NIXL_GPU_INIT_PORT:-8502}"  # NIXL GPU handshake (GPUDIRECT=1)

# ---------------------------------------------------------------------------
# Model / parallelism. Override from the environment; nothing here needs to
# be kept in sync with the CXL adapter config, which no longer declares any
# model or TP fields (one pool serves many models concurrently).
#
#   MODEL=Qwen/Qwen3-32B TP=2 ./launch_node.sh 0 cxl
#
# GPUS pins which devices this instance sees. Leave unset to use the first
# TP devices; set it to run two instances on one node, e.g.
#   GPUS=0 VLLM_PORT=8010 ./launch_node.sh 0 cxl
#   GPUS=1 VLLM_PORT=8011 ./launch_node.sh 0 cxl
# Both attach to the SAME MP server: node_id must stay 1:1 with the node,
# because it indexes a row in the shared CXL lock table. So the second
# instance also sets SKIP_MP_SERVER=1 and reuses the first's LMC_ZMQ_PORT.
# ---------------------------------------------------------------------------
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${TP:-1}"
DTYPE="${DTYPE:-auto}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
L1_SIZE_GB="${L1_SIZE_GB:-64}"          # MP server L1 (DRAM) capacity
# Default to the first TP devices (0, 0-1, 0-1-2-3, ...).
GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"

# ---------------------------------------------------------------------------
# CLI args.
# ---------------------------------------------------------------------------
NODE_ID="${1:?usage: $0 <node_id 0|1> <mode cxl|nixl>}"
MODE="${2:?usage: $0 <node_id 0|1> <mode cxl|nixl>}"
HERE="$(cd "$(dirname "$0")" && pwd)"

case "$NODE_ID" in
    0) MY_IP="$NODE0_HOST"; PEER_ID=1; PEER_IP="$NODE1_HOST" ;;
    1) MY_IP="$NODE1_HOST"; PEER_ID=0; PEER_IP="$NODE0_HOST" ;;
    *) echo "ERROR: node_id must be 0 or 1 (got '$NODE_ID')" >&2; exit 2 ;;
esac

# The spec files live under config/ and are named by mode + node id:
# e.g. config/cxl.node0.json, config/nixl.node1.json. The "nixl" mode
# still selects the adapter whose JSON declares "type": "nixl_peer";
# only the file basename is shortened.
CONFIG_DIR="$HERE/config"
case "$MODE" in
    cxl)
        BASE="$CONFIG_DIR/cxl.base.json"
        OVERRIDE="$CONFIG_DIR/cxl.node${NODE_ID}.json"
        ;;
    nixl)
        BASE="$CONFIG_DIR/nixl.base.json"
        OVERRIDE="$CONFIG_DIR/nixl.node${NODE_ID}.json"
        # Pin NIXL's UCX transport to the direct RDMA link (mlx5_0 port 1).
        # RC = reliable-connection RDMA.
        #
        # UCX_TLS is an ALLOWLIST, so it must also name the CUDA transports
        # when GPUDirect is on. With a bare "rc", UCX has no cuda_copy
        # transport, classifies the GPU staging buffer as host memory, and
        # fails registration with:
        #   "VRAM memory is detected as host by UCX ... registration cannot
        #    proceed" -> NIXL_ERR_BACKEND
        # cuda_copy is what lets UCX detect and register VRAM. (gdr_copy,
        # the GPUDirect fast path, is NOT in this UCX build — naming it just
        # produces a "transport not available" WARN on every launch, so it
        # is left out. rc already carries the RDMA path to the NIC.)
        if [[ "${GPUDIRECT:-0}" == "1" ]]; then
            export UCX_TLS="${UCX_TLS:-rc,cuda_copy}"
        else
            export UCX_TLS="${UCX_TLS:-rc}"
        fi
        export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1}"
        ;;
    *)
        echo "ERROR: mode must be cxl or nixl (got '$MODE')" >&2
        exit 2
        ;;
esac

# All logs go under ./logs as node<N>-<mode>.log and node<N>-vllm.log.
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
# INSTANCE distinguishes co-located vLLM instances in the log names; it
# defaults to the vLLM port so two instances never clobber each other.
INSTANCE="${INSTANCE:-$VLLM_PORT}"
LMC_LOG="$LOG_DIR/node${NODE_ID}-${MODE}.log"
VLLM_LOG="$LOG_DIR/node${NODE_ID}-vllm-${INSTANCE}.log"

# vLLM-side LMCache engine config (chunk size etc.).
export LMCACHE_CONFIG_FILE="$CONFIG_DIR/lmcache.yaml"

# Stable Python hash seed: token hashing must be byte-identical across
# nodes for cross-node hits, and Python's default randomized hashing would
# change the chunk_hash between processes.
export PYTHONHASHSEED=0
# Unbuffered so the tee'd MP-server log stays live (the `lmcache` console
# entry point can't take `python -u`).
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Process management: track both children, kill them on exit.
# ---------------------------------------------------------------------------
PIDS=()
cleanup() {
    trap - INT TERM EXIT
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
    sleep 2
    for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT

wait_for_http() {
    local url="$1"; local timeout="${2:-300}"; local deadline=$(( SECONDS + timeout ))
    echo "Waiting for $url (timeout ${timeout}s)..."
    while ! curl -sf "$url" >/dev/null 2>&1; do
        [[ $SECONDS -ge $deadline ]] && { echo "TIMEOUT waiting for $url" >&2; exit 1; }
        sleep 3
    done
    echo "  -> $url ready"
}


# ---------------------------------------------------------------------------
# 1. MP server -> terminal AND log file.
# ---------------------------------------------------------------------------
# Per-request stage breakdown (the PROFILE / PROFILE-L2LK log lines).
# Off by default — it adds a CUDA stream sync per retrieve, so it is
# measurement overhead on the TTFT path. Override from the terminal:
#   LMC_PROFILE=1 ./launch_node.sh 1 nixl
export LMC_PROFILE="${LMC_PROFILE:-0}"

L2_JSON="$(jq -c -s '.[0] * .[1]' "$BASE" "$OVERRIDE" \
    | sed -e "s/NODE0_HOST/${NODE0_HOST}/g" -e "s/NODE1_HOST/${NODE1_HOST}/g")"

# ---------------------------------------------------------------------------
# GPUDirect (nixl mode only, opt-in): GPUDIRECT=1 ./launch_node.sh <id> nixl
#
# Pulls a remote hit straight into the GPU staging buffer over RDMA, skipping
# the L1/DRAM landing and the H2D bounce. Injected here rather than committed
# into the JSON so the default stays the (validated) DRAM path and A/B runs
# need no file edits.
#
# The GPU side is a SEPARATE NIXL agent from the DRAM one (NIXL registers one
# region with one memory type per agent), so it needs its own bind port and
# its own peer URL — hence gpu_init_bind_url + peers[].gpu_init_url.
# ---------------------------------------------------------------------------
GPUDIRECT="${GPUDIRECT:-0}"
if [[ "$GPUDIRECT" == "1" ]]; then
    if [[ "$MODE" != "nixl" ]]; then
        echo "ERROR: GPUDIRECT=1 is only supported in 'nixl' mode (got '$MODE')" >&2
        exit 2
    fi
    L2_JSON="$(jq -c \
        --arg bind "0.0.0.0:${NIXL_GPU_INIT_PORT}" \
        --arg peer "${PEER_IP}:${NIXL_GPU_INIT_PORT}" \
        '.enable_gpu_direct = true
         | .gpu_init_bind_url = $bind
         | .peers = [.peers[] | .gpu_init_url = $peer]' <<<"$L2_JSON")"
fi

L2_STORE_POLICY="${L2_STORE_POLICY:-lazy}"
# L2_STORE_POLICY="${L2_STORE_POLICY:-default}"

L2_PREFETCH_POLICY="default"
#L2_PREFETCH_POLICY="retain"

# EVICTION_DESTINATION="${EVICTION_DESTINATION:-DISCARD}"
EVICTION_DESTINATION="${EVICTION_DESTINATION:-L2_CACHE}"

# SKIP_MP_SERVER=1 attaches this vLLM instance to an MP server another
# invocation already started on this node. That is how you run two vLLM
# instances per node: one MP server owns the pool (node_id is 1:1 with the
# node), and both instances talk to it over the same ZMQ port.
if [[ "${SKIP_MP_SERVER:-0}" == "1" ]]; then
    echo "SKIP_MP_SERVER=1: reusing the MP server on :${LMC_ZMQ_PORT}"
    wait_for_http "http://localhost:${LMC_HTTP_PORT}/healthcheck" 60
else
lmcache server \
    --host localhost --port "$LMC_ZMQ_PORT" \
    --http-host 0.0.0.0 --http-port "$LMC_HTTP_PORT" \
    --hash-algorithm builtin \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --eviction-destination "$EVICTION_DESTINATION" \
    --l2-store-policy "$L2_STORE_POLICY" \
    --l2-prefetch-policy "$L2_PREFETCH_POLICY" \
    --l2-adapter "$L2_JSON" \
    > >(tee "$LMC_LOG") 2>&1 &
PIDS+=($!)

wait_for_http "http://localhost:${LMC_HTTP_PORT}/healthcheck"
fi

# ---------------------------------------------------------------------------
# 2. vLLM -> log file only.
# ---------------------------------------------------------------------------
# Native `vllm serve` with the LMCache MP connector. Each node binds its own
# OpenAI endpoint on $VLLM_PORT (0.0.0.0) so the lightweight router on node0
# can reach both. --disable-log-stats is dropped so the router can scrape
# vLLM's /metrics for the GPU-load routing signal.
KV_CFG="{
  \"kv_connector\": \"LMCacheMPConnector\",
  \"kv_role\": \"kv_both\",
  \"kv_connector_extra_config\": {
    \"lmcache.mp.host\": \"tcp://localhost\",
    \"lmcache.mp.port\": ${LMC_ZMQ_PORT}
  }
}"

CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size "$TP" \
    --kv-transfer-config "$KV_CFG" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype "$DTYPE" \
    > "$VLLM_LOG" 2>&1 &
PIDS+=($!)

wait_for_http "http://localhost:${VLLM_PORT}/health" 600

# ---------------------------------------------------------------------------
# 3. Summary.
# ---------------------------------------------------------------------------
echo "
============================================================
 node${NODE_ID} READY — mode=${MODE}
============================================================
  MY_IP          : $MY_IP
  PEER           : node${PEER_ID} at $PEER_IP
  MODEL          : $MODEL
  TP / GPUs      : TP=$TP on CUDA_VISIBLE_DEVICES=$GPUS (dtype=$DTYPE)
  GPUDirect      : $([[ "$GPUDIRECT" == "1" ]] \
                       && echo "ENABLED (gpu handshake :${NIXL_GPU_INIT_PORT})" \
                       || echo "disabled (DRAM path)")
  LMC_PROFILE    : $([[ "$LMC_PROFILE" == "0" ]] \
                       && echo "off" \
                       || echo "ON (stage breakdown; adds a sync per retrieve)")
  MP server ZMQ  : tcp://localhost:${LMC_ZMQ_PORT}$([[ "${SKIP_MP_SERVER:-0}" == "1" ]] && echo "  (reused)" || echo "")
  MP server HTTP : http://${MY_IP}:${LMC_HTTP_PORT}  (/healthcheck, /lookup_hits)
  vLLM API       : http://${MY_IP}:${VLLM_PORT}      (/health, /v1/..., /metrics)
  MP server log  : $LMC_LOG
  vLLM log       : $VLLM_LOG
============================================================
"

# Keep both children in the foreground; cleanup() reaps them on Ctrl-C.
wait
