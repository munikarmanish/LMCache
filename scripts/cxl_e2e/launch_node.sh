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
# Peers are NOT taken from the CLI: the two-node topology is hard-coded
# from the known IPs of node0 (192.168.128.31) and node1 (192.168.128.32).
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
NODE0_HOST=192.168.128.31
NODE1_HOST=192.168.128.32

# ---------------------------------------------------------------------------
# Ports / model (same on both nodes — each binds its own).
# ---------------------------------------------------------------------------
LMC_ZMQ_PORT=5555          # MP server ZMQ (vLLM connector talks here)
LMC_HTTP_PORT=8090         # MP server HTTP (healthcheck + /lookup_hits + metrics)
VLLM_PORT=8010             # vLLM OpenAI API (/health, /v1/..., /metrics)
MODEL="meta-llama/Llama-3.1-8B-Instruct"

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
        export UCX_TLS="${UCX_TLS:-rc}"
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
LMC_LOG="$LOG_DIR/node${NODE_ID}-${MODE}.log"
VLLM_LOG="$LOG_DIR/node${NODE_ID}-vllm.log"

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
export LMC_PROFILE=0
L2_JSON="$(jq -c -s '.[0] * .[1]' "$BASE" "$OVERRIDE" \
    | sed -e "s/NODE0_HOST/${NODE0_HOST}/g" -e "s/NODE1_HOST/${NODE1_HOST}/g")"

L2_STORE_POLICY="default"
#L2_STORE_POLICY="lazy"

L2_PREFETCH_POLICY="default"
#L2_PREFETCH_POLICY="retain"

lmcache server \
    --host localhost --port "$LMC_ZMQ_PORT" \
    --http-host 0.0.0.0 --http-port "$LMC_HTTP_PORT" \
    --hash-algorithm builtin \
    --l1-size-gb 92 --eviction-policy LRU \
    --l2-store-policy "$L2_STORE_POLICY" \
    --l2-prefetch-policy "$L2_PREFETCH_POLICY" \
    --l2-adapter "$L2_JSON" \
    > >(tee "$LMC_LOG") 2>&1 &
PIDS+=($!)

wait_for_http "http://localhost:${LMC_HTTP_PORT}/healthcheck"

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

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --host 0.0.0.0 \
    --kv-transfer-config "$KV_CFG" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --dtype float16 \
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
  MP server ZMQ  : tcp://localhost:${LMC_ZMQ_PORT}
  MP server HTTP : http://${MY_IP}:${LMC_HTTP_PORT}  (/healthcheck, /lookup_hits)
  vLLM API       : http://${MY_IP}:${VLLM_PORT}      (/health, /v1/..., /metrics)
  MP server log  : $LMC_LOG
  vLLM log       : $VLLM_LOG
============================================================
"

# Keep both children in the foreground; cleanup() reaps them on Ctrl-C.
wait
