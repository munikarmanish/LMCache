#!/usr/bin/env bash
# Launch the LMCache MP server with an L2 adapter (CXL or NIXL-peer/RDMA).
#
# Usage:
#   NODE_A_HOST=10.0.0.1 NODE_B_HOST=10.0.0.2 \
#     sudo ./launch_mp_server.sh node_a [cxl|nixl_peer]   # initializer
#   NODE_A_HOST=10.0.0.1 NODE_B_HOST=10.0.0.2 \
#     sudo ./launch_mp_server.sh node_b [cxl|nixl_peer]   # attacher
#
# The L2 adapter defaults to "cxl"; pass "nixl_peer" as the 2nd arg (or
# set L2=nixl_peer in the env) to use the RDMA peer adapter instead.
#
# For each adapter the script merges a shared base spec with the per-node
# override using jq, substitutes NODE_A_HOST / NODE_B_HOST placeholders,
# then passes the result as a single --l2-adapter JSON:
#
#   cxl       -> cxl.base.json        + cxl.{node_a,node_b}.json
#   nixl_peer -> nixl_peer.base.json  + nixl_peer.{node_a,node_b}.json
#
# Requires: jq, the `lmcache` CLI on PATH (from the lmcache venv),
# NODE_A_HOST, NODE_B_HOST in env.

set -euo pipefail

NODE_A_HOST=192.168.128.31
NODE_B_HOST=192.168.128.32

NODE="${1:?usage: $0 node_a|node_b [cxl|nixl_peer]}"
# L2 adapter: 2nd positional arg wins, else the L2 env var, else "cxl".
L2="${2:-${L2:-cxl}}"
HERE="$(cd "$(dirname "$0")" && pwd)"

: "${NODE_A_HOST:?NODE_A_HOST must be set (peer URL substitution)}"
: "${NODE_B_HOST:?NODE_B_HOST must be set (peer URL substitution)}"

# Pick the base + per-node spec files for the chosen adapter.
case "$L2" in
    cxl)
        BASE="$HERE/cxl.base.json"
        OVERRIDE="$HERE/cxl.${NODE}.json"
        ;;
    nixl_peer)
        BASE="$HERE/nixl_peer.base.json"
        OVERRIDE="$HERE/nixl_peer.${NODE}.json"
        # Pin NIXL's UCX transport to the direct RDMA link (mlx5_0 port 1),
        # matching scripts/p2p. RC transport = reliable-connection RDMA.
        export UCX_TLS="${UCX_TLS:-rc}"
        export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1}"
        ;;
    *)
        echo "unknown L2 adapter: $L2 (expected cxl|nixl_peer)" >&2
        exit 2
        ;;
esac

L2_JSON="$(jq -c -s '.[0] * .[1]' "$BASE" "$OVERRIDE" \
    | sed -e "s/NODE_A_HOST/${NODE_A_HOST}/g" -e "s/NODE_B_HOST/${NODE_B_HOST}/g")"

# Stable Python hash seed: token hashing must be byte-identical across
# nodes for cross-node hits, and Python's default randomized hashing
# would change the chunk_hash between processes.
export PYTHONHASHSEED=0

# Unbuffered stdout/stderr so the tee'd log stays live (the `lmcache`
# console entry point can't take `python -u`).
export PYTHONUNBUFFERED=1

# Log to both the terminal and ${NODE}.log (e.g. node_a.log). Truncated
# on each run; `tee` (no -a) overwrites. PIPESTATUS preserves the
# server's exit code through the pipe so `set -e` still fails on crash.
LOG="$HERE/${NODE}-${L2}.log"

# --hash-algorithm builtin: produces 8-byte hashes (vs 32-byte blake3).
# The CXL donor recovers ObjectKey.chunk_hash bytes losslessly from the
# wire-form CacheEngineKey; with PYTHONHASHSEED=0 the builtin hash is
# deterministic across processes so both nodes agree on chunk identity.
# (The nixl_peer adapter carries the full hash on the wire and does not
# need 8-byte hashes, but builtin+seed keeps the two nodes consistent.)
#
# --l2-store-policy lazy: never proactively writes L1 -> L2. Chunks reach
# a peer only when that peer is asked for them (CXL push / RDMA pull).
#
# --l2-prefetch-policy retain: keep chunks pulled from a peer resident in
# L1 (the 'default' policy marks prefetched chunks temporary and drops
# them after the first read, so every repeat request would re-fetch over
# the network). With 'retain' a warm chunk is served from local L1 on
# subsequent requests.
lmcache server \
    --host localhost --port 5555 \
    --hash-algorithm builtin \
    --l1-size-gb 64 --eviction-policy LRU \
    --l2-store-policy lazy \
    --l2-prefetch-policy retain \
    --l2-adapter "$L2_JSON" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
