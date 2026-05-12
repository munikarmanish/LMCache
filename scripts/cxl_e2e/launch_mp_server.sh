#!/usr/bin/env bash
# Launch the LMCache MP server with a CXL L2 adapter.
#
# Usage:
#   NODE_A_HOST=10.0.0.1 NODE_B_HOST=10.0.0.2 \
#     sudo ./launch_mp_server.sh node_a    # initializer + lock manager
#   NODE_A_HOST=10.0.0.1 NODE_B_HOST=10.0.0.2 \
#     sudo ./launch_mp_server.sh node_b    # attacher
#
# The script merges l2_adapter.base.json with the per-node overrides
# (node_a.json or node_b.json) using jq, substitutes NODE_A_HOST /
# NODE_B_HOST placeholders, then passes the result as a single
# --l2-adapter JSON.
#
# Requires: jq, ~/.virtualenvs/lmcache/bin/python, NODE_A_HOST,
# NODE_B_HOST in env.

set -euo pipefail

NODE_A_HOST=192.168.128.31
NODE_B_HOST=192.168.128.32

NODE="${1:?usage: $0 node_a|node_b}"
HERE="$(cd "$(dirname "$0")" && pwd)"

: "${NODE_A_HOST:?NODE_A_HOST must be set (peer URL substitution)}"
: "${NODE_B_HOST:?NODE_B_HOST must be set (peer URL substitution)}"

L2_JSON="$(jq -c -s '.[0] * .[1]' "$HERE/l2_adapter.base.json" "$HERE/${NODE}.json" \
    | sed -e "s/NODE_A_HOST/${NODE_A_HOST}/g" -e "s/NODE_B_HOST/${NODE_B_HOST}/g")"

# Stable Python hash seed: token hashing must be byte-identical across
# nodes for cross-node CXL hits, and Python's default randomized
# hashing would change the chunk_hash u64 between processes.
export PYTHONHASHSEED=0

# --hash-algorithm builtin: produces 8-byte hashes (vs 32-byte blake3),
# which lets the donor side recover ObjectKey.chunk_hash bytes losslessly
# from the wire-form CacheEngineKey for L1 lookup. With PYTHONHASHSEED=0
# above, builtin hash is deterministic across processes.
#
# --store-policy lazy: never proactively writes L1 → L2 (CXL). Chunks
# reach CXL only when a peer issues PushKVToCXL.
python -m lmcache.v1.multiprocess.server \
    --host localhost --port 5555 \
    --hash-algorithm builtin \
    --l1-size-gb 16 --eviction-policy LRU \
    --store-policy lazy \
    --l2-adapter "$L2_JSON"
