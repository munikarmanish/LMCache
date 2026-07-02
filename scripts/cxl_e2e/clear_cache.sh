#!/usr/bin/env bash
# clear_cache.sh — Clear the L1 (CPU/DRAM) KV cache on every LMCache MP server.
#
# POSTs /clear-cache to each node's LMCache MP HTTP server (:8090), which
# force-clears all KV objects in L1 — including any with active read/write
# locks, so only run this when the nodes are idle (in-flight store/prefetch
# ops may be corrupted). Useful between benchmark arms to start from a cold
# cache without restarting the stack.
#
# SCOPE: this clears L1 (DRAM) ONLY. The CXL/L2 shared pool is NOT — and
# CANNOT — be cleared this way:
#   - StorageManager.clear() (behind /clear-cache) only calls the L1 manager;
#     it never touches the L2 adapters.
#   - The CXL index has no bulk-clear primitive (only per-key delete()), and
#     the pool is a single shared /dev/dax mmap both nodes read/write, so a
#     live wipe from one node would corrupt the peer's in-flight reads.
#   To get a cold CXL pool, RESTART node0 with "initialize": true in its CXL
#   config (scripts/cxl_e2e/config/cxl.node0.json already sets this) — the
#   bootstrap then zeroes the header, region bitmap, descriptors, and index.
#   i.e. re-run: ./launch_node.sh 0 cxl   (node0 is the CXL initializer)
#
# Usage:
#   ./clear_cache.sh
#
# Override the target list with LMC_URLS (space- or comma-separated), e.g.:
#   LMC_URLS="http://c1:8090 http://c2:8090" ./clear_cache.sh

set -uo pipefail

# Static topology — must match launch_node.sh / launch_router.sh.
NODE0_HOST=192.168.128.31
NODE1_HOST=192.168.128.32
LMC_HTTP_PORT=8090

# Default targets = both nodes' MP HTTP servers. LMC_URLS overrides.
if [[ -n "${LMC_URLS:-}" ]]; then
    # Accept comma- or space-separated.
    read -r -a URLS <<< "${LMC_URLS//,/ }"
else
    URLS=(
        "http://${NODE0_HOST}:${LMC_HTTP_PORT}"
        "http://${NODE1_HOST}:${LMC_HTTP_PORT}"
    )
fi

rc=0
for base in "${URLS[@]}"; do
    base="${base%/}"
    url="${base}/clear-cache"
    printf '%-40s ' "clear L1: ${url}"
    # -f: fail (non-zero) on HTTP >=400; -s: silent; capture body for status.
    if body="$(curl -fsS -X POST "$url" 2>&1)"; then
        echo "OK  ${body}"
    else
        echo "FAILED  ${body}"
        rc=1
    fi
done

exit "$rc"
