#!/usr/bin/env bash
# Launch a vLLM serving Llama-3.1-8B-Instruct, talking to the local
# LMCache MP server over tcp://localhost:5555.
#
# Usage:
#   ./launch_vllm.sh <port> [<cuda-visible-device>]
#
# Examples:
#   ./launch_vllm.sh 8010        # GPU 0, port 8010
#   ./launch_vllm.sh 8010 1      # GPU 1, port 8010 (multi-GPU node)

set -euo pipefail

NODE_A_HOST=192.168.128.31
NODE_B_HOST=192.168.128.32

PORT="${1:?usage: $0 <port> [<cuda-visible-device>]}"
GPU="${2:-0}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# vLLM-side LMCache engine config (chunk size etc.).
export LMCACHE_CONFIG_FILE="$HERE/lmcache.yaml"

# Stable Python hash seed: token hashing must be byte-identical across
# nodes for cross-node CXL hits, and Python's default randomized
# hashing would change the chunk_hash u64 between processes.
export PYTHONHASHSEED=0

MODEL="meta-llama/Llama-3.1-8B-Instruct"

# kv_connector_extra_config tells vLLM where the MP server lives.
KV_CFG='{
  "kv_connector": "LMCacheMPConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://localhost",
    "lmcache.mp.port": 5555
  }
}'

# Log to both the terminal and vllm.log (truncated each run via `tee`,
# no -a). PIPESTATUS preserves vLLM's exit code through the pipe so
# `set -e` still fails if it crashes.
LOG="$HERE/vllm.log"

# --disable-log-stats: silence vLLM's periodic throughput log lines
# ("Avg prompt throughput: ... Avg generation throughput: ...").
CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" \
    --kv-transfer-config "$KV_CFG" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --gpu-memory-utilization 0.8 \
    --dtype float16 \
    --disable-log-stats \
    --port "$PORT" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
