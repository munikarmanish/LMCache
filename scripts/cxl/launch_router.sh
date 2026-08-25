#!/usr/bin/env bash
# launch_router.sh — Bring up the lightweight KV-aware router in front of the
# two nodes (run this on c1/node0 after both nodes' launch_node.sh are up).
#
# The router forwards OpenAI requests to one of the two vLLM endpoints per a
# selectable strategy (round_robin | random | gpu_load | max_prefix |
# weighted | anti_affinity). random supports optional per-node sampling
# weights. It reads the KV-prefix signal from each node's LMCache
# /lookup_hits (:8090) and the GPU-load signal from each node's vLLM /metrics
# (:8010). See router.py for the strategy details.
#
# Usage:
#   ./launch_router.sh [strategy] [w_prefix] [w_load] [weights]
#
# Examples:
#   ./launch_router.sh                       # round_robin (default)
#   ./launch_router.sh max_prefix
#   ./launch_router.sh weighted 0.7 0.3
#   ./launch_router.sh random                # uniform random
#   ./launch_router.sh random - - c1=0.7,c2=0.3   # 70/30 weighted random
#
# For the random strategy, the 4th arg is the per-node --weights list
# (name=weight,...); every node must be listed. Pass "-" for w_prefix/w_load
# to skip them (they only matter for the weighted strategy).

set -euo pipefail

VENV="$HOME/.virtualenvs/lmcache"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

HERE="$(cd "$(dirname "$0")" && pwd)"

# Static topology — must match launch_node.sh.
NODE0_HOST=192.168.128.31
NODE1_HOST=192.168.128.32
VLLM_PORT=8010     # vLLM OpenAI API (serve + /metrics)
LMC_HTTP_PORT=8090 # LMCache MP HTTP (/lookup_hits)
MODEL="meta-llama/Llama-3.1-8B-Instruct"
ROUTER_PORT=8000   # the router's public OpenAI endpoint

STRATEGY="${1:-round_robin}"
W_PREFIX="${2:-0.7}"
W_LOAD="${3:-0.3}"
WEIGHTS="${4:-}"  # random strategy: name=weight,... (empty => uniform)

# Allow "-" as a placeholder for the weighted-only args so `random` callers can
# reach the 4th positional (weights) without setting them.
[ "$W_PREFIX" = "-" ] && W_PREFIX=0.7
[ "$W_LOAD" = "-" ] && W_LOAD=0.3

NODES="c1=http://${NODE0_HOST}:${VLLM_PORT},c2=http://${NODE1_HOST}:${VLLM_PORT}"
LOOKUP="c1=http://${NODE0_HOST}:${LMC_HTTP_PORT},c2=http://${NODE1_HOST}:${LMC_HTTP_PORT}"

LOG_LEVEL=debug  # debug | info | warning | error
LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"

echo "Starting router: strategy=$STRATEGY on :$ROUTER_PORT"
echo "  nodes : $NODES"
echo "  lookup: $LOOKUP"
[ -n "$WEIGHTS" ] && echo "  weights: $WEIGHTS"

# Pass --weights only when set, so non-random strategies don't warn about an
# ignored flag.
WEIGHTS_ARG=()
[ -n "$WEIGHTS" ] && WEIGHTS_ARG=(--weights "$WEIGHTS")

export PYTHONHASHSEED=0  # deterministic builtin hash for LMCache chunk identity
python "$HERE/router.py" \
    --nodes "$NODES" \
    --lookup "$LOOKUP" \
    --model "$MODEL" \
    --strategy "$STRATEGY" \
    --w-prefix "$W_PREFIX" \
    --w-load "$W_LOAD" \
    "${WEIGHTS_ARG[@]}" \
    --port "$ROUTER_PORT" \
    --host 0.0.0.0 \
    --log-level "$LOG_LEVEL" \
    2>&1 | tee "$LOG_DIR/router.log"
