#!/usr/bin/env bash
# lmc_bench.sh — Launch a single `lmcache bench engine` run against one node.
#
# Points the builtin engine benchmark at a node's vLLM endpoint and uses the
# node's LMCache MP server to auto-resolve tokens-per-GB (so no manual
# --tokens-per-gb-kvcache is needed). Workload parameters are kept in
# variables below so they are easy to tweak for a sweep.
#
# Usage:
#   ./lmc_bench.sh <host> <vllm_port> <lmc_port> [<seed>]
#
# Examples:
#   ./lmc_bench.sh 192.168.128.31 8010 9000        # default seed
#   ./lmc_bench.sh 192.168.128.31 8010 9000 1234   # reproducible seed
#
# Workload parameters are overridable via environment variables (each falls
# back to the default shown below). e.g. to sweep concurrency / doc length:
#   WORKLOAD=long-doc-qa DOCUMENT_LENGTH=8000 NUM_INFLIGHT_REQUESTS=16 \
#       ./lmc_bench.sh 192.168.128.31 8010 9000
#
# Overridable env vars (default):
#   WORKLOAD (long-doc-qa)          KV_CACHE_VOLUME (64)
#   DOCUMENT_LENGTH (30000)         QUERY_PER_DOCUMENT (2)
#   NUM_INFLIGHT_REQUESTS (1)
#   SHARED_PROMPT_LENGTH (2000)     CHAT_HISTORY_LENGTH (10000)
#   USER_INPUT_LENGTH (50)          OUTPUT_LENGTH (200)
#   QPS (1)                         DURATION (60)

set -euo pipefail

# ---------------------------------------------------------------------------
# Activate the lmcache virtualenv (puts the lmcache CLI on PATH).
# ---------------------------------------------------------------------------
VENV="$HOME/.virtualenvs/lmcache"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: lmcache venv not found at $VENV" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ---------------------------------------------------------------------------
# Required positional arguments.
# ---------------------------------------------------------------------------
HOST="${1:?usage: $0 <host> <vllm_port> <lmc_port> [<seed>]}"
VLLM_PORT="${2:?usage: $0 <host> <vllm_port> <lmc_port> [<seed>]}"
LMC_PORT="${3:?usage: $0 <host> <vllm_port> <lmc_port> [<seed>]}"
SEED="${4:-}"

# ---------------------------------------------------------------------------
# Benchmark parameters. Each is overridable via an environment variable of the
# same name; the value here is the default used when the var is unset.
# ---------------------------------------------------------------------------
WORKLOAD="${WORKLOAD:-long-doc-qa}"   # long-doc-qa | multi-round-chat
KV_CACHE_VOLUME="${KV_CACHE_VOLUME:-64}"

# long-doc-qa params
DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-30000}"
QUERY_PER_DOCUMENT="${QUERY_PER_DOCUMENT:-2}"
NUM_INFLIGHT_REQUESTS="${NUM_INFLIGHT_REQUESTS:-1}"

# multi-round-chat params
SHARED_PROMPT_LENGTH="${SHARED_PROMPT_LENGTH:-2000}"
CHAT_HISTORY_LENGTH="${CHAT_HISTORY_LENGTH:-10000}"
USER_INPUT_LENGTH="${USER_INPUT_LENGTH:-50}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-200}"
QPS="${QPS:-1}"
DURATION="${DURATION:-60}"

# Pass --seed only when one was given; otherwise let the CLI use its default.
SEED_ARG=()
if [[ -n "$SEED" ]]; then
    SEED_ARG=(--seed "$SEED")
fi

# ---------------------------------------------------------------------------
# Assemble workload-specific CLI arguments based on $WORKLOAD.
# Each workload exposes its own flag namespace (--ldqa-*, --mrc-*, ...); only
# the flags for the selected workload are passed.
# ---------------------------------------------------------------------------
WORKLOAD_ARGS=()
case "$WORKLOAD" in
    long-doc-qa)
        WORKLOAD_ARGS=(
            --ldqa-document-length "$DOCUMENT_LENGTH"
            --ldqa-query-per-document "$QUERY_PER_DOCUMENT"
            --ldqa-num-inflight-requests "$NUM_INFLIGHT_REQUESTS"
        )
        ;;
    multi-round-chat)
        WORKLOAD_ARGS=(
            --mrc-shared-prompt-length "$SHARED_PROMPT_LENGTH"
            --mrc-chat-history-length "$CHAT_HISTORY_LENGTH"
            --mrc-user-input-length "$USER_INPUT_LENGTH"
            --mrc-output-length "$OUTPUT_LENGTH"
            --mrc-qps "$QPS"
            --mrc-duration "$DURATION"
        )
        ;;
    *)
        echo "ERROR: unsupported WORKLOAD '$WORKLOAD' (expected: long-doc-qa, multi-round-chat)" >&2
        exit 1
        ;;
esac

# Result files live under results/.
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/results"

ENGINE_URL="http://${HOST}:${VLLM_PORT}"
LMCACHE_URL="http://${HOST}:${LMC_PORT}"

echo "[lmc_bench] engine=${ENGINE_URL} lmcache=${LMCACHE_URL} workload=${WORKLOAD}"

lmcache bench engine \
    --engine-url "$ENGINE_URL" \
    --lmcache-url "$LMCACHE_URL" \
    --kv-cache-volume "$KV_CACHE_VOLUME" \
    --workload "$WORKLOAD" \
    "${WORKLOAD_ARGS[@]}" \
    --no-interactive \
    --json \
    --output-dir "$HERE/results" \
    "${SEED_ARG[@]}"
