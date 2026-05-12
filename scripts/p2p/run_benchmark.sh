#!/bin/bash

TS="$(date +%Y%m%d-%H%M%S)"

HOST1="192.168.128.31"
HOST2="192.168.128.32"
PORT="8010"

LOG1="/tmp/c1_${TS}.log"
LOG2="/tmp/c2_${TS}.log"

# MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

# BENCHMARK_SCRIPT="/home/manish/code/LMCache/benchmarks/long_doc_qa/long_doc_qa.py"
BENCHMARK_SCRIPT="/home/manish/code/test_lmcache_p2p/long_doc_qa.py" # with p99 ttft calculation

NUM_DOCS=50
DOC_LEN=10000
OUTPUT_LEN=100
MAX_INFLIGHT=4
REPEAT_COUNT=1
REPEAT_MODE="tile"

run_benchmark() {
    local host="$1"
    local port="$2"
    local log="$3"

    python "${BENCHMARK_SCRIPT}" \
        --model "${MODEL}" \
        --host "${host}" \
        --port "${PORT}" \
        --num-documents "${NUM_DOCS}" \
        --document-length "${DOC_LEN}" \
        --output-len "${OUTPUT_LEN}" \
        --repeat-count "${REPEAT_COUNT}" \
        --repeat-mode "${REPEAT_MODE}" \
        --max-inflight-requests "${MAX_INFLIGHT}" \
        2>&1 | tee "${log}"
}

run_benchmark "${HOST1}" "${PORT}" "${LOG1}"

echo "Sleeping for 5 seconds before starting the quering c2..."
sleep 5

run_benchmark "${HOST2}" "${PORT}" "${LOG2}"
