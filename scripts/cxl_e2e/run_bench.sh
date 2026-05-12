#!/bin/bash
#
# Usage: ./run_bench.sh <label>

HOST_A=192.168.128.31
HOST_B=192.168.128.32

# get the required argument
LABEL="${1:?usage: $0 <label>}"

python bench_ttft.py \
        --node-a-url "http://${HOST_A}:8010" \
        --node-b-url "http://${HOST_B}:8010" \
        --prompt-tokens 1000 2000 4000 8000 \
        --repeat 3 \
        --label "$LABEL" \
        --out "results-$LABEL.csv"
