#!/bin/bash
#
# Usage: ./run_bench.sh <label> [<seed>]
#
# <seed> is optional; when given it is passed to bench_ttft.py so the
# generated prompts are reproducible. When omitted, bench_ttft.py uses the
# current timestamp (and prints the chosen seed so a run can be replayed).

HOST_A=192.168.128.31
HOST_B=192.168.128.32

# get the required argument
LABEL="${1:?usage: $0 <label> [<seed>]}"
SEED="${2:-}"

# Pass --seed only when a seed was given; otherwise let bench_ttft.py
# default to the current timestamp.
SEED_ARG=()
if [[ -n "$SEED" ]]; then
    SEED_ARG=(--seed "$SEED")
fi

python bench_ttft.py \
        --node-a-url "http://${HOST_A}:8010" \
        --node-b-url "http://${HOST_B}:8010" \
        --prompt-tokens 1000 3000 9000 27000 \
        --repeat 5 \
        --label "$LABEL" \
        "${SEED_ARG[@]}" \
        --out "results-$LABEL.csv"
