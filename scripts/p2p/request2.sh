#!/bin/bash

HOST="192.168.128.32"
PORT="8010"
#MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS=50
TEMPERATURE=0.5
PROMPT="$(printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..120})"

http POST $HOST:$PORT/v1/completions \
    model="$MODEL" \
    prompt="$PROMPT" \
    max_tokens:=$MAX_TOKENS \
    temperature:=$TEMPERATURE
