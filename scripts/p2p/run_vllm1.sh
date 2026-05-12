#!/bin/bash

export PYTHONHASHSEED=0
export UCX_TLS=rc
export UCX_NET_DEVICES=mlx5_0:1
#export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1
#export UCX_MAX_RNDV_RAILS=2
export CUDA_VISIBLE_DEVICES=0
export LMCACHE_CONFIG_FILE=config1.yaml

# MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
PORT=8010

vllm serve "$MODEL" \
    --port "$PORT" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --dtype float16 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
