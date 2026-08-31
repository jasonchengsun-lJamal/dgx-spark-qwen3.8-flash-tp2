#!/bin/bash
# qwen38_tp2_rank0_start.sh — Spark1 (rank0/head)
# Qwen3.8-Flash-Next-NVFP4 TP2 dual-DGX-Spark startup
# Dedicated image: vllm/vllm-openai:qwen38-flash-next (dev build, NOT PyPI)
set -x
exec > /home/jamal_spark/qwen38_tp2_rank0.log 2>&1

MODEL=/data/models/Qwen3.8-Flash-Next-NVFP4
PORT=8004
MASTER_ADDR=192.168.100.10     # 200GbE NIC IP — see Pitfall #1
MASTER_PORT=25004

export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0
export TORCHINDUCTOR_AUTOTUNE_AT_COMPILE_TIME=0

docker run --rm --gpus all --ipc=host \
  --network host \
  -v ${MODEL}:/model \
  -v /tmp/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -v /tmp/standalone_compile_patched.py:/usr/local/lib/python3.12/dist-packages/torch/_inductor/standalone_compile.py:ro \
  -e HF_HOME=/root/.cache/huggingface \
  -e NCCL_DEBUG=INFO \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_DISABLE=1 \
  -e VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0 \
  -e TORCHINDUCTOR_AUTOTUNE_AT_COMPILE_TIME=0 \
  vllm/vllm-openai:qwen38-flash-next \
  --model /model \
  --served-model-name qwen3.8-flash-next \
  --host 0.0.0.0 --port ${PORT} \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --nnodes 2 --node-rank 0 \
  --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT} \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --max-num-seqs 32 \
  --no-enable-flashinfer-autotune \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml 2>&1
