# Dual DGX Spark (GX10) Qwen3.8 Flash-Next TP2 Deployment Guide

A battle-tested field guide for running **Qwen3.8-Flash-Next-NVFP4** across **two NVIDIA DGX Spark (GB10/GX10) nodes** with vLLM TP=2 multi-node serving — including why we left DeepSeek-V4-Flash (unified-memory pressure), measured memory numbers, the 5 TP2 traps, and 3 memory traps.

## Highlights

- **Why**: DeepSeek-V4-Flash TP2 on DGX Spark leaves only ~1 GB free (weights 79.17 GiB); gpu-memory-utilization **0.78 fails to boot**, only **0.82** works. Qwen3.8 Flash-Next (3B-active MoE, 61.73 GiB) runs at **0.85 stable**, raising concurrency from **seqs 8 → 32** on the same hardware.
- **Measured numbers**: KV cache 2.78M tokens, theoretical concurrency 42.4, short-request latency ~0.5 s.
- **Pitfalls covered**: NCCL NIC selection, fp8 KV cache, PLE layer patches, `--headless` on rank1, head-first start order, gpu-memory-utilization sensitivity, UMA OOM, per-model concurrency tuning, safe restarts.

## Files

- `DEPLOYMENT_GUIDE.md` — full Qwen3.8 Flash-Next TP2 deployment guide (scripts included)
- `DEEPSEEK_TP2_NOTES.md` — DeepSeek-V4-Flash TP2 experience + crash postmortems (0.78 KV allocation failure, cron overload chain), per-model concurrency data
- `scripts/qwen38_tp2_rank0_start.sh` — rank0 (head) startup script
- `scripts/qwen38_tp2_rank1_start.sh` — rank1 (worker, `--headless`) startup script

## Reproduce

Two DGX Sparks linked by 200GbE, each with a local copy of `/data/models/Qwen3.8-Flash-Next-NVFP4`, dedicated image `vllm/vllm-openai:qwen38-flash-next`, PLE patches bind-mounted, head on Spark1 (rank0) then worker on Spark2 (rank1 with `--headless`). Full scripts in the guide.

---

*Measured 2026-08-31 on three machines (Spark1 .59 / Spark2 .226 / Spark3 .209). All numbers from real deployment logs and vLLM /metrics.*
