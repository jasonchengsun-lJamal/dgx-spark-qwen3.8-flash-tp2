# Dual DGX Spark (GB10/GX10) — Local TP2 Serving Guides

---

## 📚 本系列已发表文章 · Published Articles in This Series

| # | 文章 · Title | 链接 · Link |
|---|---|---|
| 1 | 双 DGX Spark 部署 Qwen3.8 Flash-Next TP2（部署指南 + 实测内存数据 + 崩溃复盘）| [jasonchengsun-lJamal/dgx-spark-qwen3.8-flash-tp2](https://github.com/jasonchengsun-lJamal/dgx-spark-qwen3.8-flash-tp2) |
| 2 | Windows Agent 终端弹窗四层消除方案 | [jasonchengsun-lJamal/windows-agent-popup-elimination](https://github.com/jasonchengsun-lJamal/windows-agent-popup-elimination) |
| 3 | 微信 PC 版数据导出：工具选型与实操步骤（中英文对照）| [jasonchengsun-lJamal/wechat-pc-data-export](https://github.com/jasonchengsun-lJamal/wechat-pc-data-export) |

---

Battle-tested deployment guides for running TP=2 multi-node inference across **two NVIDIA
DGX Spark** nodes (200 GbE interconnect), with measured memory numbers and real crash
postmortems. All data from production logs, `docker inspect`, and vLLM `/metrics`.

## 📄 Guides

| Guide | Covers |
|---|---|
| [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) | **Qwen3.8-Flash-Next-NVFP4** TP2 (the one we run in production: util 0.85, seqs 8→32, KV max_concurrency 42.4, 5 TP2 traps, startup timing) |
| [`DEEPSEEK_TP2_GUIDE.md`](DEEPSEEK_TP2_GUIDE.md) | **DeepSeek-V4-Flash-DSpark** TP2 (util 0.82 only — 0.78/0.70 crash postmortems, dspark-vllm-gx10 head.env, cron-storm overload chain, ops notes) |

## Scripts

- `scripts/qwen38_tp2_rank0_start.sh` — Qwen3.8 rank0 (head)
- `scripts/qwen38_tp2_rank1_start.sh` — Qwen3.8 rank1 (worker, `--headless`)

## TL;DR

- **Qwen3.8 Flash-Next**: 61.73 GiB/rank NVFP4 → `gpu-memory-utilization 0.85`, seqs **32**,
  KV max_concurrency **42.4**, short-request latency ~0.5 s. Dedicated dev image
  `vllm/vllm-openai:qwen38-flash-next` (not PyPI), PLE patches, rank1 `--headless`.
- **DeepSeek-V4-Flash**: 79.17 GiB/rank → **0.82 is the only working util** (0.78 = KV alloc
  failure: 6.92 < 7.8 GiB; 0.70 = weights don't fit). seqs 8, max-model-len 262144,
  batched 16384, MTP 5, API served by rank0 only.
- **Both**: NCCL must use the 200G NIC; head before worker; cold boot >10 min (weight load
  ~397 s + AOT/JIT compile 5–10 min); per-model `max-num-batched-tokens` (dense 8192 /
  MoE 16384); cap parallel cron jobs to avoid stampede.

---

*Measured 2026-08-31 on Spark1 (.59) + Spark2 (.226) + Spark3 (.209).*
