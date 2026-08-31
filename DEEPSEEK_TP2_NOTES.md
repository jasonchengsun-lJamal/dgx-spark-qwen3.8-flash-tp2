# DeepSeek-V4-Flash TP2 on Dual DGX Spark: Deployment Notes & Crash Postmortem

> Companion to the Qwen3.8 Flash-Next guide. Everything here was **measured on real hardware**
> (two NVIDIA DGX Spark / GB10-GX10, 200 GbE interconnect) during Aug 2026 — including two
> crash cases we lived through so you don't have to.

## Why we moved away from DeepSeek on this hardware

DeepSeek-V4-Flash-DSpark (MoE, full-precision weights) leaves almost **zero headroom** on a
DGX Spark's 128 GB unified memory:

| Model | Weights (rank-local) | gpu-memory-utilization | Boot result |
|---|---|---|---|
| DeepSeek-V4-Flash | **79.17 GiB** | 0.70 | ❌ fails — cannot even fit weights + base config |
| DeepSeek-V4-Flash | **79.17 GiB** | 0.78 | ❌ fails — KV cache allocation error (see crash #1) |
| DeepSeek-V4-Flash | **79.17 GiB** | **0.82** | ✅ boots — the only working value |
| Qwen3.8-Flash-Next | 61.73 GiB (NVFP4) | **0.85** | ✅ boots with headroom, concurrency 42.4 |

The takeaway: **gpu-memory-utilization is model-specific and must be measured per model.**
There is no universal "safe" value. On this hardware DeepSeek can only use 0.82 while the
lighter NVFP4 Qwen3.8 runs at 0.85 — and trying to "save memory" by lowering it (0.78)
actually *breaks* boot instead of helping.

## Crash case #1: gpu-memory-utilization 0.78 → KV cache allocation failure

Symptom: vLLM exits during startup with a KV cache allocation error:

```
Failed to allocate KV cache ... kv_cache_size_tokens ... available 6.92 GiB < required 7.8 GiB
```

Root cause chain:
- 79.17 GiB weights are loaded into the unified memory pool (GPU + CPU share the same 128 GB)
- At util=0.78, the *reserved* pool after weights leaves only ~6.92 GiB for the KV cache
- The model's KV cache needs ~7.8 GiB minimum at the configured max-model-len
- 6.92 < 7.8 → hard failure, no fallback

At util=0.70 it is even worse: weights (79.17 GiB) + base config don't even fit the
reserved pool, so boot dies earlier. **Lowering the utilization number never helps on this
hardware — it only shrinks the pool you can't fit into.** The fix was raising it to 0.82,
the smallest value where everything fits.

## Crash case #2: DeepSeek under cron load — 180s timeouts ×3 (the "overload chain")

Symptom: repeated 180s request timeouts, several scheduled jobs all failing around the same
minute, and the model looked "healthy" (port up, single request fast).

Root cause chain (all measured):
1. Several cron jobs fire at the same wall-clock minute (report generators, watchdogs, etc.)
2. Each carries **long contexts** (60–95k tokens) → prefill is split into **dozens of
   batches** because max-num-batched-tokens is small
3. One long-context request takes **79+ seconds** of prefill
4. Multiple such requests queue and contend → every job exceeds its 180s timeout
5. Each retry adds *another* long-context request → self-amplifying retry storm

Measured at peak: `running=4, waiting=0` on the single Qwen endpoint, with 7 failed cron
fires in one batch.

Fixes that actually worked (two-sided, both required):
- **vLLM side**: align `--max-num-batched-tokens` to the model's architecture — dense 27B
  tops out at **8192** (16384 → KV allocation failure), MoE 35B takes **16384**. Don't use
  one value for all models.
- **Scheduler side**: cap parallel cron jobs (`cron.max_parallel_jobs=2`) so the model can
  never be stampeded again. This is read live from config on every scheduler tick — no
  restart needed.

## Per-model concurrency tuning (measured)

| Model | arch | max-num-seqs | batched tokens | KV max_concurrency | latency (short req) |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash | MoE (dense-ish activ.) | 8 | 16384 | ~40 | — |
| Qwen3.8-27B | dense | 8 | **8192** | — | — |
| Qwen3.6-35B-A3B | MoE | 8 | **16384** | — | — |
| Qwen3.8-Flash-Next | 3B-active MoE | **32** | 65536 ctx | **42.4** | ~0.5 s |

Why Qwen3.8 Flash-Next can run 4× seqs on the same two boxes: its KV cache is tiny
(`kv_cache_size_tokens=2,739,548`, `kv_cache_usage_perc=0.0` at seqs=8). KV
max_concurrency read from `/metrics` is the real ceiling — use it, not guesses.

## TP2 multi-node traps (both models)

1. **NCCL NIC selection** — on dual-DGX-Spark the 200 GbE interconnect is a separate NIC;
   if NCCL picks the wrong one you get single-digit GB/s. Pin it:
   `NCCL_SOCKET_IFNAME=<100G-iface>`.
2. **fp8 KV cache** — do not enable fp8 KV for these models on this stack; it breaks or
   degrades (vLLM dev image).
3. **PLE layer patches** — the Qwen3.8 Flash-Next weights need the PLE (Post-Layer
   Embedding) patch set bind-mounted into the container. Not a PyPI vLLM — use the
   dedicated dev image `vllm/vllm-openai:qwen38-flash-next`.
4. **rank1 `--headless`** — the worker node MUST be started with `--headless` or it tries
   to bind a control plane the TP2 setup doesn't have. Symptom: rank1 hangs at startup.
5. **Head first, worker second** — rank0 (head) first, wait ~20 s, then rank1 (worker).
   Starting the worker first gives `Connection reset` during rendezvous.

## Startup timing reality (multi-node is SLOW)

| Stage | Single node | Dual node (measured) |
|---|---|---|
| Weight load rank1 | — | **61.73 GiB / 397 s** |
| Weight load rank0 | — | 198/206 shards ≈ 96% at ~7 min |
| AOT compile (first boot) | — | **5–10 min** |
| Total to ready | ~380 s | **>10 min** |

Health-check timeouts of 300 s WILL false-negative on a cold TP2 boot. Use ≥15 min or poll
the actual `/metrics` / `ready` endpoint.

## Operational notes

- rank0 serves the API; on the DeepSeek setup **rank1 (Spark2) does not listen on 8000** —
  that is normal, the API is served by rank0 only.
- Restarts need a >5 min window (vLLM container restart) — drain active requests first
  (`running=0`).
- Before any restart, make sure no long task is mid-flight (report jobs, downloads).
- Both ranks MUST run identical flags; a mismatch (e.g. seqs 8 vs 32) fails rendezvous.

---

*Measured 2026-08-31 on Spark1 (.59) + Spark2 (.226) + Spark3 (.209). Numbers from real
deployment logs, docker inspect, and vLLM /metrics.*
