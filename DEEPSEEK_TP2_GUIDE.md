# DeepSeek-V4-Flash TP2 Deployment Guide on Dual DGX Spark (GB10/GX10)

> Full deployment guide for **DeepSeek-V4-Flash-DSpark** (MoE, abliterated variant) served
> with vLLM **TP=2 across two NVIDIA DGX Spark** nodes over a 200 GbE interconnect.
> Every number below was measured on real hardware (Aug 2026), including two crash
> postmortems at the end. Companion repo: `dgx-spark-qwen3.8-flash-tp2`.

## 1. Why TP2 on DGX Spark

A single DGX Spark (128 GB unified memory, ~1 PFLOPS FP4) cannot hold DeepSeek-V4-Flash
plus a usable KV cache alone — the weights alone are **79.17 GiB per rank**. Splitting the
model across two Sparks with tensor parallelism (TP=2) halves per-rank weight memory and
doubles KV capacity and throughput.

| Item | Value (measured) |
|---|---|
| Weights (per rank) | 79.17 GiB (NVMe, local on each node) |
| gpu-memory-utilization | **0.82** (only working value — see crash #1) |
| max-model-len | 262144 |
| max-num-seqs | 8 |
| max-num-batched-tokens | 16384 |
| MTP (multi-token prediction) | 5 tokens |
| KV max_concurrency | ~40 (from /metrics) |
| API | rank0 only (rank1 does NOT listen on 8000) |

## 2. Hardware & topology

```
Spark1 (rank0/head)  192.168.1.59  100G iface: 192.168.100.10  user: jamal_spark
Spark2 (rank1/worker) 192.168.1.226 100G iface: 192.168.100.11  user: js_spark2
Interconnect: 200 GbE copper (NCCL uses enp1s0f0np0 on the 100.10/11 net)
```

- Both nodes keep a **local** copy of the weights at `/data/models/DeepSeek-V4-Flash-DSpark`
  (vLLM always reads local NVMe; never NFS-mount weights for vLLM).
- The 200 GbE NIC must be selected explicitly for NCCL/GLOO/TP (see §4.1).

## 3. What the official dspack launcher provides

We use the upstream **`dspark-vllm-gx10`** repo (version `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`)
rather than hand-rolling docker run. All serving params live in one file:

**`config/head.env`** (head node, Spark1):

```bash
NODE_RANK=0
HEADLESS=                    # empty on head
MASTER_ADDR=192.168.100.10   # 100G interconnect IP of head
MASTER_PORT=25000
VLLM_HOST_IP=192.168.100.10

# NCCL / TP transport — must point at the 200G NIC, NOT the 1G management NIC
NCCL_IB_HCA=rocep1s0f0
NCCL_SOCKET_IFNAME=enp1s0f0np0
TP_SOCKET_IFNAME=enp1s0f0np0
GLOO_SOCKET_IFNAME=enp1s0f0np0

DSPARK_VLLM_IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
DSPARK_MODEL_HOST=/data/models/DeepSeek-V4-Flash-DSpark
HF_CACHE=/home/jamal_spark/.cache/huggingface
DSPARK_TMP_HOST=/var/lib/dspark-tmp
SERVED_MODEL_NAME="deepseek-v4-flash-dspark-abliterated /data/models/DeepSeek-V4-Flash-DSpark deepseek-v4-flash"

MAX_MODEL_LEN=262144
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=16384
GPU_MEMORY_UTILIZATION=0.82
MTP_NUM_TOKENS=5
JIT_MONITOR_MODE=warn

# Worker (rank1) — reached over the 100G net
WORKER_SSH=js_spark2@192.168.100.11
WORKER_REPO_DIR=/home/js_spark2/dspark-vllm-gx10

VLLM_PORT=8000
VLLM_HOST=0.0.0.0
```

Start with the launcher script (head first, worker second — the script handles it):

```bash
cd ~/dspark-vllm-gx10
./start-node.sh        # launches rank0 (head) then rank1 (worker) via WORKER_SSH
```

## 4. Pitfalls (measured)

### 4.1 NCCL must use the 200G NIC
Default NIC discovery picks the wrong interface → single-digit GB/s. The five env vars in
`head.env` (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`,
`VLLM_HOST_IP`) must all point at the interconnect net (192.168.100.x / enp1s0f0np0).

### 4.2 gpu-memory-utilization is a knife edge — 0.82 ONLY
See crash #1. 0.78 fails KV allocation, 0.70 doesn't fit weights, 0.82 boots. Do NOT
"tune" it by trial — read the KV numbers in the log and adjust by small steps.

### 4.3 rank1 does not serve the API
Only rank0 listens on 8000. If your health check on Spark2's 8000 fails, that is NORMAL.
Check rank0.

### 4.4 Head must start before worker
Starting the worker first gives `Connection reset` at rendezvous. `start-node.sh` handles
order; if you hand-roll, wait ~20 s between ranks.

### 4.5 Cold boot is SLOW
Multi-node weight load + first-boot AOT/JIT compile takes **>10 min** (rank1 weight load
alone was 397 s for 61.73 GiB; rank0 shards ~96% at 7 min; AOT compile adds 5–10 min).
A 300 s health-check timeout will false-negative. Poll the real `/metrics` or `/health`.

### 4.6 Both ranks' flags must match
A mismatch (e.g. seqs 8 vs 32, or different max-model-len) fails rendezvous or silently
misbehaves. Keep `head.env` in sync on both nodes (worker's copy is used only as fallback;
the head pushes the real config via WORKER_SSH).

## 5. Verify

```bash
# On Spark1 (rank0):
curl -s http://192.168.1.59:8000/v1/models          # served_model_name == deepseek-v4-flash-dspark-abliterated
curl -s http://192.168.1.59:8000/metrics | grep -E 'kv_cache_max_concurrency|running' | head
```

Expected: model listed; `kv_cache_max_concurrency` ≈ 40 at seqs 8; short request latency
~0.5–1 s; long-context (60–95k tokens) prefill takes tens of seconds — budget for it.

## 6. Crash postmortems

### Crash #1: gpu-memory-utilization 0.78 → KV cache allocation failure
```
Failed to allocate KV cache ... available 6.92 GiB < required 7.8 GiB
```
- 79.17 GiB weights consume the unified-memory pool; at util=0.78 only ~6.92 GiB remains
  for KV, but the model needs ≥7.8 GiB at the configured max-model-len.
- At 0.70, weights + base config don't even fit → earlier boot failure.
- **Lowering util never helps on unified memory — it only shrinks the pool you can't fit
  into. 0.82 is the smallest value that works.**

### Crash #2: cron storm → 180 s timeouts ×3 (overload chain)
1. Multiple scheduled jobs fire the same minute, each with 60–95k-token contexts.
2. Prefill splits into dozens of batches (max-num-batched-tokens 16384) → one request
   takes **79+ s** of prefill.
3. Requests queue and contend → every job exceeds its 180 s timeout; each retry adds
   another long-context request → self-amplifying storm (7 failed fires in one batch,
   `running=4, waiting=0`).
Fixes (both required):
- vLLM: align `MAX_NUM_BATCHED_TOKENS` per architecture — dense 27B tops at 8192
  (16384 → KV allocation failure), MoE 35B handles 16384.
- Scheduler: cap concurrent jobs (`cron.max_parallel_jobs=2`, read live each tick — no
  restart needed) so the model can't be stampeded again.

## 7. Ops notes

- Restart requires a >5 min window (container restart); drain requests first (`running=0`).
- Do not restart while long tasks (report jobs, downloads) are mid-flight.
- Weight sync between nodes: rsync `--checksum` over the 200G copper (never NFS for vLLM
  reads). Spark1 is the version source.
- We later replaced DeepSeek with **Qwen3.8-Flash-Next** (61.73 GiB NVFP4, runs at
  util=0.85, seqs 8→32, KV max_concurrency 42.4) on the same two boxes — see
  `DEPLOYMENT_GUIDE.md` in this repo.

---

*Measured 2026-08-31 on Spark1 (.59) + Spark2 (.226) + Spark3 (.209). Numbers from real
deployment logs, docker inspect, and vLLM /metrics.*
