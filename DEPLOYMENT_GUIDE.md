# Deploying Qwen3.8 Flash-Next on Dual DGX Spark (GB10/GX10): A Field Guide with Pitfalls

> By Sun Cheng's team, 2026-08-31 — battle-tested on real hardware
> Scenario: Two NVIDIA DGX Spark (GB10/GX10) nodes linked by a 200GbE cable, running vLLM in TP=2 multi-node mode to serve Qwen3.8-Flash-Next-NVFP4 locally — replacing the previous DeepSeek-V4-Flash TP2 setup.

---

## 1. Backstory: Why We Moved from DeepSeek to Qwen3.8 Flash

### 1.1 The DeepSeek TP2 Pain: Extremely Tight Memory, Crash-Prone

Each DGX Spark (GX10) has 128 GB of unified memory (LPDDR5X, shared between CPU and GPU; `free -g` reports ~121 GB usable). DeepSeek-V4-Flash weights alone consume **79.17 GiB**; add KV cache, activations, CUDA context, and JIT compile cache and you are left with **~1 GB free at idle** (measured: total 121 / used 119 / free 1).

Three direct consequences:

1. **`gpu-memory-utilization` is extremely sensitive — anything below 0.78 simply fails to start**
   - At `0.78`: KV cache available = 6.92 GiB < 7.8 GiB required → **vLLM Engine init failed**, container exits with status 1.
   - At `0.70`: not enough memory for the 79.17 GiB weights plus base overhead → **fails at startup**.
   - The only viable value: **0.82** (KV cache barely fits, with razor-thin headroom).
2. **Concurrency must be kept very low**: `MAX_NUM_SEQS=8`, `MAX_NUM_BATCHED_TOKENS=16384`. A few long-context requests can blow the memory budget.
3. **OOM red line**: under unified memory, CPU processes (e.g. resident agents at ~3.2 GB each) directly squeeze vLLM's KV cache → inference OOM crashes.

### 1.2 Why Qwen3.8-Flash-Next Is a Better Fit

Qwen3.8-Flash-Next is a **3B-active MoE model** (NVFP4-quantized weights ≈ 61.7 GiB, halved across the two TP2 nodes). Its memory footprint is far smaller than DeepSeek's:

| Metric | DeepSeek-V4-Flash (TP2) | Qwen3.8-Flash-Next-NVFP4 (TP2) |
|:--|:--|:--|
| Weights loaded | 79.17 GiB/node | 61.73 GiB/node (measured from logs) |
| gpu-memory-utilization | **0.82 only** (fails below 0.78) | **0.85 stable** |
| Free memory at idle | ~1 GB (OOM-prone) | comfortable |
| KV cache capacity | tight | 2.78M tokens, theoretical concurrency 42+ |
| max-num-seqs | 8 (anything higher blows up) | **32 stable** (4×+ KV headroom) |

Bottom line: **on the same two Sparks, switching to Qwen3.8 Flash raises concurrency from 8 to 32 while dramatically lowering crash risk.**

---

## 2. Hardware & Network Topology

```
┌─────────────────┐      200GbE (192.168.100.x)      ┌─────────────────┐
│  Spark1 (rank0) │ ◄══════════════════════════════► │  Spark2 (rank1) │
│  .59 / 100.10   │                                 │  .226 / 100.11  │
│  vLLM head      │                                 │  vLLM worker    │
└─────────────────┘                                 └─────────────────┘
```

- **Nodes**: Spark1 (rank0/head), Spark2 (rank1/worker), each 128 GB unified memory
- **Interconnect**: 200GbE direct cable; NCCL must use this NIC (see Pitfall #1)
- **Model**: `/data/models/Qwen3.8-Flash-Next-NVFP4` (a local copy on each node; vLLM always reads local disk)
- **Serving port**: 8004 (same port on both nodes — TP2 semantics)

---

## 3. Deployment Steps

### 3.0 Prerequisite: Special Docker Image (Critical!)

Qwen3.8-Flash-Next NVFP4 weights need a **dev build of vLLM**; the PyPI vLLM does NOT support this model:

```bash
docker pull vllm/vllm-openai:qwen38-flash-next   # dedicated image, NOT the PyPI build
```

### 3.1 Prepare the PLE Patches (Critical! Pitfall #3)

The Flash-Next architecture's `ple_layer` and `standalone_compile` need manual patches or inference errors:

```bash
# /tmp/ple_layer_patched.py and /tmp/standalone_compile_patched.py
# Must be patched for the matching vLLM version, then bind-mounted into the
# container as read-only (:ro)
```

### 3.2 rank0 (Spark1) Start Script

```bash
#!/bin/bash
# qwen38_tp2_rank0_start.sh
set -x
exec > /home/jamal_spark/qwen38_tp2_rank0.log 2>&1

MODEL=/data/models/Qwen3.8-Flash-Next-NVFP4
PORT=8004
MASTER_ADDR=192.168.100.10     # 200GbE IP (Pitfall #1)
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
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \   # Pitfall #1: 200GbE NIC name
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
```

### 3.3 rank1 (Spark2) Start Script

Identical to rank0, with three differences:
- `--node-rank 1`
- **Must add `--headless`** (Pitfall #4)
- Log file `qwen38_tp2_rank1.log`

### 3.4 Start Order (Pitfall #5: head first, worker second)

```bash
# Start head on Spark1 first
ssh jamal_spark@192.168.1.59 "nohup bash ~/qwen38_tp2_rank0_start.sh > /dev/null 2>&1 &"
sleep 20   # give the head a head start

# Then start worker on Spark2
ssh js_spark2@192.168.1.226 "nohup bash ~/qwen38_tp2_rank1_start.sh > /dev/null 2>&1 &"
```

**Worker-first ⇒ NCCL Connection reset. Guaranteed.**

### 3.5 Verification

```bash
# Model list (first load takes 6–10 minutes, includes AOT compilation)
curl http://127.0.0.1:8004/v1/models
# → {"id":"qwen3.8-flash-next"}

# KV cache headroom (the key metric for choosing max-num-seqs)
curl http://127.0.0.1:8004/metrics | grep kv_cache_max_concurrency
# → kv_cache_max_concurrency="42.4"  ← theoretical concurrency ceiling

# Real inference smoke test
curl http://127.0.0.1:8004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"reply OK"}],"max_tokens":10}'
```

---

## 4. Pitfall Checklist (Five TP2 Traps + Three Memory Traps)

### Pitfall 1: NCCL Picks the Wrong NIC → Connection Failure / Extremely Slow
DGX Spark has multiple NICs (1 GbE management + 200GbE cable). NCCL may pick the wrong one by default.
**Fix**: set `NCCL_SOCKET_IFNAME` to the 200GbE NIC name (measured: `enp1s0f0np0`); confirm with `ip link`.

### Pitfall 2: FP8 KV Cache Breaks Flash-Next
Setting KV cache dtype to fp8 crashes on Flash-Next.
**Fix**: keep the default `auto` (log shows `kv_cache_dtype_skip_layers` applied correctly).

### Pitfall 3: PLE Layer Must Be Patched
Without the patch mounts, inference errors out.
**Fix**: bind-mount the two patch files as in §3.1.

### Pitfall 4: Forgetting `--headless` on rank1
Without `--headless`, both nodes start an API server and conflict on the port.
**Fix**: the worker node must run with `--headless`.

### Pitfall 5: Reversed Start Order → Connection Reset
Worker-first always produces `NCCL Connection reset`.
**Fix**: head 20 s first, then worker.

### Pitfall 6: gpu-memory-utilization Too Low → Engine Init Failed
On DGX Spark's unified-memory architecture this parameter is extremely sensitive:
- **DeepSeek: anything below 0.78 fails** (KV available 6.92 GiB < 7.8 GiB required); only 0.82 works
- **Qwen3.8 Flash: 0.85 is stable** (MoE, small activations, big headroom)
- Rule of thumb: **start low to confirm the model boots, then raise it; any change requires a container restart (~5 min no-service window)**

### Pitfall 7: CPU Processes Squeeze Unified Memory → OOM During Inference
Under unified memory, CPU processes (e.g. resident agents at ~3.2 GB each) directly reduce the memory available to vLLM's KV cache.
**Fix**: check `free -g` before deploying; reserve more headroom on machines hosting many agent processes; after a crash, check `nvidia-smi`/`free` before deciding whether to raise the parameter or reduce processes.

### Pitfall 8: Copying DeepSeek's max-num-seqs → Wasted Concurrency
DeepSeek is limited to seqs=8 (memory ceiling). **Qwen3.8 Flash comfortably runs 32 or higher** (KV theoretical max 42.4).
**Fix**: after swapping models, re-measure KV headroom and set concurrency for the new model — don't reuse old parameters.

### Pitfall 9: Restarting Without Checking In-Flight Requests
`docker rm -f` kills every in-flight inference.
**Fix**: before restarting, `curl /metrics | grep num_requests_running` and confirm 0.

---

## 5. Concurrency Tuning (Based on Measured KV Data)

| Parameter | DeepSeek tier | Qwen3.8 Flash tier | Evidence |
|:--|:--|:--|:--|
| gpu-memory-utilization | 0.82 (floor) | 0.85 | measured stable |
| max-num-seqs | 8 | **32** | KV theoretical max 42.4 |
| max-model-len | 262144 | 65536 | Flash-Next long-context cost is low |
| KV cache | tight | 2.78M tokens | from /metrics |
| Single-request latency | — | ~0.5 s (short) | measured |

> **Note**: after raising max-num-seqs, watch `kv_cache_usage_perc` (currently 0). If long-context workloads push usage toward 1.0, step back down to 24/16.

---

## 6. Operations Notes

- **Logs**: rank0 `~/qwen38_tp2_rank0.log`, rank1 `~/qwen38_tp2_rank1.log`
- **Restart**: stop both containers → head first → worker second → wait for `Application startup complete` (first boot 6–10 min including weight load + AOT compile)
- **Watchdogs**: disable hard-restart watchdogs during switchovers; re-enable and update the launcher script after deploy
- **Weight sync**: Spark1 is the version source; Spark2/3 sync over the 200GbE cable with rsync (`--checksum`); **never serve vLLM from NFS**; vLLM always reads local disk

---

*Based on measurements from 2026-08-31 on three machines (Spark1 .59 / Spark2 .226 / Spark3 .209). All numbers come from real deployment logs and vLLM /metrics output.*
