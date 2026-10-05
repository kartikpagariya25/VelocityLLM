<div align="center">

<img src="assets/logo.png" alt="VelocityLLM Banner" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.11%2B-FF7A00?style=for-the-badge&logo=python&logoColor=white)
![vLLM](https://img.shields.io/badge/Powered%20by-vLLM-FF8C00?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-FF9500?style=for-the-badge&logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-FFA500?style=for-the-badge&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-64%20passing-FFB347?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-FFC300?style=for-the-badge)

**A dynamic / continuous batching scheduler engine for LLM inference serving —**
**built to prove, with live numbers, that intelligent scheduling raises GPU throughput and SLA reliability over traditional static batching.**

[Overview](#overview) • [Architecture](#architecture) • [Setup](#setup-guide) • [Usage](#usage) • [Benchmark Results](#benchmark-results) • [Troubleshooting](#troubleshooting--common-fixes) • [Roadmap](#roadmap)

</div>

---

## Overview

**VelocityLLM** is an industry-grade systems engineering project that tackles a real production problem in LLM serving: **static batch sizes under-utilize GPU throughput and silently break latency promises under variable request loads.**

This project builds a **dynamic/continuous batching scheduler** on top of [vLLM](https://github.com/vllm-project/vllm) that:

- Enforces configurable **latency SLAs** via a predictive, SLA-aware admission controller
- Adapts its concurrency ceiling in real time using an **AIMD feedback controller** driven by live GPU utilization and memory headroom
- Prioritizes requests with **anti-starvation aging**, so low-priority traffic is never indefinitely starved
- Survives real failure conditions — GPU OOM, client disconnects, malformed input, traffic bursts — without crashing or silently degrading
- Is **model-agnostic** — the scheduler only reasons about request metadata (arrival time, token counts, priority), never model internals, so it works with any vLLM-supported model
- Proves its improvement with **real, reproducible benchmark data** against a static-batching baseline running through the same codebase
- Ships an opt-in **Smart policy** (`--policy smart`) that adds token-aware, self-calibrating, fully explained scheduling for mixed real-time / best-effort traffic (see [Policies at a glance](#policies-at-a-glance))

### Policies at a glance

Three schedulers live in the same codebase and are selected with `--policy`:

| Policy | What it is | Where it fits | Measured limitation |
|---|---|---|---|
| `static` | Fixed concurrency (8), FIFO, no SLA awareness | Baseline; simple, predictable at light load | Under overload everything queues and finishes late (interactive p99 ~10 s in our mixed test) |
| `dynamic` | SLA-aware admission + AIMD concurrency + priority queue with aging | Variable load where SLA protection matters | Conservative on a fresh server; results varied between repeats on our RTX 4050 |
| `smart` | `dynamic` plus ten token/KV/deadline-aware features, online self-calibration and a decision trace | Mixed real-time + best-effort traffic under overload | Sheds best-effort work by design; no benefit shown on uniform or light load |

What `smart` improves is **goodput** (requests finished inside their SLA), not raw throughput: in our tests its tokens/s is comparable to Static's and higher than Dynamic's, but not shown to exceed Static's.

> Academic/Industry Project — guided by Dr. Viomesh
> Team: Kartik R. Pagariya (scheduler architecture, infra, benchmarking) · Vikrant Kadam (Phase 2/3 engine implementation)

---

## Problem Statement

| | |
|---|---|
| **Problem** | Static batch sizes under-utilize GPU throughput under variable request loads in production LLM serving. |
| **Objective** | Build a dynamic/continuous batching scheduler that maximizes GPU utilization while meeting latency SLAs. |
| **Tech Stack** | Python, vLLM, Triton Inference Server, CUDA |
| **Expected Output** | A serving engine with measurable throughput gains at equal or better p99 latency. |

---

## Architecture

```
                   +--------------------------------------+
Client Requests -> | API Gateway (FastAPI)                |
                   | - Request validation & sanitization  |
                   | - Correlation-ID middleware           |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | Admission Controller                 |
                   | - SLA-headroom prediction              |
                   | - Tiered burst shedding (429)           |
                   | - GPU memory soft/hard limits            |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | Prioritized Request Queue             |
                   | - Priority ordering (HIGH/NORMAL/LOW) |
                   | - Anti-starvation aging                 |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | Dispatch Loop + Adaptive Controller   |
                   | - AIMD concurrency ceiling tuning     |
                   | - OOM emergency throttle                |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | Inference Backend (swappable)          |
                   | VLLMBackend (real GPU)                 |
                   | MockBackend (CPU-only test/dev)         |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | vLLM AsyncLLMEngine                    |
                   | (PagedAttention + Continuous Batching) |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | GPU (CUDA)                             |
                   +--------------------------------------+

SchedulerPolicy is an abstract interface with three swappable implementations
in the SAME codebase, selected via --policy flag:
  - StaticBatchPolicy   : fixed concurrency, no SLA awareness (baseline)
  - DynamicBatchPolicy  : everything above (the novel contribution)
  - SmartBatchPolicy    : opt-in closed-loop scheduler layered on Dynamic -- token-aware,
                          future-KV aware, self-calibrating, every decision explained
                          (see docs/SMART_SCHEDULER.md)

Observability: GET /stats (JSON) and GET /metrics (Prometheus format) expose
live scheduler telemetry -- active/queued requests, concurrency limit, SLA
compliance rate, GPU utilization, OOM/disconnect/burst-shed counters.
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference engine | [vLLM](https://github.com/vllm-project/vllm) AsyncLLMEngine (PagedAttention + continuous batching) |
| API layer | FastAPI + Uvicorn, OpenAI-compatible `/v1/completions` + native `/generate` |
| Scheduler core | Pure Python asyncio, policy-pattern (`SchedulerPolicy` ABC) |
| Admission control | Predictive SLA-headroom check, tiered priority-aware burst shedding |
| Adaptive concurrency | AIMD (Additive-Increase / Multiplicative-Decrease) feedback controller |
| Priority queue | Min-heap with time-based anti-starvation aging |
| Backend abstraction | `VLLMBackend` (production) / `MockBackend` (dependency-free testing) |
| Load generator | Python asyncio + aiohttp; flood, constant, Poisson, and burst traffic patterns |
| GPU monitoring | `nvidia-smi` polled from a dedicated background thread (never blocks the event loop) |
| Metrics | Prometheus-format counters/histograms at `/metrics` |
| Testing | pytest + pytest-asyncio + httpx, 29 tests across unit/integration/robustness |
| Model | Llama-3.2-1B-Instruct (fits comfortably on 8GB VRAM) |
| Dashboard | Planned -- Phase 5 |
| Serving infra (stretch) | Triton Inference Server + vLLM backend -- Phase 6 |

---

## Project Structure

```
velocityllm/
├── scheduler_engine/
│   ├── server.py               Unified FastAPI server; --policy static|dynamic|smart
│   ├── policy.py                SchedulerPolicy ABC, StaticBatchPolicy, DynamicBatchPolicy
│   ├── admission.py             SLA-aware admission controller, burst shedding
│   ├── adaptive_controller.py   AIMD adaptive concurrency controller
│   ├── priority_queue.py        Prioritized queue with anti-starvation aging
│   ├── backend.py               VLLMBackend / MockBackend abstraction
│   ├── validation.py            Input sanitization & bounds checking
│   ├── logging_config.py        Structured JSON logging + correlation IDs
│   ├── types.py                 Pydantic/dataclass schemas, ServerConfig
│   ├── gpu_monitor.py            Background GPU sampler (used by load generator)
│   ├── gpu_watch.py              Standalone live GPU terminal monitor
│   ├── smart_policy.py          SmartBatchPolicy: wires the 10 smart-scheduler features together
│   ├── smart_admission.py       Token-aware admission, future-KV check, early rejection (reason codes)
│   ├── smart_queue.py           Slack/deadline priority, short-prompt-first, token buckets, RT/BE lanes
│   ├── profiler.py              Per-request token estimate, bucket, traffic class, deadline
│   ├── estimators.py            Output-length predictor, KV estimator, online TTFT/ITL calibrator
│   ├── capacity.py              Memory-safe vs SLA-safe capacity planner (min wins)
│   ├── decision_trace.py        Structured, queryable event log of every scheduling decision
│   ├── smart_config.py          Tunables for the smart policy (live-editable)
│   └── smart_routes.py          /smart/* endpoints (state, queue, trace, events SSE, config, metrics)
├── load_generator/
│   ├── generate_load.py         Async load generator (flood/constant/poisson/burst)
│   ├── traffic_patterns.py      Arrival-time generators + mixed short/long prompt sets
│   └── compare_results.py       Static vs dynamic side-by-side comparison report
├── tests/                       64 tests: admission, adaptive controller, priority
│                                 queue, policies, API integration, robustness/edge cases,
│                                 plus tests/test_smart_features.py (22 tests, one+ per smart feature)
├── tools/                       sim_benchmark.py -- SIMULATED-GPU policy comparison (not a GPU measurement)
├── docs/                        PRD, implementation plan, benchmark results summary
├── results/                     Raw CSV + summary JSON from every benchmark run
├── models/                      Downloaded model weights (gitignored)
├── verify_phase2.py             Standalone Phase 2 demonstration script
├── verify_phase3.py             Standalone Phase 3 demonstration script
├── run_tests.py                 Test suite runner
├── pytest.ini                   asyncio_mode = auto
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Windows 11 with **WSL2 (Ubuntu)** installed, or native Linux
- NVIDIA GPU with a recent driver (`nvidia-smi` must work inside WSL2)
- Python 3.11+ (see [Troubleshooting](#troubleshooting--common-fixes) for Python 3.14 caveats)
- A [Hugging Face](https://huggingface.co) account and access token

---

## Setup Guide

### 1. Clone and Enter the Project

```bash
git clone https://github.com/kartikpagariya25/VelocityLLM.git
cd VelocityLLM
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Fix pip's Temp Directory (WSL2 Only)

```bash
mkdir -p ~/tmp_pip
echo 'export TMPDIR=$HOME/tmp_pip' >> ~/.bashrc
source ~/.bashrc
```

### 4. Install System Dependencies

```bash
sudo apt update
sudo apt install build-essential python3-dev -y
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> Required WSL2/Blackwell-GPU environment variables (`VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`) are set automatically at import time inside `scheduler_engine/backend.py` -- no manual shell configuration needed.

### 6. Hugging Face Authentication & Model Download

```bash
pip install "huggingface-hub>=1.5.0,<2.0"
hf auth login
mkdir -p models
hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir models/llama-3.2-1b \
  --exclude "original/*"
```

> Keep `huggingface-hub` below 2.0 (the pin above): `transformers`, which vLLM imports, rejects newer versions and the engine fails to start. Quantized models such as `Qwen/Qwen2.5-3B-Instruct-AWQ` download the same way and need no code changes.

> Gated model -- request access on its [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) first. To skip waiting, substitute a non-gated equivalent like `Qwen/Qwen2.5-1.5B-Instruct` -- the scheduler is model-agnostic.

### 7. Run the Test Suite (Verifies Everything Works, No GPU Required)

```bash
python3 run_tests.py
```

All 29 tests should pass using `MockBackend` -- no GPU or model download needed for this step.

---

## Usage

### Start the Server

```bash
# Dynamic scheduler (the novel contribution)
python3 -m scheduler_engine.server --policy dynamic --sla-ms 8000

# Static baseline (for comparison)
python3 -m scheduler_engine.server --policy static

# Smart scheduler (opt-in: token-aware, explainable, self-calibrating)
python3 -m scheduler_engine.server --policy smart --sla-ms 8000

# CPU-only / no-GPU dev mode
python3 -m scheduler_engine.server --policy dynamic --mock
```

Key flags: `--port`, `--sla-ms` (target SLA in milliseconds), `--max-concurrency`, `--burst-shed-ratio`, `--structured-logs`.

### Send a Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "max_tokens": 50}'
```

### Check Live Telemetry

```bash
curl http://localhost:8000/stats     # JSON scheduler telemetry
curl http://localhost:8000/metrics   # Prometheus-format metrics
curl http://localhost:8000/health    # Liveness/readiness
```

### Run a Benchmark

```bash
# Simple saturation test
python3 load_generator/generate_load.py \
  --url http://localhost:8000/generate \
  --num-requests 30 --concurrency 10 --max-tokens 100 \
  --output results/run.csv --label "My Run"

# Realistic Poisson traffic over 10 seconds
python3 load_generator/generate_load.py \
  --url http://localhost:8000/generate \
  --num-requests 40 --concurrency 10 --pattern poisson --duration 10 \
  --output results/poisson.csv --label "Poisson Traffic"

# Bursty traffic
python3 load_generator/generate_load.py \
  --url http://localhost:8000/generate \
  --num-requests 40 --concurrency 15 --pattern burst --duration 15 \
  --output results/burst.csv --label "Burst Traffic"

# Mixed short/long prompts (head-of-line blocking test)
python3 load_generator/generate_load.py \
  --url http://localhost:8000/generate \
  --num-requests 30 --concurrency 10 --mixed-prompts \
  --output results/mixed.csv --label "Mixed Prompts"
```

Each run prints a clean, highlighted report and saves raw data (`*.csv`) plus a machine-readable summary (`*_summary.json`).

### Compare Static vs Dynamic

```bash
python3 load_generator/compare_results.py \
  --static results/static_run_summary.json \
  --dynamic results/dynamic_run_summary.json
```

### Smart Scheduler (opt-in)

`--policy smart` adds ten research-derived scheduling features on top of the dynamic scheduler. Static and Dynamic are unchanged.

| # | Feature | What it does |
|---|---|---|
| 1 | Token-aware scheduling | Cost of a request = prompt tokens + predicted output tokens, not "one request" |
| 2 | Future KV prediction | Admission looks at the KV memory running requests *will* need, not only what they use now |
| 3 | Memory limit vs SLA limit | Capacity = min(memory-safe, SLA-safe); AIMD stays as a reactive brake. The binding constraint is reported |
| 4 | Deadline / slack priority | Requests close to missing their SLA move up; requests that can no longer meet it are *not* boosted (avoids EDF overload collapse) |
| 5 | Short-prompt-first | Short prompts jump ahead of long ones, bounded so HIGH priority stays protected |
| 6 | Token-length buckets | SHORT / MEDIUM / LONG buckets (configurable); LONG requests are capped to a share of slots |
| 7 | Early rejection | Requests predicted to miss their SLA get an immediate `429` with a reason code, before using the GPU |
| 8 | Real-time vs best-effort | `X-Traffic-Class` header; best-effort is throttled under pressure so real-time SLAs hold |
| 9 | Online self-calibration | Startup probe + live measurements learn this model/GPU's TTFT and per-token latency; no offline training |
| 10 | Decision trace | Every admit / reject / reorder / dispatch / capacity change is logged with the signals used |

```bash
curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' \
  -H 'X-Traffic-Class: real_time' -H 'X-SLA-Ms: 3000' -H 'X-Correlation-ID: demo-1' \
  -d '{"prompt":"Explain what a GPU does.","max_tokens":60}'

curl http://localhost:8000/smart/trace/demo-1   # why it was admitted, queued, dispatched
curl http://localhost:8000/smart/state          # capacity, KV, buckets, lanes, calibration, queue
curl -N http://localhost:8000/smart/events      # live decision stream (SSE)
```

When to use it: traffic that mixes interactive and batch requests with different priorities/SLAs and sometimes exceeds GPU capacity. For uniform or lightly loaded traffic `static` or `dynamic` is simpler and measured no worse.

Rejected requests return `429` with `reason_code` (`sla_risk`, `memory_risk`, `queue_overload`, `policy_limit`) and a `details` object (predicted wait/exec, SLA, KV figures).
Smart endpoints exist only with `--policy smart` (other policies return 404 on `/smart/*`).

Full reference (endpoints, tunables, trace events, validation, limitations): [`docs/SMART_SCHEDULER.md`](docs/SMART_SCHEDULER.md)

---

## Benchmark Results

**Model:** Llama-3.2-1B-Instruct · **Hardware:** RTX 5050 (8GB VRAM) · Full methodology in [`docs/baseline_results_summary.md`](docs/baseline_results_summary.md)

| Scenario | Metric | Static | Dynamic | Result |
|---|---|---|---|---|
| Saturation load (30 req, flood) | Throughput | 7.07 req/s | **8.11 req/s** | **+14.7%** |
| | p99 latency | 4.241s | **3.198s** | **-24.6%** |
| Poisson traffic (light load) | Throughput / p99 | 4.15 req/s / 0.835s | 4.10 req/s / 0.829s | Equivalent (expected -- no contention) |
| Bursty traffic | Throughput / p99 | 1.11 req/s / 0.693s | 1.11 req/s / 0.743s | Equivalent (expected -- no contention) |
| **Mixed short/long prompts** | **SLA compliance** | **80.0%** | **100.0%** | **Dynamic never breaks SLA promises** |

**The headline finding:** under mixed realistic load, the static baseline let 20% of requests silently breach their latency SLA. The dynamic scheduler maintained 100% SLA compliance by proactively shedding only 3.3% of requests (an honest, immediate HTTP 429) rather than letting everyone's latency degrade unpredictably -- at nearly identical throughput (5.13 vs 5.08 req/s).

### Smart policy on a second machine: RTX 4050 Laptop (6 GB)

**Model:** Qwen2.5-0.5B-Instruct · **Stack:** vLLM 0.28 on WSL2 · 100 concurrent requests, 30-request warm-up, 3 repeats, all policies capped at concurrency 8 · Raw data: [`docs/rtx4050_results/`](docs/rtx4050_results/)

| SLA | Policy | Goodput % mean [min-max] | p99 | tok/s | Rejected |
|---|---|---|---|---|---|
| 8 s | Static | 94 [90-97] | 8.43 s | 876 | 0 |
| 8 s | Dynamic | 55 [27-97] | 7.22 s | 556 | 44 |
| 8 s | Smart | 91 [80-96] | 7.69 s | 875 | 8 |
| 5 s | Static | 56 [56-56] | 8.58 s | 852 | 0 |
| 5 s | Dynamic | 42 [16-56] | 5.55 s | 671 | 52 |
| 5 s | Smart | 57 [56-57] | **5.20 s** | 843 | 39 |

Goodput = requests finished inside the SLA, out of 100 offered. What this supports, and what it does not:

- Smart's goodput is about equal to Static's, **not higher**. Throughput is equal at an equal concurrency ceiling.
- At a 5 s SLA Smart refuses the requests that would be late (`sla_risk`), so admitted requests finish with p99 5.2 s instead of 8.6 s.
- On this GPU, Dynamic's goodput varied much more between repeats than Smart's.
- With a ceiling of 16, Smart reached 1417 tok/s vs Static's 897 -- but Static is fixed at 8 (`initial_concurrency`), so that gap comes from concurrency, not from the smart features.
- The scheduler's KV capacity estimate was roughly 10-20% *below* vLLM's real cache on all four models tested (safe side).
- Not shown by this uniform-prompt test: other GPUs/models, or the individual effect of each of the ten features (mixed traffic is covered below). The GPU is power-capped (30 W) and its clock varied during runs; treat single runs as noise.

#### Mixed traffic with priorities and per-class SLAs (same GPU, Qwen2.5-0.5B)

Open-loop Poisson arrivals for 15 s, 3 repeats, concurrency cap 8 for every policy, identical request sequence per policy.
Mix: 55% *interactive* (short prompt, 30% HIGH priority, SLA 3 s), 30% *standard* (~250 words, SLA 6 s), 15% *batch* (~1200 words, LOW priority, best-effort, SLA 20 s).
Goodput = served **and** within the request's own SLA, out of all requests offered (rejected requests count against it).
Data: [`docs/rtx4050_results/`](docs/rtx4050_results/) · Scripts: `tools/gpu_bench/` (`compare_mixed.sh`, `mixed_load.py`).

| Load | Policy | Overall goodput % | Interactive goodput % | HIGH-priority goodput % | Interactive p99 | Batch served % | Rejected | tok/s |
|---|---|---|---|---|---|---|---|---|
| 8 req/s | Static | 97 [91-100] | 93 [80-100] | 94 [83-100] | 2.42 s | 100 | 0 | 405 |
| 8 req/s | Dynamic | 95 [84-100] | 90 [71-100] | 95 [86-100] | 2.41 s | 100 | 3 | 398 |
| 8 req/s | Smart | 99 [97-100] | 100 [100-100] | 100 [100-100] | 1.28 s | 91 [73-100] | 1 | 410 |
| 16 req/s | Static | 38 [31-43] | 17 [10-22] | 19 [14-21] | 10.34 s | 100 | 0 | 426 |
| 16 req/s | Dynamic | 25 [22-30] | 10 [2-21] | 18 [3-29] | 4.79 s | 91 [73-100] | 162 | 188 |
| 16 req/s | Smart | **84 [80-87]** | **95 [90-99]** | **100 [100-100]** | **2.24 s** | 17 [2-30] | 38 | 455 |

What this supports:

- **Under overload (16 req/s) Smart keeps real-time traffic inside its SLA** (interactive 95%, HIGH priority 100%) while Static and Dynamic fall to roughly 10-20%. The ranges across repeats do not overlap. Overall goodput, which counts the dropped batch requests against Smart, is still 84% vs 38% (Static) and 25% (Dynamic).
- **The cost is best-effort work:** at 16 req/s Smart serves only ~17% of batch requests (it sheds the rest by design); Static and Dynamic serve 100% and 91% of them.
- At light-to-moderate load (8 req/s) the policies are close and their ranges overlap; Smart is at best equal there.
- Dynamic rejected about two thirds of requests at 16 req/s and finished below Static on overall goodput in this test.

What it does not show: which of the ten features produces the gain (no ablation was run), other models or GPUs, or other traffic mixes. One model, one laptop GPU (power-capped; clock varies run to run).

---

## Environment Variables Reference

| Variable | Value | Why It's Needed |
|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER` | `0` (set automatically) | WSL2's CUDA implementation doesn't support UVA, which vLLM's V2 Model Runner requires |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` (set automatically) | FlashInfer's sampler tries to JIT-compile with `nvcc`, which isn't installed by default |
| `TMPDIR` | `$HOME/tmp_pip` | Redirects pip's temp files off WSL2's tiny `/tmp` tmpfs |
| `gpu_memory_utilization` | `0.80` (config default) | Default `0.92` is too aggressive for 8GB VRAM under WSL2's driver overhead |
| `max_model_len` | `4096` (config default) | Llama's default 131072 needs 4GiB+ of KV cache alone |
| `--sla-ms` | `8000` (recommended for this hardware) | Default 3000ms causes unrealistic over-rejection on this model/GPU combo under load |

---

## Troubleshooting / Common Fixes

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: bootstrapping phase` on script start | vLLM uses `spawn` multiprocessing on WSL2; script lacks a main guard | Wrap execution in `if __name__ == "__main__":` |
| `ValueError: Free memory ... less than desired GPU memory utilization` | Default `gpu_memory_utilization=0.92` too high for 8GB VRAM | Use `gpu_memory_utilization=0.80` |
| `RuntimeError: UVA is not available` | WSL2 doesn't support Unified Virtual Addressing | `VLLM_USE_V2_MODEL_RUNNER=0` (already set in `backend.py`) |
| `Failed to find C compiler` / `Python.h: No such file` | Missing build tools on fresh WSL2 Ubuntu | `sudo apt install build-essential python3-dev -y` |
| `Could not find nvcc and default cuda_home doesn't exist` | FlashInfer sampler needs `nvcc` to JIT-compile | `VLLM_USE_FLASHINFER_SAMPLER=0` (already set in `backend.py`) |
| `KV cache is needed, which is larger than available` | Model's default max context (131072) needs 4GiB+ KV cache | Use `max_model_len=4096` |
| `pynvml.NVMLError_Unknown` on GPU utilization query | NVML's utilization query is unreliable inside WSL2 | Use `nvidia-smi` via `subprocess` (already used throughout) |
| `No space left on device` during pip install | WSL2's `/tmp` is a small RAM-backed tmpfs | Set `TMPDIR` to a path on the real filesystem |
| Dynamic scheduler mysteriously *slower* than static | `get_gpu_telemetry()` called `nvidia-smi` synchronously inside the async request path, blocking the event loop on every call | Poll GPU stats from a dedicated background thread instead (see `backend.py::_telemetry_loop`) |
| Server fails with `address already in use` on restart | A previous server process didn't fully terminate | `lsof -i :8000` to find it, then `fuser -k 8000/tcp` before restarting |
| Many requests rejected with "Predicted latency exceeds SLA target" | Default SLA target (3000ms) too tight for this model/hardware at high concurrency | Start server with `--sla-ms 8000` for realistic benchmarking |
| `PermissionError: [Errno 13] Permission denied: 'nvcc'` during vLLM engine start (torch.compile / Inductor) | No Linux `nvcc`; WSL's PATH includes Windows folders that Python cannot execute, so the lookup fails with *permission denied* instead of *not found* | Remove `/mnt/...` entries from PATH for the server: `export PATH=$(echo "$PATH" \| tr ':' '\n' \| grep -v '^/mnt/' \| paste -sd:)` |
| `/health` says `"backend": "vllm"` but no GPU is used | If vLLM cannot be imported the server falls back to `MockBackend`; `/health` reports the *configured* backend | Check the startup log for `AsyncLLMEngine initialized successfully` (and no `Falling back to MockBackend`), and watch `nvidia-smi` |
| `404` on `/smart/...` | Smart endpoints exist only when the server runs with `--policy smart` | Restart with `--policy smart` |
| Smart rejects most of a sudden 100-request burst with `queue_overload` | Burst-shedding starts when the queue is ~90% full (`max_queue_size` is 100) | Spread arrivals out, or raise `max_queue_size` in `ServerConfig` |

---

## Roadmap

- [x] **Phase 0** -- Environment, GPU/CUDA/vLLM verified end-to-end
- [x] **Phase 1** -- Baseline static-batching server, load generator, GPU monitoring
- [x] **Phase 2** -- Core dynamic scheduler: SLA-aware admission control, adaptive AIMD batch sizing, priority queue with anti-starvation aging
- [x] **Phase 3** -- Robustness: input validation, GPU memory limits, OOM recovery, client-disconnect reclamation, burst shedding, structured logging -- 29/29 tests passing
- [x] **Phase 4** -- Advanced load generation (Poisson/burst/mixed-prompt patterns) and full comparative benchmarking -- real, reproducible static vs dynamic data captured
- [x] **Smart scheduler** -- opt-in `--policy smart`: 10 research-derived features, decision trace, `/smart/*` API, 64/64 tests passing, validated on a real GPU (RTX 4050) with 4 models
- [ ] **Smart follow-ups** -- add `smart` to the control service / Arena runner, wire the frontend to `/smart/events` and `/smart/state`, mixed-traffic benchmark to measure each feature separately
- [ ] **Phase 5** -- Live dashboard with static-vs-dynamic comparison view
- [ ] **Phase 6** -- CLI packaging, final benchmark report document, optional Triton Inference Server deployment
- [ ] **Phase 7** -- Demo rehearsal and presentation

Full detail: [`docs/VelocityLLM_Implementation_Plan.md`](docs/VelocityLLM_Implementation_Plan.md)

---

## Documentation

| Document | Description |
|---|---|
| [`docs/VelocityLLM_PRD.md`](docs/VelocityLLM_PRD.md) | Full product requirements -- goals, edge cases, test plan, KPIs |
| [`docs/VelocityLLM_Implementation_Plan.md`](docs/VelocityLLM_Implementation_Plan.md) | Phased execution plan with exit criteria |
| [`docs/baseline_results_summary.md`](docs/baseline_results_summary.md) | Full benchmark methodology, results, and root-cause log for every issue found during testing |
| [`docs/SMART_SCHEDULER.md`](docs/SMART_SCHEDULER.md) | Smart scheduler reference -- features, endpoints, tunables, decision-trace events, validation, limitations |
| [`docs/rtx4050_results/`](docs/rtx4050_results/) | Raw CSVs and notes from the RTX 4050 (6 GB) runs |

---

## Team

**Kartik R. Pagariya** -- Scheduler architecture, infrastructure, benchmarking, testing
**Vikrant Kadam** -- Phase 2/3 scheduler engine implementation
B.Tech, Artificial Intelligence and Data Science -- Vishwakarma Institute of Technology, Pune
GitHub: [@kartikpagariya25](https://github.com/kartikpagariya25)

---

<div align="center">

*Built as an industry-guided academic project under Dr. Viomesh.*

</div>