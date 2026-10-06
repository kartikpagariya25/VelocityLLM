<div align="center">

<img src="assets/logo.png" alt="VelocityLLM Banner" width="100%"/>

<br/>

[![PyPI](https://img.shields.io/pypi/v/velocityllm?style=for-the-badge&logo=pypi&logoColor=white&color=FF7A00)](https://pypi.org/project/velocityllm/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-FF7A00?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
![vLLM](https://img.shields.io/badge/Powered%20by-vLLM-FF8C00?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-FF9500?style=for-the-badge&logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-FFA500?style=for-the-badge&logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-131%20passing-FFB347?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-FFC300?style=for-the-badge)

**A dynamic / continuous batching scheduler engine for LLM inference serving —**
**built to prove, with live numbers, that intelligent scheduling raises GPU throughput and SLA reliability over traditional static batching.**

```bash
pip install velocityllm && velocityllm bench --mock --models-dir ./models
```

[Results](#headline-results) • [Install](#install-from-pypi) • [Commands](#command-reference) • [Reproduce](#reproduce-the-benchmark-on-your-gpu) • [Architecture](#architecture) • [Troubleshooting](#troubleshooting--common-fixes) • [Limitations](#limitations-stated-up-front) • [Roadmap](#roadmap)

</div>

---

## Headline results

Same model, same GPU, same 100-user flood, same SLA (8 s). The only change is the scheduler: **Static** (fixed concurrency 8) vs **VelocityLLM Dynamic**. Median of 3 repeats, run with the published `velocityllm` PyPI package on an **RTX 5050 Laptop GPU (8 GB), WSL2**.

| Model | Policy | Served | p50 | p95 | p99 | On time (SLA 8 s) | Tokens/s |
|---|---|---|---|---|---|---|---|
| tinyllama-1.1b | Static | 100/100 | 1.32 s | 2.12 s | 2.29 s | 100% | 610 |
| | **Dynamic** | 100/100 | 1.24 s | 1.84 s | 1.94 s | 100% | **814** |
| llama-3.2-1b | Static | 100/100 | 3.73 s | 6.34 s | 6.51 s | 100% | 732 |
| | **Dynamic** | 100/100 | 2.17 s | 2.99 s | 3.05 s | 100% | **1550** |
| qwen2.5-1.5b | Static | 100/100 | 4.71 s | 7.61 s | 8.08 s | 99% | 548 |
| | **Dynamic** | 100/100 | 2.39 s | 3.79 s | 3.85 s | 100% | **1172** |
| stablelm-2-1.6b | Static | 100/100 | 5.36 s | 9.08 s | 9.69 s | 80% | 516 |
| | **Dynamic** | 100/100 | 3.30 s | 4.50 s | 4.51 s | 100% | **1109** |
| gemma-2-2b | Static | 100/100 | 9.44 s | 13.09 s | 13.48 s | 23% | 196 |
| | **Dynamic** | 100/100 | 3.42 s | 4.78 s | 4.78 s | 100% | **571** |

| Dynamic vs Static | tinyllama | llama-3.2-1b | qwen2.5-1.5b | stablelm-2-1.6b | gemma-2-2b |
|---|---|---|---|---|---|
| Throughput (tokens/s) | +34% | +112% | +114% | +115% | +191% |
| p50 latency | -6% | -42% | -49% | -38% | -64% |
| p95 latency | -13% | -53% | -50% | -50% | -64% |
| p99 latency | -15% | -53% | -52% | -53% | -65% |
| Requests rejected | 0 | 0 | 0 | 0 | 0 |

- Dynamic raised throughput and cut p50, p95 and p99 on **all five models**, with **zero rejected requests**, so the gain does not come from dropping work.
- The gain grows with model size and KV-cache pressure. On the smallest model (tinyllama) it is modest; on gemma-2-2b, Static met the 8 s SLA for only 23% of users and Dynamic met it for 100%.
- *on time* counts every offered request, and a rejected request counts as a miss.
- Reproduce this table yourself in one command: see [Reproduce the benchmark](#reproduce-the-benchmark-on-your-gpu). Scope and caveats are in [Limitations](#limitations-stated-up-front).

---

## Overview

**VelocityLLM** is an industry-grade systems engineering project that tackles a real production problem in LLM serving: **static batch sizes under-utilize GPU throughput and silently break latency promises under variable request loads.**

It builds a **dynamic/continuous batching scheduler** on top of [vLLM](https://github.com/vllm-project/vllm) that:

- Enforces configurable **latency SLAs** via a predictive, SLA-aware admission controller
- Adapts its concurrency ceiling in real time using an **AIMD feedback controller** driven by live GPU utilization and memory headroom
- Prioritizes requests with **anti-starvation aging**, so low-priority traffic is never indefinitely starved
- Survives real failure conditions — GPU OOM, client disconnects, malformed input, traffic bursts — without crashing or silently degrading
- Is **model-agnostic** — the scheduler only reasons about request metadata (arrival time, token counts, priority), never model internals, so it works with any vLLM-supported model
- Proves its improvement with **real, reproducible benchmark data** against a static-batching baseline running through the same codebase
- Ships an opt-in **Smart policy** (`--policy smart`) that adds token-aware, KV-cache-aware, self-calibrating, fully explained scheduling for mixed real-time / best-effort traffic
- Ships as a **pip package** with a terminal benchmark (`velocityllm bench`), a Python API, a scheduler server, a load generator and a web dashboard (Arena)

### Why Dynamic is faster (what the numbers point to)

1. **Decoding is memory-bandwidth-bound.** Each step reads the model weights once no matter how many sequences are in the batch, so running more sequences together raises throughput at little extra cost per step, until compute or KV memory is the limit. A fixed concurrency of 8 stops short of that limit.
2. **Queueing makes the tail.** With 100 users and 8 slots, requests run in about a dozen waves, and the last wave waits for all the earlier ones. Opening more slots means fewer waves, so p95 and p99 fall sharply.
3. **KV memory is the real ceiling.** On models with large per-token KV footprints (gemma-2-2b on an 8 GB card) the scheduler must stay inside memory, which is what the capacity planner and AIMD brake do.

This explanation is consistent with the measurements above; it is not a profiler trace.

### Policies at a glance

Three schedulers live in the same codebase and are selected with `--policy`:

| Policy | What it is | Where it fits | Measured limitation |
|---|---|---|---|
| `static` | Fixed concurrency (8), FIFO, no SLA awareness | Baseline; simple, predictable at light load | Under overload everything queues and finishes late (p99 above 13 s on gemma-2-2b in the flood test) |
| `dynamic` | SLA-aware admission + AIMD concurrency + priority queue with aging | Variable or bursty load where throughput and SLA protection matter | Conservative on a fresh server; on a smaller GPU its results varied between repeats |
| `smart` | `dynamic` plus ten token/KV/deadline-aware features, online self-calibration and a decision trace | Mixed real-time + best-effort traffic under overload | Sheds best-effort work by design; no benefit shown on uniform or light load |

`smart` improves **goodput** (requests finished inside their own SLA) under overload, not raw throughput.

> Academic/Industry Project — guided by Dr. Viomesh · Sponsored by Single Core Labs
> Team: Kartik R. Pagariya (scheduler architecture, infra, benchmarking, packaging) · Vikrant Kadam (Phase 2/3 engine implementation) · Aditya Dengale (contributor)

---

## Problem Statement

| | |
|---|---|
| **Problem** | Static batch sizes under-utilize GPU throughput under variable request loads in production LLM serving. |
| **Objective** | Build a dynamic/continuous batching scheduler that maximizes GPU utilization while meeting latency SLAs. |
| **Tech Stack** | Python, vLLM, FastAPI, CUDA (Triton Inference Server deployment is a stretch goal) |
| **Expected Output** | A serving engine with measurable throughput gains at equal or better p99 latency. |
| **Delivered** | Published PyPI package, reproducible terminal benchmark, scheduler server, load generator, Arena dashboard. |

---

## Install from PyPI

Requirements: Python 3.11+ (verified on 3.14 under WSL2), Linux or WSL2. A real model needs an NVIDIA GPU with a working `nvidia-smi`.

### Option A: try it with no GPU (simulated engine, about one minute)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install velocityllm
velocityllm version
velocityllm detect
```

Make a tiny fake model folder and run the benchmark against the simulated engine. The numbers are simulated and only prove the pipeline works, never GPU performance:

```bash
mkdir -p models/demo && echo '{"max_position_embeddings": 4096}' > models/demo/config.json
velocityllm bench --mock --models-dir ./models --requests 20 --repeats 1
```

### Option B: real GPU

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "velocityllm[gpu]"      # adds vLLM and PyTorch, several GB
velocityllm detect                   # should print your GPU and the model it found
```

If vLLM is already installed in your environment, plain `pip install velocityllm` is enough and downloads almost nothing (the wheel is about 3 MB).

### Upgrade, pin, or roll back

```bash
pip install --no-cache-dir -U velocityllm            # latest
pip install --no-cache-dir "velocityllm==0.4.1"      # an exact version (also rolls back)
pip index versions velocityllm                       # list published versions
velocityllm version                                  # confirm what is installed
```

Run `velocityllm` from outside a source checkout of this repository so the installed package is the one that loads.

---

## Command reference

```text
velocityllm <command> [options]        'velocityllm <command> --help' lists every option
```

| Command | What it does |
|---|---|
| `velocityllm version` | Print the installed version |
| `velocityllm detect` | Show the GPU and the local model VelocityLLM would use |
| `velocityllm chargeVelocity` | Detect GPU and model, then start the scheduler server |
| `velocityllm serve` | Run the scheduler server (`--policy static\|dynamic\|smart`, `--mock` for no GPU) |
| `velocityllm bench` | Compare policies on one or more models: p50/p95/p99, tokens/s, on-time % |
| `velocityllm loadgen` | Send test traffic (flood, constant, poisson, burst) to a running server |
| `velocityllm arena` | Run the benchmark service behind the Arena web dashboard |

### `velocityllm bench` (the main benchmark)

```bash
# every model folder found in ./models, Static vs Dynamic
velocityllm bench --models-dir ./models --requests 100 --repeats 3 --sla-ms 8000

# pick models and policies
velocityllm bench --models-dir ~/velocityllm/models \
  --only tinyllama-1.1b,llama-3.2-1b,qwen2.5-1.5b,stablelm-2-1.6b,gemma-2-2b \
  --policies static,dynamic --requests 100 --repeats 3 --sla-ms 8000

# add the Smart policy
velocityllm bench --models-dir ./models --only qwen2.5-1.5b --policies static,dynamic,smart

# explicit NAME=PATH pairs instead of a models folder
velocityllm bench --model qwen=/data/qwen2.5-1.5b --model llama=/data/llama-3.2-1b

# mixed short/long prompt scenario, custom output folder
velocityllm bench --models-dir ./models --only qwen2.5-1.5b --scenario flood,mixed --out-dir ./qwen_run

# print the table again from saved runs, without re-running anything
velocityllm bench --models-dir ./models --report-only
```

| Option | Default | Meaning |
|---|---|---|
| `--models-dir` | `./models` | Folder with one sub-folder per model (each must contain `config.json`) |
| `--model NAME=PATH` | none | Add one model explicitly; repeat for more |
| `--only a,b,c` | all | Use only these model folder names |
| `--limit N` | 0 (no limit) | Use at most N models from `--models-dir` |
| `--policies` | `static,dynamic` | Any of `static`, `dynamic`, `smart` |
| `--scenario` | `flood` | `flood` (all users at once), `mixed` (short and long prompts) |
| `--sla-ms` | `8000` | Latency target; used for the on-time column |
| `--requests` | `100` | Simultaneous users per run |
| `--repeats` | `3` | Runs per policy; the table shows the median |
| `--seed` | `42` | Seed for the request set (the same requests go to every policy) |
| `--port` | `8000` | Port for the temporary benchmark server |
| `--max-model-len` | `4096` | Context length cap (lower it if KV memory is tight) |
| `--settle` | `5.0` | Seconds the GPU rests between runs |
| `--startup-timeout` | `600` | Seconds to wait for a model to load |
| `--out-dir` | `./velocity_bench/latest` | Where results are written |
| `--mock` | off | Simulated engine, no GPU |
| `--report-only` | off | Only re-print the table from saved runs |

Output files in the output folder: `summary.csv`, `summary.json`, `raw/` (one CSV per run, with per-request status, latency and tokens) and `server.log`.

Policies run in an alternating order across repeats, so slow drift on a laptop GPU is not charged to a single policy.

### `velocityllm serve` and `loadgen`

```bash
velocityllm serve --policy dynamic --model-path ~/models/qwen2.5-1.5b --sla-ms 8000
velocityllm serve --policy static  --model-path ~/models/qwen2.5-1.5b
velocityllm serve --policy smart   --model-path ~/models/qwen2.5-1.5b --sla-ms 8000
velocityllm serve --mock --policy dynamic            # no GPU needed

velocityllm loadgen --url http://localhost:8000/generate \
  --num-requests 100 --concurrency 100 --pattern flood --max-tokens 100 \
  --output results/run.csv --label "My Run"
```

Key `serve` flags: `--port`, `--sla-ms`, `--max-concurrency`, `--burst-shed-ratio`, `--structured-logs`, `--kv-cache-dtype`.

### `velocityllm arena` (web dashboard)

```bash
velocityllm arena --models-dir ./models --port 9000
```

Then open `http://localhost:9000/`. Pick a model, a user count and an SLA, and watch the policies run side by side. Model folders default to `./models` and results to `./velocity_runs` in the folder where you run the command.

### Python API

```python
import velocityllm

rows = velocityllm.benchmark(
    ["qwen2.5-1.5b", "gemma-2-2b"],            # model folder names, or paths, or {"name": "path"}
    models_dir="/home/me/velocityllm/models",   # where the names are looked up
    policies=("static", "dynamic"),
    requests=100, repeats=3, sla_ms=8000.0,
    out_dir="/home/me/velocity_bench/run1",
)
for r in rows:
    print(r["model"], r["policy"], r["p50"], r["p95"], r["p99"], r["tokens_per_sec"])
```

`benchmark()` returns one dict per (model, scenario, policy) and writes the same files as the CLI. Put the code in a `.py` file and run `python file.py`; it is Python, not a shell command.

---

## Reproduce the benchmark on your GPU

This is the exact sequence behind the [headline results](#headline-results).

**1. Environment (WSL2 or Linux, NVIDIA driver working)**

```bash
nvidia-smi                                   # must list your GPU
sudo apt update && sudo apt install build-essential python3-dev -y
mkdir -p ~/tmp_pip && echo 'export TMPDIR=$HOME/tmp_pip' >> ~/.bashrc && source ~/.bashrc   # WSL2 only
python3 -m venv ~/vl-env && source ~/vl-env/bin/activate
pip install --upgrade pip
pip install "velocityllm[gpu]" "huggingface-hub>=1.5.0,<2.0"
```

**2. Download the models** (one folder per model; folder names are what `--only` uses)

```bash
hf auth login
mkdir -p ~/velocityllm/models && cd ~/velocityllm/models
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0   --local-dir tinyllama-1.1b
hf download meta-llama/Llama-3.2-1B-Instruct     --local-dir llama-3.2-1b --exclude "original/*"
hf download Qwen/Qwen2.5-1.5B-Instruct           --local-dir qwen2.5-1.5b
hf download stabilityai/stablelm-2-1_6b-chat     --local-dir stablelm-2-1.6b
hf download google/gemma-2-2b-it                 --local-dir gemma-2-2b
```

Llama and Gemma are gated: request access on their Hugging Face model pages first, or substitute any non-gated model, because the scheduler is model-agnostic.

**3. Free the GPU, then run**

```bash
nvidia-smi                     # nothing else should be using the GPU
cd ~                           # run from outside a source checkout
velocityllm bench --models-dir ~/velocityllm/models \
  --only tinyllama-1.1b,llama-3.2-1b,qwen2.5-1.5b,stablelm-2-1.6b,gemma-2-2b \
  --policies static,dynamic --requests 100 --repeats 3 --sla-ms 8000
```

A one-model smoke test first: `--only tinyllama-1.1b --repeats 1`.

**4. Read the results**

```bash
cd ~/velocity_bench/latest
column -s, -t < summary.csv | less -S
cp -r ~/velocity_bench/latest ~/velocity_bench/my_run_1     # keep it: the next run overwrites 'latest'
```

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

### Tech stack

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
| Dashboard | React + Vite web UI (Arena) served by the control service |
| Packaging | `pyproject.toml`, console script `velocityllm`, bundled web UI, published on PyPI |
| Testing | pytest + pytest-asyncio + httpx, 131 tests across unit, integration, robustness and packaging |

### Project structure

```
velocityllm/
├── velocityllm/                 The pip package entry point
│   ├── cli.py                    `velocityllm` command: serve, bench, arena, loadgen, detect, chargeVelocity, version
│   ├── bench.py                  Terminal benchmark + Python API (benchmark())
│   └── __main__.py               python -m velocityllm
├── scheduler_engine/
│   ├── server.py                 Unified FastAPI server; --policy static|dynamic|smart
│   ├── policy.py                 SchedulerPolicy ABC, StaticBatchPolicy, DynamicBatchPolicy
│   ├── admission.py              SLA-aware admission controller, burst shedding
│   ├── adaptive_controller.py    AIMD adaptive concurrency controller
│   ├── priority_queue.py         Prioritized queue with anti-starvation aging
│   ├── backend.py                VLLMBackend / MockBackend abstraction
│   ├── hardware.py               GPU and local-model detection
│   ├── kv_model.py               KV bytes/token and weight size derived from a model's config.json
│   ├── validation.py             Input sanitization & bounds checking
│   ├── logging_config.py         Structured JSON logging + correlation IDs
│   ├── types.py                  Pydantic/dataclass schemas, ServerConfig
│   ├── gpu_monitor.py            Background GPU sampler (used by load generator)
│   ├── smart_policy.py           SmartBatchPolicy: wires the smart-scheduler features together
│   ├── smart_admission.py        Token-aware admission, future-KV check, early rejection (reason codes)
│   ├── smart_queue.py            Slack/deadline priority, short-prompt-first, token buckets, RT/BE lanes
│   ├── profiler.py               Per-request token estimate, bucket, traffic class, deadline
│   ├── estimators.py             Output-length predictor, KV estimator, online TTFT/ITL calibrator
│   ├── capacity.py               Memory-safe vs SLA-safe capacity planner (min wins)
│   ├── decision_trace.py         Structured, queryable event log of every scheduling decision
│   ├── smart_config.py           Tunables for the smart policy (live-editable)
│   └── smart_routes.py           /smart/* endpoints (state, queue, trace, events SSE, config, metrics)
├── control_service/              Arena backend: runs benchmarks and serves the web UI
├── frontend/                     React + Vite web UI (built into control_service/web for the pip package)
├── load_generator/               Async load generator, traffic patterns, static-vs-dynamic compare report
├── tests/                        131 tests: admission, AIMD, queue, policies, API, robustness, smart, packaging
├── tools/                        build_package.sh, GPU bench scripts, simulated-GPU comparison
├── docs/                         PRD, implementation plan, benchmark methodology, smart-scheduler reference
├── results/                      Raw CSV + summary JSON from benchmark runs
├── pyproject.toml                Package metadata (name, version, dependencies, console script)
├── requirements.txt              Pinned dependencies for running from source
└── README.md
```

---

## Run from source (development)

```bash
git clone https://github.com/kartikpagariya25/VelocityLLM.git
cd VelocityLLM
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
sudo apt update && sudo apt install build-essential python3-dev -y
pip install -r requirements.txt
pip install -e .
python3 run_tests.py            # 131 tests, uses MockBackend, no GPU or model needed
```

> The WSL2/Blackwell environment variables (`VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`) are set automatically at import time in `scheduler_engine/backend.py`.

Run the server from source:

```bash
python3 -m scheduler_engine.server --policy dynamic --sla-ms 8000     # Dynamic
python3 -m scheduler_engine.server --policy static                    # Static baseline
python3 -m scheduler_engine.server --policy smart --sla-ms 8000       # Smart (opt-in)
python3 -m scheduler_engine.server --policy dynamic --mock            # CPU-only dev mode
```

Build the wheel yourself (frontend build, bundle, `python -m build`, `twine check`):

```bash
bash tools/build_package.sh
ls -lh dist/
```

---

## Server API

```bash
# Generate
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "max_tokens": 50}'

# Telemetry
curl http://localhost:8000/stats      # JSON scheduler telemetry
curl http://localhost:8000/metrics    # Prometheus-format metrics
curl http://localhost:8000/health     # liveness / readiness
```

Rejected requests return `429` with a reason (`Predicted latency exceeds SLA target`, burst shedding, and with Smart a `reason_code`).

### Other load patterns

```bash
velocityllm loadgen --url http://localhost:8000/generate --num-requests 40 --concurrency 10 \
  --pattern poisson --duration 10 --output results/poisson.csv --label "Poisson"
velocityllm loadgen --url http://localhost:8000/generate --num-requests 40 --concurrency 15 \
  --pattern burst --duration 15 --output results/burst.csv --label "Burst"
velocityllm loadgen --url http://localhost:8000/generate --num-requests 30 --concurrency 10 \
  --mixed-prompts --output results/mixed.csv --label "Mixed prompts"
python3 load_generator/compare_results.py --static results/static_run_summary.json --dynamic results/dynamic_run_summary.json
```

---

## Smart scheduler (opt-in)

`--policy smart` adds ten scheduling features on top of the dynamic scheduler. Static and Dynamic are unchanged.

| # | Feature | What it does |
|---|---|---|
| 1 | Token-aware scheduling | Cost of a request = prompt tokens + predicted output tokens, not "one request" |
| 2 | Future KV prediction | Admission looks at the KV memory running requests *will* need, not only what they use now |
| 3 | Memory limit vs SLA limit | Capacity = min(memory-safe, SLA-safe); AIMD stays as a reactive brake. The binding constraint is reported |
| 4 | Deadline / slack priority | Requests close to missing their SLA move up; requests that can no longer meet it are *not* boosted (avoids EDF overload collapse) |
| 5 | Short-prompt-first | Short prompts jump ahead of long ones, bounded so HIGH priority stays protected |
| 6 | Token-length buckets | SHORT / MEDIUM / LONG buckets; LONG requests are capped to a share of slots |
| 7 | Early rejection | Requests predicted to miss their SLA get an immediate `429` with a reason code, before using the GPU |
| 8 | Real-time vs best-effort | `X-Traffic-Class` header; best-effort is throttled under pressure so real-time SLAs hold |
| 9 | Online self-calibration | Startup probe + live measurements learn this model/GPU's TTFT and per-token latency; no offline training |
| 10 | Decision trace | Every admit / reject / reorder / dispatch / capacity change is logged with the signals used |

Additions in 0.4.x: KV bytes per token and weight size derived from the model's `config.json`, the engine's real KV capacity read from vLLM, real token counts from the backend, and a predictive ramp that jumps to the planned capacity when the queue is non-empty and there is no SLA breach. An opt-in fp8 KV cache (`VELOCITY_KV_CACHE_DTYPE=fp8`) and `VELOCITY_SMART_OVERRIDES=key=value,...` allow ablation runs.

```bash
curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' \
  -H 'X-Traffic-Class: real_time' -H 'X-SLA-Ms: 3000' -H 'X-Correlation-ID: demo-1' \
  -d '{"prompt":"Explain what a GPU does.","max_tokens":60}'

curl http://localhost:8000/smart/trace/demo-1   # why it was admitted, queued, dispatched
curl http://localhost:8000/smart/state          # capacity, KV, buckets, lanes, calibration, queue
curl -N http://localhost:8000/smart/events      # live decision stream (SSE)
```

Rejected requests return `429` with `reason_code` (`sla_risk`, `memory_risk`, `queue_overload`, `policy_limit`) and a `details` object. `/smart/*` endpoints exist only with `--policy smart`. Full reference: [`docs/SMART_SCHEDULER.md`](docs/SMART_SCHEDULER.md).

---

## More benchmark evidence

Every table below is from real GPU runs; raw CSVs are in [`results/`](results/) and [`docs/rtx4050_results/`](docs/rtx4050_results/). Methodology: [`docs/baseline_results_summary.md`](docs/baseline_results_summary.md).

### Early baseline: Llama-3.2-1B on RTX 5050 (8 GB), 30 requests

| Scenario | Metric | Static | Dynamic | Result |
|---|---|---|---|---|
| Saturation load (30 req, flood) | Throughput | 7.07 req/s | **8.11 req/s** | **+14.7%** |
| | p99 latency | 4.241 s | **3.198 s** | **-24.6%** |
| Poisson traffic (light load) | Throughput / p99 | 4.15 req/s / 0.835 s | 4.10 req/s / 0.829 s | Equivalent (no contention) |
| Bursty traffic | Throughput / p99 | 1.11 req/s / 0.693 s | 1.11 req/s / 0.743 s | Equivalent (no contention) |
| **Mixed short/long prompts** | **SLA compliance** | **80.0%** | **100.0%** | Dynamic held every SLA by shedding 3.3% of requests |

### Smart on a second machine: RTX 4050 Laptop (6 GB), Qwen2.5-0.5B

100 concurrent requests, 3 repeats, all policies capped at concurrency 8. Goodput = requests finished inside the SLA out of 100 offered.

| SLA | Policy | Goodput % mean [min-max] | p99 | tok/s | Rejected |
|---|---|---|---|---|---|
| 8 s | Static | 94 [90-97] | 8.43 s | 876 | 0 |
| 8 s | Dynamic | 55 [27-97] | 7.22 s | 556 | 44 |
| 8 s | Smart | 91 [80-96] | 7.69 s | 875 | 8 |
| 5 s | Static | 56 [56-56] | 8.58 s | 852 | 0 |
| 5 s | Dynamic | 42 [16-56] | 5.55 s | 671 | 52 |
| 5 s | Smart | 57 [56-57] | **5.20 s** | 843 | 39 |

At an equal concurrency cap, Smart's goodput is about equal to Static's, not higher. At a 5 s SLA it refuses requests that would be late, so admitted requests finish at p99 5.2 s instead of 8.6 s.

### Mixed traffic with priorities and per-class SLAs (same GPU)

Open-loop Poisson arrivals for 15 s, 3 repeats, concurrency cap 8 for every policy. Mix: 55% interactive (SLA 3 s), 30% standard (SLA 6 s), 15% batch (LOW priority, SLA 20 s).

| Load | Policy | Overall goodput % | Interactive goodput % | HIGH-priority goodput % | Interactive p99 | Batch served % | Rejected |
|---|---|---|---|---|---|---|---|
| 16 req/s | Static | 38 [31-43] | 17 [10-22] | 19 [14-21] | 10.34 s | 100 | 0 |
| 16 req/s | Dynamic | 25 [22-30] | 10 [2-21] | 18 [3-29] | 4.79 s | 91 [73-100] | 162 |
| 16 req/s | Smart | **84 [80-87]** | **95 [90-99]** | **100 [100-100]** | **2.24 s** | 17 [2-30] | 38 |

Under overload Smart keeps real-time traffic inside its SLA while Static and Dynamic fall to roughly 10-20%. The cost is best-effort work: Smart serves only about 17% of batch requests at that load by design. At 8 req/s all policies are close.

### Check any number yourself

```bash
velocityllm bench --models-dir ./models --report-only        # re-print the last saved table
ls results/ docs/rtx4050_results/                            # raw data behind the tables above
python3 run_tests.py                                         # 131 tests, no GPU needed
```

---

## Arena dashboard and live demo

```bash
velocityllm arena --models-dir ./models --port 9000          # then open http://localhost:9000/
```

A **live multi-device demo** (everyone in a room joins from a phone, a Static run and a Dynamic run hit the same flood, and each device is scored) lives on the `live-demo` branch:

```bash
git checkout live-demo && git pull origin live-demo
cd frontend && npm install && npm run build && cd ..
python3 -m control_service --host 0.0.0.0 --port 9000
```

- Presenter screen: `http://localhost:9000/#/live`
- Phones: `http://<your-laptop-wifi-ip>:9000/#/join` (the `#/join` part is required; without it the marketing site opens)
- On WSL2, forward the port from Windows (PowerShell, as Administrator):

```powershell
$wsl = ((wsl hostname -I).Trim() -split " ")[0]
netsh interface portproxy delete v4tov4 listenport=9000 listenaddress=0.0.0.0
netsh interface portproxy add v4tov4 listenport=9000 listenaddress=0.0.0.0 connectport=$wsl
New-NetFirewallRule -DisplayName "VelocityLLM 9000" -Direction Inbound -Protocol TCP -LocalPort 9000 -Action Allow
```

Use your phone hotspot or Wi-Fi so the laptop and phones share one network. If a hotspot blocks device-to-device traffic (client isolation), use a Cloudflare tunnel with `--protocol http2`.

---

## Environment variables and defaults

| Variable | Value | Why it is needed |
|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER` | `0` (set automatically) | WSL2's CUDA does not support UVA, which vLLM's V2 model runner requires |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` (set automatically) | FlashInfer's sampler tries to JIT-compile with `nvcc`, which is not installed by default |
| `TMPDIR` | `$HOME/tmp_pip` | Moves pip's temp files off WSL2's small `/tmp` |
| `VELOCITY_KV_CACHE_DTYPE` | `fp8` (optional) | Opt-in fp8 KV cache to fit more tokens in memory |
| `VELOCITY_SMART_OVERRIDES` | `key=value,...` (optional) | Override Smart tunables for ablation runs |
| `gpu_memory_utilization` | `0.80` (config default) | `0.92` is too aggressive for 8 GB VRAM under WSL2 |
| `max_model_len` | `4096` (config default) | A model's native 131072 context needs gigabytes of KV cache alone |
| `--sla-ms` | `8000` (recommended here) | The 3000 ms default over-rejects on this model/GPU class under load |

---

## Troubleshooting / Common fixes

### Install and Python errors

| Symptom | Cause | Fix |
|---|---|---|
| `command not found: velocityllm` | Virtualenv not active, or installed in another environment | `source .venv/bin/activate`, then `pip show velocityllm`; reinstall with `pip install velocityllm` |
| `AttributeError: module 'velocityllm' has no attribute 'benchmark'` | An old release (before 0.4.0) is installed, or a different `velocityllm` is being imported | `pip uninstall -y velocityllm && pip install --no-cache-dir -U velocityllm`; check with `python -c "import velocityllm; print(velocityllm.__version__, velocityllm.__file__)"` (the path must be in `site-packages`) |
| `ValueError: dictionary update sequence element #0 has length 14; 2 is required` | Release 0.4.0 expects a `{name: path}` dict, and a list of model names was passed | `pip install --no-cache-dir -U "velocityllm>=0.4.1"`; or pass a dict |
| `import: command not found`, `syntax error near unexpected token` | Python code was pasted into the shell | Put it in a file (`cat > run.py <<'EOF' ... EOF`) and run `python run.py`, or start `python` first |
| `ModuleNotFoundError: No module named 'PIL'` | Pillow was missing from the dependencies of an early build | `pip install -U velocityllm` (0.4.0 and later declare Pillow) |
| `ModuleNotFoundError: No module named 'scheduler_engine'` | Benchmark subprocess could not find the package | Upgrade to 0.4.0+ (it sets `PYTHONPATH` for the subprocess); run from outside a source checkout |
| `pip install` downloads gigabytes | The `[gpu]` extra installs vLLM and PyTorch | Expected for a fresh environment; if vLLM is already installed use plain `pip install velocityllm` |
| `No matching distribution found for velocityllm==X` | That version is not on PyPI yet, or the index has not refreshed | `pip index versions velocityllm`; wait a minute and retry with `--no-cache-dir` |
| `No space left on device` during pip install | WSL2's `/tmp` is a small RAM-backed tmpfs | `export TMPDIR=$HOME/tmp_pip` (see Setup) |

### Benchmark and model errors

| Symptom | Cause | Fix |
|---|---|---|
| `models folder not found` / `no model folders (with config.json)` | `--models-dir` is wrong or the model folders lack `config.json` | Point `--models-dir` at the parent folder; each model folder needs a `config.json` |
| `not in <dir>: <name> (available: ...)` | A name in `--only` does not match a folder | Use the exact folder names listed in `available:` |
| `FileNotFoundError ... raw/warmup.csv` with a relative `--out-dir` | Old releases resolved the path inside a subprocess | Upgrade to 0.4.0+; or pass an absolute `--out-dir` |
| Benchmark times out while loading a model | Slow disk or large model | Raise `--startup-timeout 1200` and make sure no other process is using the GPU |
| `address already in use` on port 8000 | An earlier server did not exit | `fuser -k 8000/tcp` (or `lsof -i :8000`), or pass `--port 8010` |
| Numbers differ a lot from one run to the next | Laptop GPU power capping or thermal throttling | Close other GPU programs, plug in power, raise `--repeats`, keep `--settle 5` or more, compare policies only inside the same run |
| `ValueError: Free memory ... less than desired GPU memory utilization` | `gpu_memory_utilization` too high for the card | Use `0.80`; stop other processes (`nvidia-smi`) |
| `KV cache is needed, which is larger than available` | The model's native context needs too much KV memory | Use `--max-model-len 4096` or lower |
| Many requests rejected with `Predicted latency exceeds SLA target` | SLA target too tight for the model and load | Use `--sla-ms 8000` |
| `/health` says `"backend": "vllm"` but no GPU is used | vLLM could not be imported and the server fell back to `MockBackend` | Look for `AsyncLLMEngine initialized successfully` in the startup log; watch `nvidia-smi` |

### WSL2 and GPU errors

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: bootstrapping phase` | vLLM uses `spawn` multiprocessing on WSL2 and the script has no main guard | Wrap execution in `if __name__ == "__main__":` |
| `RuntimeError: UVA is not available` | WSL2 lacks Unified Virtual Addressing | `VLLM_USE_V2_MODEL_RUNNER=0` (already set automatically) |
| `Failed to find C compiler` / `Python.h: No such file` | Missing build tools | `sudo apt install build-essential python3-dev -y` |
| `Could not find nvcc` | FlashInfer sampler needs `nvcc` | `VLLM_USE_FLASHINFER_SAMPLER=0` (already set automatically) |
| `PermissionError: ... 'nvcc'` during engine start | WSL's PATH includes Windows folders that Python cannot execute | `export PATH=$(echo "$PATH" \| tr ':' '\n' \| grep -v '^/mnt/' \| paste -sd:)` |
| `pynvml.NVMLError_Unknown` | NVML utilization query is unreliable in WSL2 | VelocityLLM uses `nvidia-smi` through `subprocess` instead |
| vLLM or `transformers` fails to import after upgrading `huggingface-hub` | `transformers` rejects `huggingface-hub` 2.x | `pip install "huggingface-hub>=1.5.0,<2.0"` |
| `nvidia-smi` not found in WSL | Windows NVIDIA driver is not WSL-enabled | Install the current Windows NVIDIA driver; `nvidia-smi` must work in WSL |

### Phone, Arena and live demo

| Symptom | Cause | Fix |
|---|---|---|
| Phone shows the marketing website, not the join page | The link has no hash route | Use `http://<ip>:9000/#/join` (presenter: `#/live`) |
| Phone cannot reach the laptop | Wrong IP, different network, or no port forward | Take the IPv4 of the Windows `Wi-Fi` adapter from `ipconfig`; put laptop and phones on the same hotspot; recreate the `portproxy` rule |
| `portproxy` worked yesterday, not today | WSL's IP changes after every restart | Re-run the `$wsl = ...` PowerShell block above |
| `http://<ip>:9000/api/preflight` works on the laptop but not the phone | Windows firewall or hotspot client isolation | Add the firewall rule; if the hotspot isolates clients, use a Cloudflare tunnel (`--protocol http2`) |
| `404` on `/smart/...` | Smart endpoints exist only with `--policy smart` | Restart the server with `--policy smart` |
| Smart rejects most of a sudden 100-request burst with `queue_overload` | Burst shedding starts when the queue is about 90% full | Spread arrivals out, or raise `max_queue_size` in `ServerConfig` |
| Dynamic scheduler looks *slower* than static in a custom test | A blocking GPU query inside the async path | Fixed in this codebase (GPU stats are polled from a background thread); make sure you run a current version |

---

## Limitations (stated up front)

- **One GPU class.** The five-model table is from one RTX 5050 Laptop GPU (8 GB, power-limited, WSL2); the earlier Smart tables are from an RTX 4050 Laptop (6 GB). Other GPUs and larger models (7B and up) are not yet measured.
- **One headline workload.** The five-model table uses a 100-user simultaneous flood. The `mixed` scenario and long-context workloads are supported by the tool but not part of that table.
- **Median of 3.** The summary shows the median; per-repeat spread is in `raw/`. A laptop GPU can drift between runs, so compare policies inside the same run only.
- **Small models gain less.** tinyllama-1.1b improved by +34% throughput and -6% p50; the gain grows with model size and KV pressure.
- **Smart is not a throughput feature.** It protects real-time SLAs under overload by shedding best-effort work; at equal concurrency caps its throughput and goodput match Static's on uniform load.
- **Mock numbers are simulated.** `--mock` exercises the pipeline and says nothing about GPU performance.
- **Alpha software.** PyPI classifier `Development Status :: 3 - Alpha`; APIs may still change.

---

## Roadmap

- [x] **Phase 0** -- Environment, GPU/CUDA/vLLM verified end-to-end
- [x] **Phase 1** -- Baseline static-batching server, load generator, GPU monitoring
- [x] **Phase 2** -- Core dynamic scheduler: SLA-aware admission control, adaptive AIMD batch sizing, priority queue with anti-starvation aging
- [x] **Phase 3** -- Robustness: input validation, GPU memory limits, OOM recovery, client-disconnect reclamation, burst shedding, structured logging
- [x] **Phase 4** -- Advanced load generation (Poisson/burst/mixed-prompt patterns) and comparative benchmarking against static batching
- [x] **Smart scheduler** -- opt-in `--policy smart`: ten features, decision trace, `/smart/*` API, predictive KV-aware capacity
- [x] **Phase 5** -- Arena web dashboard and live multi-device demo (`live-demo` branch)
- [x] **Phase 6 (packaging)** -- `pip install velocityllm`, terminal benchmark, Python API, 131 tests
- [ ] **Larger models and more GPUs** -- 7B-class models, data-center GPUs, mixed and long-context scenarios
- [ ] **Per-feature ablation** -- isolate the contribution of each Smart feature with interleaved runs
- [ ] **Triton Inference Server deployment** (stretch)

Full detail: [`docs/VelocityLLM_Implementation_Plan.md`](docs/VelocityLLM_Implementation_Plan.md)

---

## Documentation

| Document | Description |
|---|---|
| [`docs/VelocityLLM_PRD.md`](docs/VelocityLLM_PRD.md) | Product requirements: goals, edge cases, test plan, KPIs |
| [`docs/VelocityLLM_Implementation_Plan.md`](docs/VelocityLLM_Implementation_Plan.md) | Phased execution plan with exit criteria |
| [`docs/baseline_results_summary.md`](docs/baseline_results_summary.md) | Benchmark methodology, results, and root-cause log for every issue found in testing |
| [`docs/SMART_SCHEDULER.md`](docs/SMART_SCHEDULER.md) | Smart scheduler reference: features, endpoints, tunables, trace events, validation, limitations |
| [`docs/Control_Service.md`](docs/Control_Service.md) | Arena / control service reference |
| [`docs/rtx4050_results/`](docs/rtx4050_results/) | Raw CSVs and notes from the RTX 4050 (6 GB) runs |

---

## Team

**Kartik R. Pagariya** -- Scheduler architecture, infrastructure, benchmarking, testing, packaging
**Vikrant Kadam** -- Phase 2/3 scheduler engine implementation
**Aditya Dengale** -- Contributor
B.Tech, Artificial Intelligence and Data Science -- Vishwakarma Institute of Technology, Pune
GitHub: [@kartikpagariya25](https://github.com/kartikpagariya25) · PyPI: [`velocityllm`](https://pypi.org/project/velocityllm/)

---

<div align="center">

*Built as an industry-guided academic project under Dr. Viomesh, sponsored by Single Core Labs.*

</div>
