<div align="center">

<img src="assets/logo.png" alt="VelocityLLM Banner" width="60%" style="border-radius: 20px;"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.11%2B-FF7A00?style=for-the-badge&logo=python&logoColor=white)
![vLLM](https://img.shields.io/badge/Powered%20by-vLLM-FF8C00?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-FF9500?style=for-the-badge&logo=nvidia&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-FFA500?style=for-the-badge&logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-FFB347?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-FFC300?style=for-the-badge)

**A dynamic / continuous batching scheduler engine for LLM inference serving —**
**built to prove, with live numbers, that intelligent scheduling raises GPU throughput without breaking latency SLAs.**

[Overview](#overview) • [Architecture](#architecture) • [Setup](#setup-guide) • [Usage](#usage) • [Troubleshooting](#troubleshooting--common-fixes) • [Roadmap](#roadmap)

</div>

---

## Overview

**VelocityLLM** is an industry-grade systems engineering project that tackles a real production problem in LLM serving: **static batch sizes under-utilize GPU throughput under variable request loads.**

This project builds a **dynamic/continuous batching scheduler** on top of [vLLM](https://github.com/vllm-project/vllm) that:

- Maximizes GPU utilization under bursty, real-world traffic patterns
- Enforces configurable **latency SLAs** (p50 / p95 / p99 targets) via SLA-aware admission control
- Is **model-agnostic** — works with any vLLM-supported causal LM without code changes
- Proves its improvement with a **live benchmarking dashboard**, comparing itself against a traditional static-batching baseline

> Academic/Industry Project — guided by Dr. Viomesh | Author: Kartik R. Pagariya

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
                   | - Request validation                 |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | Admission Controller                 |
                   | - SLA-aware gating                   |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | Dynamic Batching Scheduler            |
                   | - Iteration-level queueing            |
                   | - Adaptive batch sizing               |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | vLLM Inference Engine                |
                   | (PagedAttention + Continuous          |
                   |  Batching)                            |
                   +--------------------------------------+
                                       |
                                       v
                   +--------------------------------------+
                   | GPU (CUDA)                            |
                   +--------------------------------------+

Observability path:
GPU / Scheduler / API  -->  Metrics (Prometheus format)  -->  Streamlit Dashboard

Comparison path:
Static Batching Server (separate policy, same codebase)  -->  same dashboard, second series
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Inference engine | [vLLM](https://github.com/vllm-project/vllm) (PagedAttention + continuous batching) |
| API layer | FastAPI + Uvicorn |
| Scheduler core | Pure Python (asyncio), policy-pattern design |
| Load generator | Python asyncio + aiohttp (custom CLI) |
| GPU monitoring | `nvidia-smi` via subprocess (see [Troubleshooting](#troubleshooting--common-fixes) for why) |
| Metrics | Prometheus-format counters/histograms |
| Dashboard | Streamlit |
| Serving infra (stretch) | Triton Inference Server + vLLM backend |
| Testing | pytest, pytest-asyncio, httpx |
| Model | Llama-3.2-1B-Instruct (small, fits 8GB VRAM) |

---

## Project Structure

```
velocityllm/
├── scheduler_engine/       Core FastAPI server, scheduler policies, GPU monitor
│   ├── baseline_server.py  Static-batching baseline (AsyncLLMEngine)
│   ├── gpu_monitor.py      Background GPU sampling (used during load tests)
│   └── gpu_watch.py        Standalone live GPU terminal monitor
├── load_generator/         Async traffic simulation CLI
│   └── generate_load.py
├── dashboard/              Streamlit live/replay comparison dashboard
├── tests/                  Unit, integration, load, and chaos tests
├── docs/                   PRD, implementation plan, benchmark reports
├── models/                 Downloaded model weights (gitignored)
├── requirements.txt
├── .gitignore
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
git clone https://github.com/kartikpagariya25/velocityllm.git
cd velocityllm
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> If venv creation fails with `ensurepip is not available`, run:
> `sudo apt install python3.14-venv -y` (match the package name to your exact `python3` version).

### 3. Fix pip's Temp Directory (WSL2 Only — Avoids "No Space Left on Device")

WSL's `/tmp` is a small RAM-backed filesystem (~3.7GB) even when your real disk has hundreds of GB free.

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

### 6. Set Required Environment Variables (WSL2 + Blackwell GPU Specific)

```bash
echo 'export VLLM_USE_V2_MODEL_RUNNER=0' >> ~/.bashrc
echo 'export VLLM_USE_FLASHINFER_SAMPLER=0' >> ~/.bashrc
source ~/.bashrc
```

See [Troubleshooting](#troubleshooting--common-fixes) for exactly why each of these is required.

### 7. Hugging Face Authentication

```bash
pip install -U huggingface_hub
hf auth login
```

You'll be prompted for a token — create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (type = **Read**).

### 8. Download the Model

```bash
mkdir -p models
hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir models/llama-3.2-1b \
  --exclude "original/*"
```

> `Llama-3.2-1B-Instruct` is gated — request access on its [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) first (approval can take a few minutes to a day). To skip waiting, substitute a non-gated equivalent like `Qwen/Qwen2.5-1.5B-Instruct` — the scheduler is model-agnostic and works identically either way.
>
> The `--exclude "original/*"` flag is important — without it you download a duplicate `.pth` copy alongside the `.safetensors` copy, roughly doubling download size for no benefit.

### 9. Verify the Full Stack Works

```bash
python3 scheduler_engine/test_inference.py
```

You should see a generated response printed at the end (e.g., "The capital of France is Paris.").

---

## Usage

### Run the Dynamic Scheduler Server (Production / Phase 2)

```bash
# Run Dynamic Continuous Batching Server (Default)
python3 -m scheduler_engine.server --policy dynamic --port 8000

# Run in Mock mode (for testing or development without a GPU)
python3 -m scheduler_engine.server --policy dynamic --mock --port 8000

# Or run Static Baseline Policy on the unified server
python3 -m scheduler_engine.server --policy static --port 8000
```

### Run the Legacy Baseline Server (Phase 1 Reference)

```bash
uvicorn scheduler_engine.baseline_server:app --host 0.0.0.0 --port 8000
```

### Send a Test Request (in a separate terminal)

```bash
# VelocityLLM Native Generation Endpoint
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain dynamic continuous batching.", "max_tokens": 60, "priority": 1}'

# OpenAI-Compatible Completions Endpoint
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the speed of light?", "max_tokens": 50, "priority": "high"}'
```

### Check Scheduler Telemetry & Prometheus Metrics

```bash
# Real-time scheduler internal stats
curl http://localhost:8000/stats

# Prometheus scraper metrics
curl http://localhost:8000/metrics
```

### Run the Phase 2 Verification Suite

```bash
python3 verify_phase2.py
```

### Run Automated Unit & Integration Tests

```bash
pytest tests/ -v
# or
python3 run_tests.py
```

### Run a Load Test with Live GPU Monitoring & Priority Traffic

```bash
python3 load_generator/generate_load.py \
  --num-requests 30 \
  --concurrency 8 \
  --max-tokens 100 \
  --priority mixed \
  --mixed-tokens \
  --output results.csv
```

This prints a clearly highlighted summary block:

```
==============================================================
               VELOCITY-LLM BENCHMARK RESULTS
==============================================================
  Total Requests Submitted    : 30
  Completed Successfully (200): 30 (100.0%)
  Shed under SLA Policy (429) : 0 (0.0%)
  Wall Elapsed Time           : 4.12s
--------------------------------------------------------------
  LATENCY & THROUGHPUT (SUCCESSFUL REQUESTS)
--------------------------------------------------------------
  Request Throughput          : 7.28 req/sec
  Token Throughput            : 364.00 tokens/sec
  p50 Latency                 : 1.250s
  p95 Latency                 : 1.840s
  p99 Latency                 : 2.110s
  SLA Compliance Rate         : 100.0%
==============================================================
```

### Watch GPU Usage Live (Separate Terminal, Optional)

```bash
python3 scheduler_engine/gpu_watch.py
```

---

## Environment Variables Reference

| Variable | Value | Why It's Needed |
|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER` | `0` | WSL2's CUDA implementation doesn't support UVA, which vLLM's V2 Model Runner requires |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | FlashInfer's sampler tries to JIT-compile with `nvcc`, which isn't installed by default |
| `TMPDIR` | `$HOME/tmp_pip` | Redirects pip's temp files off WSL2's tiny `/tmp` tmpfs |
| `gpu_memory_utilization` (engine arg) | `0.80` | Default `0.92` is too aggressive for 8GB VRAM under WSL2's driver overhead |
| `max_model_len` (engine arg) | `4096` | Llama's default 131072 needs 4GiB+ of KV cache alone |

---

## Troubleshooting / Common Fixes

Every one of these was hit and resolved during this project's own setup — applying them upfront avoids repeating the same debugging cycle.

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | vLLM uses `spawn` multiprocessing on WSL2; script lacks a main guard | Wrap execution in `if __name__ == "__main__":` |
| `ValueError: Free memory on device ... is less than desired GPU memory utilization` | Default `gpu_memory_utilization=0.92` too high for 8GB VRAM | Pass `gpu_memory_utilization=0.80` |
| `RuntimeError: UVA is not available` | WSL2 doesn't support Unified Virtual Addressing, needed by vLLM's V2 Model Runner | `export VLLM_USE_V2_MODEL_RUNNER=0` |
| `fatal error: Python.h: No such file or directory` | Missing Python development headers | `sudo apt install python3-dev -y` |
| `RuntimeError: Failed to find C compiler` | No `gcc` on a fresh WSL2 Ubuntu install | `sudo apt install build-essential -y` |
| `RuntimeError: Could not find nvcc and default cuda_home doesn't exist` | FlashInfer sampler needs `nvcc` to JIT-compile a kernel | `export VLLM_USE_FLASHINFER_SAMPLER=0` |
| `ValueError: ... KV cache is needed, which is larger than the available KV cache memory` | Model's default max context (131072 tokens) needs 4GiB+ KV cache | Pass `max_model_len=4096` |
| `pynvml.NVMLError_Unknown` when querying GPU utilization | NVML's utilization-query function is unreliable inside WSL2 | Use `nvidia-smi` via `subprocess` instead of `pynvml` for utilization |
| `ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device` | WSL2's `/tmp` is a small RAM-backed tmpfs, unrelated to real disk space | Set `TMPDIR` to a path on the real filesystem (see Setup step 3) |

---

## Results (Baseline — Static Batching)

**Model:** Llama-3.2-1B-Instruct · **Config:** `max_num_seqs=8`, `gpu_memory_utilization=0.80`, `max_model_len=4096`

| Run | Requests | Concurrency | Throughput | p50 | p99 | Avg GPU % | Max GPU % |
|---|---|---|---|---|---|---|---|
| Light Load | 20 | 5 | 7.20 req/s | 1.888s | 2.771s | 83.6% | 99.0% |
| Heavy / Stress | 30 | 10 | 4.25 req/s | 4.657s | 7.056s | 94.7% | 100.0% |

As load increases, the GPU saturates toward 100% utilization, but end-to-end p99 latency nearly triples. This throughput-vs-latency trade-off is exactly what the dynamic scheduler is designed to manage more intelligently through adaptive batch sizing and SLA-aware admission control.

Full logs and methodology: [`docs/baseline_results_summary.md`](docs/baseline_results_summary.md)

---

## Roadmap

- [x] **Phase 0** — Environment, GPU/CUDA/vLLM verified end-to-end
- [x] **Phase 1** — Baseline static-batching server, load generator, GPU monitoring
- [x] **Phase 2** — Core dynamic scheduler: SLA-aware admission control, adaptive batch sizing, priority + aging, swappable policy architecture
- [ ] **Phase 3** — Robustness: input validation, OOM protection, structured logging, full test suite
- [ ] **Phase 4** — Advanced load generation (bursty/Poisson traffic) and full comparative benchmarking
- [ ] **Phase 5** — Live Streamlit dashboard with static-vs-dynamic comparison view
- [ ] **Phase 6** — CLI packaging, final benchmark report, optional Triton Inference Server deployment
- [ ] **Phase 7** — Demo rehearsal and presentation

Full detail: [`docs/VelocityLLM_Implementation_Plan.md`](docs/VelocityLLM_Implementation_Plan.md)

---

## Documentation

| Document | Description |
|---|---|
| [`docs/VelocityLLM_PRD.md`](docs/VelocityLLM_PRD.md) | Full product requirements — goals, edge cases, test plan, KPIs |
| [`docs/VelocityLLM_Implementation_Plan.md`](docs/VelocityLLM_Implementation_Plan.md) | Phased execution plan with exit criteria |
| [`docs/baseline_results_summary.md`](docs/baseline_results_summary.md) | Baseline benchmark reference numbers |

---

## Author

**Kartik R. Pagariya**
B.Tech, Artificial Intelligence and Data Science — Vishwakarma Institute of Technology, Pune
GitHub: [@kartikpagariya25](https://github.com/kartikpagariya25)

---

<div align="center">

*Built as an industry-guided academic project under Dr. Viomesh.*

</div>
