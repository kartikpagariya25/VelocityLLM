# VelocityLLM — Dynamic Batching Engine for LLM Inference Serving
## Product Requirements Document

**Version:** 1.0
**Author:** Kartik R. Pagariya
**Sponsor:** Dr. Viomesh (Academic/Industry Mentor)
**Document Type:** PRD

---

## 1. Executive Summary

VelocityLLM is a dynamic/continuous batching scheduler engine built on top of vLLM that maximizes GPU utilization for LLM inference serving while enforcing strict latency SLAs. It is designed to be model-agnostic, production-grade, and independently verifiable through a live benchmarking dashboard that quantifies throughput and utilization gains against a traditional static-batching baseline.

**One-line pitch:** *A drop-in scheduling layer that proves, with live numbers, that intelligent batching can raise GPU throughput without breaking latency promises.*

---

## 2. Problem Statement

Static batch sizing in production LLM serving causes GPUs to idle whenever request completion times vary within a batch (short vs. long generations), and to reject or queue requests when traffic bursts exceed the fixed batch capacity. This results in:
- Under-utilized, expensive GPU hardware (industry data shows 30-40% SM utilization under static batching at moderate load)
- Inconsistent latency under bursty, real-world traffic patterns
- No adaptive mechanism to trade off throughput vs. latency based on live system state

## 3. Goals & Non-Goals

### 3.1 Goals
- G1: Build a continuous/dynamic batching scheduler achieving measurably higher throughput than static batching at equal or better p99 latency.
- G2: Enforce configurable latency SLAs (p50/p95/p99 targets, TTFT targets) via admission control and adaptive batch sizing.
- G3: Be architecture-agnostic — work with any vLLM-supported causal LM without model-specific code.
- G4: Provide a live, side-by-side benchmarking dashboard proving utilization/throughput gains with real percentages.
- G5: Handle production-relevant failure modes gracefully (OOM, GPU contention, malformed requests, timeouts, overload).
- G6: Ship as a deployable service (CLI-launched backend + REST API + dashboard), not a notebook/script.

### 3.2 Non-Goals (explicitly out of scope, stated for scope honesty)
- Multi-GPU / distributed tensor-parallel scheduling (mentioned as future work)
- Multi-modal (image/audio + text) request handling
- Custom CUDA kernel development or modifying vLLM internals/attention kernels
- Fine-tuning, quantization research, or model training of any kind
- Multi-tenant billing/auth systems (basic API-key gating only, if time permits)

---

## 4. Users & Use Cases

| User                                 | Use Case                                                                                  |
|---------------------------------------|--------------------------------------------------------------------------------------------|
| ML Platform Engineer (persona)        | Deploys VelocityLLM in front of vLLM to serve a chat/completion API under variable load    |
| SRE / On-call (persona)               | Monitors dashboard for SLA violations, GPU saturation, error rates                          |
| Evaluator / Mentor / Recruiter        | Reviews benchmark report + live demo to assess engineering rigor                            |

---

## 5. System Architecture

```
                     +--------------------------+
   Client Requests → |   API Gateway (FastAPI)   |
                     |   - Request validation     |
                     |   - Auth (API key, opt.)   |
                     +-------------+--------------+
                                   |
                     +-------------v--------------+
                     |   Admission Controller       |
                     |   - SLA-aware gating          |
                     |   - Backpressure / 429s       |
                     +-------------+--------------+
                                   |
                     +-------------v--------------+
                     |   Dynamic Batching            |
                     |   Scheduler (Core Engine)      |
                     |   - Iteration-level queue        |
                     |   - Priority policy                |
                     |   - Adaptive batch sizing            |
                     +-------------+--------------+
                                   |
                     +-------------v--------------+
                     |   vLLM Inference Engine        |
                     |   (PagedAttention +              |
                     |    continuous batching)            |
                     +-------------+--------------+
                                   |
                     +-------------v--------------+
                     |   GPU (CUDA)                     |
                     +--------------------------------+

Parallel path (observability):
GPU / Scheduler / API  -->  Metrics Exporter (Prometheus format)
                        -->  Time-series store
                        -->  Streamlit / Grafana Dashboard (live comparison view)

Baseline path (for comparison):
Static Batching Server (separate process, fixed max-num-seqs, same model)
                        -->  Metrics Exporter --> same dashboard, second series
```

### 5.1 Core Components

1. **API Gateway** — FastAPI service exposing OpenAI-compatible-ish `/v1/completions` endpoint.
2. **Admission Controller** — decides accept/queue/reject based on current load and SLA headroom.
3. **Scheduler Core** — the actual novel contribution: request queue, iteration-level batch assembly, eviction/insertion logic, priority policy, adaptive batch-size controller (feedback loop using live GPU/queue metrics).
4. **Inference Backend** — vLLM engine (`AsyncLLMEngine`), abstracted behind an interface so it is swappable.
5. **Metrics & Observability** — structured logging + Prometheus-format metrics (latency histograms, throughput counters, GPU util gauges via NVML).
6. **Baseline Server** — a separate, intentionally "dumb" static-batching server used purely for comparison; same codebase, different scheduler policy (`StaticBatchPolicy` vs `DynamicBatchPolicy`), so the comparison is apples-to-apples.
7. **Load Generator (CLI)** — configurable synthetic traffic tool (constant, ramp, burst, Poisson-distributed patterns).
8. **Dashboard** — live comparison UI (Streamlit primary, optional Grafana for polish).
9. **Triton Deployment Layer (stretch goal)** — optional packaging of the vLLM backend behind Triton Inference Server to demonstrate production-deployment maturity.

### 5.2 Design Principle: Model-Agnosticism

The scheduler operates only on request-level metadata (arrival time, prompt token count, generated token count so far, priority class, deadline). It never touches model weights, tokenizer internals, or logits. This guarantees compatibility with any HuggingFace-format causal LM supported by vLLM (Llama, Mistral, Qwen, Gemma, Phi, etc.) with zero code change — only a config swap.

---

## 6. Functional Requirements

| ID   | Requirement                                                                                                         |
|------|----------------------------------------------------------------------------------------------------------------------|
| FR1  | System accepts inference requests via REST API with prompt, max_tokens, priority (optional), and streaming flag.     |
| FR2  | Scheduler performs iteration-level (continuous) batching — new requests join an in-flight batch as capacity frees up. |
| FR3  | Scheduler enforces a configurable SLA target (e.g., p99 latency <= N ms); requests are prioritized/admission-controlled to protect this target. |
| FR4  | Adaptive batch-size controller adjusts effective concurrency based on live GPU memory headroom and queue depth.       |
| FR5  | System exposes real-time metrics: throughput (tokens/sec, req/sec), latency percentiles (p50/p95/p99), TTFT, GPU utilization %, GPU memory used, queue depth, active batch size. |
| FR6  | Baseline static-batching mode available via config flag for controlled comparison.                                    |
| FR7  | Load generator supports constant, ramp-up, and Poisson/bursty traffic patterns with reproducible seeding.             |
| FR8  | Dashboard displays live and historical (replayed) comparison between baseline and dynamic scheduler with computed % improvement figures. |
| FR9  | System gracefully rejects/queues requests under GPU memory pressure instead of crashing (OOM protection).             |
| FR10 | All requests, decisions, and errors are structured-logged with correlation/request IDs for traceability.              |

---

## 7. Non-Functional Requirements

| Category      | Requirement                                                                                          |
|----------------|--------------------------------------------------------------------------------------------------------|
| Performance    | Scheduler overhead must be < 5ms per admission decision (not the bottleneck itself)                     |
| Reliability    | No single malformed/oversized request should crash the server process                                   |
| Observability  | 100% of requests must be traceable via logs + metrics with request ID                                    |
| Reproducibility| Benchmarks must be re-runnable with fixed seeds and produce consistent results (+/-5% variance)          |
| Portability    | Must run on a single consumer/prosumer GPU (8-24GB VRAM range) using a small model                        |
| Code Quality   | Type-hinted Python, modular (policy pattern for scheduler strategies), no dead code, tested               |
| Security       | Input validation on all API fields, request size limits, optional API-key gating                          |

---

## 8. Comprehensive Edge Cases & Failure Modes

### 8.1 Traffic / Load Edge Cases
- E1: Sudden 10x traffic spike within 1 second (burst) → admission controller sheds load (HTTP 429) instead of degrading everyone.
- E2: Traffic drops to zero after sustained load → scheduler scales batch size back down, no stale reservations.
- E3: Single client sends requests in a tight loop (abuse) → per-client rate limiting.
- E4: Thundering herd — many requests arrive within the same millisecond (reconnect storm).

### 8.2 Request-Content Edge Cases
- E5: Extremely long prompt (near/exceeding max context) → reject with clear error, never silently truncate.
- E6: Extremely short / empty prompt → validated, rejected with 400, no tokenizer crash.
- E7: Unreasonably large `max_tokens` requested → clamp to server-configured maximum.
- E8: Malformed JSON / missing fields → clean 422 response, no stack trace leak.
- E9: Non-UTF8 / adversarial unicode input → sanitized without breaking tokenization.
- E10: One very-long-generation request mixed with many short ones (classic head-of-line blocking) — the headline scenario dynamic batching must win at.

### 8.3 Resource / GPU Edge Cases
- E11: GPU VRAM approaching exhaustion mid-batch → preemptive soft-limit stop on new admissions before hard OOM.
- E12: Actual CUDA OOM despite safeguards → caught, in-flight batch fails gracefully, service recovers without restart.
- E13: GPU shared with another process (contention) → utilization metrics documented honestly as a known limitation.
- E14: KV cache exhaustion under many long concurrent sequences → explicit, documented preemption policy.

### 8.4 Concurrency / System Edge Cases
- E15: Duplicate request IDs (client bug/replay) → deduplicated or explicitly rejected.
- E16: Client disconnects mid-stream → server detects and frees the compute slot early.
- E17: Scheduler/metrics race conditions → proper async locks/atomic counters, verified via concurrency tests.
- E18: Server restart/crash mid-request → in-flight requests fail cleanly with retriable error, no zombie state.

### 8.5 SLA / Policy Edge Cases
- E19: SLA target impossible at current load (saturation) → transparently reported as "SLA breach" in metrics, not silently absorbed.
- E20: Priority starvation under sustained high-priority load → aging mechanism boosts long-waiting low-priority requests.
- E21: Conflicting config (SLA target lower than model's achievable minimum) → fail fast at startup with clear error.

### 8.6 Comparison / Benchmarking Integrity Edge Cases
- E22: Baseline and dynamic runs must share identical seed, model, and hardware thermal state — documented methodology to preempt "unfair comparison" criticism.
- E23: Cold-start / CUDA graph warmup effects excluded from reported metrics, and this exclusion is explicitly documented.

---

## 9. Test Plan

### 9.1 Unit Tests
- Admission controller decision logic (accept/queue/reject) under mocked load states
- Scheduler queue insertion/eviction correctness (FIFO, priority ordering, aging logic)
- Config validation logic (rejects invalid SLA/config combos)
- Metrics computation correctness (percentile calculations against known datasets)
- Input validation/sanitization functions (E5-E9, each as a discrete test)

### 9.2 Integration Tests
- End-to-end request lifecycle: submit → schedule → generate → respond, with latency bound assertions
- Static vs dynamic scheduler produce correctness-parity completions (batching must not change output quality, only timing)
- Graceful degradation under simulated GPU-memory-pressure (mocked NVML readings)
- Client-disconnect handling (E16) verified via forced connection drops

### 9.3 Load / Performance Tests
- Constant load at increasing RPS until saturation — plot throughput vs. latency curve
- Burst pattern test (E1) — verify shedding behavior and recovery time
- Long-duration soak test (30-60 min) — check for memory leaks, metric drift
- Mixed short/long prompt workload (E10) — the headline benchmark scenario

### 9.4 Chaos / Resilience Tests
- Kill scheduler mid-batch, verify clean recovery (E18)
- Force a simulated OOM, verify graceful handling (E12)
- Inject malformed/adversarial requests continuously alongside normal traffic, verify no crash / no SLA impact on well-formed requests

### 9.5 Acceptance Criteria (Definition of Done)
- All unit + integration tests passing
- Load test demonstrates measurable throughput improvement at equal-or-better p99 latency vs. static baseline
- Zero unhandled crashes across the full edge-case test suite
- Dashboard renders comparison correctly from both live and replayed benchmark data

---

## 10. Tech Stack

| Layer                | Tool/Framework                          | Notes                                                                                   |
|-----------------------|------------------------------------------|-------------------------------------------------------------------------------------------|
| Inference engine       | vLLM                                     | Core PagedAttention + continuous batching primitives                                       |
| API layer              | FastAPI + Uvicorn                        | Async-native, matches vLLM's async engine well                                              |
| Scheduler core         | Pure Python (asyncio)                    | Policy-pattern design: `SchedulerPolicy` interface with Static/Dynamic implementations       |
| Load generator         | Python asyncio + aiohttp (custom)        | Full control over traffic shaping; `locust` as backup option                                |
| GPU monitoring         | `pynvml` (NVIDIA Management Library)     | Programmatic GPU util/memory reads for metrics + admission control                           |
| Metrics                | `prometheus_client`                      | Standard, dashboard-agnostic format                                                          |
| Dashboard              | Streamlit (primary)                      | Fast, Python-native, good for live-updating comparison views                                 |
| Serving infra (stretch)| Triton Inference Server + vLLM backend   | Demonstrates production-deployment maturity                                                  |
| Testing                | pytest, pytest-asyncio, httpx            | Async test client for API-level tests                                                        |
| Model                  | Llama-3.2-1B-Instruct                    | Small enough to run comfortably on 8GB VRAM                                                   |

### 10.1 NVIDIA Tools Reference

- **NVIDIA GPU Driver + CUDA Toolkit** — foundation for everything below
- **`nvidia-smi`** — first sanity check, GPU visibility
- **`nvitop`** — nicer terminal-based live GPU monitor (pip installable)
- **`pynvml`** — Python bindings to NVML, used by the metrics exporter
- **NVIDIA Nsight Systems** (optional/stretch) — kernel-level GPU profiling for report credibility
- **NVIDIA DCGM** (mention-only) — real datacenter GPU telemetry tool, referenced in report as "how this scales in production"
- **Triton Inference Server** (Docker image) — for the stretch deployment goal

---

## 11. Success Metrics (KPIs)

| Metric                                   | Baseline (Static)                  | Target (Dynamic)                         |
|--------------------------------------------|--------------------------------------|---------------------------------------------|
| GPU SM Utilization (%)                      | ~30-40% (expected, to be measured)   | >= 75-85%                                     |
| Throughput (tokens/sec)                     | Measured baseline value              | >= 1.5-2x baseline (refine after Phase 2)     |
| p99 Latency                                 | Measured baseline value              | Equal or better than baseline                  |
| SLA Violation Rate under burst              | High (expected)                      | Near-zero, with transparent shedding           |
| Crash rate across edge-case suite           | N/A                                   | 0                                               |

*(Exact target percentages will be locked in after the baseline measurement phase — presenting a target range now, backed by real numbers later, is more credible than inventing a number upfront.)*

---

## 12. Risks & Mitigations

| Risk                                                  | Mitigation                                                                                     |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Limited GPU VRAM (8GB)                                    | Use smallest viable model (Llama-3.2-1B); leaves headroom for KV cache/batching experiments         |
| vLLM API changes between versions                          | Pin exact vLLM version in requirements; document version used                                        |
| Unfair/non-reproducible benchmark accusations               | Fixed seeds, documented methodology, excluded warmup window (E22/E23), raw data saved and shareable  |
| Scope creep (trying to handle every LLM/every edge case)      | Section 3.2 Non-Goals explicitly locked in; stretch goals clearly separated from core deliverable    |
| Dashboard becomes a time sink                                  | Streamlit first (fast), Grafana only if time remains — never block core engine work for UI polish     |
| Blackwell (RTX 50-series) GPU compatibility issues               | Use latest vLLM release with confirmed SM120 support; WSL2/Ubuntu preferred over Windows-native install |

---

## 13. Deliverables Checklist

- [ ] Scheduler engine source code (modular, tested, type-hinted)
- [ ] Baseline static-batching server (for comparison)
- [ ] Load generator CLI tool
- [ ] Live comparison dashboard
- [ ] Full test suite (unit + integration + load + chaos)
- [ ] Benchmark report (numbers, graphs, methodology, conclusions)
- [ ] Architecture diagram (final polished version)
- [ ] README + setup/run instructions
- [ ] Demo script/runbook for presentation day

---

*End of PRD. See the companion document "VelocityLLM_Implementation_Plan.md" for the phased execution plan.*
