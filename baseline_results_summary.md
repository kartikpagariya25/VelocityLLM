# VelocityLLM — Benchmark Results Summary

**For:** Team (Kartik, Vikrant, Aditya, Pranali) & Dr. Viomesh
**Last updated:** Phase 4 completion
**Model:** Llama-3.2-1B-Instruct
**Hardware:** NVIDIA RTX 5050 (8GB VRAM), WSL2 Ubuntu
**Server config:** `gpu_memory_utilization=0.80`, `max_model_len=4096`

---

## 1. What Is Being Compared

Two scheduler policies running through the **same unified server codebase** (`scheduler_engine/server.py`), switched via `--policy static` or `--policy dynamic`:

- **Static Baseline** (`StaticBatchPolicy`) — fixed concurrency (8 slots), first-come-first-served, no SLA awareness, no admission control. Represents a traditional serving setup.
- **Dynamic Scheduler** (`DynamicBatchPolicy`) — SLA-aware admission control, adaptive AIMD batch-size controller, priority queue with anti-starvation aging, GPU memory soft/hard limit protection, OOM recovery, client-disconnect slot reclamation.

Both sit on top of the same vLLM `AsyncLLMEngine` (PagedAttention + continuous batching), so the comparison isolates the value added by the scheduling layer itself, not the underlying inference engine.

---

## 2. Test 1 — Simple Saturation Load ("Flood" Pattern)

All requests fired at once, no pacing. This tests raw throughput ceiling under sustained pressure.

| Run | Requests | Concurrency | Max Tokens | Throughput | p50 | p99 | Avg GPU % | Max GPU % |
|---|---|---|---|---|---|---|---|---|
| Static Baseline | 30 | 10 | 100 | 7.07 req/s | 2.396s | 4.241s | 89.3% | 99% |
| Dynamic Scheduler | 30 | 10 | 100 | **8.11 req/s** | **1.871s** | **3.198s** | 83.4% | 99% |

**Result: Dynamic gave +14.7% higher throughput, -21.9% better p50, -24.6% better p99**, with 4 requests proactively shed (HTTP 429) to protect SLA for the rest — static accepted all 30 but let latency degrade unchecked.

### Diagnostic note (important for the team to know)
An earlier version of this test showed the *opposite* result (dynamic slower than static). Root cause: `VLLMBackend.get_gpu_telemetry()` was calling `nvidia-smi` as a blocking `subprocess.run()` call directly inside the async request-handling path, freezing the entire event loop for 20-50ms on every call. Fixed by moving GPU polling into a dedicated background thread (`_telemetry_loop`) that updates a cached value every `gpu_sample_interval` (0.2s); the async path now only reads memory, never blocks. **This fix is required — without it, the dynamic scheduler's real advantage is masked by unrelated I/O overhead.**

---

## 3. Test 2 — Poisson (Realistic Random) Traffic

40 requests spread across a 10-second window using Poisson-distributed inter-arrival times — simulates organic, non-bursty real-world traffic.

| Run | Throughput | p50 | p99 | SLA compliance | Avg GPU % |
|---|---|---|---|---|---|
| Static Baseline | 4.15 req/s | 0.530s | 0.835s | 100.0% | 84.0% |
| Dynamic Scheduler | 4.10 req/s | 0.531s | 0.829s | 100.0% | 80.5% |

**Result: Statistically equivalent.** At this load level (below saturation), both policies perform the same — expected, since the scheduling layer only matters when there's contention to manage. This is a useful negative control: it shows the dynamic scheduler adds no overhead/regression under light load.

---

## 4. Test 3 — Bursty Traffic

40 requests over 15 seconds with two deliberate high-rate burst windows, rest at low background rate — simulates traffic spikes (e.g. a link going viral).

| Run | Throughput | p50 | p99 | Avg GPU % |
|---|---|---|---|---|
| Static Baseline | 1.11 req/s | 0.509s | 0.693s | 34.2% |
| Dynamic Scheduler | 1.11 req/s | 0.525s | 0.743s | 33.4% |

**Result: Statistically equivalent.** Low average GPU utilization in both is expected and correct — most of the 15-second window has no traffic (only the burst windows do), so the GPU is legitimately idle between bursts. Neither policy was pushed into contention here since burst amplitude was moderate relative to capacity.

---

## 5. Test 4 — Mixed Short/Long Prompts (Headline Scenario)

30 requests, ~25% long (300-500 max tokens) mixed with ~75% short (30-80 max tokens), all fired at once. This is the classic **head-of-line blocking test**: does a long-running request starve short requests queued behind it?

| Run | Throughput | p99 (overall) | **SLA compliance** | Long-prompt p99 | Short-prompt p99 |
|---|---|---|---|---|---|
| Static Baseline | 5.13 req/s | 5.503s | **80.0%** | 5.503s | 1.373s |
| Dynamic Scheduler | 5.08 req/s | 5.218s | **100.0%** | 5.218s | 1.222s |

**This is the most important result in the whole benchmark.**

- Static let **20% of requests silently breach their SLA target** — real users would have experienced worse-than-promised latency with zero system-level warning.
- Dynamic maintained **100% SLA compliance** by proactively rejecting 1 request (3.3% of load) that it predicted could not be served within its SLA window — the client gets an immediate, honest "try again" (HTTP 429) instead of a slow, silent failure.
- Short prompts stayed fast in both cases here (no severe head-of-line blocking observed at this specific queue depth), but the **SLA-compliance gap (80% vs 100%) is the real, statistically meaningful finding** — throughput was nearly identical (5.13 vs 5.08 req/s).

**Takeaway for the report:** The value of the dynamic scheduler is not "it's always faster" — it's that **it never lies about latency**. Static either serves fast or serves slow with no distinction; Dynamic serves fast, or explicitly declines, but never silently breaks its promise.

---

## 6. Root Cause Log (Issues Found & Fixed During Benchmarking)

Documented here so nobody re-discovers these independently:

1. **Event-loop-blocking GPU telemetry** (see Test 1 note above) — fixed via background polling thread.
2. **Default SLA target (3000ms) was too tight** for this model/hardware combination under 10-concurrency load — caused unrealistic over-rejection (17/30 shed). Raised to 8000ms (`--sla-ms 8000`) for fair, realistic comparisons. This value should be revisited once a production SLA requirement is actually specified.
3. **Server must be fully terminated before restarting on the same port** — a lingering background process caused a silent RAM/VRAM split between two vLLM instances, corrupting one benchmark run's numbers (`Available RAM: 1.55 GiB` instead of the normal ~4 GiB). Always verify with `lsof -i :8000` before restarting.

---

## 7. What This Proves (For the Final Report)

- The dynamic scheduler adds **no measurable regression** under light/moderate load (Poisson, burst tests — statistically equivalent to static).
- Under saturating load, it delivers **measurably higher throughput and lower latency** (+14.7% throughput, -24.6% p99).
- Under mixed workload (the realistic production scenario), it delivers **materially better SLA reliability** (100% vs 80% compliance) at comparable throughput — this is the core "meets latency SLAs while maximizing utilization" claim from the original problem statement.

## 8. What's Still Open (Honest Limitations)

- Burst test did not push GPU into genuine contention — a stronger burst amplitude test would better demonstrate adaptive concurrency scaling under spike conditions.
- Only tested on a single model (Llama-3.2-1B) and single GPU — model-agnosticism and multi-GPU scaling remain claims to verify empirically if time permits (see PRD Section 3.2 Non-Goals).
- SLA target (8000ms) was tuned for this hardware; a production deployment would calibrate this against real business requirements, not benchmark convenience.

---

*Raw data for every run above is in `results/*.csv` and `results/*_summary.json` in the repository.*
