# VelocityLLM — Phased Implementation Plan

**Version:** 1.0
**Companion document to:** VelocityLLM_PRD.md
**Author:** Kartik R. Pagariya

---

## How to Use This Plan

Each phase has an explicit **exit criterion**. You do not move to the next phase until it's met — this is intentional. Production engineering runs on completed, verified milestones, not arbitrary calendar weeks.

---

### Phase 0 — Environment & Foundations

**Goal:** Prove the GPU/CUDA/vLLM stack works end-to-end before writing any scheduler logic.

- Verify GPU, driver, CUDA toolkit installation (`nvidia-smi` sanity check, in both Windows and WSL2)
- Set up WSL2 Ubuntu as the primary development environment (better vLLM/Linux support)
- Install Python environment, pin dependencies
- Install vLLM, download Llama-3.2-1B-Instruct from Hugging Face, run a single manual inference request successfully
- Install `pynvml`, confirm programmatic GPU metric reads work
- Initialize git repository with proper structure

**Exit criterion:** You can manually send one prompt to vLLM and get a correct completion back, and you can read live GPU utilization via Python.

---

### Phase 1 — Baseline System (Static Batching)

**Goal:** Build the "traditional system" you will spend the rest of the project outperforming.

- Wrap vLLM in a minimal FastAPI service with static/fixed batch configuration
- Build the load generator CLI (constant pattern first, simplest case)
- Build basic metrics collection (latency, throughput, GPU util) and log to file
- Run first baseline benchmark, save raw results

**Exit criterion:** You have real numbers proving the static system's throughput/latency/utilization ceiling — this becomes your comparison anchor for everything after.

---

### Phase 2 — Core Dynamic Scheduler Engine

**Goal:** Build the actual novel contribution.

- Design the `SchedulerPolicy` interface (static vs dynamic as swappable implementations, not separate codebases)
- Implement iteration-level continuous batching logic on top of vLLM's async engine
- Implement admission controller with SLA-aware accept/queue/reject logic
- Implement adaptive batch-size controller (feedback loop from GPU/queue metrics)
- Implement priority + aging logic (prevents starvation)

**Exit criterion:** Dynamic scheduler runs correctly, produces valid completions, and initial informal testing shows better GPU utilization than the Phase 1 baseline under identical simple load.

---

### Phase 3 — Robustness & Edge Case Hardening

**Goal:** Make it production-grade, not just "works on the happy path."

- Implement input validation and sanitization
- Implement GPU memory soft-limit protection and OOM recovery
- Implement client-disconnect detection and slot reclamation
- Implement burst-shedding / backpressure
- Implement structured logging with request correlation IDs
- Write the full unit + integration test suite against all of the above

**Exit criterion:** Full edge-case test suite passes; deliberately feeding malformed/adversarial/overload traffic does not crash the system.

---

### Phase 4 — Advanced Load Generation & Full Benchmarking

**Goal:** Produce the numbers that will headline your presentation.

- Extend load generator: ramp-up and Poisson/bursty patterns, mixed short/long prompt scenario (the headline test)
- Run full comparative benchmark suite: static vs dynamic, across all traffic patterns, with fixed seeds
- Run soak test for stability verification
- Compute final KPI numbers with real data, not placeholders

**Exit criterion:** You have a complete, reproducible dataset showing throughput gain %, GPU utilization gain %, and equal-or-better p99 latency, across multiple traffic scenarios.

---

### Phase 5 — Dashboard & Visualization

**Goal:** Make the results self-evidently clear to a non-technical evaluator within seconds of looking at the screen.

- Design dashboard layout (deep UI discussion: layout, graph placement, live vs. replay mode, color coding for baseline vs. dynamic)
- Build Streamlit app: live metrics streaming view + historical benchmark replay view
- Build the "headline" comparison visualization (side-by-side or overlaid throughput/latency/utilization curves with computed % improvement badges)
- (Stretch) Grafana + Prometheus integration for a more "industry SRE dashboard" aesthetic

**Exit criterion:** Someone with zero context on your project can look at the dashboard for 30 seconds and correctly state "the dynamic system uses the GPU more efficiently and generates more throughput without hurting response time."

---

### Phase 6 — Deployment Packaging & Documentation

**Goal:** Package it as something that looks like it could actually ship.

- CLI entry points for scheduler, baseline server, and load generator (proper argparse/click-based CLIs)
- (Stretch) Package/deploy via Triton Inference Server with vLLM backend
- Write README with architecture diagram, setup instructions, and how-to-reproduce-benchmarks guide
- Write the final benchmark report document (methodology, results, graphs, conclusions, honest discussion of limitations and future work)

**Exit criterion:** A stranger could clone the repo, follow the README, and reproduce your core benchmark result independently.

---

### Phase 7 — Presentation & Demo Rehearsal

**Goal:** Make sure the live demo actually works under presentation pressure.

- Prepare a demo runbook (exact commands, exact order, fallback plan if live GPU demo fails — pre-recorded backup clip)
- Rehearse the "killer number" narrative (e.g., "X% more throughput, same p99 latency")
- Prepare answers for likely questions: "why not just increase static batch size?", "how does this generalize to other models?", "what happens at 10x this load?", "what's your SLA enforcement mechanism exactly?"

**Exit criterion:** You can run the entire demo start-to-finish without needing to explain away a bug live.

---

## Open Discussion Items (Resolve With Mentor / In Follow-Up Sessions)

- Final SLA target numbers (locked in after Phase 1 baseline data exists)
- Whether Triton + Grafana stretch goals are pursued, based on time remaining after Phase 4
- Whether a second model is benchmarked to prove model-agnosticism claim empirically (Phase 4 stretch)

---

*Next step: Phase 0 kickoff — environment verification, git repo setup, and vLLM installation walkthrough.*
