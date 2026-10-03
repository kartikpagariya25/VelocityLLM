# Smart Scheduler (`--policy smart`)

An opt-in scheduling policy layered on the dynamic scheduler. It is **token-aware**, **future-KV aware**,
**self-calibrating**, and **explains every decision**. Static and Dynamic are unchanged; pick the policy with `--policy`.

```bash
python3 -m scheduler_engine.server --policy smart --sla-ms 8000            # real GPU
python3 -m scheduler_engine.server --policy smart --sla-ms 8000 --mock     # no GPU
```

## How a request flows

```
Request -> Profiler -> Future-resource estimate -> Admission (early reject) -> Smart queue
        -> Capacity planner (min(memory-safe, SLA-safe), AIMD as brake) -> Backend (vLLM)
        -> runtime telemetry -> Self-calibration (updates the estimators) -> next request
                                  \-> Decision trace (every step above is logged)
```

Streaming (`/generate/stream`) and non-streaming requests share this same lifecycle (admission, queue, concurrency
limit, accounting). Duplicate `X-Correlation-ID`s are made unique (`id#n`) instead of colliding.

## The ten features

| # | Feature | Behaviour | Code |
|---|---|---|---|
| 1 | Token-aware scheduling | Request cost = prompt tokens + predicted output tokens. Prompt tokens use the repo's conservative estimator (no tokenizer needed) | `profiler.py` |
| 2 | Future KV prediction | Running requests are projected to their predicted length (EMA + p90 per prompt-length bin); admission checks *predicted* KV pressure, not only current usage | `estimators.py`, `smart_admission.py` |
| 3 | Memory limit vs SLA limit | `memory_limit_capacity` (KV budget / average footprint) and `sla_limit_capacity` (largest concurrency whose predicted latency is within `sla_safety` x SLA) are computed separately; `selected = min(...)`; AIMD remains a reactive brake. The binding constraint is reported | `capacity.py` |
| 4 | Deadline / slack priority | Urgency rises as a request's slack shrinks. Requests that can no longer meet their deadline are *de-prioritized*, not boosted (plain EDF collapses under overload) | `smart_queue.py` |
| 5 | Short-prompt-first | Small bonus for short prompts, smaller than one priority level, so HIGH-priority long prompts stay ahead. Switchable at runtime | `smart_queue.py` |
| 6 | Token-length buckets | SHORT <= 256, MEDIUM <= 1024, LONG above (configurable, estimated total tokens). LONG requests are capped to `max_long_fraction` of slots; dispatch prefers the active batch's bucket | `profiler.py`, `smart_queue.py` |
| 7 | Early rejection | Checks in order: `queue_overload`, `memory_risk`, `policy_limit`, `sla_risk`. Predicted completion is a small simulation over running + queued work using the calibrated latency models. Rejected requests never touch the GPU | `smart_admission.py` |
| 8 | Real-time vs best-effort | `X-Traffic-Class: real_time \| best_effort` (default: LOW priority = best-effort). Real-time is served first; best-effort is throttled when pressure is high and resumes with hysteresis; a starvation guard releases old best-effort work | `smart_queue.py`, `smart_policy.py` |
| 9 | Online self-calibration | A startup probe measures TTFT and per-token latency at concurrency 1/4/8; live requests keep updating exponentially-weighted models. Tracks its own prediction error (MAPE) and a learned queue-wait bias. No offline training | `estimators.py` |
| 10 | Decision trace | Every admit / reject / reorder / dispatch / complete / cancel / capacity change / throttle / calibration / config change is a structured event with the signals used | `decision_trace.py` |

## Using it

Per-request hints (all optional):

| Header | Meaning |
|---|---|
| `X-Traffic-Class` | `real_time` or `best_effort` |
| `X-SLA-Ms` | SLA for this request (ms); `sla_target_ms` in the JSON body also works |
| `X-Correlation-ID` | Your own request id; use it to look the request up in the trace |

Rejection (`429`) body (abridged):

```json
{"detail": {"error": "Rate limit / SLA protection shed", "status": "rejected_sla_impossible",
  "reason": "Predicted latency 0.82s exceeds SLA 0.10s (wait 0.00s + exec 0.82s).",
  "retry_after_seconds": 0.2, "reason_code": "sla_risk",
  "details": {"predicted_latency_ms": 816.5, "sla_ms": 100.0, "predicted_wait_ms": 0.0,
              "predicted_exec_ms": 816.5, "expected_concurrency": 9, "queue_size": 0, "operating_limit": 11,
              "kv_predicted_future_tokens": 0, "kv_safe_tokens": 250582, "gpu_memory_ratio": 0.8166}}}
```

Reason codes: `sla_risk`, `memory_risk`, `queue_overload`, `policy_limit`.

## Endpoints (only with `--policy smart`; other policies return 404)

| Endpoint | Purpose |
|---|---|
| `GET /smart/state` | Everything a dashboard needs in one JSON: capacity (memory/SLA/AIMD/operating + binding constraint), tokens and KV (marked `"source": "estimate"`), buckets, lanes, pressure, calibration, metrics, queue |
| `GET /smart/queue` | Waiting requests with live slack, urgency, bucket, score terms |
| `GET /smart/trace` | Recent decision events (`?limit&event&request_id&since_seq`) |
| `GET /smart/trace/{request_id}` | Every decision made about one request |
| `GET /smart/events` | Server-Sent Events: live trace; add `?state=1` to also stream full state |
| `GET /smart/metrics` | Prometheus: SLA goodput, TTFT/ITL percentiles, rejections by reason, capacity, KV pressure, calibration error |
| `GET /smart/config`, `PATCH /smart/config` | Read / change tunables live, e.g. `{"short_prompt_first": false}` |

Trace event types: `admit`, `reject`, `reorder`, `dispatch`, `complete`, `cancel`, `capacity_change`, `throttle`,
`calibration`, `config_change`. CORS is enabled for browser dashboards (`VELOCITY_CORS_ORIGINS`, default `*`).

## Configuration

Optional `smart:` section in the YAML passed with `--config` (every key is optional):

```yaml
smart:
  kv_bytes_per_token: 12288     # = 2 * layers * kv_heads * head_dim * dtype_bytes  (from the model's config.json)
  model_weights_mb: 1642        # weights + runtime overhead; used only to estimate the KV budget
  bucket_short_max: 256
  bucket_medium_max: 1024
  calibration_state_path: ./calibration.json   # warm-start across restarts
```

**Set `kv_bytes_per_token` for your model** (default 32768 is Llama-3.2-1B). Examples: Qwen2.5-0.5B 12288,
Qwen3-0.6B 114688, SmolLM2-360M 40960. A wrong value makes the memory-capacity maths wrong.

| Group | Keys (defaults) |
|---|---|
| KV model | `kv_bytes_per_token` (32768), `model_weights_mb` (2600), `kv_safe_fraction` (0.90), `kv_capacity_tokens_override` (0 = auto) |
| Buckets | `bucket_short_max` (256), `bucket_medium_max` (1024), `max_long_fraction` (0.50) |
| Queue scoring | `deadline_priority`, `short_prompt_first`, `bucket_compat` (all true), `w_priority` (1.0), `w_urgency` (1.5), `w_short` (0.6), `w_doomed` (1.2), `w_bucket` (0.25), `short_ref_tokens` (1024) |
| Early rejection | `early_reject_margin` (0.05), `early_reject_margin_high` (0.35), `memory_risk_pressure` (1.0) |
| Capacity | `sla_safety` (0.85), `aimd_ramp_step` (2) |
| RT / BE | `rt_reserve_fraction` (0.25), `be_throttle_pressure` (0.80), `be_resume_pressure` (0.60), `be_max_wait_seconds` (20) |
| Calibration | `warmup_probe` (true), `warmup_probe_tokens` (10), `calibration_warmup_samples` (20), `calibration_converged_mape` (0.30), `calibration_state_path` (none) |
| Trace | `trace_buffer_size` (2000) |

Calibration status in `/smart/state`: `warming_up` (< 20 requests), `converging`, `calibrated` (execution-latency MAPE <= 30%).

## Metrics definitions

- **Goodput** = requests that finished inside their SLA, divided by requests *offered* (rejected ones count against it).
- **TTFT** = time from arrival to first token; **ITL** = time between tokens.
- `sla_goodput_of_offered` and `rejected_by_reason` are in `/smart/state` -> `metrics`.

## Tests

```bash
python3 -m pytest tests -q                       # 64 tests, no GPU needed
python3 -m pytest tests/test_smart_features.py   # the 22 smart-scheduler tests
```

`tools/sim_benchmark.py` compares policies on a **simulated** GPU (a mock whose per-token latency rises with
concurrency). It is for exercising scheduling logic, not a hardware measurement; do not quote its numbers as GPU results.

## Validation on a real GPU

Run on an RTX 4050 Laptop (6 GB) with vLLM 0.28 on WSL2, four models (Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-0.6B,
SmolLM2-360M): no crashes, calibration converged, and the KV capacity estimate was roughly 10-20% below vLLM's
real cache (safe side). Equal-concurrency comparison and its caveats: see README -> *Benchmark Results* and
[`rtx4050_results/`](rtx4050_results/).

## Known limitations

- KV figures are **estimates** derived from token counts and `config.json`; they are not read from the engine.
- Static runs at a fixed concurrency of 8 (`initial_concurrency`, no CLI flag), so Smart-vs-Static throughput comparisons are only fair with `--max-concurrency 8`.
- Burst shedding (inherited from the dynamic policy) starts at ~90% queue fill (`max_queue_size` = 100) and can reject part of a sudden 100-request burst even when the GPU could serve it.
- The mixed-traffic benchmark (README -> Benchmark Results) shows Smart protecting interactive traffic under overload at the cost of best-effort throughput, but it does not isolate which of the ten features is responsible (no ablation yet). One model, one GPU.
- Not yet wired into the control service / Arena runner or the frontend (`/smart/events` and `/smart/state` are ready for it).
- GPU-memory pressure for best-effort throttling is measured as growth above the idle baseline (vLLM pre-allocates most VRAM at startup, so the absolute ratio is always high). The baseline is the lowest ratio observed since start.
- `/health` reports the *configured* backend; confirm the real engine in the startup log (`AsyncLLMEngine initialized successfully`).
