# VelocityLLM Benchmark Lab: Frontend Requirements

Status: draft for implementation
Owner: Kartik Pagariya | Frontend implementation: Vikrant Kadam

## 1. Purpose

A single page in the VelocityLLM frontend that lets a presenter run the same load against the Static and the Dynamic scheduler from the browser, watch each engine's logs and metrics live, and get a comparison table when both finish. Everything that is run from the CLI today (start a server with a policy, generate load, score the CSV, summarise) must be triggerable and observable from this page. No terminal is used during a demo.

## 2. Scope

In scope
- One new route (`/lab`) inside the existing frontend, split into a Static half and a Dynamic half.
- A control backend that starts and stops engines, runs load, streams logs and metrics, and stores results.
- Live log console, live metrics, live timeline per half.
- Comparison table, export, run history.
- Replay of recorded real runs when no GPU is available.

Out of scope
- Visual design, theme and palette (the frontend team applies the shared site theme).
- Authentication and multi-user accounts.
- Changes to scheduling logic.

## 3. Run modes

| Mode | Behaviour | When |
|---|---|---|
| Sequential (default) | Static runs first, then Dynamic, one engine at a time on the GPU. Same rules as `fair_benchmark.py`. | Default. The only mode whose numbers are reported as the benchmark. |
| Side by side (demo) | Both engines run at once, each with half of GPU memory, same load fired at both. Labelled "Demo mode, shared GPU". | Optional, for visual contrast on stage. Numbers are marked indicative. |
| Replay | Plays back a stored real run (logs, metrics, requests) at original or faster speed. Labelled "Recorded run". | No GPU, no network, or as backup. |

Every run shows its mode, model, GPU name, SLA, request count and repeat count in a header strip. A recorded run must never look like a live one.

## 4. Page layout and functional requirements

### 4.1 Control bar (FR-1x)
- FR-10 Model select, populated from `GET /api/models`.
- FR-11 Scenario select: Flood, Mixed (descriptions shown on hover).
- FR-12 SLA control in milliseconds (default 3000, range 500 to 15000, step 250). The same value is sent to both engines and shown on both halves.
- FR-13 Requests count (default 30) and repeats (default 1 for live, 3 for "Full benchmark").
- FR-14 Mode select (Section 3).
- FR-15 Actions: Pre-flight check, Start, Stop, Reset. Start is disabled until pre-flight passes. Stop cancels the run and shuts engines down cleanly.
- FR-16 Pre-flight panel: backend reachable, GPU detected and free memory, ports free, model path exists, previous run not still active. Each item shows pass or fail with a reason.

### 4.2 Engine panels (FR-2x), one per scheduler
- FR-20 Header: policy name, status badge (Idle, Starting, Loading model, Warming up, Running, Settling, Done, Failed, Cancelled).
- FR-21 Live metric tiles, updated at least every 500 ms: in-flight requests, queue length, concurrency limit (dynamic: adaptive limit; static: fixed), served, rejected, tokens per second, GPU memory used.
- FR-22 Live latency view: each request as a bar or dot on a time axis from arrival to completion, coloured by outcome (served within SLA, served beyond SLA, rejected). SLA line drawn.
- FR-23 Live log console (Section 5.3): line by line, timestamped, level and category tagged, auto scroll with pause and resume, text filter, category toggles, copy and download.
- FR-24 Both panels use the same layout and scale so they can be compared by eye. When one engine is not running, its panel shows an idle state, not an empty area.

### 4.3 Run timeline (FR-3x)
- FR-30 Phase stepper across the top: Pre-flight, Static start, Static warm-up, Static load, Dynamic start, Dynamic warm-up, Dynamic load, Scoring, Done. Current phase highlighted, elapsed time shown.
- FR-31 Warm-up requests are labelled as discarded and are not counted in any metric.

### 4.4 Results and comparison (FR-4x)
- FR-40 Appears when both engines are done; partial results appear per engine as soon as that engine finishes.
- FR-41 Table rows: p50 latency, p95 latency, p99 latency, tokens per second, served, rejected, within SLA (share of served), within SLA (share of offered, rejected counted as miss).
- FR-42 Columns: Static, Dynamic, Change. Change is a signed percentage; direction rules in Section 7.
- FR-43 Cells where Dynamic is worse are shown as clearly as cells where it is better. No hiding or reordering.
- FR-44 Footnote lists SLA, model, GPU, requests, repeats, seed, mode and a warning if more than 20% of replies had zero tokens.
- FR-45 When repeats is greater than 1, values are the median and the spread (min to max) is available on hover.
- FR-46 Export buttons: CSV, JSON, and a summary card image.
- FR-47 A short generated sentence under the table states the measured result in plain words, built only from the numbers in the table.

### 4.5 History (FR-5x)
- FR-50 List of past runs (time, model, scenario, SLA, mode, headline deltas), newest first.
- FR-51 Open any past run to see its table, logs and timelines in read only mode.
- FR-52 Select two runs to compare their tables side by side.

### 4.6 Request inspector (FR-6x)
- FR-60 Clicking a request in the timeline opens its details: id, priority, arrival, queue wait, execution time, total latency, tokens, outcome.
- FR-61 For a rejected request, show the admission reason (queue full, SLA impossible, burst shed, memory) and the retry-after value.

### 4.7 States and errors (FR-7x)
- FR-70 Engine start failure, port in use, model load failure, GPU out of memory, load generator failure, lost connection to backend: each shows a specific message and a Retry action. No raw stack traces in the UI; full text stays available in the log console.
- FR-71 If the log stream disconnects, the UI reconnects automatically and resumes from the last event id without duplicating lines.
- FR-72 Reset clears panels and returns to Idle only when no run is active.

## 5. Backend requirements

The frontend talks to a new control service. Existing engine endpoints are used by the control service, not exposed directly to the browser.

### 5.1 Gaps in the current backend
- No endpoint to start or stop an engine or run a benchmark (done today by `fair_benchmark.py` and `python3 -m scheduler_engine.server`).
- Engine logs go to stdout only. They need to be captured and streamed.
- No CORS configuration on the engine server.
- Per request events are not emitted; they have to be derived from log lines and from the load generator's CSV rows.

### 5.2 Control service
A small FastAPI app (suggested port 9000) that reuses the logic in `load_generator/fair_benchmark.py` (`start_server`, `stop_server`, `run_load`, `score`, `aggregate`) rather than duplicating it. It spawns one engine process per policy, captures its stdout line by line, polls the engine's `/stats` every 500 ms, runs the load generator, and writes raw CSVs and the summary under `results/<run_id>/`.

Endpoints

| Method and path | Purpose |
|---|---|
| `GET /api/preflight` | GPU name, free and total memory, ports, model paths, active run. |
| `GET /api/models` | Available models with name and path, flagged if known to return zero tokens. |
| `GET /api/scenarios` | Flood, Mixed with description and request mix. |
| `POST /api/runs` | Start a run. Body: `model`, `scenario`, `sla_ms`, `requests`, `repeats`, `seed`, `mode`. Returns `run_id`. Rejects if a run is active. |
| `GET /api/runs` | History list. |
| `GET /api/runs/{id}` | Full state: config, phase, results, per policy raw summary. |
| `GET /api/runs/{id}/events` | Server-Sent Events stream (5.3). Supports `Last-Event-ID`. |
| `POST /api/runs/{id}/cancel` | Cancel and clean up engines. |
| `GET /api/runs/{id}/export?format=csv\|json` | Download results. |
| `POST /api/replay/{recorded_id}` | Start a replay of a stored run. Same event stream as a live run. |

### 5.3 Event stream
Each event has an incrementing `id`, an `event` name and a JSON `data` payload.

| Event | Payload |
|---|---|
| `phase` | `{ "phase": "static_load", "policy": "static", "repeat": 1, "ts": 1730000000.12 }` |
| `log` | `{ "policy": "dynamic", "ts": ..., "level": "INFO", "category": "admission", "request_id": "r-17", "message": "Request r-17 rejected: Predicted latency 6.10s exceeds SLA target 5.00s." }` |
| `metrics` | `{ "policy": "dynamic", "ts": ..., "active": 12, "queued": 3, "concurrency_limit": 14, "completed": 20, "rejected": 1, "tokens": 5120, "gpu_mem_used_mb": 20480, "gpu_mem_total_mb": 81920 }` (mapped from `/stats`) |
| `request` | `{ "policy": "dynamic", "request_id": "r-17", "arrival_s": 0.0, "start_s": 1.2, "end_s": 3.4, "status": "served\|rejected", "http_status": 200, "tokens": 128, "priority": "HIGH", "reject_reason": null }` |
| `result` | `{ "policy": "dynamic", "p50_s": ..., "p95_s": ..., "p99_s": ..., "tokens_per_s": ..., "served": 30, "offered": 30, "rejected": 0, "within_sla_served": 1.0, "within_sla_offered": 1.0, "zero_token_share": 0.0 }` |
| `done` | `{ "run_id": "...", "status": "completed" }` |
| `error` | `{ "code": "ENGINE_START_FAILED", "message": "...", "policy": "static" }` |

Log categories derived from engine log lines:
- `admission`: lines such as "Request ... rejected: Queue full", "rejected: Predicted latency ... exceeds SLA target", "shed by severe burst-shedding".
- `controller`: "Adaptive Batch Controller: concurrency A -> B (reason)".
- `lifecycle`: "StaticBatchPolicy initialized", "DynamicBatchPolicy initialized", shutdown lines.
- `cancel`: requests cancelled or aborted, slot reclaimed.
- `error`: warnings and errors including GPU memory recovery.
- `system`: control service messages (model load, warm-up start and end, scoring).

Rule: the console shows real engine log lines unchanged in `message`. Control service lines are tagged `system` so they are never mistaken for engine output.

### 5.4 Engine server changes
- Allow CORS from the frontend origin only (configurable).
- Keep `/generate`, `/health`, `/stats`, `/metrics` as they are; the control service is the only client of `/stats`.
- Optional: emit one structured log line per request completion (id, queue wait, execution time, tokens) so the live timeline does not depend on parsing the CSV after the fact.

### 5.5 Fairness rules enforced by the control service
- Same SLA, model, request count, seed and traffic for both policies.
- Each engine starts cold with the same settle time between runs.
- Warm-up requests are sent and discarded.
- With repeats greater than 1, the order of the two policies is randomised per repeat and the median is reported.
- Percentiles are computed on served requests; "within SLA, offered" counts rejected requests as misses.
- Static and Dynamic never run concurrently in Sequential mode.

## 6. Replay data
- Recorded runs live under `results/recorded/<id>/` as the raw CSVs, the event log (NDJSON of the Section 5.3 events with relative timestamps) and a manifest (model, GPU, SLA, date).
- Replay speed: 1x, 2x, 4x.
- At least one recorded run per model and scenario from the fair benchmark is included before the demo.

## 7. Metric and comparison rules
- Latency percentiles and queue wait: lower is better. Tokens per second, served, within SLA: higher is better. Rejected: lower is better but is shown beside the SLA row, because rejecting is a deliberate trade.
- Change % = (Dynamic minus Static) divided by Static, signed.
- If Static value is zero, show "n/a".
- Tokens per second = total generated tokens of served requests divided by total wall time of the load phase.
- All numbers on the page come from the run's raw data. No hardcoded results anywhere in the frontend.

## 8. Non-functional
- Log console handles at least 5,000 lines without lag (virtualised list), keeps the last 20,000 in memory, and the full log is downloadable.
- Metrics update at 2 Hz without frame drops on a 1080p laptop.
- Stream reconnect within 3 seconds after a drop.
- A single active run at a time across all browser tabs; a second tab joins the same run read only.
- Control service binds to localhost by default; if exposed, it requires an API key header. No credentials of any kind are stored in the repo, the frontend bundle or the logs.
- Works in current Chrome and Edge at 1366 px width and above.

## 9. Acceptance criteria
1. From a cold start, a presenter selects model, scenario and SLA and presses Start; both engines run one after the other with no terminal use.
2. Each half shows its own live logs, tiles and timeline while it runs.
3. After the run, the comparison table appears with every row populated, matching the numbers in the exported JSON and in `results/<run_id>/fair_summary.csv` produced by the CLI for the same configuration.
4. A rejected request shows its reason in the inspector and in the log.
5. Pressing Stop mid run leaves no engine process or occupied port behind.
6. With the GPU disabled, Replay plays a recorded run with identical UI and a visible "Recorded run" label.
7. Refreshing the page during a run restores the panels and continues streaming without duplicate log lines.
8. All error cases in FR-70 produce a readable message and a working Retry.

## 10. Open decisions
- Whether Side by side mode ships in the first version or after Sequential and Replay are stable.
- GPU memory split used in Side by side mode (default 50% each).
- Which recorded runs are bundled for the demo, and on which GPU they are recorded.
- Where the control service runs during the event (presenter laptop or the GPU machine) and how the browser reaches it.
