# Control service

Runs the real Static vs Dynamic benchmark from the Arena page. It starts one engine per policy on this machine, streams the engine logs, per-request results and GPU/queue metrics to the browser, and stores every run under `results/runs/<run_id>/`.

## Run on a PC with an NVIDIA GPU

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
python3 -m control_service --model llama=models/llama-3.2-1b
```

Open `http://localhost:9000/#/lab`, choose **PC GPU (this machine)** and press Run comparison. Repeat `--model NAME=PATH` for more models, or drop model folders (each with a `config.json`) into `models/`.

## Try the whole pipeline without a GPU

```bash
python3 -m control_service --mock
```

The page labels this run "Mock engine"; its numbers say nothing about real hardware.

## Options

| Flag | Meaning |
| --- | --- |
| `--host 0.0.0.0` | Reach the service from another machine (college GPU) |
| `--cors-origin URL` | Allow another browser origin, `*` for any |
| `--api-key KEY` | Require `X-API-Key` on every `/api` call |
| `--settle S` | Rest between engine runs (default 5) |
| `--startup-timeout S` | Wait for model load (default 600) |
| `--results-dir`, `--models-dir`, `--engine-port` | Locations and base port |

## Fairness

Both policies get the same model, SLA (`--sla-ms`), request plan and seed, built by the same `generate_load.build_request_plan` the CLI uses. Each policy starts a fresh engine, takes 8 warm-up requests that are discarded, then runs the load. Tokens per second is generated tokens of served requests divided by the wall time of the load phase. With `repeats > 1` the order alternates and the median is reported.

## API

`GET /api/preflight`, `/api/models`, `/api/scenarios`, `POST /api/runs`, `GET /api/runs`, `/api/runs/{id}`, `/api/runs/{id}/events` (SSE, resumes with `Last-Event-ID`), `POST /api/runs/{id}/cancel`, `GET /api/runs/{id}/export?format=csv|json`, `POST /api/replay/{id}`, `GET /api/runs/{id}/recording`, `GET /api/benchmarks/export`.

Failures (missing GPU or model, engine that will not start, engine crash during load, lost browser connection) end the run with an `error` event that carries the last engine log lines.

## Pressure test (load sweep)

`POST /api/runs` accepts `levels` (2 to 8 whole numbers, each 1 to 200). The engine starts once per policy, warms up once, then serves each level in turn with the same request plan for both policies. The page draws p99 and the share answered within the SLA per level, and reports levels where Dynamic does not win as plainly as the ones where it does.

## Saving and replaying benchmarks

Every completed run is stored in `results/runs/<id>/` (config, events, raw CSVs, server logs, `results.json` with the machine's GPU, driver, vllm, torch and git commit). `GET /api/runs/{id}/recording` returns one self-contained file; the Arena replays it with no GPU and no control service via Saved benchmarks, Import recording. `GET /api/benchmarks/export` downloads every run plus `benchmarks_summary.csv` as one zip.

## College GPU

On the college machine run `python3 -m control_service --host 0.0.0.0 --cors-origin '*'`, run the benchmarks from the Arena, then download the zip. For the demo, import the recordings from the zip on any laptop. A site served over HTTPS cannot call an `http://` control service (mixed content), so use the local site for live runs.

## Self-checks and error handling

- The admission estimator bounds how much a single completion can change its latency estimate, decays stale estimates toward baseline when nothing completes for 10 s, and always admits one request to an idle server, so a slow cold start can no longer make an engine reject every request.
- Each policy is warmed up twice before scoring.
- After every load level the service checks the result. If nothing was answered, 90% or more of the requests were rejected, more than 20% failed, or most replies had zero tokens, it logs a warning, stores it in `results.json` under `warnings`, and the Arena shows it above the verdict.
- If the Arena itself hits an unexpected error it shows a recovery screen instead of a blank page.

## Fair comparison settings

Every load request is sent with temperature 0, so both schedulers get the same prompts and write the same text; without it the two engines produce different amounts of output and latency and throughput stop being comparable. The Arena also has a Repeats selector (1, 3 or 5): the comparison runs that many times, alternating which scheduler goes first, and reports the median. If the two schedulers still wrote more than 10% different amounts of text, the verdict says so.

## Reading the results
- **Answered on time** counts the users who got a reply inside the SLA. A declined request counts as not on time.
- **Throughput** is shown as a comparison only when both schedulers answered a similar amount of text. If Dynamic declines most requests its run is shorter and covers different work, so the card says it is not comparable.
- Dynamic raises its concurrency limit quickly while a queue builds and memory allows, and stops raising it once replies themselves (not queue waiting) approach the SLA.
- GPU memory limits only apply once usage has grown after the model loaded, so other programs holding the GPU do not make Dynamic refuse everything.
- Token counts come from the engine's own count of generated tokens, so busy runs are not under-counted when the engine batches several tokens into one streamed update.
