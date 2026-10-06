# VelocityLLM

SLA-aware dynamic scheduling for LLM serving on vLLM. VelocityLLM sits in front of your model, decides how many requests the GPU should run at once, and keeps latency inside the promise you set. It ships with a terminal benchmark so you can see the difference on your own hardware.

## Install

```bash
pip install velocityllm            # scheduler, benchmark, simulated engine
pip install "velocityllm[gpu]"     # adds vLLM for real models on an NVIDIA GPU
```

Python 3.11 or newer.

## Compare static and dynamic scheduling on your models

Put each model in its own folder under `./models`, then:

```bash
velocityllm bench --models-dir ./models --requests 100 --repeats 3 --sla-ms 8000
```

Every model is served twice, once with a fixed batch size (static) and once with VelocityLLM (dynamic), under the same traffic. The table shows p50, p95 and p99 latency, tokens per second and the share of requests answered inside the SLA, with the change against static on a line of its own. Results are saved as CSV and JSON in `./velocity_bench/latest`.

Pick models and policies explicitly:

```bash
velocityllm bench --models-dir ./models --only qwen2.5-1.5b,llama-3.2-1b,gemma-2-2b --policies static,dynamic,smart
velocityllm bench --model qwen=/data/qwen2.5-1.5b --model llama=/data/llama-3.2-1b
velocityllm bench --mock --models-dir ./models      # no GPU, simulated engine
```

## Use it from Python

```python
import velocityllm

rows = velocityllm.benchmark(
    ["qwen2.5-1.5b", "gemma-2-2b"], models_dir="/data/models",
    requests=100, repeats=3, sla_ms=8000,
)
for r in rows:
    print(r["model"], r["policy"], r["p50"], r["p95"], r["p99"], r["tokens_per_sec"])
```

## Serve a model

```bash
velocityllm serve --policy dynamic --model-path /data/qwen2.5-1.5b --sla-ms 8000
velocityllm serve --mock                                  # try it without a GPU
velocityllm loadgen --url http://localhost:8000/generate --num-requests 100 --concurrency 100 --pattern flood
```

Policies: `static` (fixed concurrency baseline), `dynamic` (SLA-aware admission with an adaptive concurrency limit) and `smart` (token-aware, KV-cache-aware, explains every decision).

## Arena dashboard

```bash
velocityllm arena --models-dir ./models
```

Opens the web dashboard where you pick a model, a user count and an SLA and watch the policies run side by side.
