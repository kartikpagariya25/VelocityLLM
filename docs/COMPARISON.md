# Comparing VelocityLLM with vLLM and Triton

`tools/rate_sweep.py` replays the same seeded Poisson workload against every target and scores them identically (TTFT, per-token latency, p50/p95/p99 end-to-end, throughput, goodput, SLA attainment). Use the same model, GPU and SLA for every system.

Record with each run: GPU model, driver, CUDA, PyTorch, vLLM and Triton versions (`nvidia-smi`, `pip list`).

## 1. VelocityLLM policies (local, started by the tool)

```bash
python tools/rate_sweep.py --policies static,static-batch,dynamic \
  --model-path /path/to/llama-3.2-1b --sla-ms 8000 --rps 1,2,4,6,8,12,16 \
  --out-dir results/sweep_gpu
```

Scheduling-policy experiment on the dynamic engine:

```bash
python tools/rate_sweep.py --policies dynamic:fcfs,dynamic:sjf,dynamic:priority \
  --model-path /path/to/llama-3.2-1b --out-dir results/policy_order_gpu
```

Token budget and chunked prefill:

```bash
python tools/rate_sweep.py --policies dynamic --max-batch-tokens 2048 --out-dir results/token_budget
python tools/rate_sweep.py --policies dynamic --chunked-prefill --max-num-batched-tokens 2048 --out-dir results/chunked_on
python tools/rate_sweep.py --policies dynamic --out-dir results/chunked_off
```

## 2. vLLM server

Start vLLM yourself, with the same model and a comparable concurrency ceiling:

```bash
vllm serve /path/to/llama-3.2-1b --port 8001 --max-model-len 4096 \
  --gpu-memory-utilization 0.80 --max-num-seqs 32
```

Then sweep it alone, or next to the local policies in one run so they share the plots:

```bash
python tools/rate_sweep.py --policies static,dynamic \
  --target vllm=openai@http://localhost:8001#/path/to/llama-3.2-1b \
  --model-path /path/to/llama-3.2-1b --out-dir results/sweep_vs_vllm
```

The `#model` suffix is the model name vLLM reports (the path you passed to `vllm serve`).
Stop VelocityLLM's own server before starting vLLM on the same GPU; both cannot hold 80% of memory at once, so run the vLLM target in a separate pass if memory is tight and merge the CSVs.

## 3. Triton Inference Server

The repository in `tools/triton/model_repository` has two models.

**`hf_generate`** is the dynamic-batcher comparison the guide asks for: a Python-backend model that runs Hugging Face `generate` over whatever batch Triton's dynamic batcher assembles. `config.pbtxt` sets `max_batch_size`, `preferred_batch_size` and `max_queue_delay_microseconds`; sweep that delay (for example 50 ms, 200 ms, 500 ms) to reproduce the "max wait time" experiment. It needs `transformers` and `torch` inside the Triton container, and the `model_path` parameter pointing at a mounted model directory.

**`vllm_model`** is Triton's vLLM backend, a continuous-batching reference served through Triton.

```bash
docker run --gpus all --rm -p 8002:8000 \
  -v $PWD/tools/triton/model_repository:/models/repo -v /path/to/llama-3.2-1b:/models/llama-3.2-1b \
  nvcr.io/nvidia/tritonserver:<tag>-py3 tritonserver --model-repository=/models/repo
```

Use a `-py3` image that includes the backend you need (the vLLM image for `vllm_model`). Check NVIDIA's current release notes for tags.

```bash
python tools/rate_sweep.py --policies "" \
  --target triton_batcher=triton-infer@http://localhost:8002#hf_generate \
  --out-dir results/sweep_triton_batcher

python tools/rate_sweep.py --policies "" \
  --target triton_vllm=triton-generate@http://localhost:8002#vllm_model \
  --out-dir results/sweep_triton_vllm
```

`triton-infer` is non-streaming, so its TTFT equals its total latency; the sweep report flags that. `triton-generate` streams, so its TTFT is measured.

## Reading the results

Compare systems at equal p99, not equal throughput: the capacity table lists, per target, the highest offered rate whose p99 stays within the SLA with at most 1% rejected requests, next to peak goodput. Load shedding is not credited as capacity. Report where vLLM wins as plainly as where VelocityLLM does.

These harnesses were validated against fake servers in `tests/test_sweep_tools.py`; the Triton model files themselves have not been run on a GPU yet.
