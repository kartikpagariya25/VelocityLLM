# Project 1 Compliance: Dynamic Batching Engine for LLM Inference Serving

Requirement-by-requirement check against the Single Core Labs "AI Inference Optimization Track" guide, Project 1.

Status key: **Done** = implemented and covered by tests. **Done (sim)** = implemented and measured on the simulated backend; the GPU run is still to be recorded. **Harness ready** = scripts and configs exist, the run itself needs the named server or GPU.

## What to build

| PDF requirement | Status | Where |
|---|---|---|
| Workload generator: Poisson arrivals, variable prompt and output lengths, configurable requests per second | Done | `load_generator/stream_client.py` (`build_workload`), `load_generator/traffic_patterns.py` |
| Baseline A: static batching (wait for N requests or a timeout) | Done | `scheduler_engine/static_batching.py` (`--policy static-batch`); the fixed-concurrency cap baseline remains as `--policy static` |
| Iteration-level batching with admission control | Done | `scheduler_engine/policy.py` (`DynamicBatchPolicy`), `admission.py`, `adaptive_controller.py` |
| Max-batch-tokens limit | Done | `--max-batch-tokens`, enforced in the dispatch loop; an empty batch always accepts one request so an oversized request cannot deadlock |
| Optional priority or shortest-job-first policies | Done | `--queue-order fcfs / sjf / priority`, `scheduler_engine/priority_queue.py`; SJF and priority both age waiting requests so nothing starves |
| Simulated model to debug scheduling before the GPU | Done | `MockBackend` with `--mock-tps` and `--mock-contention` for a realistic batch-scaling curve |
| Model runner on Hugging Face Transformers with KV cache | Done | `scheduler_engine/hf_backend.py` (`--runner hf`): per-sequence KV cache, one forward pass per sequence per iteration, prefill and decode separated. `VLLMBackend` (`--runner vllm`, default) remains the production runner |
| Comparison point: vLLM server | Harness ready | `tools/rate_sweep.py --target NAME=openai@URL`; steps in `docs/COMPARISON.md` |
| Comparison point: Triton dynamic batcher (`max_queue_delay`) | Harness ready | `tools/triton/model_repository/hf_generate` (Python backend, `dynamic_batching`, `max_queue_delay_microseconds`) and `vllm_model`; steps in `docs/COMPARISON.md` |
| Metrics collector and SLA definition (p99 TTFT, p99 end-to-end) | Done | `scheduler_engine/metrics.py`, `/stats`, `/metrics`, `load_generator/stream_client.py` |
| Optional chunked prefill | Harness ready | `--chunked-prefill` and `--max-num-batched-tokens` pass through to vLLM; the on/off measurement needs the GPU |

## Experiments to run

| Experiment | Status | How |
|---|---|---|
| Request-rate sweep, low load to saturation | Done (sim) | `tools/rate_sweep.py --rps 1,2,4,...` |
| Max batch size / max batch tokens | Done (sim) | `--batch-size`, `--max-concurrency`, `--max-batch-tokens` |
| Max wait time | Done (sim) | `--static-batch-timeout-ms` |
| Scheduling policy: FCFS vs SJF vs priority | Done (sim) | `--policies dynamic:fcfs,dynamic:sjf,dynamic:priority` |
| Prompt and output length distributions | Done | built-in log-normal chat-like lengths, or `--lengths-file` with pairs exported from a chat dataset |
| Chunked prefill on / off | Harness ready | needs the GPU |
| Model size | Done | earlier results in `results/` cover TinyLlama, Qwen and StableLM next to Llama-3.2-1B |

## Evaluation metrics

| Metric | Status | Where reported |
|---|---|---|
| Throughput (tokens/s, requests/s) | Done | sweep summary, `/stats` |
| Time to first token p50 / p95 / p99 | Done | sweep summary, `/stats` (`p50_ttft_seconds` and friends), `/metrics` |
| Per-token latency | Done | `per_token_latency_ms` on every response and in `/stats` |
| End-to-end latency p99 | Done | sweep summary, `/stats` |
| GPU utilization | Done | `scheduler_engine/gpu_monitor.py`, `/stats` |
| Goodput (requests meeting the SLA) | Done | `goodput_requests`, `goodput_tokens` in `/stats`; on-time tokens/s and requests/s in the sweep |
| Queue wait time | Done | `avg_queue_wait_seconds` in `/stats`, `queue_time_seconds` on every response |

## Definition of done

| Item | Status |
|---|---|
| Working scheduler serving a real model | Done (earlier GPU runs in `results/`) |
| Static vs dynamic vs vLLM comparison | Static vs dynamic Done; vLLM and Triton runs are Harness ready |
| Throughput-latency curves | Done (sim); `results/sweep_mock/throughput_latency_curves.png` |
| SLA / goodput result | Done (sim); capacity table in the sweep output |
| Reproducible benchmark scripts | Done: fixed seeds, raw per-request CSVs saved next to every summary |

## Deviations from the guide, stated plainly

1. **The Transformers runner is validated on CPU only.** `--runner hf` is tested against `generate()` on a tiny randomly initialised model; it has not been run on a GPU with a real 0.5B to 1.5B model. It steps each sequence with its own forward pass rather than one padded tensor batch, so GPU throughput will be below vLLM; vLLM remains the default runner for the main results.
2. **Numbers in `results/sweep_mock` are simulated.** They validate the scheduling logic and show how the policies behave under the same workload; they are not GPU measurements. The guide's rule applies: report real GPU numbers from the vLLM and Triton comparison before drawing conclusions about vLLM.
3. **vLLM already implements continuous batching.** As the guide notes, the goal is to build, measure and explain, not to claim to beat vLLM unless measured.
