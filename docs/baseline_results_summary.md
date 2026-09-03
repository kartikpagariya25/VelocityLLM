# Baseline (Static Batching) — Reference Results

**Model:** Llama-3.2-1B-Instruct
**Config:** max_num_seqs=8, gpu_memory_utilization=0.80, max_model_len=4096
**Hardware:** RTX 5050, 8GB VRAM, WSL2 Ubuntu

## Run 1 — Light Load
- Requests: 20, Concurrency: 5, Max tokens: 50
- Throughput: 7.20 req/sec
- p50 latency: 1.888s
- p99 latency: 2.771s
- Avg GPU utilization: 83.6%
- Max GPU utilization: 99.0%

## Run 2 — Heavy Load (Stress Test)
- Requests: 30, Concurrency: 10, Max tokens: 200
- Throughput: 4.25 req/sec
- p50 latency: 4.657s
- p99 latency: 7.056s
- Avg GPU utilization: 94.7%
- Max GPU utilization: 100.0%

## Observation
As load increases, GPU saturates (near 100% utilization) but end-to-end
latency degrades sharply (p99 nearly triples). This demonstrates the
throughput-vs-latency trade-off that the dynamic scheduler (Phase 2+)
is designed to manage more intelligently via adaptive batch sizing
and SLA-aware admission control.
