# RTX 4050 Laptop (6 GB) results, Qwen2.5-0.5B

Setup: vLLM 0.28, WSL2, 100 concurrent requests, 30-request warm-up, 3 repeats, `max_tokens=80`.
GPU is power-capped (30 W) and its clock varied ~1350-2460 MHz during runs.

## Same concurrency ceiling (8) for all policies
| SLA | Policy | Goodput % mean [min-max] | p99 s | tok/s | Rejected |
|---|---|---|---|---|---|
| 8s | Static | 94 [90-97] | 8.43 | 876 | 0 |
| 8s | Dynamic | 55 [27-97] | 7.22 | 556 | 44 |
| 8s | Smart | 91 [80-96] | 7.69 | 875 | 8 |
| 5s | Static | 56 [56-56] | 8.58 | 852 | 0 |
| 5s | Dynamic | 42 [16-56] | 5.55 | 671 | 52 |
| 5s | Smart | 57 [56-57] | 5.20 | 843 | 39 |

## What this supports
- Smart's goodput is about equal to Static's, not higher.
- At SLA 5s Smart refuses the requests that would be late; admitted requests finish with p99 5.2s (Static: 8.6s).
- Smart's goodput is far more stable than Dynamic's across repeats.
- With a ceiling of 16, Smart reached 1417 tok/s vs Static 897 (Static is fixed at 8), but at an equal ceiling of 8 throughput is the same.

## What this does NOT show
- No claim about other GPUs, larger models, or mixed traffic.
- Every prompt was similar, so the short-prompt-first, bucket and real-time/best-effort features were not exercised.
- results_models.csv used max-concurrency 16 (Static stayed at 8).
