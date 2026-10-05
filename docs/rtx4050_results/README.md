# RTX 4050 Laptop (6 GB) results -- Qwen2.5-0.5B

Setup: vLLM 0.28, WSL2, 100 concurrent requests, 30-request warm-up, 3 repeats, `max_tokens=80`.
The GPU is power-capped (30 W) and its clock varied roughly 1350-2460 MHz during the runs.

## Same concurrency ceiling (8) for all policies (`results_cap8_*.csv`)
| SLA | Policy | Goodput % mean [min-max] | p99 s | tok/s | Rejected |
|---|---|---|---|---|---|
| 8 s | Static | 94 [90-97] | 8.43 | 876 | 0 |
| 8 s | Dynamic | 55 [27-97] | 7.22 | 556 | 44 |
| 8 s | Smart | 91 [80-96] | 7.69 | 875 | 8 |
| 5 s | Static | 56 [56-56] | 8.58 | 852 | 0 |
| 5 s | Dynamic | 42 [16-56] | 5.55 | 671 | 52 |
| 5 s | Smart | 57 [56-57] | 5.20 | 843 | 39 |

## Four models, max-concurrency 16, SLA 8 s (`results_models.csv`, earlier cold-start run)
Single runs per cell; Static is fixed at concurrency 8 while Dynamic/Smart may go to 16, so Static-vs-others is not like for like.

## What this supports
- Smart's goodput is about equal to Static's, not higher; throughput is equal at an equal ceiling.
- At a 5 s SLA Smart rejects requests that would be late; admitted requests finish with p99 5.2 s (Static: 8.6 s).
- Dynamic's goodput varied much more across repeats than Smart's on this GPU.

## Mixed traffic (`results_mixed2_r8.csv`, `results_mixed2_r16.csv`)
Poisson arrivals, 15 s x 3 repeats, cap 8, per-class SLAs (interactive 3 s / standard 6 s / batch 20 s), identical requests for every policy.
| Load | Policy | Overall % | Interactive % | HIGH % | Interactive p99 | Batch served % | Rejected |
|---|---|---|---|---|---|---|---|
| 8 | Static | 97 [91-100] | 93 [80-100] | 94 [83-100] | 2.42 s | 100 | 0 |
| 8 | Dynamic | 95 [84-100] | 90 [71-100] | 95 [86-100] | 2.41 s | 100 | 3 |
| 8 | Smart | 99 [97-100] | 100 [100-100] | 100 [100-100] | 1.28 s | 91 [73-100] | 1 |
| 16 | Static | 38 [31-43] | 17 [10-22] | 19 [14-21] | 10.34 s | 100 | 0 |
| 16 | Dynamic | 25 [22-30] | 10 [2-21] | 18 [3-29] | 4.79 s | 91 [73-100] | 162 |
| 16 | Smart | 84 [80-87] | 95 [90-99] | 100 [100-100] | 2.24 s | 17 [2-30] | 38 |

Under overload Smart protects interactive/HIGH traffic by shedding best-effort batch work. At 8 req/s the policies are close.
(An earlier mixed run, before a fix to the best-effort memory-pressure signal, served 0% of batch even at light load and is not reported.)

## What it does NOT show
- Other GPUs or larger models; other traffic mixes; the separate effect of each of the ten features (no ablation).
- Run-to-run variance on this laptop GPU is large (the same config gave 48 and 90 accepted requests on two runs), so single runs should not be compared.
