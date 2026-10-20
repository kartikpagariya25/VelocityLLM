export const SWEEP_META = {
  gpu: 'RTX 5050 laptop GPU, 8 GB',
  model: 'Llama-3.2-1B',
  slaSeconds: 8,
  secondsPerLevel: 30,
  seed: 42,
}

export const SWEEP_RATES = [2, 4, 8, 12, 16, 24]

export const SWEEP_SERIES = [
  {
    id: 'static',
    name: 'Static',
    note: 'Fixed batch of 8',
    rows: [
      { p99: 2.52, tps: 220, good: 220, ok: 100 },
      { p99: 2.49, tps: 430, good: 430, ok: 100 },
      { p99: 4.94, tps: 764, good: 764, ok: 100 },
      { p99: 19.79, tps: 747, good: 289, ok: 41 },
      { p99: 34.39, tps: 765, good: 169, ok: 23 },
      { p99: 71.27, tps: 706, good: 80, ok: 12 },
    ],
  },
  {
    id: 'static-batch',
    name: 'Static, wait to fill',
    note: 'Waits for 8 requests or 200 ms',
    rows: [
      { p99: 5.6, tps: 208, good: 208, ok: 100 },
      { p99: 9.68, tps: 331, good: 259, ok: 80 },
      { p99: 42.22, tps: 348, good: 42, ok: 13 },
      { p99: 72.69, tps: 352, good: 31, ok: 9 },
      { p99: 94.6, tps: 395, good: 24, ok: 6 },
      { p99: 200.27, tps: 322, good: 13, ok: 4 },
    ],
  },
  {
    id: 'dynamic',
    name: 'VelocityLLM',
    note: 'Dynamic, up to 32 in flight',
    rows: [
      { p99: 2.57, tps: 212, good: 212, ok: 100 },
      { p99: 2.58, tps: 437, good: 437, ok: 100 },
      { p99: 2.77, tps: 834, good: 834, ok: 100 },
      { p99: 3.05, tps: 1196, good: 1196, ok: 100 },
      { p99: 3.6, tps: 1588, good: 1588, ok: 100 },
      { p99: 6.06, tps: 2109, good: 2109, ok: 100 },
    ],
  },
]

export const SWEEP_METRICS = [
  { id: 'p99', label: 'p99 latency', unit: 's', log: true, better: 'lower is better' },
  { id: 'tps', label: 'Throughput', unit: 'tok/s', log: false, better: 'higher is better' },
  { id: 'good', label: 'Goodput', unit: 'tok/s', log: false, better: 'higher is better' },
]
