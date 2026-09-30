import traces from '../data/traces.json'

export const pair = (model, scenario) => traces[model][scenario]

export function delta(model, scenario) {
  const { static: s, dynamic: d } = pair(model, scenario)
  return Math.round(((d.p99 - s.p99) / s.p99) * 100)
}

export function summary() {
  const rows = []
  Object.keys(traces).forEach((model) =>
    Object.keys(traces[model]).forEach((scenario) => rows.push(delta(model, scenario))),
  )
  const shedTimes = []
  Object.values(traces).forEach((byScenario) =>
    Object.values(byScenario).forEach(({ dynamic }) =>
      dynamic.requests.filter((r) => r.shed).forEach((r) => shedTimes.push(r.end)),
    ),
  )
  shedTimes.sort((a, b) => a - b)
  return {
    runs: rows.length,
    better: rows.filter((v) => v < 0).length,
    shedMedian: shedTimes[Math.floor(shedTimes.length / 2)] || 0,
  }
}

export function inFlight(trace, tau) {
  return trace.requests.reduce((n, r) => (!r.shed && r.start <= tau && tau < r.end ? n + 1 : n), 0)
}

export function liveStats(trace, t) {
  let finished = 0
  let rejected = 0
  let running = 0
  let waiting = 0
  let slowest = 0
  trace.requests.forEach((r) => {
    if (r.shed) {
      if (r.end <= t) rejected += 1
      return
    }
    if (r.end <= t) {
      finished += 1
      slowest = Math.max(slowest, r.end)
    } else if (r.start <= t) running += 1
    else waiting += 1
  })
  return { finished, rejected, running, waiting, slowest }
}
