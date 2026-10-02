export const emptyPolicy = () => ({ metrics: null, series: [], reqs: [], result: null })

export function newStore(cfg, steps, source) {
  return {
    cfg,
    source,
    steps,
    status: 'running',
    phase: null,
    phaseAt: {},
    startedAt: performance.now(),
    base: null,
    logs: [],
    error: null,
    static: emptyPolicy(),
    dynamic: emptyPolicy(),
  }
}

export function apply(s, { event, data }) {
  const p = data.policy ? s[data.policy] : null
  if (event === 'phase') {
    s.phase = data.phase
    s.phaseAt[data.phase] = performance.now() - s.startedAt
  } else if (event === 'log') {
    let ts = data.ts || 0
    if (ts > 1e9) {
      s.base ??= ts
      ts -= s.base
    }
    s.logs.push({ ...data, ts, n: s.logs.length })
  } else if (event === 'metrics' && p) {
    p.metrics = data
    p.series.push({
      t: data.ts > 1e9 ? data.ts - (s.base ?? data.ts) : data.ts,
      active: data.active,
      queued: data.queued,
      limit: data.concurrency_limit,
    })
  } else if (event === 'request' && p) {
    p.reqs.push(data)
  } else if (event === 'result' && p) {
    p.result = data
  } else if (event === 'done') {
    s.status = data.status === 'cancelled' ? 'cancelled' : 'done'
  } else if (event === 'error') {
    s.status = 'failed'
    s.error = data.message
  }
}

export const f2 = (v) => (v == null || Number.isNaN(v) ? '-' : v.toFixed(2))
export const pctStr = (v) => `${(v * 100).toFixed(0)}%`

export const ROWS = [
  ['p50_s', 'p50 latency', (v) => `${f2(v)} s`, 'low'],
  ['p95_s', 'p95 latency', (v) => `${f2(v)} s`, 'low'],
  ['p99_s', 'p99 latency', (v) => `${f2(v)} s`, 'low'],
  ['tokens_per_s', 'Tokens / s', (v) => v.toFixed(0), 'high'],
  ['within_sla_served', 'Within SLA (of served)', pctStr, 'high'],
  ['within_sla_offered', 'Within SLA (of offered)', pctStr, 'high'],
  ['served', 'Served', (v) => v, null],
  ['rejected', 'Rejected', (v) => v, null],
]

export const change = (a, b) => (a ? ((b - a) / a) * 100 : 0)

export function sentence(cfg, s, d) {
  if (!s || !d) return ''
  const p99 = change(s.p99_s, d.p99_s)
  const tok = change(s.tokens_per_s, d.tokens_per_s)
  const dir = (v, up, down) => (Math.abs(v) < 1 ? 'about the same' : v < 0 ? `${Math.abs(v).toFixed(0)}% ${down}` : `${v.toFixed(0)}% ${up}`)
  const shed = d.rejected
    ? ` It rejected ${d.rejected} of ${d.offered} requests to get there; Static rejected ${s.rejected}.`
    : ' It rejected no requests.'
  const offered = d.within_sla_offered - s.within_sla_offered
  const slaNote =
    Math.abs(offered) < 0.01
      ? ` Counted against all offered requests, the share finishing within the SLA was the same (${pctStr(d.within_sla_offered)}): Dynamic turned late replies into early rejections rather than into extra on-time replies.`
      : offered > 0
        ? ` Counted against all offered requests, ${pctStr(d.within_sla_offered)} finished within the SLA against ${pctStr(s.within_sla_offered)} for Static.`
        : ` Counted against all offered requests, only ${pctStr(d.within_sla_offered)} finished within the SLA against ${pctStr(s.within_sla_offered)} for Static.`
  return `With ${cfg.users} users on ${cfg.traffic} traffic and a ${cfg.sla.toFixed(1)} s SLA, Dynamic's p99 latency was ${f2(d.p99_s)} s against ${f2(s.p99_s)} s for Static (${dir(p99, 'higher', 'lower')}), and its throughput was ${dir(tok, 'higher', 'lower')}.${shed}${slaNote}`
}

export function toCsv(s) {
  const head = 'metric,static,dynamic'
  const lines = ROWS.map(([k, label]) => `${label},${s.static.result?.[k] ?? ''},${s.dynamic.result?.[k] ?? ''}`)
  return [head, ...lines].join('\n')
}

export function download(name, text, type = 'text/plain') {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([text], { type }))
  a.download = name
  a.click()
  setTimeout(() => URL.revokeObjectURL(a.href), 1000)
}

export const logLine = (l) =>
  `[${l.ts.toFixed(2).padStart(7)}s] [${(l.policy || 'sys').slice(0, 3).toUpperCase()}] ${l.level.padEnd(7)} ${l.category.padEnd(10)} ${l.message}`
