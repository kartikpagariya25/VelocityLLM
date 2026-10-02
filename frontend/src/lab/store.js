const LOG_CAP = 5000

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
    info: null,
    error: null,
    errorDetail: [],
    static: emptyPolicy(),
    dynamic: emptyPolicy(),
    levels: { static: {}, dynamic: {} },
    runId: null,
  }
}

export function apply(s, { event, data }) {
  const p = data.policy ? s[data.policy] : null
  if (event === 'phase') {
    s.phase = data.phase
    s.phaseAt[data.phase] = performance.now() - s.startedAt
    if (data.policy && data.phase.endsWith('_start')) s[data.policy] = emptyPolicy()
    if (data.policy && data.level != null && data.phase.endsWith('_load')) {
      s[data.policy] = { ...emptyPolicy(), result: s[data.policy].result }
      s.currentLevel = data.level
    }
  } else if (event === 'info') {
    s.info = data
    s.users = data.users
    s.repeats = data.repeats || 1
  } else if (event === 'log') {
    let ts = data.t ?? data.ts ?? 0
    if (data.t == null && ts > 1e9) {
      s.base ??= ts
      ts -= s.base
    }
    s.logs.push({ ...data, ts, n: (s.logCount = (s.logCount || 0) + 1) })
    if (s.logs.length > LOG_CAP) s.logs.splice(0, s.logs.length - LOG_CAP)
  } else if (event === 'metrics' && p) {
    p.metrics = data
    p.series.push({
      t: data.t ?? data.ts,
      active: data.active,
      queued: data.queued,
      limit: data.concurrency_limit,
    })
  } else if (event === 'request' && p) {
    p.reqs.push(data)
  } else if (event === 'level_result' && p) {
    s.levels[data.policy][data.level] = data
  } else if (event === 'result' && p) {
    p.result = data
  } else if (event === 'done') {
    if (data.status === 'cancelled') s.status = 'cancelled'
    else if (data.status === 'failed') s.status = 'failed'
    else s.status = 'done'
  } else if (event === 'error') {
    s.status = 'failed'
    s.error = data.message
    s.errorDetail = data.detail || []
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
  ['errors', 'Failed requests', (v) => v, null],
]

export const change = (a, b) => (a ? ((b - a) / a) * 100 : 0)

const NOISE = 5

export const verdictOf = (v, lowerIsBetter) => {
  if (Math.abs(v) < NOISE) return 'same'
  return (v < 0) === lowerIsBetter ? 'better' : 'worse'
}

const phrase = (v, up, down) => (Math.abs(v) < NOISE ? 'about the same' : v < 0 ? `${Math.abs(v).toFixed(0)}% ${down}` : `${v.toFixed(0)}% ${up}`)

export function sentence(cfg, s, d, users, repeats = 1) {
  if (!s || !d) return ''
  if (!s.served || !d.served) {
    const who = !s.served && !d.served ? 'Neither scheduler' : !d.served ? 'Dynamic' : 'Static'
    return `${who} answered no requests at ${users ?? cfg.users} users, so there is nothing to compare. Check the warnings and the execution log, then run again.`
  }
  const n = users ?? cfg.users
  const p99 = change(s.p99_s, d.p99_s)
  const p50 = change(s.p50_s, d.p50_s)
  const tok = change(s.tokens_per_s, d.tokens_per_s)
  const head = `With ${n} users on ${cfg.traffic} traffic and a ${cfg.sla.toFixed(1)} s SLA, Dynamic's p99 latency was ${f2(d.p99_s)} s against ${f2(s.p99_s)} s for Static (${phrase(p99, 'higher', 'lower')}), its median latency was ${phrase(p50, 'higher', 'lower')}, and its throughput was ${phrase(tok, 'higher', 'lower')}.`
  const shed = d.rejected ? ` It rejected ${d.rejected} of ${d.offered} requests; Static rejected ${s.rejected}.` : ' It rejected no requests.'
  const gap = d.within_sla_offered - s.within_sla_offered
  const sla =
    Math.abs(gap) < 0.01
      ? ` Counted against all offered requests, the share finishing within the SLA was the same (${pctStr(d.within_sla_offered)})${d.rejected > s.rejected ? ': Dynamic turned late replies into early rejections rather than into extra on-time replies' : ''}.`
      : ` Counted against all offered requests, ${pctStr(d.within_sla_offered)} finished within the SLA against ${pctStr(s.within_sla_offered)} for Static.`
  const zero = Math.max(s.zero_token_share || 0, d.zero_token_share || 0) > 0.5 ? ' Warning: most replies had zero tokens, so these numbers are not meaningful for this model.' : ''
  const once = repeats > 1 ? '' : ' This is a single run; differences under 5% are normal run-to-run variation.'
  return head + shed + sla + zero + once
}

export function sweepSentence(cfg, levels, data) {
  const done = levels.filter((n) => data.static[n] && data.dynamic[n])
  if (done.length < 2) return ''
  const top = done[done.length - 1]
  const s = data.static[top]
  const d = data.dynamic[top]
  if (!s.served || !d.served) return `At ${top} users ${!d.served ? 'Dynamic' : 'Static'} answered no requests, so the heaviest level cannot be compared. Check the warnings and the execution log, then run again.`
  const p99 = change(s.p99_s, d.p99_s)
  const sServed = s.within_sla_served
  const dServed = d.within_sla_served
  const wins = done.filter((n) => verdictOf(change(data.static[n].p99_s, data.dynamic[n].p99_s), true) === 'better')
  const lead = wins.length ? `Dynamic had clearly lower p99 latency at ${wins.join(', ')} users.` : 'Dynamic did not have clearly lower p99 latency at any tested load.'
  return `${lead} At the heaviest load (${top} users) the p99 latency of answered requests was ${f2(d.p99_s)} s for Dynamic and ${f2(s.p99_s)} s for Static (${phrase(p99, 'higher', 'lower')}). ${pctStr(dServed)} of Dynamic's answered requests met the ${cfg.sla.toFixed(1)} s SLA against ${pctStr(sServed)} for Static, and Dynamic rejected ${d.rejected} of ${d.offered} requests while Static rejected ${s.rejected}. Counted against every user, ${pctStr(d.within_sla_offered)} of Dynamic and ${pctStr(s.within_sla_offered)} of Static finished on time.`
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
