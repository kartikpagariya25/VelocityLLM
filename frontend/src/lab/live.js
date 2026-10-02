export async function startLiveRun(base, cfg, emit) {
  const res = await fetch(`${base}/api/runs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: cfg.model,
      scenario: cfg.traffic,
      sla_ms: Math.round(cfg.sla * 1000),
      requests: cfg.users,
      repeats: 1,
      mode: cfg.mode === 'parallel' ? 'side_by_side' : 'sequential',
    }),
  })
  if (!res.ok) throw new Error(`Control service returned ${res.status}`)
  const { run_id } = await res.json()
  const es = new EventSource(`${base}/api/runs/${run_id}/events`)
  const names = ['phase', 'log', 'metrics', 'request', 'result', 'done', 'error']
  names.forEach((n) =>
    es.addEventListener(n, (e) => {
      const data = JSON.parse(e.data)
      emit({ event: n, data })
      if (n === 'done') es.close()
    }),
  )
  return {
    cancel: () => {
      es.close()
      fetch(`${base}/api/runs/${run_id}/cancel`, { method: 'POST' })
    },
    canBurst: () => false,
    burst: () => {},
  }
}

export async function probe(base) {
  try {
    const r = await fetch(`${base}/api/preflight`, { signal: AbortSignal.timeout(1500) })
    return r.ok ? await r.json() : null
  } catch {
    return null
  }
}
