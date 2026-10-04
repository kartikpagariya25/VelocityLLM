const EVENTS = ['phase', 'log', 'metrics', 'request', 'result', 'level_result', 'done', 'error', 'info']

const clean = (base) => base.trim().replace(/\/+$/, '')

async function readError(res) {
  try {
    const body = await res.json()
    return typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
  } catch {
    return `Control service returned HTTP ${res.status}`
  }
}

export async function probe(base) {
  try {
    const r = await fetch(`${clean(base)}/api/preflight`, { signal: AbortSignal.timeout(6000) })
    return r.ok ? await r.json() : null
  } catch {
    return null
  }
}

export async function startLiveRun(base, cfg, emit) {
  const root = clean(base)
  let res
  try {
    res = await fetch(`${root}/api/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: cfg.model,
        scenario: cfg.traffic,
        sla_ms: Math.round(cfg.sla * 1000),
        requests: cfg.users,
        repeats: cfg.repeats || 1,
        levels: cfg.levels && cfg.levels.length > 1 ? cfg.levels : null,
        mode: 'sequential',
        prompt_preset: cfg.prompt,
        prompt_text: cfg.prompt === 'custom' ? cfg.text : null,
      }),
    })
  } catch {
    throw new Error(`Cannot reach the control service at ${root}. Is it running, and is the address right?`)
  }
  if (!res.ok) throw new Error(await readError(res))
  const { run_id } = await res.json()

  const es = new EventSource(`${root}/api/runs/${run_id}/events`)
  let finished = false
  let failures = 0
  es.onopen = () => {
    failures = 0
  }
  es.onerror = () => {
    if (finished) return
    failures += 1
    if (es.readyState === 2 || failures > 8) {
      finished = true
      es.close()
      emit({ event: 'error', data: { message: 'Lost the connection to the control service. The run may still be going on the machine; reload the page to check.' } })
    }
  }
  EVENTS.forEach((n) =>
    es.addEventListener(n, (e) => {
      let data
      try {
        data = JSON.parse(e.data)
      } catch {
        return
      }
      if (n === 'done') {
        finished = true
        es.close()
      }
      emit({ event: n, data })
    }),
  )

  return {
    runId: run_id,
    cancel: async () => {
      try {
        const r = await fetch(`${root}/api/runs/${run_id}/cancel`, { method: 'POST' })
        if (!r.ok && r.status !== 409) throw new Error(await readError(r))
      } catch (e) {
        emit({ event: 'error', data: { message: `Cancel failed: ${e.message}` } })
      }
    },
    canBurst: () => false,
    burst: () => {},
  }
}
