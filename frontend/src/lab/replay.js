const sleep = (ms, signal) =>
  new Promise((res) => {
    const id = setTimeout(res, ms)
    signal.addEventListener('abort', () => (clearTimeout(id), res()), { once: true })
  })

export function parseRecording(obj) {
  if (!obj || obj.format !== 'velocityllm-recording-1' || !Array.isArray(obj.events) || !obj.events.length || !obj.config) {
    throw new Error('This file is not a VelocityLLM recording. Download one from the Saved benchmarks list.')
  }
  return obj
}

export function configOf(rec) {
  const c = rec.config
  return {
    id: rec.id,
    users: Math.max(...(rec.results?.levels || [c.requests])),
    levels: rec.results?.levels || [c.requests],
    traffic: c.scenario,
    sla: c.sla_ms / 1000,
    model: c.model,
    prompt: c.prompt_preset,
    text: c.prompt_text || '',
    mode: 'sequential',
    speed: 1,
  }
}

export function startReplay(rec, speed, emit) {
  const ac = new AbortController()
  const { signal } = ac
  const run = async () => {
    let prev = 0
    for (const ev of rec.events) {
      if (signal.aborted) return
      const gap = ((ev.at - prev) * 1000) / speed
      prev = ev.at
      if (gap > 4) await sleep(Math.min(gap, 1200), signal)
      if (signal.aborted) return
      emit({ event: ev.event, data: ev.event === 'info' ? { ...ev.data, recorded: true, recorded_at: rec.results?.created } : ev.data })
    }
  }
  run()
  return {
    cancel: () => {
      ac.abort()
      emit({ event: 'done', data: { run_id: rec.id, status: 'cancelled' } })
    },
    canBurst: () => false,
    burst: () => {},
  }
}
