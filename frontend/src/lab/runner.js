import { Sim, buildArrivals, promptTokens, MODELS } from './sim'

const POLICIES = ['static', 'dynamic']

const sleep = (ms, signal) =>
  new Promise((res) => {
    const id = setTimeout(res, ms)
    signal.addEventListener('abort', () => (clearTimeout(id), res()), { once: true })
  })

const frame = (signal) =>
  new Promise((res) => {
    if (signal.aborted) return res(0)
    const id = setTimeout(() => res(performance.now()), 33)
    signal.addEventListener('abort', () => (clearTimeout(id), res(0)), { once: true })
  })

export const stepsFor = (mode) =>
  mode === 'parallel'
    ? [
        ['preflight', 'Pre-flight'],
        ['start', 'Engines start'],
        ['warmup', 'Warm-up'],
        ['load', 'Load'],
        ['scoring', 'Scoring'],
        ['done', 'Done'],
      ]
    : [
        ['preflight', 'Pre-flight'],
        ['static_start', 'Static start'],
        ['static_warmup', 'Static warm-up'],
        ['static_load', 'Static load'],
        ['dynamic_start', 'Dynamic start'],
        ['dynamic_warmup', 'Dynamic warm-up'],
        ['dynamic_load', 'Dynamic load'],
        ['scoring', 'Scoring'],
        ['done', 'Done'],
      ]

export function startSimRun(cfg, emit) {
  const ac = new AbortController()
  const { signal } = ac
  const model = MODELS.find((m) => m.id === cfg.model)
  const master = buildArrivals(cfg)
  let nextId = master.length + 1
  let sims = []
  const burstTokens = promptTokens(cfg.prompt, cfg.text)
  const ts = () => performance.now() / 1000

  const phase = (name, policy = null, level = null) => emit({ event: 'phase', data: { phase: name, policy, repeat: 1, level, ts: ts() } })
  const sys = (policy, message, category = 'system', level = 'INFO') =>
    emit({ event: 'log', data: { policy, ts: 0, level, category, request_id: null, message } })

  const preflight = async () => {
    phase('preflight')
    const lines = [
      'Control service reachable (simulated)',
      `Model ${model.name} resolved`,
      'GPU memory free: 40960 MB (simulated)',
      'Port 8000 free',
      `Scenario ready: ${cfg.users} users, ${cfg.traffic} traffic, SLA ${cfg.sla.toFixed(1)} s`,
    ]
    for (const l of lines) {
      sys(null, `Pre-flight: ${l}`)
      await sleep(180, signal)
    }
  }

  const boot = async (policy, startName, warmName) => {
    phase(startName, policy)
    const flags = policy === 'static' ? '--policy static' : `--policy dynamic --sla-ms ${Math.round(cfg.sla * 1000)}`
    sys(policy, `python3 -m scheduler_engine.server ${flags} --max-concurrency 16`)
    await sleep(350, signal)
    sys(policy, `Loading model ${model.name} (simulated)`)
    await sleep(350, signal)
    phase(warmName, policy)
    sys(policy, 'Warm-up request sent, discarded from scoring')
    await sleep(350, signal)
  }

  const play = async (list, label) => {
    phase(label, list.length === 1 ? list[0].policy : null)
    let last = performance.now()
    let simT = 0
    while (list.some((s) => !s.done) && !signal.aborted) {
      const now = await frame(signal)
      if (!now) break
      simT += ((now - last) / 1000) * cfg.speed
      last = now
      list.forEach((s) => s.advance(simT))
    }
  }

  const finish = async (list) => {
    phase('scoring')
    await sleep(400, signal)
    list.forEach((s) => emit({ event: 'result', data: { policy: s.policy, ...s.result() } }))
  }

  const make = (policy, arrivals) =>
    new Sim({ policy, arrivals, sla: cfg.sla, tpt: model.tpt, emit })

  const run = async () => {
    try {
      await preflight()
      if (signal.aborted) return
      if (cfg.mode === 'parallel') {
        phase('start')
        sims = [make('static', master), make('dynamic', master)]
        phase('warmup')
        await sleep(500, signal)
        await play(sims, 'load')
        if (signal.aborted) return
        await finish(sims)
      } else {
        await boot('static', 'static_start', 'static_warmup')
        if (signal.aborted) return
        sims = [make('static', master)]
        await play(sims, 'static_load')
        if (signal.aborted) return
        await boot('dynamic', 'dynamic_start', 'dynamic_warmup')
        if (signal.aborted) return
        const s = sims[0]
        sims = [make('dynamic', master)]
        await play(sims, 'dynamic_load')
        if (signal.aborted) return
        await finish([s, sims[0]])
      }
      phase('done')
      emit({ event: 'done', data: { run_id: cfg.id, status: 'completed' } })
    } catch (e) {
      emit({ event: 'error', data: { message: String(e.message || e) } })
    }
  }

  run()

  return {
    cancel: () => {
      ac.abort()
      emit({ event: 'done', data: { run_id: cfg.id, status: 'cancelled' } })
    },
    canBurst: () => sims.length > 0 && (cfg.mode === 'parallel' || sims[0].policy === 'static') && sims.some((s) => !s.done),
    burst: (n) => {
      const t = sims[0].t
      const list = Array.from({ length: n }, (_, i) => ({ id: `r-${nextId + i}`, t: t + i * 0.005, tokens: burstTokens, priority: 'NORMAL' }))
      nextId += n
      list.forEach((a) => master.push(a))
      sims.forEach((s) => s.inject(list))
      sys(null, `Judge burst: ${n} extra requests injected at t=${t.toFixed(1)}s`, 'system', 'WARNING')
    },
  }
}

export function startSimSweep(cfg, levels, emit) {
  const ac = new AbortController()
  const { signal } = ac
  const model = MODELS.find((m) => m.id === cfg.model)
  const phase = (name, policy = null, level = null) => emit({ event: 'phase', data: { phase: name, policy, repeat: 1, level, ts: performance.now() / 1000 } })
  const sys = (policy, message) => emit({ event: 'log', data: { policy, ts: 0, level: 'INFO', category: 'system', request_id: null, message } })
  const top = Math.max(...levels)
  const results = {}

  const run = async () => {
    try {
      phase('preflight')
      sys(null, `Pre-flight: pressure test of ${levels.join(', ')} users on ${model.name}, SLA ${cfg.sla.toFixed(1)} s (simulated)`)
      await sleep(300, signal)
      for (const policy of POLICIES) {
        phase(`${policy}_start`, policy)
        sys(policy, `python3 -m scheduler_engine.server --policy ${policy}${policy === 'dynamic' ? ` --sla-ms ${Math.round(cfg.sla * 1000)}` : ''} --max-concurrency 16`)
        await sleep(300, signal)
        phase(`${policy}_warmup`, policy)
        sys(policy, 'Warm-up request sent, discarded from scoring')
        await sleep(300, signal)
        for (const n of levels) {
          if (signal.aborted) return
          phase(`${policy}_load`, policy, n)
          sys(policy, `Load level: ${n} users`)
          const sim = new Sim({ policy, arrivals: buildArrivals({ ...cfg, users: n }), sla: cfg.sla, tpt: model.tpt, emit })
          sim.advance(1e6)
          const result = { ...sim.result(), errors: 0 }
          emit({ event: 'level_result', data: { policy, level: n, repeat: 1, ...result } })
          if (n === top) results[policy] = result
          await sleep(350, signal)
        }
      }
      if (signal.aborted) return
      phase('scoring')
      await sleep(300, signal)
      POLICIES.forEach((policy) => emit({ event: 'result', data: { policy, level: top, ...results[policy] } }))
      phase('done')
      emit({ event: 'done', data: { run_id: cfg.id, status: 'completed' } })
    } catch (e) {
      emit({ event: 'error', data: { message: String(e.message || e) } })
    }
  }
  run()

  return {
    cancel: () => {
      ac.abort()
      emit({ event: 'done', data: { run_id: cfg.id, status: 'cancelled' } })
    },
    canBurst: () => false,
    burst: () => {},
  }
}
