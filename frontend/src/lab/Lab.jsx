import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MODELS, PROMPTS, TRAFFIC, LIMITS, HEAVY_IMAGE, customTokens, registerModels } from './sim'
import { startSimRun, startSimSweep, stepsFor, stepsForPolicies } from './runner'
import { startLiveRun, attachRun, probe, listModels } from './live'
import { newStore, apply, ROWS, change, sentence, smartSentence, verdictOf, toCsv, download, logLine, f2, onTime, unequalText, hasGoodput } from './store'
import { drawTimeline, drawConcurrency, niceMax } from './charts'
import Info from './Info'
import Count from './Count'
import Pressure from './Pressure'
import Saved from './Saved'
import Vision from './Vision'
import SessionPanel from '../live/SessionPanel'
import { startReplay, configOf } from './replay'
import './Lab.css'

const has = (store, policy) => !store.cfg.policies || store.cfg.policies.includes(policy)

const PC_DEFAULT = window.location.port === '9000' ? window.location.origin : 'http://localhost:9000'
const SOURCE_BADGE = { sim: 'Simulated run', pc: 'PC GPU', cloud: 'College GPU', mock: 'Mock engine', replay: 'Recorded run' }
const INFO_KEY = { p50_s: 'p50', p95_s: 'p95', p99_s: 'p99', tokens_per_s: 'tokps', goodput_tokens_per_s: 'goodput', image_tokens_per_s: 'imageTokens', within_sla_served: 'slaServed', within_sla_offered: 'slaOffered', served: 'served', rejected: 'rejected', errors: 'errors' }
const CATS = ['admission', 'controller', 'lifecycle', 'system']
const BURSTS = [20, 50, 100]
const DEF = { users: 30, prompt: 'medium', text: '', traffic: 'flood', sla: 8, model: 'llama', mode: 'sequential', speed: 4, repeats: 1, imageMix: 'mixed' }

const loadHistory = () => {
  try {
    return JSON.parse(localStorage.getItem('vllm-lab-history') || '[]')
  } catch {
    return []
  }
}

function DeviceTable({ store }) {
  const dr = store.deviceResults
  const pols = ['static', 'dynamic', 'smart'].filter((p) => dr && dr[p] && Object.keys(dr[p]).length)
  if (!pols.length) return null
  const names = [...new Set(pols.flatMap((p) => Object.keys(dr[p])))]
  return (
    <div className="tablewrap devtable">
      <h4>Per device</h4>
      <table>
        <thead>
          <tr>
            <th>Device</th>
            {pols.map((p) => (
              <th key={p}>{p[0].toUpperCase() + p.slice(1)} on time / p99 / tok/s</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {names.map((n) => (
            <tr key={n}>
              <td>{n}</td>
              {pols.map((p) => {
                const r = dr[p][n]
                return <td key={p}>{r ? `${onTime(r)}/${r.offered} · ${f2(r.p99_s)} s · ${(r.tokens_per_s || 0).toFixed(0)}` : '-'}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Tile({ label, value, tone, info }) {
  return (
    <div className={`tile ${tone || ''}`}>
      <span>
        {label}
        {info && <Info id={info} />}
      </span>
      <b>{value}</b>
    </div>
  )
}

export function Panel({ name, policy, store, onPick }) {
  const cv = useRef(null)
  const hit = useRef([])
  const p = store[policy]
  const m = p.metrics
  const tMax = niceMax(Math.max(0, ...store.static.series.map((x) => x.t), ...store.dynamic.series.map((x) => x.t), ...(store.smart?.series || []).map((x) => x.t)))

  useEffect(() => {
    if (cv.current) drawTimeline(cv.current, p.reqs, store.cfg.sla, tMax, hit.current)
  })

  const click = (e) => {
    const rect = cv.current.getBoundingClientRect()
    const y = e.clientY - rect.top
    const row = hit.current.find((r) => y >= r.y && y < r.y + Math.max(r.h, 3))
    if (row) onPick({ policy, ...row.r })
  }

  const served = p.reqs.filter((r) => r.status === 'served')
  const lat = served.map((r) => r.end_s - r.arrival_s).sort((a, b) => a - b)
  const p95 = lat.length ? lat[Math.min(lat.length - 1, Math.ceil(0.95 * lat.length) - 1)] : null
  const state = p.result ? 'Done' : m ? 'Running' : store.status === 'running' ? 'Waiting' : 'Idle'
  const vision = !!store.cfg.vision
  const imageTok = served.reduce((n, r) => n + (r.image_tokens || 0), 0)
  const heavy = p.reqs.filter((r) => (r.image_tokens || 0) >= HEAVY_IMAGE)
  const heavyOnTime = heavy.filter((r) => r.status === 'served' && r.end_s - r.arrival_s <= store.cfg.sla).length

  return (
    <section className={`panel panel--${policy}`}>
      <header>
        <h3>
          {name}
          <Info id={policy} />
        </h3>
        <span className={`badge badge--${state.toLowerCase()}`}>{state}</span>
      </header>
      <div className={`tiles ${vision ? 'tiles--vision' : ''}`}>
        <Tile info="active" label="Active" value={m?.active ?? 0} />
        <Tile info="queued" label="Queued" value={m?.queued ?? 0} />
        <Tile info="limit" label="Limit" value={m?.concurrency_limit ?? '-'} tone={policy === 'dynamic' ? 'hot' : policy === 'smart' ? 'smart' : ''} />
        <Tile info="done" label="Done" value={m?.completed ?? 0} tone="ok" />
        <Tile info="rejected" label="Rejected" value={m?.rejected ?? 0} tone="rej" />
        <Tile info="p95" label="p95" value={p95 == null ? '-' : `${f2(p95)}s`} />
        <Tile info="mem" label="GPU mem" value={m ? `${Math.round((m.gpu_mem_used_mb / m.gpu_mem_total_mb) * 100)}%` : '-'} />
        <Tile info="tokens" label="Tokens" value={m?.tokens ?? 0} />
        {vision && <Tile info="imageTokens" label="Image tok" value={imageTok} />}
        {vision && <Tile info="bigImages" label="Big images" value={heavy.length ? `${heavyOnTime}/${heavy.length}` : '-'} tone="hot" />}
      </div>
      <p className="cap">
        Request timeline <Info id="timeline" />
      </p>
      <p className="key">
        <i className="dot dot--ok" />
        on time
        <i className="dot dot--late" />
        late
        <i className="dot dot--rej" />
        declined
        {vision && (
          <>
            <i className="dot dot--big" />
            large image
          </>
        )}
      </p>
      <canvas ref={cv} className="timeline" onClick={click} aria-label={`${name} request timeline`} />
    </section>
  )
}

export function Console({ logs, running }) {
  const [pol, setPol] = useState('all')
  const [cats, setCats] = useState(new Set(CATS))
  const [q, setQ] = useState('')
  const [paused, setPaused] = useState(false)
  const frozen = useRef(0)
  const box = useRef(null)

  if (!paused) frozen.current = logs.length
  const view = useMemo(() => {
    const needle = q.toLowerCase()
    return logs
      .slice(0, frozen.current)
      .filter((l) => (pol === 'all' || l.policy === pol || (!l.policy && pol !== 'all')) && cats.has(l.category) && (!needle || l.message.toLowerCase().includes(needle)))
  }, [logs, logs.length, pol, cats, q, paused])
  const shown = view.slice(-500)

  useEffect(() => {
    if (!paused && box.current) box.current.scrollTop = box.current.scrollHeight
  }, [shown.length, paused])

  const toggle = (c) =>
    setCats((s) => {
      const n = new Set(s)
      n.has(c) ? n.delete(c) : n.add(c)
      return n
    })
  const text = () => view.map(logLine).join('\n')

  return (
    <section className="console">
      <header>
        <h3>
          Execution log <Info id="log" />
        </h3>
        <span className="count">{view.length} lines</span>
        {running && <span className="live-dot" />}
        <div className="seg">
          {['all', 'static', 'dynamic', 'smart'].map((x) => (
            <button key={x} className={pol === x ? 'on' : ''} onClick={() => setPol(x)}>
              {x}
            </button>
          ))}
        </div>
        <div className="chips">
          {CATS.map((c) => (
            <button key={c} className={`cat cat--${c} ${cats.has(c) ? 'on' : ''}`} onClick={() => toggle(c)}>
              {c}
            </button>
          ))}
        </div>
        <input className="search" placeholder="Filter text" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="mini" onClick={() => setPaused((v) => !v)}>
          {paused ? 'Resume' : 'Pause'}
        </button>
        <button className="mini" onClick={() => navigator.clipboard?.writeText(text())}>
          Copy
        </button>
        <button className="mini" onClick={() => download('velocityllm-run.log', text())}>
          Download
        </button>
      </header>
      <div className="logbox" ref={box} role="log">
        {shown.length === 0 && <p className="empty">Logs from both engines appear here, line by line, as soon as a run starts.</p>}
        {shown.map((l) => (
          <div key={l.n} className={`ln ln--${l.category} ln--${l.level.toLowerCase()}`}>
            <i>{l.ts.toFixed(2).padStart(7)}s</i>
            <b className={`tag tag--${l.policy || 'sys'}`}>{(l.policy || 'sys').slice(0, 3)}</b>
            <em>{l.category}</em>
            <span>{l.message}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

export function Results({ store, label, recordingUrl }) {
  const s = store.static.result
  const d = store.dynamic.result
  const m = store.smart?.result
  if (!s && !d && !m) return null
  return (
    <section className="results">
      <header>
        <h3>
          Results {label && <small>{label}</small>}
          {store.users > 0 && store.cfg.levels?.length > 1 && <small>at {store.users} users</small>}
        </h3>
        {s && d && (
          <div className="exports">
            {recordingUrl && (
              <a className="mini" href={recordingUrl}>
                Recording
              </a>
            )}
            <button className="mini" onClick={() => download('velocityllm-results.csv', toCsv(store), 'text/csv')}>
              CSV
            </button>
            <button
              className="mini"
              onClick={() => download('velocityllm-results.json', JSON.stringify({ config: store.cfg, static: s, dynamic: d, smart: m }, null, 2), 'application/json')}
            >
              JSON
            </button>
          </div>
        )}
      </header>
      {s && d && (
        <div className="heads">
          {[
            ['Answered on time', `of ${s.offered || d.offered || '?'} users`, onTime(s), onTime(d), m && onTime(m), (v) => `${Math.round(v)}`, false, 'slaOffered'],
            ['p99 latency', 'seconds, lower is better', s.p99_s, d.p99_s, m && m.p99_s, (v) => `${v.toFixed(2)} s`, true, 'p99'],
            unequalText(s, d) && hasGoodput(s, d)
              ? ['On-time tokens/s', 'text delivered within the SLA', s.goodput_tokens_per_s, d.goodput_tokens_per_s, m && m.goodput_tokens_per_s, (v) => v.toFixed(0), false, 'goodput']
              : ['Throughput (tokens/s)', 'tokens per second', s.tokens_per_s, d.tokens_per_s, m && m.tokens_per_s, (v) => v.toFixed(0), false, 'tokps'],
          ].map(([name, hint, a, b, c, fmt, low, info]) => {
            const same = info === 'tokps' && unequalText(s, d)
            const delta = change(a, b)
            const v = same || delta == null ? 'same' : verdictOf(delta, low)
            return (
              <div key={name} className={`head head--${v}`}>
                <span>
                  {name} <Info id={info} />
                </span>
                <b>
                  <Count value={a} fmt={fmt} /> <i>to</i> <Count value={b} fmt={fmt} />
                </b>
                <em>
                  {same ? 'Not comparable: different amount of text' : delta == null || Math.abs(delta) < 0.5 ? hint : `${delta > 0 ? '+' : ''}${delta.toFixed(0)}% for Dynamic`}
                </em>
                {c != null && (
                  <em className="head__smart">
                    Smart <b>{fmt(c)}</b>
                  </em>
                )}
              </div>
            )
          })}
        </div>
      )}
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Static</th>
              <th>Dynamic</th>
              {m && <th>Smart</th>}
              <th>Dynamic vs Static</th>
              {m && <th>Smart vs Static</th>}
            </tr>
          </thead>
          <tbody>
            {ROWS.filter(([k]) => (k !== 'errors' || s?.errors || d?.errors || m?.errors) && (k !== 'goodput_tokens_per_s' || hasGoodput(s, d)) && (k !== 'image_tokens_per_s' || s?.image_tokens > 0 || d?.image_tokens > 0 || m?.image_tokens > 0)).map(([k, name, fmt, good]) => {
              const cell = (other) => {
                const delta = s && other ? change(s[k], other[k]) : null
                if (delta == null) return { text: '', cls: '' }
                const verdict = good ? verdictOf(delta, good === 'low') : 'same'
                const cls = !good || verdict === 'same' ? '' : verdict === 'better' ? 'good' : 'bad'
                const text = good ? `${delta > 0 ? '+' : ''}${delta.toFixed(0)}%` : s[k] === other[k] ? '' : `${other[k] - s[k] > 0 ? '+' : ''}${other[k] - s[k]}`
                return { text, cls }
              }
              const dCell = cell(d)
              const mCell = cell(m)
              return (
                <tr key={k}>
                  <td>
                    {name}
                    {INFO_KEY[k] && <Info id={INFO_KEY[k]} />}
                  </td>
                  <td>{s ? fmt(s[k]) : '-'}</td>
                  <td>{d ? fmt(d[k]) : '-'}</td>
                  {m && <td>{fmt(m[k])}</td>}
                  <td className={dCell.cls}>{dCell.text}</td>
                  {m && <td className={mCell.cls}>{mCell.text}</td>}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <DeviceTable store={store} />
      {[...(s?.warnings || []).map((w) => `Static: ${w}`), ...(d?.warnings || []).map((w) => `Dynamic: ${w}`), ...(m?.warnings || []).map((w) => `Smart: ${w}`)].map((w) => (
        <p key={w} className="warnline">
          {w}
        </p>
      ))}
      {s && d && <p className="verdict">{sentence(store.cfg, s, d, store.users, store.repeats)}</p>}
      {s && m && <p className="verdict">{smartSentence(store.cfg, s, m)}</p>}
    </section>
  )
}

export function Inspector({ req, sla, onClose }) {
  if (!req) return null
  const wait = req.start_s != null ? req.start_s - req.arrival_s : null
  const exec = req.start_s != null ? req.end_s - req.start_s : null
  const total = req.end_s - req.arrival_s
  const rows = [
    ['Request', req.request_id],
    ['Engine', req.policy],
    ...(req.device ? [['Phone', req.device]] : []),
    ['Priority', req.priority],
    ['Outcome', req.status === 'served' ? (total <= sla ? 'Served within SLA' : 'Served beyond SLA') : req.status === 'error' ? `Failed (${String(req.reject_reason).replace('_', ' ')})` : `Rejected (${String(req.reject_reason).replace('_', ' ')})`],
    ['Arrival', `${f2(req.arrival_s)} s`],
    ['Queue wait', wait == null ? '-' : `${f2(wait)} s`],
    ['Execution', exec == null ? '-' : `${f2(exec)} s`],
    ['Total latency', req.status === 'served' ? `${f2(total)} s` : '-'],
    ['Tokens', req.tokens],
    ...(req.image_tokens ? [['Image tokens', req.image_tokens]] : []),
    ['HTTP', req.http_status],
  ]
  if (req.status === 'rejected') rows.push(['Retry after', `${f2(req.retry_after_s)} s`])
  if (req.error) rows.push(['Error', req.error])
  return (
    <aside className="inspector" role="dialog" aria-label="Request inspector">
      <button className="x" onClick={onClose} aria-label="Close">
        ×
      </button>
      <h3>Request inspector</h3>
      <dl>
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </aside>
  )
}

export default function Lab({ session = false }) {
  const [cfg, setCfg] = useState(DEF)
  const [source, setSource] = useState(session ? 'pc' : 'sim')
  const [urls, setUrls] = useState(() => {
    try {
      return { pc: session ? window.location.origin : PC_DEFAULT, cloud: localStorage.getItem('vllm-cloud-url') || '' }
    } catch {
      return { pc: session ? window.location.origin : PC_DEFAULT, cloud: '' }
    }
  })
  const live = source !== 'sim'
  const url = urls[source] || ''
  const setUrl = (v) => {
    setUrls((u) => ({ ...u, [source]: v }))
    if (source === 'cloud') {
      try {
        localStorage.setItem('vllm-cloud-url', v)
      } catch {
        return
      }
    }
  }
  const [probeState, setProbeState] = useState(null)
  const [liveInfo, setLiveInfo] = useState(null)
  const [simVer, setSimVer] = useState(0)
  const [, setTick] = useState(0)
  const [burstN, setBurstN] = useState(50)
  const [recheck, setRecheck] = useState(0)
  const [pick, setPick] = useState(null)
  const [history, setHistory] = useState(loadHistory)
  const [viewing, setViewing] = useState(null)
  const [err, setErr] = useState('')
  const store = useRef(null)
  const ctl = useRef(null)
  const dirty = useRef(false)
  const cc = useRef(null)

  useEffect(() => {
    const id = setInterval(() => {
      if (dirty.current) {
        dirty.current = false
        setTick((n) => n + 1)
      }
    }, 80)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    if (cc.current && store.current) drawConcurrency(cc.current, store.current)
  })

  useEffect(() => {
    setLiveInfo(null)
    if (!live || !url) return setProbeState(live ? 'down' : null)
    setProbeState('checking')
    let stale = false
    const id = setTimeout(
      () =>
        probe(url).then((r) => {
          if (stale) return
          setLiveInfo(r)
          setProbeState(r ? 'ok' : 'down')
        }),
      400,
    )
    return () => {
      stale = true
      clearTimeout(id)
    }
  }, [source, url, live, recheck])

  useEffect(() => {
    if (!live || !liveModels.length) return
    if (!liveModels.some((m) => m.id === cfg.model)) setCfg((c) => ({ ...c, model: liveModels[0].id }))
  }, [liveInfo, live])

  useEffect(() => {
    if (live) return
    let stale = false
    listModels(urls.pc).then((found) => {
      if (stale) return
      registerModels(found)
      setSimVer((v) => v + 1)
    })
    return () => {
      stale = true
    }
  }, [live])

  useEffect(() => {
    if (!live && !MODELS.some((m) => m.id === cfg.model)) setCfg((c) => ({ ...c, model: MODELS[0].id }))
  }, [live, simVer])

  const set = (k) => (e) => setCfg((c) => ({ ...c, [k]: e.target.type === 'range' || e.target.type === 'number' || k === 'repeats' ? Number(e.target.value) : e.target.value }))
  const st = store.current
  const shownSource = st ? (st.info?.mock ? 'mock' : st.source) : source
  const liveModels = (liveInfo?.models || []).filter((m) => m.available)
  const modelOptions = live ? liveModels.map((m) => ({ id: m.id, name: m.name, vision: !!m.vision })) : MODELS
  const isVision = !!modelOptions.find((m) => m.id === cfg.model)?.vision
  const blocked = live && (probeState !== 'ok' || !liveInfo?.ok)
  const running = st?.status === 'running'

  const finishRecord = useCallback((s) => {
    if (!s.static.result || !s.dynamic.result) return
    const rec = { id: s.cfg.id, at: new Date().toISOString(), cfg: s.cfg, source: s.source, static: s.static.result, dynamic: s.dynamic.result, smart: s.smart.result || null }
    setHistory((h) => {
      const next = [rec, ...h].slice(0, 12)
      try {
        localStorage.setItem('vllm-lab-history', JSON.stringify(next))
      } catch {
        return next
      }
      return next
    })
  }, [])

  const launch = async (levels) => {
    setErr('')
    setPick(null)
    setViewing(null)
    const sweep = levels && levels.length > 1
    if (!sweep && (cfg.users < LIMITS.users[0] || cfg.users > LIMITS.users[1])) return setErr(`Users must be between ${LIMITS.users[0]} and ${LIMITS.users[1]}.`)
    if (cfg.prompt === 'custom' && !cfg.text.trim()) return setErr('Type a prompt or pick a preset.')
    const full = { ...cfg, vision: isVision, id: `run-${Date.now().toString(36)}`, levels: sweep ? levels : [cfg.users], users: sweep ? Math.max(...levels) : cfg.users }
    const s = newStore(full, stepsFor(cfg.mode), source)
    s.users = full.users
    s.repeats = live ? cfg.repeats : 1
    store.current = s
    const emit = (ev) => {
      apply(s, ev)
      if (ev.event === 'done' && ev.data.status === 'completed' && !sweep) finishRecord(s)
      dirty.current = true
    }
    try {
      if (live) {
        ctl.current = await startLiveRun(url, full, emit)
        s.runId = ctl.current.runId || null
      } else {
        ctl.current = sweep ? startSimSweep(full, levels, emit) : startSimRun(full, emit)
      }
    } catch (e) {
      s.status = 'failed'
      s.error = e.message
    }
    dirty.current = true
  }
  const run = () => launch(null)

  const replay = (rec, speed) => {
    setErr('')
    setPick(null)
    setViewing(null)
    const full = configOf(rec)
    const s = newStore(full, full.policies ? stepsForPolicies(full.policies) : stepsFor('sequential'), 'replay')
    s.users = full.users
    s.repeats = rec.config.repeats || 1
    store.current = s
    const emit = (ev) => {
      apply(s, ev)
      dirty.current = true
    }
    ctl.current = startReplay(rec, speed, emit)
    dirty.current = true
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const judgePreset = (setLevels) => {
    setCfg((c) => ({ ...c, traffic: 'mixed', sla: 5 }))
    setLevels([30, 60, 100, 150])
  }

  const attach = useCallback((run) => {
    ctl.current?.stop?.()
    setPick(null)
    setViewing(null)
    const c = run.config
    const policies = c.policies || ['static', 'dynamic']
    const full = { id: run.id, users: c.requests, levels: [c.requests], traffic: 'live phone', sla: c.sla_ms / 1000, model: c.model, prompt: 'medium', text: '', imageMix: 'mixed', mode: 'sequential', speed: 1, policies, vision: false }
    const s = newStore(full, stepsForPolicies(policies), 'pc')
    s.users = c.requests
    s.repeats = c.repeats || 1
    s.runId = run.id
    store.current = s
    const base = window.location.origin
    const emit = (ev) => {
      apply(s, ev)
      dirty.current = true
    }
    const handle = attachRun(base, run.id, emit)
    ctl.current = {
      ...handle,
      cancel: () => fetch(`${base}/api/runs/${run.id}/cancel`, { method: 'POST' }),
      canBurst: () => false,
      burst: () => {},
    }
    dirty.current = true
  }, [])

  const cancel = () => ctl.current?.cancel()
  const canBurst = running && ctl.current?.canBurst()
  const doBurst = () => ctl.current.burst(burstN)

  const phaseIdx = st ? Math.max(0, st.steps.findIndex(([id]) => id === st.phase)) : -1
  const shown = viewing ? { cfg: viewing.cfg, static: { result: viewing.static }, dynamic: { result: viewing.dynamic }, smart: { result: viewing.smart || null } } : st
  const tokensEst = cfg.prompt === 'custom' ? customTokens(cfg.text) : PROMPTS[cfg.prompt].tokens

  return (
    <div className="lab">
      <header className="lab__bar">
        <a href="#/" className="lab__brand">
          <img src="/logo.png" alt="" className="lab__logo" />
          <span className="sr-only">Velocity</span>
          <span>LLM</span>
        </a>
        <span className="lab__title">{session ? 'Velocity Live' : 'Velocity Arena'}</span>
        <span className={`src src--${shownSource}`}>{SOURCE_BADGE[shownSource]}</span>
        <a className="lab__back" href="#/">
          Back to site
        </a>
      </header>

      <main className="lab__main">
        {session && <SessionPanel onRun={attach} running={running} />}
        {!session && <section className="controls">
          <div className="grid">
            <label>
              <span className="lt">Users (1 request each) <Info id="users" /></span>
              <div className="row">
                <input type="range" min={LIMITS.users[0]} max={LIMITS.users[1]} value={cfg.users} onChange={set('users')} />
                <input type="number" min={LIMITS.users[0]} max={LIMITS.users[1]} value={cfg.users} onChange={set('users')} />
              </div>
            </label>
            <label>
              <span className="lt">Traffic type <Info id="traffic" /></span>
              <select value={cfg.traffic} onChange={set('traffic')}>
                {Object.entries(TRAFFIC).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v.label}
                  </option>
                ))}
              </select>
              <small>{TRAFFIC[cfg.traffic].hint}</small>
            </label>
            <label>
              <span className="lt">Prompt <Info id="prompt" /></span>
              <select value={cfg.prompt} onChange={set('prompt')} disabled={cfg.traffic === 'mixed'}>
                {Object.entries(PROMPTS).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v.label}
                  </option>
                ))}
              </select>
              <small>{cfg.traffic === 'mixed' ? '70% short, 30% long' : `about ${tokensEst} output tokens per request`}{isVision ? ', the question is about the image' : ''}</small>
            </label>
            <label>
              <span className="lt">SLA target <Info id="sla" /></span>
              <div className="row">
                <input type="range" min={LIMITS.sla[0]} max={LIMITS.sla[1]} step="0.5" value={cfg.sla} onChange={set('sla')} />
                <output>{cfg.sla.toFixed(1)} s</output>
              </div>
            </label>
            <label>
              <span className="lt">Model <Info id="model" /></span>
              <select value={cfg.model} onChange={set('model')}>
                {modelOptions.length === 0 && <option value="">No models available</option>}
                {modelOptions.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                    {m.vision && !/vision/i.test(m.name) ? ' (vision)' : ''}
                  </option>
                ))}
              </select>
              {isVision && <small>Vision model: every request carries an image.</small>}
            </label>
            <label>
              <span className="lt">Run mode <Info id="mode" /></span>
              <select value={live ? 'sequential' : cfg.mode} onChange={set('mode')} disabled={live}>
                <option value="sequential">Sequential (one GPU, fair)</option>
                <option value="parallel">Side by side (simulation only)</option>
              </select>
            </label>
            <label>
              <span className="lt">Data source <Info id="source" /></span>
              <select value={source} onChange={(e) => setSource(e.target.value)}>
                <option value="sim">Simulation</option>
                <option value="pc">PC GPU (this machine)</option>
                <option value="cloud">Cloud GPU (college)</option>
              </select>
              {live && (
                <>
                  <input
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder={source === 'cloud' ? 'https://college-gpu-host:9000' : 'http://localhost:9000'}
                    aria-label="Control service URL"
                  />
                  <small className={`probe probe--${probeState}`} onClick={() => setRecheck((n) => n + 1)} title="Click to check again">
                    {probeState === 'ok' ? 'Control service reachable' : probeState === 'checking' ? 'Checking...' : url ? 'Control service not reachable' : 'Enter the control service URL'}
                  </small>
                </>
              )}
            </label>
            {live && (
              <label>
                <span className="lt">
                  Repeats (median) <Info id="repeats" />
                </span>
                <select value={cfg.repeats} onChange={set('repeats')}>
                  {[1, 3, 5].map((x) => (
                    <option key={x} value={x}>
                      {x === 1 ? '1 (quick)' : `${x} (fairer, slower)`}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {source === 'sim' && (
              <label>
                <span className="lt">Playback speed <Info id="speed" /></span>
                <select value={cfg.speed} onChange={set('speed')}>
                  {[1, 2, 4, 8].map((x) => (
                    <option key={x} value={x}>
                      {x}x
                    </option>
                  ))}
                </select>
              </label>
            )}
          </div>
          {cfg.prompt === 'custom' && cfg.traffic !== 'mixed' && (
            <label className="custom">
              Custom prompt
              <textarea
                maxLength={LIMITS.promptChars}
                value={cfg.text}
                onChange={set('text')}
                placeholder="Type what every user will ask"
                rows={2}
              />
              <small>
                {cfg.text.length}/{LIMITS.promptChars} characters
              </small>
            </label>
          )}
          {isVision && <Vision mix={cfg.imageMix} onChange={(imageMix) => setCfg((c) => ({ ...c, imageMix }))} />}
          <div className="actions">
            {!running ? (
              <button className="go" onClick={run} disabled={blocked || (live && modelOptions.length === 0)}>
                Run comparison
              </button>
            ) : (
              <button className="stop" onClick={cancel}>
                Cancel run
              </button>
            )}
            <div className="burst">
              <select value={burstN} onChange={(e) => setBurstN(Number(e.target.value))} aria-label="Burst size">
                {BURSTS.map((n) => (
                  <option key={n} value={n}>
                    +{n} requests
                  </option>
                ))}
              </select>
              <button className="boom" onClick={doBurst} disabled={!canBurst}>
                Burst
              </button>
              <Info id="burst" />
            </div>
            {err && <span className="err">{err}</span>}
            {st?.status === 'failed' && <span className="err">{st.error}</span>}
            {live && !running && <span className="hint">Burst works in the simulation only.</span>}
            {running && cfg.mode === 'sequential' && (st.phase === 'dynamic_load' || st.phase === 'smart_load') && <span className="hint">Burst is locked here so every engine faces the same traffic.</span>}
          </div>
          {live && liveInfo && (
            <ul className="checks">
              {liveInfo.checks
                .filter((c) => c.level !== 'info' || !c.ok)
                .map((c) => (
                  <li key={c.id} className={c.ok ? 'ok' : c.level === 'warn' ? 'warn' : 'bad'}>
                    <b>{c.label}</b> {c.detail}
                  </li>
                ))}
            </ul>
          )}
          {live && liveInfo?.mock && <p className="note">This control service uses the mock engine: the whole pipeline is real, but no GPU or model is involved, so the numbers say nothing about real hardware.</p>}
          {source === 'sim' && (
            <p className="note">
              Simulation: all three engines run on one modelled GPU (concurrency-dependent decode speed, 8-slot static batch, AIMD controller, and the admission, capacity and queue rules from the repo). Numbers are illustrative, not measured. Pick PC GPU or Cloud GPU for real runs.
            </p>
          )}
        </section>}

        {st?.status === 'failed' && st.errorDetail.length > 0 && (
          <pre className="errdetail">{st.errorDetail.join('\n')}</pre>
        )}
        {st?.info && (
          <p className="strip">
            {st.info.model} · {st.info.mock ? 'mock engine' : st.info.gpu?.name || 'GPU not detected'} · SLA {st.info.sla_s.toFixed(1)} s · {st.info.users} users · {st.info.scenario}{st.info.vision ? ` · images: ${st.info.image_mix || cfg.imageMix}` : ''}
            {st.info.repeats > 1 ? ` · ${st.info.repeats} repeats (median)` : ''}
          </p>
        )}
        {st && (
          <ol className="stepper" aria-label="Run phases">
            {st.steps.map(([id, label], i) => (
              <li key={id} className={i < phaseIdx || st.status === 'done' ? 'done' : i === phaseIdx ? 'now' : ''}>
                <span>{i + 1}</span>
                {label}
              </li>
            ))}
          </ol>
        )}

        {st ? (
          <>
            <div className={`duo ${has(st, 'smart') ? 'duo--3' : ''}`}>
              <Panel name="Static scheduler" policy="static" store={st} onPick={setPick} />
              <Panel name="VelocityLLM Dynamic" policy="dynamic" store={st} onPick={setPick} />
              {has(st, 'smart') && <Panel name="VelocityLLM Smart" policy="smart" store={st} onPick={setPick} />}
            </div>
            {live && !session && st.status === 'done' && !st.smart.result && (
              <p className="note">This run has no Smart data: it came from a control service without Smart support. Update the control service to compare Smart on the GPU.</p>
            )}
            <section className="aimd">
              <header>
                <h3>
                  Concurrency over time <Info id="concurrency" />
                </h3>
                <span className="legend">
                  <i className="lg lg--s" />
                  Static limit (fixed {8})<i className="lg lg--d" />
                  Dynamic limit (AIMD)
                  {has(st, 'smart') && (
                    <>
                      <i className="lg lg--m" />
                      Smart limit (capacity planner)
                    </>
                  )}
                </span>
              </header>
              <canvas ref={cc} className="aimd__cv" />
            </section>
            <Console logs={st.logs} running={running} />
            <Results store={shown} label={viewing ? 'from history' : st.source === 'replay' ? 'recorded run' : ''} recordingUrl={live && st.runId && st.status === 'done' ? `${url.replace(/\/+$/, '')}/api/runs/${st.runId}/recording` : null} />
          </>
        ) : (
          <p className="idle">{session ? 'Waiting for the first phone. The comparison appears here the moment a run starts.' : 'Choose the traffic, set the SLA and press Run comparison. Static, Dynamic and Smart face identical requests, and every scheduler decision shows up in the log below.'}</p>
        )}

        {!session && <Pressure store={st} running={running} disabled={blocked || (live && modelOptions.length === 0)} onRun={launch} onJudgePreset={judgePreset} />}
        <Saved base={live ? url : urls.pc} refreshKey={st?.status} onReplay={replay} busy={running} />

        {!session && history.length > 0 && (
          <section className="history">
            <header>
              <h3>Recent runs</h3>
              <button
                className="mini"
                onClick={() => {
                  setHistory([])
                  try {
                    localStorage.removeItem('vllm-lab-history')
                  } catch {
                    setViewing(null)
                  }
                }}
              >
                Clear
              </button>
            </header>
            <ul>
              {history.map((h) => (
                <li key={h.id}>
                  <button onClick={() => (setViewing(h), window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }))}>
                    <b>{h.cfg.users} users</b> {h.cfg.traffic}, SLA {h.cfg.sla.toFixed(1)} s, {SOURCE_BADGE[h.source] || 'simulated'}
                    <span>
                      p99 {f2(h.static.p99_s)} s to {f2(h.dynamic.p99_s)} s{h.smart ? ` to ${f2(h.smart.p99_s)} s` : ''}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>
      <Inspector req={pick} sla={st?.cfg.sla ?? 5} onClose={() => setPick(null)} />
    </div>
  )
}
