import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MODELS, PROMPTS, TRAFFIC, LIMITS, customTokens } from './sim'
import { startSimRun, stepsFor } from './runner'
import { startLiveRun, probe } from './live'
import { newStore, apply, ROWS, change, sentence, toCsv, download, logLine, f2 } from './store'
import { drawTimeline, drawConcurrency } from './charts'
import './Lab.css'

const CATS = ['admission', 'controller', 'lifecycle', 'system']
const BURSTS = [20, 50, 100]
const DEF = { users: 30, prompt: 'medium', text: '', traffic: 'flood', sla: 8, model: 'llama', mode: 'sequential', speed: 4 }

const loadHistory = () => {
  try {
    return JSON.parse(localStorage.getItem('vllm-lab-history') || '[]')
  } catch {
    return []
  }
}

function Tile({ label, value, tone }) {
  return (
    <div className={`tile ${tone || ''}`}>
      <span>{label}</span>
      <b>{value}</b>
    </div>
  )
}

function Panel({ name, policy, store, onPick }) {
  const cv = useRef(null)
  const hit = useRef([])
  const p = store[policy]
  const m = p.metrics
  const tMax = Math.max(10, Math.ceil(Math.max(0, ...store.static.series.map((x) => x.t), ...store.dynamic.series.map((x) => x.t)) / 5) * 5)

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

  return (
    <section className={`panel panel--${policy}`}>
      <header>
        <h3>{name}</h3>
        <span className={`badge badge--${state.toLowerCase()}`}>{state}</span>
      </header>
      <div className="tiles">
        <Tile label="Active" value={m?.active ?? 0} />
        <Tile label="Queued" value={m?.queued ?? 0} />
        <Tile label="Limit" value={m?.concurrency_limit ?? '-'} tone={policy === 'dynamic' ? 'hot' : ''} />
        <Tile label="Done" value={m?.completed ?? 0} tone="ok" />
        <Tile label="Rejected" value={m?.rejected ?? 0} tone="rej" />
        <Tile label="p95" value={p95 == null ? '-' : `${f2(p95)}s`} />
        <Tile label="GPU mem" value={m ? `${Math.round((m.gpu_mem_used_mb / m.gpu_mem_total_mb) * 100)}%` : '-'} />
        <Tile label="Tokens" value={m?.tokens ?? 0} />
      </div>
      <canvas ref={cv} className="timeline" onClick={click} aria-label={`${name} request timeline`} />
    </section>
  )
}

function Console({ logs, running }) {
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
        <h3>Execution log</h3>
        <span className="count">{view.length} lines</span>
        {running && <span className="live-dot" />}
        <div className="seg">
          {['all', 'static', 'dynamic'].map((x) => (
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

function Results({ store, label }) {
  const s = store.static.result
  const d = store.dynamic.result
  if (!s && !d) return null
  return (
    <section className="results">
      <header>
        <h3>Results {label && <small>{label}</small>}</h3>
        {s && d && (
          <div className="exports">
            <button className="mini" onClick={() => download('velocityllm-results.csv', toCsv(store), 'text/csv')}>
              CSV
            </button>
            <button
              className="mini"
              onClick={() => download('velocityllm-results.json', JSON.stringify({ config: store.cfg, static: s, dynamic: d }, null, 2), 'application/json')}
            >
              JSON
            </button>
          </div>
        )}
      </header>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Static</th>
              <th>Dynamic</th>
              <th>Change</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map(([k, name, fmt, good]) => {
              const delta = s && d ? change(s[k], d[k]) : null
              const better = good && delta != null && Math.abs(delta) >= 1 ? (good === 'low' ? delta < 0 : delta > 0) : null
              return (
                <tr key={k}>
                  <td>{name}</td>
                  <td>{s ? fmt(s[k]) : '-'}</td>
                  <td>{d ? fmt(d[k]) : '-'}</td>
                  <td className={better == null ? '' : better ? 'good' : 'bad'}>
                    {delta == null || (!good && s[k] === d[k]) ? '' : `${delta > 0 ? '+' : ''}${delta.toFixed(0)}%`}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {s && d && <p className="verdict">{sentence(store.cfg, s, d)}</p>}
    </section>
  )
}

function Inspector({ req, sla, onClose }) {
  if (!req) return null
  const wait = req.start_s != null ? req.start_s - req.arrival_s : null
  const exec = req.start_s != null ? req.end_s - req.start_s : null
  const total = req.end_s - req.arrival_s
  const rows = [
    ['Request', req.request_id],
    ['Engine', req.policy],
    ['Priority', req.priority],
    ['Outcome', req.status === 'served' ? (total <= sla ? 'Served within SLA' : 'Served beyond SLA') : `Rejected (${String(req.reject_reason).replace('_', ' ')})`],
    ['Arrival', `${f2(req.arrival_s)} s`],
    ['Queue wait', wait == null ? '-' : `${f2(wait)} s`],
    ['Execution', exec == null ? '-' : `${f2(exec)} s`],
    ['Total latency', req.status === 'served' ? `${f2(total)} s` : '-'],
    ['Tokens', req.tokens],
    ['HTTP', req.http_status],
  ]
  if (req.status === 'rejected') rows.push(['Retry after', `${f2(req.retry_after_s)} s`])
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

export default function Lab() {
  const [cfg, setCfg] = useState(DEF)
  const [source, setSource] = useState('sim')
  const [url, setUrl] = useState('http://localhost:9000')
  const [probeState, setProbeState] = useState(null)
  const [, setTick] = useState(0)
  const [burstN, setBurstN] = useState(50)
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
    if (source !== 'live') return setProbeState(null)
    setProbeState('checking')
    probe(url).then((r) => setProbeState(r ? 'ok' : 'down'))
  }, [source, url])

  const set = (k) => (e) => setCfg((c) => ({ ...c, [k]: e.target.type === 'range' || e.target.type === 'number' ? Number(e.target.value) : e.target.value }))
  const st = store.current
  const running = st?.status === 'running'

  const finishRecord = useCallback((s) => {
    if (!s.static.result || !s.dynamic.result) return
    const rec = { id: s.cfg.id, at: new Date().toISOString(), cfg: s.cfg, source: s.source, static: s.static.result, dynamic: s.dynamic.result }
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

  const run = async () => {
    setErr('')
    setPick(null)
    setViewing(null)
    if (cfg.users < LIMITS.users[0] || cfg.users > LIMITS.users[1]) return setErr(`Users must be between ${LIMITS.users[0]} and ${LIMITS.users[1]}.`)
    if (cfg.prompt === 'custom' && !cfg.text.trim()) return setErr('Type a prompt or pick a preset.')
    const full = { ...cfg, id: `run-${Date.now().toString(36)}` }
    const s = newStore(full, stepsFor(cfg.mode), source)
    store.current = s
    const emit = (ev) => {
      apply(s, ev)
      if (ev.event === 'done' && ev.data.status === 'completed') finishRecord(s)
      dirty.current = true
    }
    try {
      ctl.current = source === 'live' ? await startLiveRun(url, full, emit) : startSimRun(full, emit)
    } catch (e) {
      s.status = 'failed'
      s.error = e.message
    }
    dirty.current = true
  }

  const cancel = () => ctl.current?.cancel()
  const canBurst = running && ctl.current?.canBurst()
  const doBurst = () => ctl.current.burst(burstN)

  const phaseIdx = st ? Math.max(0, st.steps.findIndex(([id]) => id === st.phase)) : -1
  const shown = viewing ? { cfg: viewing.cfg, static: { result: viewing.static }, dynamic: { result: viewing.dynamic } } : st
  const tokensEst = cfg.prompt === 'custom' ? customTokens(cfg.text) : PROMPTS[cfg.prompt].tokens

  return (
    <div className="lab">
      <header className="lab__bar">
        <a href="#/" className="lab__brand">
          Velocity<span>LLM</span>
        </a>
        <span className="lab__title">Velocity Arena</span>
        <span className={`src src--${source}`}>{source === 'sim' ? 'Simulated run' : 'Live backend'}</span>
        <a className="lab__back" href="#/">
          Back to site
        </a>
      </header>

      <main className="lab__main">
        <section className="controls">
          <div className="grid">
            <label>
              Users (1 request each)
              <div className="row">
                <input type="range" min={LIMITS.users[0]} max={LIMITS.users[1]} value={cfg.users} onChange={set('users')} />
                <input type="number" min={LIMITS.users[0]} max={LIMITS.users[1]} value={cfg.users} onChange={set('users')} />
              </div>
            </label>
            <label>
              Traffic type
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
              Prompt
              <select value={cfg.prompt} onChange={set('prompt')} disabled={cfg.traffic === 'mixed'}>
                {Object.entries(PROMPTS).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v.label}
                  </option>
                ))}
              </select>
              <small>{cfg.traffic === 'mixed' ? '70% short, 30% long' : `about ${tokensEst} output tokens per request`}</small>
            </label>
            <label>
              SLA target
              <div className="row">
                <input type="range" min={LIMITS.sla[0]} max={LIMITS.sla[1]} step="0.5" value={cfg.sla} onChange={set('sla')} />
                <output>{cfg.sla.toFixed(1)} s</output>
              </div>
            </label>
            <label>
              Model
              <select value={cfg.model} onChange={set('model')}>
                {MODELS.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Run mode
              <select value={cfg.mode} onChange={set('mode')}>
                <option value="sequential">Sequential (one GPU, fair)</option>
                <option value="parallel">Side by side (live demo)</option>
              </select>
            </label>
            <label>
              Data source
              <select value={source} onChange={(e) => setSource(e.target.value)}>
                <option value="sim">Simulation</option>
                <option value="live">Live backend (GPU)</option>
              </select>
              {source === 'live' && (
                <>
                  <input value={url} onChange={(e) => setUrl(e.target.value)} aria-label="Control service URL" />
                  <small className={`probe probe--${probeState}`}>
                    {probeState === 'ok' ? 'Control service reachable' : probeState === 'checking' ? 'Checking...' : 'Control service not reachable'}
                  </small>
                </>
              )}
            </label>
            {source === 'sim' && (
              <label>
                Playback speed
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
          <div className="actions">
            {!running ? (
              <button className="go" onClick={run} disabled={source === 'live' && probeState !== 'ok'}>
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
            </div>
            {err && <span className="err">{err}</span>}
            {st?.status === 'failed' && <span className="err">{st.error}</span>}
            {running && cfg.mode === 'sequential' && st.phase === 'dynamic_load' && <span className="hint">Burst is locked here so both engines face the same traffic.</span>}
          </div>
          {source === 'sim' && (
            <p className="note">
              Simulation: both engines run on one modelled GPU (concurrency-dependent decode speed, 8-slot static batch, AIMD controller and admission rules from the repo). Numbers are illustrative, not measured. Switch to Live backend for real GPU runs.
            </p>
          )}
        </section>

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
            <div className="duo">
              <Panel name="Static scheduler" policy="static" store={st} onPick={setPick} />
              <Panel name="VelocityLLM Dynamic" policy="dynamic" store={st} onPick={setPick} />
            </div>
            <section className="aimd">
              <header>
                <h3>Concurrency over time</h3>
                <span className="legend">
                  <i className="lg lg--s" />
                  Static limit (fixed {8})<i className="lg lg--d" />
                  Dynamic limit (AIMD)
                </span>
              </header>
              <canvas ref={cc} className="aimd__cv" />
            </section>
            <Console logs={st.logs} running={running} />
            <Results store={shown} label={viewing ? 'from history' : ''} />
          </>
        ) : (
          <p className="idle">Choose the traffic, set the SLA and press Run comparison. Static and Dynamic face identical requests, and every scheduler decision shows up in the log below.</p>
        )}

        {history.length > 0 && (
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
                    <b>{h.cfg.users} users</b> {h.cfg.traffic}, SLA {h.cfg.sla.toFixed(1)} s, {h.source === 'sim' ? 'simulated' : 'live'}
                    <span>
                      p99 {f2(h.static.p99_s)} s to {f2(h.dynamic.p99_s)} s
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
