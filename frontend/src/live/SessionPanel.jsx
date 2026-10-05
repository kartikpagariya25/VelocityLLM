import { useEffect, useRef, useState } from 'react'
import { clearSpecs, liveState, saveConfig, startNow } from './api'
import './live.css'

const SCEN = { flood: 'flood', mixed: 'mixed', steady: 'steady', burst: 'burst' }

export default function SessionPanel({ onRun, running }) {
  const [state, setState] = useState(null)
  const [host, setHost] = useState('')
  const [err, setErr] = useState('')
  const seen = useRef(null)

  useEffect(() => {
    let stop = false
    const tick = async () => {
      try {
        const s = await liveState()
        if (stop) return
        setState(s)
        if (s.run && s.run.id !== seen.current) {
          seen.current = s.run.id
          onRun(s.run)
        }
      } catch (e) {
        if (!stop) setErr(e.message)
      }
    }
    tick()
    const id = setInterval(tick, 1500)
    return () => {
      stop = true
      clearInterval(id)
    }
  }, [onRun])

  const fallback = window.location.hostname && !['localhost', '127.0.0.1'].includes(window.location.hostname) ? window.location.hostname : state?.hosts?.[0] || ''
  const address = host || fallback
  const joinUrl = address ? `http://${address}:${state?.port || window.location.port || 9000}/#/join` : 'Enter the PC address'
  const cfg = state?.config

  const update = async (patch) => {
    setErr('')
    try {
      await saveConfig({ ...cfg, ...patch })
      setState(await liveState())
    } catch (e) {
      setErr(e.message)
    }
  }

  const act = (fn) => async () => {
    setErr('')
    try {
      await fn()
      setState(await liveState())
    } catch (e) {
      setErr(e.message)
    }
  }

  const smart = cfg?.policies?.includes('smart')
  const togglePolicies = (on) => update({ policies: on ? ['static', 'dynamic', 'smart'] : ['static', 'dynamic'] })
  const total = (state?.devices || []).filter((d) => d.ready).reduce((n, d) => n + d.spec.requests, 0)

  return (
    <section className="controls sess">
      <div className="sess__top">
        <div>
          <p className="lt">Open this on every phone (same Wi-Fi or hotspot)</p>
          <div className="sess__url">{joinUrl}</div>
          <p className="sess__note">
            Engine: {state ? `${state.gpu}${state.mock ? '' : ''} · ${state.model || 'no model'}` : '...'} · Phones: {state?.hosts?.length ? `try ${state.hosts.join(' or ')}` : 'PC address not detected'}. If the phone cannot open it, see docs/LIVE_DEMO.md.
          </p>
        </div>
        <label className="sess__cfg">
          PC address
          <input className="sess__addr" value={host} placeholder={fallback || '192.168.x.x'} onChange={(e) => setHost(e.target.value.trim())} />
        </label>
      </div>

      <div className="sess__devs">
        {(state?.devices || []).length === 0 && <p className="hint">No phone has joined yet. Each phone appears here as soon as it presses Send prompts.</p>}
        {(state?.devices || []).map((d) => (
          <div key={d.name} className={`sess__dev ${d.online ? '' : 'sess__dev--off'}`}>
            <b>{d.name}</b>
            <span>{d.ready ? (d.online ? 'ready' : 'offline') : 'joined'}</span>
            {d.spec && (
              <p>
                {d.spec.requests} requests · {SCEN[d.spec.scenario]} · {d.spec.prompt_preset} prompt
              </p>
            )}
          </div>
        ))}
      </div>

      {cfg && (
        <div className="sess__cfg">
          <label>
            Model
            <select value={state.model || ''} onChange={(e) => update({ model: e.target.value })}>
              {state.models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            SLA (s)
            <input type="number" min="1" max="60" step="0.5" defaultValue={cfg.sla_ms / 1000} key={cfg.sla_ms} onBlur={(e) => update({ sla_ms: Number(e.target.value) * 1000 })} />
          </label>
          <label>
            Repeats (median)
            <select value={cfg.repeats} onChange={(e) => update({ repeats: Number(e.target.value) })}>
              {[1, 3, 5].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label>
            Join window (s)
            <input type="number" min="0.5" max="20" step="0.5" defaultValue={cfg.window_s} key={cfg.window_s} onBlur={(e) => update({ window_s: Number(e.target.value) })} />
          </label>
          <label className="chk">
            <input type="checkbox" checked={cfg.auto_start} onChange={(e) => update({ auto_start: e.target.checked })} />
            Start automatically after a phone sends
          </label>
          <label className="chk">
            <input type="checkbox" checked={!!smart} onChange={(e) => togglePolicies(e.target.checked)} />
            Also run Smart
          </label>
        </div>
      )}

      <div className="sess__act">
        <button className="go" disabled={running || total === 0} onClick={act(startNow)}>
          Start now{total ? ` (${total} requests)` : ''}
        </button>
        <button className="mini" disabled={running} onClick={act(clearSpecs)}>
          Clear phones
        </button>
        {state?.countdown_s != null && <span className="hint">Auto-start in {Math.ceil(state.countdown_s)} s</span>}
        {running && <span className="hint">Run in progress</span>}
        {(err || state?.error) && <span className="err">{err || state.error}</span>}
      </div>
    </section>
  )
}
