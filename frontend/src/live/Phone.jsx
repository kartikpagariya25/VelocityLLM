import { useCallback, useEffect, useRef, useState } from 'react'
import { joinSession, liveState, sendPrompts, store } from './api'
import './live.css'

const KEY = 'velocity.live.device'
const NAME = 'velocity.live.name'
const SIZES = [10, 25, 50, 100]
const TRAFFIC = [
  ['flood', 'Flood'],
  ['mixed', 'Mixed'],
  ['steady', 'Steady'],
  ['burst', 'Burst'],
]
const PROMPTS = [
  ['short', 'Short'],
  ['medium', 'Medium'],
  ['long', 'Long'],
  ['custom', 'My own'],
]

const loadDevice = () => {
  try {
    return JSON.parse(store.get(KEY) || 'null')
  } catch {
    return null
  }
}

export default function Phone() {
  const [dev, setDev] = useState(loadDevice)
  const [name, setName] = useState(() => store.get(NAME) || '')
  const [state, setState] = useState(null)
  const [spec, setSpec] = useState({ requests: 20, scenario: 'flood', prompt_preset: 'medium', prompt_text: '' })
  const [error, setError] = useState('')
  const [sending, setSending] = useState(false)
  const [sent, setSent] = useState(null)
  const alive = useRef(true)

  const poll = useCallback(async () => {
    try {
      const s = await liveState(dev?.id)
      if (!alive.current) return
      setState(s)
      if (dev && s.joined === false) {
        store.set(KEY, null)
        setDev(null)
      }
    } catch (e) {
      if (alive.current) setError(e.message)
    }
  }, [dev])

  useEffect(() => {
    alive.current = true
    poll()
    const id = setInterval(poll, 2000)
    return () => {
      alive.current = false
      clearInterval(id)
    }
  }, [poll])

  const patch = (p) => setSpec((s) => ({ ...s, ...p }))

  const send = async () => {
    setError('')
    if (spec.prompt_preset === 'custom' && !spec.prompt_text.trim()) return setError('Type a prompt or pick a preset.')
    setSending(true)
    try {
      let device = dev
      const wanted = name.trim() || 'Phone'
      if (!device) {
        const joined = await joinSession(wanted)
        device = { id: joined.device_id, name: joined.name }
        store.set(KEY, JSON.stringify(device))
        setDev(device)
      }
      store.set(NAME, name.trim())
      const body = { ...spec, prompt_text: spec.prompt_preset === 'custom' ? spec.prompt_text.trim() : null }
      setSent(await sendPrompts(device.id, body))
    } catch (e) {
      setError(e.message)
    } finally {
      setSending(false)
    }
  }

  const run = state?.run
  const busy = state?.busy
  const countdown = state?.countdown_s
  let status = ''
  if (busy) status = 'The engine is busy. Watch the PC dashboard.'
  else if (countdown != null) status = `Starting in ${Math.ceil(countdown)} s. Other phones can still join.`
  else if (sent && !sent.armed) status = 'Sent. Waiting for the host to press Start.'
  else if (sent && run?.status === 'completed') status = 'Done. Results are on the PC dashboard.'
  else if (state?.error) status = state.error

  return (
    <div className="lv-phone">
      <div>
        <h1>Velocity Live</h1>
        <p className="lv-sub">Flood the PC's LLM and watch the schedulers compete.</p>
      </div>

      <div className="lv-card">
        <div className="lv-source">
          <span>Data source</span>
          <b>PC GPU{state ? ` · ${state.gpu}` : ''}</b>
          {state?.model && <small>{state.model}</small>}
        </div>

        <label className="lv-field">
          Your name
          <input value={name} maxLength={24} placeholder="Phone" onChange={(e) => setName(e.target.value)} disabled={!!dev} />
        </label>

        <div className="lv-field">
          Number of requests
          <div className="lv-big">{spec.requests}</div>
          <input type="range" min="1" max={state?.limits?.per_device || 100} value={spec.requests} onChange={(e) => patch({ requests: Number(e.target.value) })} />
          <div className="lv-chips">
            {SIZES.map((n) => (
              <button key={n} className={spec.requests === n ? 'on' : ''} onClick={() => patch({ requests: n })}>
                {n}
              </button>
            ))}
          </div>
        </div>

        <div className="lv-field">
          Traffic type
          <div className="lv-chips">
            {TRAFFIC.map(([k, label]) => (
              <button key={k} className={spec.scenario === k ? 'on' : ''} onClick={() => patch({ scenario: k })}>
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="lv-field">
          Prompt type
          <div className="lv-chips">
            {PROMPTS.map(([k, label]) => (
              <button key={k} className={spec.prompt_preset === k ? 'on' : ''} onClick={() => patch({ prompt_preset: k })}>
                {label}
              </button>
            ))}
          </div>
          {spec.prompt_preset === 'custom' && (
            <textarea rows={3} maxLength={500} value={spec.prompt_text} placeholder="What should every request ask?" onChange={(e) => patch({ prompt_text: e.target.value })} />
          )}
        </div>

        <button className="lv-go" onClick={send} disabled={sending || busy}>
          {sending ? 'Sending...' : 'Send prompts'}
        </button>
        {error && <p className="lv-err">{error}</p>}
        {status && <p className="lv-status">{status}</p>}
      </div>
    </div>
  )
}
