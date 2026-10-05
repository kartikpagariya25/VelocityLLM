import { useCallback, useEffect, useRef, useState } from 'react'
import Info from './Info'
import { parseRecording } from './replay'
import { f2 } from './store'

const when = (t) => new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })

export default function Saved({ base, refreshKey, onReplay, busy }) {
  const [runs, setRuns] = useState(null)
  const [msg, setMsg] = useState('')
  const [speed, setSpeed] = useState(1)
  const file = useRef(null)

  const load = useCallback(async () => {
    if (!base) return setRuns(null)
    try {
      const r = await fetch(`${base.replace(/\/+$/, '')}/api/runs`, { signal: AbortSignal.timeout(5000) })
      if (!r.ok) throw new Error()
      const all = await r.json()
      setRuns(all.filter((x) => x.has_results && !x.recorded))
    } catch {
      setRuns(null)
    }
  }, [base])

  useEffect(() => {
    load()
  }, [load, refreshKey])

  const play = async (id) => {
    setMsg('')
    try {
      const r = await fetch(`${base.replace(/\/+$/, '')}/api/runs/${id}/recording`)
      if (!r.ok) throw new Error(`The control service returned HTTP ${r.status}`)
      onReplay(parseRecording(await r.json()), speed)
    } catch (e) {
      setMsg(e.message)
    }
  }

  const pick = async (e) => {
    const f = e.target.files?.[0]
    e.target.value = ''
    if (!f) return
    setMsg('')
    try {
      onReplay(parseRecording(JSON.parse(await f.text())), speed)
    } catch (err) {
      setMsg(err instanceof SyntaxError ? 'That file is not valid JSON.' : err.message)
    }
  }

  const root = base?.replace(/\/+$/, '')
  return (
    <section className="saved">
      <header>
        <h3>
          Saved benchmarks <Info id="saved" />
        </h3>
        <label className="speed">
          <span className="lt">Replay speed <Info id="replay" /></span>
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {[1, 2, 4, 8].map((x) => (
              <option key={x} value={x}>
                {x}x
              </option>
            ))}
          </select>
        </label>
        <button className="mini" onClick={() => file.current?.click()}>
          Import recording
        </button>
        <input ref={file} type="file" accept="application/json,.json" hidden onChange={pick} />
        {runs?.length > 0 && (
          <a className="mini" href={`${root}/api/benchmarks/export`}>
            Download all (zip)
          </a>
        )}
      </header>
      {msg && <p className="err">{msg}</p>}
      {runs === null && <p className="hint">Start the control service to list saved runs here, or import a recording file.</p>}
      {runs?.length === 0 && <p className="hint">No completed runs saved yet. Every finished PC GPU or college GPU run appears here.</p>}
      {runs?.length > 0 && (
        <ul>
          {runs.map((r) => (
            <li key={r.id}>
              <div>
                <b>{r.config.model}</b> {r.config.levels?.length > 1 ? `${r.config.levels.join(', ')} users` : `${r.config.requests} users`}, {r.config.scenario}, SLA {(r.config.sla_ms / 1000).toFixed(1)} s{r.config.mock ? ', mock engine' : ''}
                <span>{when(r.created)}</span>
              </div>
              <button className="mini" disabled={busy} onClick={() => play(r.id)}>
                Replay
              </button>
              <a className="mini" href={`${root}/api/runs/${r.id}/recording`}>
                Recording
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
