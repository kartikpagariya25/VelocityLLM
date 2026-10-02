import { useEffect, useRef, useState } from 'react'
import Info from './Info'
import { drawSweep } from './charts'
import { sweepSentence, f2, pctStr, change, verdictOf } from './store'

export const PRESETS = [
  { id: 'gentle', label: 'Gentle', levels: [10, 20, 40] },
  { id: 'heavy', label: 'Heavy', levels: [30, 60, 100, 150] },
  { id: 'extreme', label: 'Extreme', levels: [50, 100, 150, 200] },
]

export function parseLevels(text) {
  const parts = text.split(',').map((x) => x.trim()).filter(Boolean)
  if (!parts.length) throw new Error('Enter at least two load levels, for example 30, 60, 100.')
  const nums = parts.map(Number)
  if (nums.some((n) => !Number.isInteger(n) || n < 1 || n > 200)) throw new Error('Each level must be a whole number from 1 to 200.')
  const uniq = [...new Set(nums)].sort((a, b) => a - b)
  if (uniq.length < 2) throw new Error('Use at least two different levels.')
  if (uniq.length > 8) throw new Error('Use at most 8 levels.')
  return uniq
}

function Chart({ levels, data, k, opts }) {
  const cv = useRef(null)
  useEffect(() => {
    if (cv.current) drawSweep(cv.current, levels, data, k, opts)
  })
  return <canvas ref={cv} className="sweep__cv" />
}

export default function Pressure({ store, running, disabled, onRun, onJudgePreset }) {
  const [text, setText] = useState('30, 60, 100, 150')
  const [err, setErr] = useState('')
  const levels = store?.cfg.levels?.length > 1 ? store.cfg.levels : null
  const data = store?.levels
  const have = levels && levels.some((n) => data.static[n] || data.dynamic[n])

  const go = () => {
    try {
      setErr('')
      onRun(parseLevels(text))
    } catch (e) {
      setErr(e.message)
    }
  }

  return (
    <section className="pressure">
      <header>
        <h3>
          Pressure test <Info id="pressure" />
        </h3>
        <span className="sub">Same traffic at growing load, both schedulers. See where Static breaks its promise.</span>
      </header>
      <div className="pressure__row">
        <label>
          <span className="lt">Load levels (users) <Info id="levels" /></span>
          <input value={text} onChange={(e) => setText(e.target.value)} aria-label="Load levels" />
        </label>
        <div className="chips2">
          {PRESETS.map((p) => (
            <button key={p.id} className="mini" onClick={() => setText(p.levels.join(', '))}>
              {p.label}
            </button>
          ))}
          <button className="mini accent" onClick={() => onJudgePreset((lv) => setText(lv.join(', ')))} title="Mixed traffic, 5 s SLA, 30 to 150 users">
            Judge demo setup
          </button>
        </div>
        <button className="go" onClick={go} disabled={running || disabled}>
          Run pressure test
        </button>
      </div>
      {err && <p className="err">{err}</p>}
      {have && (
        <>
          <div className="sweep">
            <div>
              <h4>
                p99 latency of answered requests <Info id="sweepP99" />
              </h4>
              <Chart levels={levels} data={data} k="p99_s" opts={{ title: 'seconds', sla: store.cfg.sla, fmt: (v) => v.toFixed(1) }} />
            </div>
            <div>
              <h4>
                Answered within the SLA <Info id="sweepSla" />
              </h4>
              <Chart levels={levels} data={data} k="within_sla_served" opts={{ title: '% of answered', scale: 100, fixedMax: 110, fmt: (v) => `${Math.round(v)}%` }} />
            </div>
          </div>
          <p className="legend">
            <i className="lg lg--s" />
            Static
            <i className="lg lg--d" />
            Dynamic
          </p>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Users</th>
                  <th>Static p99</th>
                  <th>Dynamic p99</th>
                  <th>Change</th>
                  <th>Static on time</th>
                  <th>Dynamic on time</th>
                  <th>Dynamic rejected</th>
                </tr>
              </thead>
              <tbody>
                {levels.map((n) => {
                  const s = data.static[n]
                  const d = data.dynamic[n]
                  const v = s && d ? change(s.p99_s, d.p99_s) : null
                  const tone = v == null ? '' : { better: 'good', worse: 'bad', same: '' }[verdictOf(v, true)]
                  return (
                    <tr key={n}>
                      <td>{n}</td>
                      <td>{s ? `${f2(s.p99_s)} s` : '...'}</td>
                      <td>{d ? `${f2(d.p99_s)} s` : '...'}</td>
                      <td className={tone}>{v == null ? '' : `${v > 0 ? '+' : ''}${v.toFixed(0)}%`}</td>
                      <td>{s ? pctStr(s.within_sla_served) : '...'}</td>
                      <td>{d ? pctStr(d.within_sla_served) : '...'}</td>
                      <td>{d ? `${d.rejected} of ${d.offered}` : '...'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {store.status !== 'running' && <p className="verdict">{sweepSentence(store.cfg, levels, data)}</p>}
        </>
      )}
    </section>
  )
}
