import { useEffect, useRef, useState } from 'react'
import { MODELS, SCENARIOS } from '../data/content'
import { pair, delta, summary } from '../lib/metrics'
import { useReveal } from '../lib/motion'
import './Benchmarks.css'

const MARK = { 'qwen-flood': '†', 'tinyllama-flood': '‡', 'tinyllama-mixed': '‡' }
const stats = summary()

function Row({ model, scenario, shown }) {
  const { static: s, dynamic: d } = pair(model.id, scenario)
  const top = Math.max(s.p99, d.p99)
  const change = delta(model.id, scenario)
  const mark = MARK[`${model.id}-${scenario}`]
  return (
    <div className="row" data-reveal>
      <div className="row__name">
        <b>
          {model.name}
          {mark && <sup>{mark}</sup>}
        </b>
        <span className="mono">{model.size}</span>
      </div>
      <div className="row__bars">
        <div className="bar">
          <i className="bar__fill bar__fill--static" style={{ width: shown ? `${(s.p99 / top) * 100}%` : 0 }} />
          <span className="mono">{s.p99.toFixed(2)} s</span>
        </div>
        <div className="bar">
          <i className="bar__fill bar__fill--dynamic" style={{ width: shown ? `${(d.p99 / top) * 100}%` : 0 }} />
          <span className="mono">{d.p99.toFixed(2)} s</span>
        </div>
      </div>
      <div className="row__meta">
        <b className={change < 0 ? 'good' : ''}>{change > 0 ? `+${change}` : change}%</b>
        <span className="mono">{d.shed} turned away</span>
      </div>
    </div>
  )
}

export default function Benchmarks() {
  const root = useReveal()
  const [scenario, setScenario] = useState('flood')
  const [shown, setShown] = useState(false)
  const table = useRef(null)

  useEffect(() => {
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setShown(true)
          io.disconnect()
        }
      },
      { threshold: 0.05 },
    )
    io.observe(table.current)
    return () => io.disconnect()
  }, [])

  return (
    <section className="results section" id="results" ref={root}>
      <div className="wrap">
        <p className="kicker" data-reveal>Results</p>
        <h2 className="h2" data-lines>Same load. Shorter tail.</h2>

        <div className="big" data-reveal>
          <div>
            <b>
              {stats.better}
              <small> of {stats.runs}</small>
            </b>
            <span>runs finished with a lower tail latency</span>
          </div>
          <div>
            <b>
              {stats.shedMedian.toFixed(2)}
              <small> s</small>
            </b>
            <span>median time to turn a request away, instead of making it wait</span>
          </div>
          <div>
            <b>4</b>
            <span>models, one scheduler, no code changes</span>
          </div>
        </div>

        <div className="results__top" data-reveal>
          <div className="seg" role="tablist" aria-label="Traffic pattern">
            {Object.entries(SCENARIOS).map(([id, s]) => (
              <button key={id} role="tab" aria-selected={scenario === id} className={scenario === id ? 'on' : ''} onClick={() => setScenario(id)}>
                {s.label}
              </button>
            ))}
          </div>
          <p className="legend mono">
            <i className="dot dot--static" /> Static <i className="dot dot--dynamic" /> VelocityLLM · p99 latency, shorter is better
          </p>
        </div>

        <div className="table" ref={table}>
          {MODELS.map((m) => (
            <Row key={`${m.id}-${scenario}`} model={m} scenario={scenario} shown={shown} />
          ))}
        </div>

        <ul className="foot">
          <li>Latency is measured on requests that were served. Turned-away requests are counted separately and shown on every row.</li>
          <li>
            <sup>{'†'}</sup> The static run stalled on its first wave, partly engine warm-up, so this gap is a best case.
          </li>
          <li>
            <sup>{'‡'}</sup> Most replies ended after zero tokens, so both schedulers finish almost instantly.
          </li>
        </ul>
      </div>
    </section>
  )
}
