import { useEffect, useRef, useState } from 'react'
import { useReveal } from '../lib/motion'
import { SWEEP_META, SWEEP_METRICS, SWEEP_RATES, SWEEP_SERIES } from '../data/sweep'
import './Sweep.css'

const W = 640
const H = 340
const PAD = { l: 52, r: 20, t: 20, b: 40 }
const LOG_TICKS = [2, 5, 10, 20, 50, 100, 200]

const niceMax = (v) => {
  const step = 10 ** Math.floor(Math.log10(v))
  return Math.ceil(v / step) * step
}

function scales(metric) {
  const values = SWEEP_SERIES.flatMap((s) => s.rows.map((r) => r[metric.id]))
  const xs = (i) => PAD.l + (i / (SWEEP_RATES.length - 1)) * (W - PAD.l - PAD.r)
  if (metric.log) {
    const lo = Math.log10(2)
    const hi = Math.log10(Math.max(...values, SWEEP_META.slaSeconds) * 1.15)
    const ys = (v) => H - PAD.b - ((Math.log10(Math.max(v, 2)) - lo) / (hi - lo)) * (H - PAD.t - PAD.b)
    return { xs, ys, ticks: LOG_TICKS.filter((t) => Math.log10(t) <= hi) }
  }
  const top = niceMax(Math.max(...values))
  const ys = (v) => H - PAD.b - (v / top) * (H - PAD.t - PAD.b)
  return { xs, ys, ticks: [0, top / 4, top / 2, (top * 3) / 4, top] }
}

const fmt = (v, metric) => (metric.id === 'p99' ? v.toFixed(2) : Math.round(v).toLocaleString('en-US'))

export default function Sweep() {
  const root = useReveal()
  const chart = useRef(null)
  const [metricId, setMetricId] = useState('p99')
  const [rateIndex, setRateIndex] = useState(5)
  const [drawn, setDrawn] = useState(false)
  const metric = SWEEP_METRICS.find((m) => m.id === metricId)
  const { xs, ys, ticks } = scales(metric)

  useEffect(() => {
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setDrawn(true)
          io.disconnect()
        }
      },
      { threshold: 0.25 },
    )
    io.observe(chart.current)
    return () => io.disconnect()
  }, [])

  const path = (s) => s.rows.map((r, i) => `${i ? 'L' : 'M'}${xs(i).toFixed(1)} ${ys(r[metric.id]).toFixed(1)}`).join(' ')
  const rate = SWEEP_RATES[rateIndex]

  return (
    <section className="sweep section" id="sweep" ref={root}>
      <div className="wrap">
        <h2 className="h2" data-lines>Same GPU, same traffic, flatter tail.</h2>
        <p className="lede" data-reveal>
          A request-rate sweep on a laptop GPU with {SWEEP_META.model}. Latency is the p99 of served requests against an {SWEEP_META.slaSeconds} second SLA.
        </p>

        <div className="sweep__grid" data-reveal>
          <div className="sweep__chart" ref={chart}>
            <div className="seg" role="tablist" aria-label="Metric">
              {SWEEP_METRICS.map((m) => (
                <button key={m.id} role="tab" aria-selected={metricId === m.id} className={metricId === m.id ? 'on' : ''} onClick={() => setMetricId(m.id)}>
                  {m.label}
                </button>
              ))}
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${metric.label} against offered request rate for three schedulers. Values are listed in the table beside the chart.`}>
              {ticks.map((t) => (
                <g key={t}>
                  <line x1={PAD.l} x2={W - PAD.r} y1={ys(t)} y2={ys(t)} className="sweep__grid-line" />
                  <text x={PAD.l - 10} y={ys(t) + 4} textAnchor="end" className="sweep__tick">
                    {fmt(t, metric)}
                  </text>
                </g>
              ))}
              {SWEEP_RATES.map((r, i) => (
                <text key={r} x={xs(i)} y={H - 14} textAnchor="middle" className="sweep__tick">
                  {r}
                </text>
              ))}
              <text x={W / 2} y={H - 0} textAnchor="middle" className="sweep__axis">
                offered requests per second
              </text>
              {metric.id === 'p99' && (
                <g>
                  <line x1={PAD.l} x2={W - PAD.r} y1={ys(SWEEP_META.slaSeconds)} y2={ys(SWEEP_META.slaSeconds)} className="sweep__sla" />
                  <text x={W - PAD.r} y={ys(SWEEP_META.slaSeconds) - 6} textAnchor="end" className="sweep__tick">
                    {SWEEP_META.slaSeconds} s SLA
                  </text>
                </g>
              )}
              <line x1={xs(rateIndex)} x2={xs(rateIndex)} y1={PAD.t} y2={H - PAD.b} className="sweep__marker" />
              {SWEEP_SERIES.map((s) => (
                <g key={s.id} className={`sweep__series sweep__series--${s.id}`}>
                  <path d={path(s)} pathLength="1" className={drawn ? 'is-drawn' : ''} />
                  <circle cx={xs(rateIndex)} cy={ys(s.rows[rateIndex][metric.id])} r="5" />
                </g>
              ))}
            </svg>
            <p className="sweep__hint mono">{metric.better}</p>
          </div>

          <div className="sweep__read">
            <div className="sweep__rates" role="group" aria-label="Offered rate">
              {SWEEP_RATES.map((r, i) => (
                <button key={r} className={i === rateIndex ? 'on' : ''} aria-pressed={i === rateIndex} onClick={() => setRateIndex(i)}>
                  {r}
                </button>
              ))}
              <span className="mono">requests per second</span>
            </div>
            <dl className="sweep__table" aria-live="polite">
              {SWEEP_SERIES.map((s) => {
                const row = s.rows[rateIndex]
                return (
                  <div key={s.id} className={`sweep__row sweep__row--${s.id} ${row.p99 > SWEEP_META.slaSeconds ? 'is-late' : ''}`}>
                    <dt>
                      <b>{s.name}</b>
                      <span>{s.note}</span>
                    </dt>
                    <dd>
                      <b>{fmt(row[metric.id], metric)}</b>
                      <span className="mono">{metric.unit}</span>
                      <em className="mono">{row.ok}% on time</em>
                    </dd>
                  </div>
                )
              })}
            </dl>
          </div>
        </div>

        <ul className="foot" data-reveal>
          <li>
            {SWEEP_META.gpu}, {SWEEP_META.secondsPerLevel} seconds of Poisson traffic per level, one run each, seed {SWEEP_META.seed}. Nothing was turned away in any run.
          </li>
          <li>The static baseline is capped at 8 requests in flight, its batch size, while VelocityLLM may grow to 32. A cap-32 static baseline and a vLLM server run are next.</li>
        </ul>
      </div>
    </section>
  )
}
