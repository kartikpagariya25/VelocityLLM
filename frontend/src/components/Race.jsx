import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { MODELS, SCENARIOS, NOTES } from '../data/content'
import { pair, liveStats } from '../lib/metrics'
import { prefersReducedMotion, useReveal } from '../lib/motion'
import RaceCanvas, { maxInFlight } from './RaceCanvas'
import { Play, Pause, Replay } from './Icons'
import './Race.css'

const playSeconds = (T) => (T <= 3 ? 6 : T <= 8 ? 9 : 11)
const fmt = (v) => `${v.toFixed(2)} s`

function Stat({ label, value, tone }) {
  return (
    <div className={`stat ${tone ? `stat--${tone}` : ''}`}>
      <span className="mono">{label}</span>
      <b>{value}</b>
    </div>
  )
}

function Lane({ variant, title, sub, trace, t, T, scale }) {
  const s = liveStats(trace, t)
  return (
    <article className={`lane lane--${variant}`}>
      <header className="lane__head">
        <h3>{title}</h3>
        <p>{sub}</p>
      </header>
      <RaceCanvas trace={trace} t={t} T={T} scale={scale} variant={variant} />
      <div className="lane__stats">
        <Stat label="Finished" value={`${s.finished}/${trace.total}`} />
        <Stat label="Waiting" value={s.waiting} />
        <Stat label="Turned away" value={s.rejected} tone={s.rejected ? 'bad' : ''} />
        <Stat label="Slowest" value={s.slowest ? fmt(s.slowest) : '-'} />
      </div>
    </article>
  )
}

export default function Race() {
  const root = useReveal()
  const [model, setModel] = useState('llama')
  const [scenario, setScenario] = useState('mixed')
  const [progress, setProgress] = useState(0)
  const [playing, setPlaying] = useState(false)
  const lanes = useRef(null)
  const started = useRef(false)
  const still = useRef(prefersReducedMotion())

  const { static: st, dynamic: dy } = pair(model, scenario)
  const T = useMemo(() => Math.max(st.slowest, dy.slowest) * 1.05 + 0.05, [st, dy])
  const scale = useMemo(() => Math.max(maxInFlight(st, T), maxInFlight(dy, T)), [st, dy, T])
  const D = playSeconds(T)
  const t = progress * T
  const done = progress >= 1

  useEffect(() => {
    if (!playing) return undefined
    let raf
    let last = performance.now()
    const tick = (now) => {
      const dt = (now - last) / 1000
      last = now
      let reached = false
      setProgress((p) => {
        const n = p + dt / D
        if (n >= 1) reached = true
        return Math.min(1, n)
      })
      if (reached) setPlaying(false)
      else raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, D])

  useEffect(() => {
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting && !started.current) {
          started.current = true
          if (still.current) setProgress(1)
          else setPlaying(true)
        }
      },
      { threshold: 0.4 },
    )
    io.observe(lanes.current)
    return () => io.disconnect()
  }, [])

  const restart = useCallback(() => {
    setProgress(0)
    setPlaying(!still.current)
    if (still.current) setProgress(1)
  }, [])

  const choose = (setter) => (value) => {
    setter(value)
    restart()
  }

  const change = Math.round(((dy.p99 - st.p99) / st.p99) * 100)
  const note = NOTES[`${model}-${scenario}`]

  return (
    <section className="race section" id="race" ref={root}>
      <div className="wrap">
        <p className="kicker" data-reveal>The race</p>
        <h2 className="h2" data-lines>Same traffic. Two schedulers.</h2>
        <p className="lede" data-reveal>
          Real requests from our test runs on an RTX 5050, replayed side by side. Each bar is one request: pale is waiting, solid is running.
        </p>

        <div className="race__controls" data-reveal>
          <div className="seg" role="tablist" aria-label="Model">
            {MODELS.map((m) => (
              <button key={m.id} role="tab" aria-selected={model === m.id} className={model === m.id ? 'on' : ''} onClick={() => choose(setModel)(m.id)}>
                {m.name} <span className="mono">{m.size}</span>
              </button>
            ))}
          </div>
          <div className="seg" role="tablist" aria-label="Traffic pattern">
            {Object.entries(SCENARIOS).map(([id, s]) => (
              <button key={id} role="tab" aria-selected={scenario === id} className={scenario === id ? 'on' : ''} onClick={() => choose(setScenario)(id)}>
                {s.label}
              </button>
            ))}
          </div>
        </div>
        <p className="race__hint mono">{SCENARIOS[scenario].hint}</p>

        <div className="race__lanes" ref={lanes}>
          <Lane variant="static" title="Static scheduler" sub="Fixed batch, first come first served" trace={st} t={t} T={T} scale={scale} />
          <Lane variant="dynamic" title="VelocityLLM" sub="Batch resized live, overload turned away early" trace={dy} t={t} T={T} scale={scale} />
        </div>

        <div className="race__bar" data-reveal>
          <button className="race__play" onClick={() => (done ? restart() : setPlaying((p) => !p))} aria-label={done ? 'Replay' : playing ? 'Pause' : 'Play'}>
            {done ? <Replay /> : playing ? <Pause /> : <Play />}
          </button>
          <input
            type="range"
            min="0"
            max="1"
            step="0.001"
            value={progress}
            onChange={(e) => {
              setPlaying(false)
              setProgress(Number(e.target.value))
            }}
            aria-label="Replay position"
          />
          <span className="mono race__clock">{t.toFixed(1)} s</span>
        </div>

        <div className={`race__verdict ${done ? 'race__verdict--on' : ''}`} aria-live="polite">
          <div>
            <span className="mono">Slowest answer</span>
            <b>
              {fmt(st.slowest)} <i>to</i> <em>{fmt(dy.slowest)}</em>
            </b>
          </div>
          <div>
            <span className="mono">Tail latency p99</span>
            <b className={change < 0 ? 'good' : ''}>{change > 0 ? `+${change}` : change}%</b>
          </div>
          <div>
            <span className="mono">Turned away</span>
            <b>{dy.shed} of {dy.total}</b>
          </div>
        </div>
        {note && <p className="race__note">{note}</p>}
      </div>
    </section>
  )
}
