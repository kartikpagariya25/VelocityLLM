import { useRef, useState } from 'react'
import { gsap, ScrollTrigger, useGSAP, prefersReducedMotion } from '../lib/motion'
import { STAGES } from '../data/content'
import { STAGE_ICONS } from './Icons'
import './HowItWorks.css'

const smooth = (v) => {
  const c = Math.min(1, Math.max(0, v))
  return c * c * (3 - 2 * c)
}

function Sawtooth({ refEl }) {
  return (
    <svg className="saw" viewBox="0 0 300 90" role="img" aria-label="Illustration: batch limit climbs one step at a time and drops by a quarter on a breach">
      <path
        ref={refEl}
        d="M4 70 L34 60 L64 50 L94 40 L124 30 L124 52 L154 42 L184 32 L214 22 L214 44 L244 34 L274 24"
        pathLength="1"
        fill="none"
        stroke="#ff4d1c"
        strokeWidth="2.5"
        strokeLinejoin="round"
        strokeDasharray="1"
        strokeDashoffset="0"
      />
      <text x="4" y="86" className="saw__t">concurrency limit over time (illustration)</text>
    </svg>
  )
}

export default function HowItWorks() {
  const root = useRef(null)
  const packet = useRef(null)
  const saw = useRef(null)
  const [active, setActive] = useState(0)
  const still = useRef(prefersReducedMotion())

  useGSAP(
    () => {
      if (still.current) return
      const rail = packet.current.parentNode
      const setX = gsap.quickTo(packet.current, 'x', { duration: 0.5, ease: 'power3.out' })
      gsap.set(packet.current, { x: (0.5 / STAGES.length) * rail.offsetWidth })
      const n = STAGES.length
      let current = -1
      ScrollTrigger.create({
        trigger: root.current,
        start: 'top top',
        end: 'bottom bottom',
        onUpdate: (self) => {
          const f = self.progress * n
          const i = Math.min(n - 1, Math.floor(f))
          const u = f - i
          const move = i < n - 1 ? smooth((u - 0.6) / 0.4) : 0
          setX(((i + 0.5 + move) / n) * rail.offsetWidth)
          if (i !== current) {
            current = i
            setActive(i)
          }
          if (saw.current) saw.current.style.strokeDashoffset = i === 3 ? 1 - smooth(u * 1.4) : i > 3 ? 0 : 1
        },
      })
    },
    { scope: root },
  )

  const stage = STAGES[active]

  return (
    <section className={`engine ${still.current ? 'engine--still' : ''}`} id="engine" ref={root}>
      <div className="engine__sticky wrap">
        <div className="engine__head">
          <p className="kicker">Under the hood</p>
          <h2 className="h2">One request, six decisions.</h2>
        </div>

        <div className="rail" aria-hidden="true">
          <div className="rail__line" />
          <div className="rail__packet" ref={packet} />
          {STAGES.map((s, i) => {
            const Icon = STAGE_ICONS[s.id]
            return (
              <div className={`node ${i === active ? 'node--on' : ''} ${i < active ? 'node--done' : ''}`} key={s.id}>
                <span className="node__icon">
                  <Icon width={26} height={26} />
                </span>
                <span className="node__name">{s.name}</span>
              </div>
            )
          })}
        </div>

        {still.current ? (
          STAGES.map((s) => (
            <div className="panel" key={s.id}>
              <h3>{s.name}</h3>
              <p className="panel__line">{s.line}</p>
              <ul className="panel__chips">
                {s.chips.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            </div>
          ))
        ) : (
        <div className="panel" aria-live="polite">
          <p className="panel__step mono">
            {String(active + 1).padStart(2, '0')} / {String(STAGES.length).padStart(2, '0')}
          </p>
          <h3 key={stage.id}>{stage.name}</h3>
          <p className="panel__line">{stage.line}</p>
          <ul className="panel__chips">
            {stage.chips.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
          {stage.id === 'controller' && <Sawtooth refEl={saw} />}
          {stage.id === 'gpu' && (
            <p className="panel__flag mono">
              Same server, one flag: <b>--policy static</b> or <b>--policy dynamic</b>. That is how the race was run.
            </p>
          )}
        </div>
        )}
      </div>
    </section>
  )
}
