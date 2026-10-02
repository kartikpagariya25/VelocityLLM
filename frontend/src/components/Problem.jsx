import { useRef } from 'react'
import { gsap, useGSAP, prefersReducedMotion } from '../lib/motion'
import './Problem.css'

const BEATS = [
  ['Traffic is never steady.', 'Real users arrive in waves. A batch size picked in advance is right for one moment and wrong for the next.'],
  ['A fixed batch cannot bend.', 'Static batching serves a set number at a time and lets the rest queue. Latency climbs and deadlines slip.'],
  ['Then the GPU runs out of room.', 'Nothing tells the scheduler to stop. Memory fills, requests fail and the whole server goes down with them.'],
]

const DOTS = Array.from({ length: 28 }, (_, i) => i)
const SLOTS = Array.from({ length: 8 }, (_, i) => i)

export default function Problem() {
  const root = useRef(null)
  const still = useRef(prefersReducedMotion())

  useGSAP(
    () => {
      if (still.current) return
      const count = { v: 0 }
      const label = root.current.querySelector('.ps__count')
      const tl = gsap.timeline({
        defaults: { ease: 'none' },
        scrollTrigger: { trigger: root.current, start: 'top top', end: 'bottom bottom', scrub: 0.6 },
      })
      gsap.set('.beat', { opacity: 0, y: 24 })
      gsap.set('.beat:first-child', { opacity: 1, y: 0 })
      gsap.set('.ps__dot', { opacity: 0, scale: 0.4 })
      gsap.set('.ps__slot', { opacity: 0.25 })
      tl.to('.ps__slot', { opacity: 1, stagger: 0.1, duration: 0.6 }, 0)
        .to('.ps__dot:nth-child(-n+5)', { opacity: 1, scale: 1, stagger: 0.12, duration: 0.3 }, 0.3)
        .to('.ps__fill', { scaleX: 0.42, duration: 1 }, 0)
        .to('.beat:nth-child(1)', { opacity: 0, y: -24, duration: 0.3 }, 0.95)
        .to('.beat:nth-child(2)', { opacity: 1, y: 0, duration: 0.4 }, 1.05)
        .to('.ps__dot:nth-child(n+6)', { opacity: 1, scale: 1, stagger: 0.03, duration: 0.25 }, 1.05)
        .to(count, { v: DOTS.length, duration: 0.9, onUpdate: () => (label.textContent = Math.round(count.v)) }, 1.05)
        .to('.ps__fill', { scaleX: 0.78, duration: 1 }, 1)
        .to('.beat:nth-child(2)', { opacity: 0, y: -24, duration: 0.3 }, 1.95)
        .to('.beat:nth-child(3)', { opacity: 1, y: 0, duration: 0.4 }, 2.05)
        .to('.ps__fill', { scaleX: 1, backgroundColor: '#e8384f', duration: 0.6 }, 2)
        .to('.ps__dot', { backgroundColor: '#e8384f', duration: 0.4 }, 2.1)
        .to('.ps__stamp', { opacity: 1, scale: 1, duration: 0.3, ease: 'power3.out' }, 2.45)
        .to('.ps', { x: 3, duration: 0.04, repeat: 5, yoyo: true }, 2.5)
        .to({}, { duration: 0.4 }, 2.6)
    },
    { scope: root },
  )

  return (
    <section className={`problem ${still.current ? 'problem--still' : ''}`} id="problem" ref={root}>
      <div className="problem__sticky wrap">
        <div className="problem__copy">
          <p className="kicker">The problem</p>
          <div className="problem__beats">
            {BEATS.map(([title, text]) => (
              <div className="beat" key={title}>
                <h2 className="h2">{title}</h2>
                <p className="lede">{text}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="ps" role="img" aria-label="Illustration: a static scheduler with eight slots, a growing queue and a full memory bar">
          <div className="ps__head">
            <span>Static scheduler</span>
            <span className="mono">batch size 8</span>
          </div>
          <div className="ps__slots">
            {SLOTS.map((s) => (
              <i className="ps__slot" key={s} />
            ))}
          </div>
          <div className="ps__row">
            <span>Waiting</span>
            <b className="ps__count mono">{still.current ? DOTS.length : 0}</b>
          </div>
          <div className="ps__queue">
            {DOTS.map((d) => (
              <i className="ps__dot" key={d} />
            ))}
          </div>
          <div className="ps__row">
            <span>GPU memory</span>
          </div>
          <div className="ps__meter">
            <div className="ps__fill" />
          </div>
          <div className="ps__stamp mono">Out of memory</div>
          <p className="ps__note mono">Illustration</p>
        </div>
      </div>
    </section>
  )
}
