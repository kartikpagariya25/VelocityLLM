import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import { gsap, SplitText, useGSAP, ease, prefersReducedMotion, scrollToId } from '../lib/motion'
import { REPO } from '../data/content'
import { Arrow, Github } from './Icons'
import './Hero.css'

const Die = lazy(() => import('./Die'))

function hasWebGL() {
  try {
    const c = document.createElement('canvas')
    return Boolean(c.getContext('webgl2') || c.getContext('webgl'))
  } catch {
    return false
  }
}

export default function Hero() {
  const root = useRef(null)
  const stage = useRef(null)
  const [gl] = useState(hasWebGL)
  const [visible, setVisible] = useState(true)
  const still = prefersReducedMotion()

  useEffect(() => {
    const io = new IntersectionObserver(([e]) => setVisible(e.isIntersecting), { threshold: 0.05 })
    io.observe(root.current)
    return () => io.disconnect()
  }, [])

  useGSAP(
    () => {
      if (prefersReducedMotion()) return
      const title = SplitText.create('.hero__title', { type: 'chars', mask: 'chars' })
      const tl = gsap.timeline({ defaults: { ease: ease.expo } })
      tl.from(title.chars, { yPercent: 115, duration: 1.3, stagger: 0.035 }, 0.1)
        .from('.hero__kicker', { opacity: 0, y: 16, duration: 0.9 }, 0.2)
        .from('.hero__lede, .hero__cta', { opacity: 0, y: 24, duration: 1, stagger: 0.12 }, 0.7)
        .from('.hero__stage', { opacity: 0, scale: 0.92, duration: 1.8 }, 0.3)
        .from('.hero__foot', { opacity: 0, duration: 1 }, 1.4)
    },
    { scope: root },
  )

  return (
    <section className="hero" id="top" ref={root}>
      <div className="hero__stage" ref={stage} aria-hidden="true">
        <div className="hero__glow" />
        {gl && (
          <Suspense fallback={null}>
            <Die active={visible} still={still} />
          </Suspense>
        )}
      </div>
      <div className="wrap hero__content">
        <p className="kicker hero__kicker">SLA-aware dynamic batching · LLM inference serving</p>
        <h1 className="hero__title">
          Velocity<span>LLM</span>
        </h1>
        <p className="lede hero__lede">
          Static batching lets requests pile up until the GPU chokes. VelocityLLM watches the queue, the deadline and GPU memory, then resizes the batch on the fly.
        </p>
        <div className="hero__cta">
          <button className="btn btn--hot" onClick={() => scrollToId('race')}>
            Watch the race <Arrow width={18} height={18} />
          </button>
          <a className="btn btn--ghost" href={REPO} target="_blank" rel="noopener noreferrer">
            <Github width={18} height={18} /> Source
          </a>
        </div>
      </div>
      <p className="hero__foot mono">Scroll</p>
    </section>
  )
}
