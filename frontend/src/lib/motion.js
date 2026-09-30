import { useRef } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { SplitText } from 'gsap/SplitText'
import { useGSAP } from '@gsap/react'
import Lenis from 'lenis'

gsap.registerPlugin(ScrollTrigger, SplitText, useGSAP)

export const ease = { out: 'power3.out', inOut: 'power2.inOut', expo: 'expo.out' }
export const dist = { reveal: 28, stagger: 0.08, duration: 0.9 }

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

let lenis = null

export function startSmoothScroll() {
  if (prefersReducedMotion() || lenis) return () => {}
  lenis = new Lenis({ lerp: 0.1, wheelMultiplier: 1 })
  lenis.on('scroll', ScrollTrigger.update)
  const tick = (time) => lenis.raf(time * 1000)
  gsap.ticker.add(tick)
  gsap.ticker.lagSmoothing(0)
  return () => {
    gsap.ticker.remove(tick)
    lenis.destroy()
    lenis = null
  }
}

export function scrollToId(id) {
  const el = document.getElementById(id)
  if (!el) return
  if (lenis) lenis.scrollTo(el, { offset: 0, duration: 1.4 })
  else el.scrollIntoView({ behavior: 'auto' })
}

export function useReveal() {
  const scope = useRef(null)
  useGSAP(
    () => {
      if (prefersReducedMotion()) return
      const items = gsap.utils.toArray('[data-reveal]', scope.current)
      gsap.set(items, { opacity: 0, y: dist.reveal })
      ScrollTrigger.batch(items, {
        start: 'top 90%',
        once: true,
        onEnter: (batch) =>
          gsap.to(batch, { opacity: 1, y: 0, duration: dist.duration, ease: ease.out, stagger: dist.stagger }),
      })
      gsap.utils.toArray('[data-lines]', scope.current).forEach((el) => {
        SplitText.create(el, {
          type: 'lines',
          mask: 'lines',
          autoSplit: true,
          onSplit: (self) =>
            gsap.from(self.lines, {
              yPercent: 110,
              duration: 1.1,
              ease: ease.expo,
              stagger: 0.09,
              scrollTrigger: { trigger: el, start: 'top 88%', once: true },
            }),
        })
      })
    },
    { scope },
  )
  return scope
}

export { gsap, ScrollTrigger, SplitText, useGSAP }
