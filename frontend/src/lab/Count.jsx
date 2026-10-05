import { useEffect, useRef, useState } from 'react'

export default function Count({ value, fmt = (v) => String(v) }) {
  const [shown, setShown] = useState(value)
  const from = useRef(value)
  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return setShown(value)
    const start = performance.now()
    const a = from.current
    let id
    const tick = (now) => {
      const k = Math.min(1, (now - start) / 700)
      setShown(a + (value - a) * (1 - Math.pow(1 - k, 3)))
      if (k < 1) id = requestAnimationFrame(tick)
      else from.current = value
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [value])
  return <>{fmt(shown)}</>
}
