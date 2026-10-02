import { useEffect, useRef, useState } from 'react'
import { INFO } from './infoText'

export default function Info({ id }) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState(null)
  const btn = useRef(null)
  const pinned = useRef(false)
  const shut = () => {
    pinned.current = false
    setOpen(false)
  }
  const entry = INFO[id]

  const place = () => {
    const r = btn.current?.getBoundingClientRect()
    if (!r) return
    const width = Math.min(300, window.innerWidth - 24)
    const left = Math.min(Math.max(12, r.left + r.width / 2 - width / 2), window.innerWidth - width - 12)
    const below = r.bottom + 10
    const flip = below + 180 > window.innerHeight && r.top > 200
    setPos({ left, width, top: flip ? undefined : below, bottom: flip ? window.innerHeight - r.top + 10 : undefined })
  }

  useEffect(() => {
    if (!open) return
    place()
    const close = shut
    const key = (e) => e.key === 'Escape' && shut()
    const out = (e) => !btn.current?.parentElement.contains(e.target) && shut()
    window.addEventListener('scroll', close, true)
    window.addEventListener('resize', close)
    window.addEventListener('keydown', key)
    document.addEventListener('pointerdown', out)
    return () => {
      window.removeEventListener('scroll', close, true)
      window.removeEventListener('resize', close)
      window.removeEventListener('keydown', key)
      document.removeEventListener('pointerdown', out)
    }
  }, [open])

  if (!entry) return null
  return (
    <span className="info" onMouseEnter={() => setOpen(true)} onMouseLeave={() => !pinned.current && setOpen(false)}>
      <button type="button" ref={btn} className="info__btn" aria-label={`About ${entry[0]}`} aria-expanded={open} onClick={(e) => (e.preventDefault(), pinned.current ? shut() : ((pinned.current = true), setOpen(true)))}>
        i
      </button>
      {open && pos && (
        <span className="info__pop" role="tooltip" style={pos}>
          <b>{entry[0]}</b>
          {entry[1]}
        </span>
      )}
    </span>
  )
}
