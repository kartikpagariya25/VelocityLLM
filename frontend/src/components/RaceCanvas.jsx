import { useEffect, useMemo, useRef, useState } from 'react'
import { inFlight } from '../lib/metrics'

const STEPS = 160
const PALETTE = {
  static: { queue: '#dde1e6', run: '#8793a3', area: 'rgba(135,147,163,0.28)', line: '#6f7c8d' },
  dynamic: { queue: '#ffd9bd', run: '#e85d00', area: 'rgba(232,93,0,0.25)', line: '#d95200' },
}

export function maxInFlight(trace, T) {
  let m = 1
  for (let k = 0; k <= STEPS; k += 1) m = Math.max(m, inFlight(trace, (k / STEPS) * T))
  return m
}

export default function RaceCanvas({ trace, t, T, scale, variant, reference = 3 }) {
  const wrap = useRef(null)
  const cv = useRef(null)
  const [w, setW] = useState(560)
  const h = w < 520 ? 250 : 310

  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.floor(e.contentRect.width))))
    ro.observe(wrap.current)
    return () => ro.disconnect()
  }, [])

  const series = useMemo(
    () => Array.from({ length: STEPS + 1 }, (_, k) => inFlight(trace, (k / STEPS) * T)),
    [trace, T],
  )

  useEffect(() => {
    const c = cv.current
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = w * dpr
    c.height = h * dpr
    const ctx = c.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

    const pal = PALETTE[variant]
    const padX = 6
    const top = 6
    const chartH = 58
    const axis = 18
    const barsH = h - top - chartH - axis - 26
    const x = (s) => padX + (Math.min(s, T) / T) * (w - padX * 2)
    const reqs = trace.requests
    const rowH = barsH / reqs.length
    const barH = Math.max(2, rowH * 0.62)

    reqs.forEach((r, i) => {
      const y = top + i * rowH + (rowH - barH) / 2
      if (r.shed) {
        if (t >= r.end) {
          ctx.fillStyle = '#d92d20'
          ctx.fillRect(x(0), y, Math.max(5, x(r.end) - x(0)), barH)
        }
        return
      }
      const q = Math.min(t, r.start)
      if (q > 0) {
        ctx.fillStyle = pal.queue
        ctx.fillRect(x(0), y, x(q) - x(0), barH)
      }
      if (t > r.start) {
        ctx.fillStyle = pal.run
        ctx.fillRect(x(r.start), y, Math.max(1, x(Math.min(t, r.end)) - x(r.start)), barH)
      }
    })

    ctx.font = '10px "JetBrains Mono", monospace'
    if (reference < T) {
      ctx.strokeStyle = 'rgba(28,23,18,0.4)'
      ctx.setLineDash([3, 4])
      ctx.beginPath()
      ctx.moveTo(x(reference), top)
      ctx.lineTo(x(reference), top + barsH)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = 'rgba(28,23,18,0.65)'
      ctx.fillText(`${reference} s`, x(reference) + 4, top + 10)
    }

    ctx.strokeStyle = 'rgba(28,23,18,0.45)'
    ctx.beginPath()
    ctx.moveTo(x(t), top)
    ctx.lineTo(x(t), top + barsH + 10)
    ctx.stroke()

    const base = top + barsH + 26 + chartH
    ctx.fillStyle = 'rgba(28,23,18,0.55)'
    ctx.fillText('Requests on the GPU', padX, base - chartH - 6)
    ctx.beginPath()
    ctx.moveTo(x(0), base)
    let last = 0
    for (let k = 0; k <= STEPS; k += 1) {
      const tau = (k / STEPS) * T
      if (tau > t) break
      last = k
      ctx.lineTo(x(tau), base - (series[k] / scale) * chartH)
    }
    ctx.lineTo(x((last / STEPS) * T), base)
    ctx.closePath()
    ctx.fillStyle = pal.area
    ctx.fill()
    ctx.strokeStyle = pal.line
    ctx.lineWidth = 1.5
    ctx.stroke()
    ctx.lineWidth = 1

    ctx.fillStyle = 'rgba(28,23,18,0.5)'
    ctx.fillText('0 s', padX, h - 4)
    const end = `${T.toFixed(T < 10 ? 1 : 0)} s`
    ctx.fillText(end, w - padX - ctx.measureText(end).width, h - 4)
  }, [trace, t, T, scale, variant, reference, w, h, series])

  return (
    <div ref={wrap} className="rc">
      <canvas ref={cv} style={{ width: '100%', height: h }} role="img" aria-label={`${variant} scheduler request timeline`} />
    </div>
  )
}
