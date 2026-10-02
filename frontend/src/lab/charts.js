const C = { ink: '#0a0a0a', mute: '#6f6f69', line: '#e7e7e2', stat: '#8c8c86', dyn: '#ff4d1c', dynSoft: 'rgba(255,77,28,.16)', statSoft: 'rgba(140,140,134,.18)', ok: '#17b06b', miss: '#f0a500', rej: '#e8384f' }

export function setup(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  const w = canvas.clientWidth
  const h = canvas.clientHeight
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
  }
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)
  return { ctx, w, h }
}

const axis = (ctx, w, h, pad, tMax, label) => {
  ctx.font = '10px JetBrains Mono, monospace'
  ctx.fillStyle = C.mute
  ctx.strokeStyle = C.line
  ctx.lineWidth = 1
  const step = tMax > 60 ? 20 : tMax > 30 ? 10 : 5
  for (let t = 0; t <= tMax; t += step) {
    const x = pad.l + (t / tMax) * (w - pad.l - pad.r)
    ctx.beginPath()
    ctx.moveTo(x, pad.t)
    ctx.lineTo(x, h - pad.b)
    ctx.stroke()
    ctx.fillText(`${t}s`, x - 8, h - 6)
  }
  if (label) ctx.fillText(label, pad.l, 10)
}

export const rowLayout = (n, h, pad) => {
  const rows = Math.max(n, 1)
  const rh = Math.max(2, Math.min(10, (h - pad.t - pad.b) / rows))
  return { rh }
}

export const TL_PAD = { l: 8, r: 8, t: 14, b: 18 }

export function drawTimeline(canvas, reqs, sla, tMax, hit) {
  const { ctx, w, h } = setup(canvas)
  const pad = TL_PAD
  axis(ctx, w, h, pad, tMax, 'request timeline')
  const ordered = [...reqs].sort((a, b) => a.arrival_s - b.arrival_s)
  const { rh } = rowLayout(ordered.length, h, pad)
  const X = (t) => pad.l + (t / tMax) * (w - pad.l - pad.r)
  hit.length = 0
  ordered.forEach((r, i) => {
    const y = pad.t + i * rh
    if (y + rh > h - pad.b) return
    const bh = Math.max(1.5, rh - 1)
    if (r.status === 'rejected') {
      ctx.fillStyle = C.rej
      ctx.fillRect(X(r.arrival_s), y, 3, bh)
    } else {
      ctx.fillStyle = r.end_s - r.arrival_s <= sla ? C.ok : C.miss
      ctx.fillRect(X(r.arrival_s), y, Math.max(2, X(r.end_s) - X(r.arrival_s)), bh)
      ctx.fillStyle = 'rgba(10,10,10,.5)'
      ctx.fillRect(X(r.arrival_s + sla), y, 1, bh)
    }
    hit.push({ y, h: rh, r })
  })
}

export function drawConcurrency(canvas, store) {
  const { ctx, w, h } = setup(canvas)
  const pad = { l: 30, r: 10, t: 14, b: 18 }
  const all = [...store.static.series, ...store.dynamic.series]
  const tMax = Math.max(10, Math.ceil(Math.max(0, ...all.map((p) => p.t)) / 5) * 5)
  const yMax = Math.max(16, ...all.map((p) => Math.max(p.limit, p.active))) + 1
  axis(ctx, w, h, pad, tMax, 'concurrency limit and requests in flight')
  const X = (t) => pad.l + (t / tMax) * (w - pad.l - pad.r)
  const Y = (v) => h - pad.b - (v / yMax) * (h - pad.t - pad.b)
  ctx.fillStyle = C.mute
  for (let v = 0; v <= yMax; v += 4) ctx.fillText(String(v), 6, Y(v) + 3)
  const area = (series, key, fill) => {
    if (series.length < 2) return
    ctx.beginPath()
    ctx.moveTo(X(series[0].t), Y(0))
    series.forEach((p) => ctx.lineTo(X(p.t), Y(p[key])))
    ctx.lineTo(X(series[series.length - 1].t), Y(0))
    ctx.fillStyle = fill
    ctx.fill()
  }
  const step = (series, color, dash, width) => {
    if (!series.length) return
    ctx.beginPath()
    series.forEach((p, i) => {
      if (i === 0) ctx.moveTo(X(p.t), Y(p.limit))
      else {
        ctx.lineTo(X(p.t), Y(series[i - 1].limit))
        ctx.lineTo(X(p.t), Y(p.limit))
      }
    })
    ctx.setLineDash(dash)
    ctx.strokeStyle = color
    ctx.lineWidth = width
    ctx.stroke()
    ctx.setLineDash([])
  }
  area(store.static.series, 'active', C.statSoft)
  area(store.dynamic.series, 'active', C.dynSoft)
  step(store.static.series, C.stat, [5, 4], 1.6)
  step(store.dynamic.series, C.dyn, [], 2.2)
  const last = store.dynamic.series[store.dynamic.series.length - 1]
  if (last) {
    ctx.fillStyle = C.dyn
    ctx.beginPath()
    ctx.arc(X(last.t), Y(last.limit), 4, 0, 7)
    ctx.fill()
  }
}
