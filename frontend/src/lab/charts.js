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

const calm = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
const ease = (x) => 1 - Math.pow(1 - Math.min(1, Math.max(0, x)), 3)
const guard = (fn) => (...args) => {
  try {
    return fn(...args)
  } catch (e) {
    console.warn('chart skipped:', e)
  }
}
const later = (canvas, fn) => {
  if (canvas.__raf) return
  canvas.__raf = requestAnimationFrame(() => {
    canvas.__raf = null
    fn()
  })
}

const axis = (ctx, w, h, pad, tMax, label) => {
  ctx.font = '10px JetBrains Mono, monospace'
  ctx.fillStyle = C.mute
  ctx.strokeStyle = C.line
  ctx.lineWidth = 1
  const step = tMax > 60 ? 20 : tMax > 30 ? 10 : tMax > 12 ? 5 : tMax > 6 ? 2 : 1
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

const niceTop = (v) => {
  const pow = Math.pow(10, Math.floor(Math.log10(v)))
  return [1, 2, 2.5, 5, 10].map((m) => m * pow).find((x) => x >= v) || v
}

export const niceMax = (t) => Math.max(4, Math.ceil(t * 1.04))

export const rowLayout = (n, h, pad) => {
  const rows = Math.max(n, 1)
  const rh = Math.max(1.4, Math.min(10, (h - pad.t - pad.b) / rows))
  return { rh }
}

export const TL_PAD = { l: 8, r: 8, t: 14, b: 18 }

export const drawTimeline = guard(function drawTimeline(canvas, reqs, sla, tMax, hit) {
  const { ctx, w, h } = setup(canvas)
  const pad = TL_PAD
  axis(ctx, w, h, pad, tMax, '')
  const born = (canvas.__born ||= new Map())
  if (reqs.length < (canvas.__n || 0)) born.clear()
  canvas.__n = reqs.length
  const now = performance.now()
  const still = !calm()
  let moving = false
  const ordered = [...reqs].sort((a, b) => a.end_s - b.end_s || a.arrival_s - b.arrival_s)
  const { rh } = rowLayout(ordered.length, h, pad)
  const X = (t) => pad.l + (t / tMax) * (w - pad.l - pad.r)
  const flat = ordered.every((r) => r.arrival_s < 0.05)
  if (flat) {
    ctx.strokeStyle = 'rgba(10,10,10,.55)'
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(X(sla), pad.t)
    ctx.lineTo(X(sla), h - pad.b)
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = C.ink
    ctx.fillText(`SLA ${sla}s`, Math.min(X(sla) + 4, w - 60), pad.t + 8)
  }
  hit.length = 0
  ordered.forEach((r, i) => {
    const y = pad.t + i * rh
    if (y + rh > h - pad.b + 1) return
    const key = r.request_id ?? `${r.index}-${i}`
    if (!born.has(key)) born.set(key, now)
    const k = still ? ease((now - born.get(key)) / 420) : 1
    if (k < 1) moving = true
    const bh = Math.max(1.2, rh - (rh > 3 ? 1 : 0.4))
    if (r.status !== 'served') {
      ctx.fillStyle = C.rej
      ctx.globalAlpha = k
      ctx.fillRect(X(r.arrival_s), y, 3 + 5 * (1 - k), bh)
      ctx.globalAlpha = 1
    } else {
      const late = r.end_s - r.arrival_s > sla
      ctx.fillStyle = late ? C.miss : C.ok
      ctx.fillRect(X(r.arrival_s), y, Math.max(2, (X(r.end_s) - X(r.arrival_s)) * k), bh)
      if (!flat) {
        ctx.fillStyle = 'rgba(10,10,10,.5)'
        ctx.fillRect(X(r.arrival_s + sla), y, 1, bh)
      }
    }
    if ((r.image_tokens || 0) >= 768) {
      ctx.fillStyle = C.ink
      ctx.fillRect(X(r.arrival_s), y, 2, bh)
    }
    hit.push({ y, h: rh, r })
  })
  if (moving) later(canvas, () => drawTimeline(canvas, reqs, sla, tMax, hit))
})

export const drawConcurrency = guard(function drawConcurrency(canvas, store) {
  const { ctx, w, h } = setup(canvas)
  const pad = { l: 30, r: 96, t: 14, b: 18 }
  const all = [...store.static.series, ...store.dynamic.series]
  const tMax = niceMax(Math.max(0, ...all.map((p) => p.t)))
  const yMax = Math.max(16, ...all.map((p) => Math.max(p.limit, p.active))) + 1
  axis(ctx, w, h, pad, tMax, 'limit (line) and requests running (shaded)')
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
    ctx.lineJoin = 'round'
    ctx.stroke()
    ctx.setLineDash([])
  }
  const tag = (series, color, name) => {
    const last = series[series.length - 1]
    if (!last) return
    ctx.fillStyle = color
    ctx.font = '600 10px JetBrains Mono, monospace'
    ctx.fillText(`${name} ${last.limit}`, X(last.t) + 8, Y(last.limit) + 3)
    ctx.font = '10px JetBrains Mono, monospace'
  }
  area(store.static.series, 'active', C.statSoft)
  area(store.dynamic.series, 'active', C.dynSoft)
  step(store.static.series, C.stat, [5, 4], 1.6)
  step(store.dynamic.series, C.dyn, [], 2.4)
  tag(store.static.series, C.stat, 'Static')
  tag(store.dynamic.series, C.dyn, 'Dynamic')
  const last = store.dynamic.series[store.dynamic.series.length - 1]
  if (last) {
    const live = store.status === 'running' && !calm()
    const pulse = live ? ((performance.now() % 1400) / 1400) : 0
    ctx.fillStyle = C.dyn
    ctx.beginPath()
    ctx.arc(X(last.t), Y(last.limit), 4, 0, 7)
    ctx.fill()
    if (live) {
      ctx.strokeStyle = `rgba(255,77,28,${0.5 * (1 - pulse)})`
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(X(last.t), Y(last.limit), 4 + 10 * pulse, 0, 7)
      ctx.stroke()
      later(canvas, () => drawConcurrency(canvas, store))
    }
  }
})

export const drawSweep = guard(function drawSweep(canvas, levels, data, key, opts) {
  const { ctx, w, h } = setup(canvas)
  const pad = { l: 44, r: 14, t: 18, b: 24 }
  const pts = (pol) => levels.filter((n) => data[pol][n]).map((n) => [n, data[pol][n][key] * (opts.scale || 1)])
  const all = [...pts('static'), ...pts('dynamic')]
  const xMax = Math.max(...levels)
  const yMax = opts.fixedMax || niceTop(Math.max(opts.sla || 0, ...all.map((p) => p[1]), 1) * 1.05)
  const X = (n) => pad.l + (n / xMax) * (w - pad.l - pad.r)
  const Y = (v) => h - pad.b - (v / yMax) * (h - pad.t - pad.b)
  ctx.font = '10px JetBrains Mono, monospace'
  ctx.fillStyle = C.mute
  ctx.strokeStyle = C.line
  ctx.lineWidth = 1
  for (let i = 0; i <= 4; i++) {
    const v = (yMax / 4) * i
    ctx.beginPath()
    ctx.moveTo(pad.l, Y(v))
    ctx.lineTo(w - pad.r, Y(v))
    ctx.stroke()
    ctx.fillText(opts.fmt ? opts.fmt(v) : v.toFixed(0), 4, Y(v) + 3)
  }
  levels.forEach((n) => ctx.fillText(String(n), X(n) - 8, h - 8))
  ctx.fillText(opts.title, pad.l, 10)
  if (opts.sla) {
    ctx.fillStyle = 'rgba(232,56,79,.045)'
    ctx.fillRect(pad.l, pad.t, w - pad.l - pad.r, Y(opts.sla) - pad.t)
    ctx.strokeStyle = C.ink
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(pad.l, Y(opts.sla))
    ctx.lineTo(w - pad.r, Y(opts.sla))
    ctx.stroke()
    ctx.setLineDash([])
    ctx.fillStyle = C.ink
    ctx.fillText('SLA: above is late', w - pad.r - 112, Y(opts.sla) + 12)
  }
  const count = all.length
  if (canvas.__count !== count) {
    canvas.__count = count
    canvas.__since = performance.now()
  }
  const k = calm() ? 1 : ease((performance.now() - (canvas.__since || 0)) / 650)
  ctx.save()
  ctx.beginPath()
  ctx.rect(0, 0, pad.l + (w - pad.l) * k, h)
  ctx.clip()
  const line = (list, color, width) => {
    if (!list.length) return
    ctx.strokeStyle = color
    ctx.fillStyle = color
    ctx.lineWidth = width
    ctx.lineJoin = 'round'
    ctx.beginPath()
    list.forEach(([n, v], i) => (i ? ctx.lineTo(X(n), Y(v)) : ctx.moveTo(X(n), Y(v))))
    ctx.stroke()
    list.forEach(([n, v]) => {
      ctx.beginPath()
      ctx.arc(X(n), Y(v), 4, 0, 7)
      ctx.fill()
    })
  }
  line(pts('static'), C.stat, 2)
  line(pts('dynamic'), C.dyn, 3)
  ctx.restore()
  if (k < 1) later(canvas, () => drawSweep(canvas, levels, data, key, opts))
})
