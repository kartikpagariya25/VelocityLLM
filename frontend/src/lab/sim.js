export const MODELS = [
  { id: 'llama', name: 'Llama-3.2-1B', tpt: 0.012 },
  { id: 'stablelm', name: 'StableLM-2-1.6B', tpt: 0.016 },
  { id: 'qwen', name: 'Qwen2.5-1.5B', tpt: 0.015 },
]

export const PROMPTS = {
  short: { label: 'Short answer', tokens: 64 },
  medium: { label: 'Explanation', tokens: 128 },
  long: { label: 'Long essay', tokens: 256 },
  custom: { label: 'Custom prompt', tokens: 0 },
}

export const TRAFFIC = {
  flood: { label: 'Flood', hint: 'Every user hits at the same instant' },
  steady: { label: 'Steady', hint: 'Requests spread evenly over 30 s' },
  burst: { label: 'Burst', hint: 'Light traffic, then a sudden spike at 12 s' },
  mixed: { label: 'Mixed', hint: 'Short and long prompts over 15 s' },
}

export const LIMITS = { users: [1, 200], sla: [2, 20], burst: [1, 100], promptChars: 500 }

const MEM_BASE = 0.55
const MEM_PER = 0.04
const MEM_TOTAL_MB = 40960
const STATIC_C = 8
const MIN_C = 1
const MAX_C = 16
const QUEUE_LIMIT = 100
const DT = 0.05

const mulberry = (a) => () => {
  a |= 0
  a = (a + 0x6d2b79f5) | 0
  let t = Math.imul(a ^ (a >>> 15), 1 | a)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}

export const customTokens = (text) => Math.min(256, Math.max(32, Math.round(text.trim().length / 4) * 2))

export function promptTokens(prompt, text) {
  return prompt === 'custom' ? customTokens(text || '') : PROMPTS[prompt].tokens
}

export function buildArrivals({ users, traffic, prompt, text, seed = 7 }) {
  const rnd = mulberry(seed)
  const base = promptTokens(prompt, text)
  const out = []
  const push = (t, tokens, priority = 'NORMAL') => out.push({ t, tokens, priority })
  for (let i = 0; i < users; i++) {
    if (traffic === 'flood') push(i * 0.01, base)
    else if (traffic === 'steady') push((i / users) * 30 + rnd() * 0.4, base)
    else if (traffic === 'burst') {
      const calm = Math.round(users * 0.35)
      if (i < calm) push((i / Math.max(1, calm)) * 10 + rnd() * 0.3, base)
      else push(12 + rnd() * 0.6, base)
    } else {
      const long = rnd() < 0.3
      push((i / users) * 15 + rnd() * 0.3, long ? 256 : 64, long ? 'NORMAL' : 'HIGH')
    }
  }
  out.sort((a, b) => a.t - b.t)
  return out.map((a, i) => ({ ...a, id: `r-${i + 1}` }))
}

const pct = (arr, p) => {
  if (!arr.length) return 0
  const s = [...arr].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.ceil((p / 100) * s.length) - 1)]
}

export function score(requests, sla) {
  const served = requests.filter((r) => r.status === 'served')
  const lat = served.map((r) => r.end_s - r.arrival_s)
  const within = served.filter((r) => r.end_s - r.arrival_s <= sla).length
  const span = served.length ? Math.max(...served.map((r) => r.end_s)) - Math.min(...requests.map((r) => r.arrival_s)) : 0
  const tokens = served.reduce((n, r) => n + r.tokens, 0)
  return {
    p50_s: pct(lat, 50),
    p95_s: pct(lat, 95),
    p99_s: pct(lat, 99),
    tokens_per_s: span > 0 ? tokens / span : 0,
    served: served.length,
    offered: requests.length,
    rejected: requests.length - served.length,
    within_sla_served: served.length ? within / served.length : 0,
    within_sla_offered: requests.length ? within / requests.length : 0,
    zero_token_share: 0,
  }
}

export class Sim {
  constructor({ policy, arrivals, sla, tpt, emit }) {
    this.policy = policy
    this.sla = sla
    this.tptBase = tpt
    this.emit = emit
    this.pending = arrivals.map((a) => ({ ...a }))
    this.t = 0
    this.limit = STATIC_C
    this.queue = []
    this.active = []
    this.log = []
    this.lat = []
    this.completed = 0
    this.rejected = 0
    this.tokens = 0
    this.lastCtl = -1
    this.lastMetric = -1
    this.execAvg = null
    this.perTok = null
    this.done = false
    this.reqs = []
    this.sys(
      policy === 'static'
        ? `StaticBatchPolicy initialized (fixed concurrency: ${STATIC_C}).`
        : `DynamicBatchPolicy initialized (initial concurrency: ${STATIC_C}, min: ${MIN_C}, max: ${MAX_C}, SLA target: ${Math.round(sla * 1000)} ms).`,
    )
  }

  tpt(n) {
    return this.tptBase * (1 + 0.1 * (n - 1) + 0.25 * Math.max(0, n - 6))
  }

  get mem() {
    return MEM_BASE + MEM_PER * this.active.length
  }

  line(category, level, message, request_id = null) {
    this.emit({
      event: 'log',
      data: { policy: this.policy, ts: this.t, level, category, request_id, message },
    })
  }

  sys(m) {
    this.line('system', 'INFO', m)
  }

  inject(list) {
    list.forEach((a) => this.pending.push({ ...a, t: Math.max(a.t, this.t) }))
    this.pending.sort((a, b) => a.t - b.t)
    this.done = false
  }

  predict(tokens) {
    const perTok = this.perTok ?? this.tpt(this.limit)
    const exec = this.execAvg ?? 128 * perTok
    const ahead = this.queue.length + this.active.length
    const waves = Math.floor(ahead / this.limit)
    return { pred: waves * exec + tokens * perTok, perTok }
  }

  arrive(a) {
    const r = { ...a, arrival_s: a.t, start_s: null, end_s: null, left: a.tokens, status: 'pending' }
    if (this.policy === 'dynamic') {
      if (this.queue.length >= QUEUE_LIMIT) return this.reject(r, 'queue_full', `Request ${r.id} rejected: Queue full (${this.queue.length}/${QUEUE_LIMIT}). Retry after 1.00s`, 1)
      const { pred } = this.predict(r.tokens)
      if (pred > this.sla)
        return this.reject(r, 'sla_impossible', `Request ${r.id} rejected: Predicted latency ${pred.toFixed(2)}s exceeds SLA target ${this.sla.toFixed(2)}s.`, Math.max(0.5, pred - this.sla))
      this.line('admission', 'INFO', `Request ${r.id} admitted (predicted ${pred.toFixed(2)}s <= SLA ${this.sla.toFixed(2)}s, queue ${this.queue.length}, active ${this.active.length}/${this.limit})`, r.id)
    } else {
      this.line('admission', 'INFO', `Request ${r.id} accepted (queue ${this.queue.length}, active ${this.active.length}/${this.limit})`, r.id)
    }
    this.queue.push(r)
  }

  reject(r, reason, message, retry) {
    this.rejected++
    this.line('admission', 'WARNING', message, r.id)
    this.emit({
      event: 'request',
      data: {
        policy: this.policy,
        request_id: r.id,
        arrival_s: r.arrival_s,
        start_s: null,
        end_s: this.t,
        status: 'rejected',
        http_status: 429,
        tokens: r.tokens,
        priority: r.priority,
        reject_reason: reason,
        retry_after_s: retry,
      },
    })
    this.reqs.push({ request_id: r.id, arrival_s: r.arrival_s, end_s: this.t, status: 'rejected', tokens: r.tokens })
  }

  control() {
    if (this.policy !== 'dynamic' || this.t - this.lastCtl < 0.25) return
    const recent = this.lat.slice(-20)
    const oldest = [...this.queue, ...this.active].reduce((m, r) => Math.max(m, this.t - r.arrival_s), 0)
    const pressure = Math.max(pct(recent, 95), oldest)
    const prev = this.limit
    let why = null
    this.streak = this.streak || 0
    if (this.mem > 0.94) {
      this.limit = MIN_C
      why = 'emergency: GPU memory above 94%'
    } else if (pressure > this.sla * 0.9 && this.limit > MIN_C) {
      this.limit = Math.max(MIN_C, Math.floor(this.limit * 0.75))
      why = `decrease: p95 ${pressure.toFixed(2)}s near SLA ${this.sla.toFixed(2)}s`
    } else if (this.queue.length > 0 && pressure < this.sla * 0.6 && this.limit < MAX_C && MEM_BASE + MEM_PER * (this.limit + 1) <= 0.9) {
      this.streak = Math.min((this.streak || 0) + 1, 4)
      this.limit = Math.min(MAX_C, this.limit + Math.min(2 ** (this.streak - 1), Math.max(1, this.limit >> 1)))
      why = `increase: p95 ${pressure.toFixed(2)}s well under SLA, queue ${this.queue.length}`
    }
    if (this.limit <= prev) this.streak = 0
    if (why && this.limit !== prev) {
      this.lastCtl = this.t
      this.line('controller', 'INFO', `Adaptive Batch Controller: concurrency ${prev} -> ${this.limit} (${why})`)
    }
  }

  metrics() {
    this.emit({
      event: 'metrics',
      data: {
        policy: this.policy,
        ts: this.t,
        active: this.active.length,
        queued: this.queue.length,
        concurrency_limit: this.limit,
        completed: this.completed,
        rejected: this.rejected,
        tokens: this.tokens,
        gpu_mem_used_mb: Math.round(this.mem * MEM_TOTAL_MB),
        gpu_mem_total_mb: MEM_TOTAL_MB,
      },
    })
  }

  step() {
    this.t += DT
    while (this.pending.length && this.pending[0].t <= this.t) this.arrive(this.pending.shift())
    while (this.active.length < this.limit && this.queue.length) {
      const r = this.queue.shift()
      r.start_s = this.t
      this.active.push(r)
      this.line('lifecycle', 'INFO', `Request ${r.id} started after ${(this.t - r.arrival_s).toFixed(2)}s in queue`, r.id)
    }
    const n = this.active.length
    if (n) {
      const adv = DT / this.tpt(n)
      for (const r of this.active) r.left -= adv
      const fin = this.active.filter((r) => r.left <= 0)
      this.active = this.active.filter((r) => r.left > 0)
      for (const r of fin) {
        r.end_s = this.t
        const exec = r.end_s - r.start_s
        this.execAvg = this.execAvg == null ? exec : this.execAvg * 0.7 + exec * 0.3
        this.perTok = this.perTok == null ? exec / r.tokens : this.perTok * 0.7 + (exec / r.tokens) * 0.3
        this.completed++
        this.tokens += r.tokens
        this.lat.push(r.end_s - r.arrival_s)
        this.line('lifecycle', 'INFO', `Request ${r.id} completed in ${(r.end_s - r.arrival_s).toFixed(2)}s (${r.tokens} tokens, exec ${exec.toFixed(2)}s)`, r.id)
        this.emit({
          event: 'request',
          data: {
            policy: this.policy,
            request_id: r.id,
            arrival_s: r.arrival_s,
            start_s: r.start_s,
            end_s: r.end_s,
            status: 'served',
            http_status: 200,
            tokens: r.tokens,
            priority: r.priority,
            reject_reason: null,
            retry_after_s: null,
          },
        })
        this.reqs.push({ request_id: r.id, arrival_s: r.arrival_s, end_s: r.end_s, status: 'served', tokens: r.tokens })
      }
    }
    this.control()
    if (this.t - this.lastMetric >= 0.5) {
      this.lastMetric = this.t
      this.metrics()
    }
    if (!this.pending.length && !this.queue.length && !this.active.length) {
      this.metrics()
      this.done = true
    }
  }

  advance(to) {
    while (this.t < to && !this.done) this.step()
  }

  result() {
    return score(this.reqs, this.sla)
  }
}

