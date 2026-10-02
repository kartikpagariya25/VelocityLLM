export const REPO = 'https://github.com/kartikpagariya25/VelocityLLM'

export const MODELS = [
  { id: 'llama', name: 'Llama-3.2', size: '1B' },
  { id: 'stablelm', name: 'StableLM-2', size: '1.6B' },
  { id: 'qwen', name: 'Qwen2.5', size: '1.5B' },
  { id: 'tinyllama', name: 'TinyLlama', size: '1.1B' },
]

export const SCENARIOS = {
  flood: { label: 'Flood', hint: '30 requests arrive at the same instant' },
  mixed: { label: 'Mixed', hint: 'Short and long prompts arrive together' },
}

export const NOTES = {
  'qwen-flood': 'The static run stalled for about 25 seconds on its first wave, partly engine warm-up. Read this pairing as a best case.',
  'tinyllama-flood': 'Most replies from this model ended after zero tokens, so both schedulers finish almost instantly.',
  'tinyllama-mixed': 'Most replies from this model ended after zero tokens, so both schedulers finish almost instantly.',
}

export const STAGES = [
  {
    id: 'gateway',
    name: 'Gateway',
    line: 'Every request enters through one validated door.',
    chips: ['FastAPI', 'OpenAI-compatible /v1/completions', 'Correlation ID on every request'],
  },
  {
    id: 'admission',
    name: 'Admission',
    line: 'Says no early, instead of failing late.',
    chips: ['Queue full: 429', 'GPU memory above 94%: shed', 'Predicted wait beyond the SLA: shed', 'Bursts shed low priority first'],
  },
  {
    id: 'queue',
    name: 'Priority queue',
    line: 'Urgent work goes first, and nothing waits forever.',
    chips: ['Priority classes', 'Aging lifts old requests', 'Cancel while waiting'],
  },
  {
    id: 'controller',
    name: 'Batch controller',
    line: 'Grows the batch while the GPU has room, and cuts it the moment it does not.',
    chips: ['Add one when the queue waits', 'Cut by a quarter on an SLA breach', 'Step down near the memory limit', 'Collapse on OOM risk'],
  },
  {
    id: 'backend',
    name: 'Backend',
    line: 'The scheduler never touches model weights.',
    chips: ['vLLM async engine', 'Mock backend for tests', 'Swap models with a config change'],
  },
  {
    id: 'gpu',
    name: 'GPU',
    line: 'Stays busy, stays inside its memory.',
    chips: ['Continuous batching', 'Live utilisation and memory feed the controller'],
  },
]

export const TEAM = [
  {
    name: 'Kartik R. Pagariya',
    role: 'Scheduler architecture, infra, benchmarking',
    photo: '/team/kartik.jpg',
    github: 'https://github.com/kartikpagariya25',
    linkedin: 'https://www.linkedin.com/in/kartikpagariya1911/',
  },
  {
    name: 'Vikrant K. Kadam',
    role: 'Core scheduler engine',
    photo: '/team/vikrant.jpg',
    github: 'https://github.com/VikrantKadam028',
    linkedin: 'https://linkedin.com/in/vikrantkadam028/',
  },
  {
    name: 'Aditya D. Dengale',
    role: 'Documentation and benchmarking support',
    photo: '/team/aditya.jpg',
    github: 'https://github.com/DevXDividends',
    linkedin: 'https://www.linkedin.com/in/adityadengale/',
  },
]
