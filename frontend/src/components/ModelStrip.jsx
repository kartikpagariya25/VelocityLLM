import { MODELS } from '../data/content'
import { useReveal } from '../lib/motion'
import './ModelStrip.css'

export default function ModelStrip() {
  const root = useReveal()
  return (
    <section className="models section" ref={root}>
      <div className="wrap models__inner">
        <div>
          <p className="kicker" data-reveal>Model-agnostic</p>
          <h2 className="h2" data-lines>Swap the model. Keep the scheduler.</h2>
          <p className="lede" data-reveal>
            The scheduler only reads request metadata: arrival time, prompt length, priority and deadline. It never touches weights, tokenizers or logits, so changing the model is a config change. Tested on an RTX 5050 with 8 GB of memory.
          </p>
        </div>
        <ul className="models__list">
          {MODELS.map((m) => (
            <li key={m.id} data-reveal>
              <b>{m.name}</b>
              <span className="mono">{m.size} parameters</span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
