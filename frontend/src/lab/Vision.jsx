import Info from './Info'
import { IMAGE_MIXES, imageTokens } from './sim'

const SAMPLES = ['bar_chart', 'sunset_mountains', 'line_chart', 'request_flow', 'heatmap', 'pie_chart', 'scatter_clusters', 'concentric_rings']
const SIZE_NAME = { 224: 'Small', 448: 'Medium', 896: 'Large' }

export const meanImageTokens = (mix) => IMAGE_MIXES[mix].sizes.reduce((n, [side, w]) => n + w * imageTokens(side), 0)

export default function Vision({ mix, onChange }) {
  const spec = IMAGE_MIXES[mix]
  return (
    <section className="vision" aria-label="Vision workload">
      <header>
        <h3>
          Vision workload <Info id="vision" />
        </h3>
        <div className="seg seg--light" role="radiogroup" aria-label="Image mix">
          {Object.entries(IMAGE_MIXES).map(([k, v]) => (
            <button key={k} type="button" role="radio" aria-checked={mix === k} className={mix === k ? 'on' : ''} onClick={() => onChange(k)}>
              {v.label}
            </button>
          ))}
        </div>
      </header>
      <div className="vision__body">
        <ul className="vision__strip" aria-label="Bundled sample images">
          {SAMPLES.map((n) => (
            <li key={n}>
              <img src={`${import.meta.env.BASE_URL}vision/${n}.jpg`} alt="" loading="lazy" width="64" height="64" />
            </li>
          ))}
        </ul>
        <div className="vision__mix">
          <p className="cap">
            Image sizes in this run <Info id="imageMix" />
          </p>
          <div className="mixbar" role="img" aria-label={spec.sizes.map(([s, w]) => `${Math.round(w * 100)}% ${SIZE_NAME[s]}`).join(', ')}>
            {spec.sizes.map(([side, w]) => (
              <span key={side} className={`mixbar__seg mixbar__seg--${side}`} style={{ flexGrow: w }}>
                {Math.round(w * 100)}% <i>{imageTokens(side)} tok</i>
              </span>
            ))}
          </div>
          <small>
            {spec.hint}. Each user asks one question about one image, about {Math.round(meanImageTokens(mix))} vision tokens on average. A large image costs as much prefill and memory as 16 thumbnails, and Dynamic counts it that way.
          </small>
        </div>
      </div>
    </section>
  )
}
