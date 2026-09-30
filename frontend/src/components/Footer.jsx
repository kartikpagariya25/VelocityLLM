import { REPO } from '../data/content'
import { Arrow } from './Icons'
import './Footer.css'

export default function Footer() {
  return (
    <footer className="foot-wrap">
      <div className="wrap footer">
        <a className="btn btn--hot" href={REPO} target="_blank" rel="noopener noreferrer">
          Read the source <Arrow width={18} height={18} />
        </a>
        <p className="mono">Single Core Labs · VIT Pune</p>
      </div>
      <div className="mark" aria-hidden="true">
        Velocity<span>LLM</span>
      </div>
    </footer>
  )
}
