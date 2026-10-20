import { REPO } from '../data/content'
import { Arrow } from './Icons'
import './Footer.css'

const COLUMNS = [
  {
    title: 'Explore',
    links: [
      ['#/lab', 'Open Arena'],
      ['#race', 'The race'],
      ['#sweep', 'Load test'],
    ],
  },
  {
    title: 'Project',
    links: [
      [REPO, 'Source', true],
      [`${REPO}/tree/main/docs`, 'Documentation', true],
      [`${REPO}/issues`, 'Issues', true],
    ],
  },
]

export default function Footer() {
  return (
    <footer className="foot-wrap">
      <div className="wrap footer">
        <div className="footer__lead">
          <p className="footer__line">An SLA-aware batching engine for LLM inference, built on vLLM.</p>
          <a className="btn btn--hot" href="#/lab">
            Open Arena <Arrow width={18} height={18} />
          </a>
        </div>
        <nav className="footer__cols" aria-label="Footer">
          {COLUMNS.map((col) => (
            <div key={col.title}>
              <h3>{col.title}</h3>
              <ul>
                {col.links.map(([href, label, external]) => (
                  <li key={label}>
                    <a href={href} {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}>
                      {label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>
        <p className="footer__small mono">Single Core Labs, VIT Pune</p>
      </div>
      <div className="mark" aria-hidden="true">
        Velocity<span>LLM</span>
      </div>
    </footer>
  )
}
