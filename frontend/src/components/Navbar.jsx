import { useEffect, useRef, useState } from 'react'
import { ScrollTrigger, scrollToId } from '../lib/motion'
import { Github } from './Icons'
import { REPO } from '../data/content'
import './Navbar.css'

const LINKS = [
  ['problem', 'Problem'],
  ['race', 'The race'],
  ['engine', 'Engine'],
  ['results', 'Results'],
  ['team', 'Team'],
]

export default function Navbar() {
  const bar = useRef(null)
  const [hidden, setHidden] = useState(false)

  useEffect(() => {
    const st = ScrollTrigger.create({
      start: 0,
      end: 'max',
      onUpdate: (self) => {
        bar.current.style.transform = `scaleX(${self.progress})`
        setHidden(self.direction === 1 && self.scroll() > 240)
      },
    })
    return () => st.kill()
  }, [])

  const go = (id) => (e) => {
    e.preventDefault()
    scrollToId(id)
  }

  return (
    <header className={`nav ${hidden ? 'nav--hidden' : ''}`}>
      <div className="nav__inner wrap">
        <a href="#top" className="nav__brand" onClick={go('top')}>
          <img src="/logo.png" alt="" className="nav__logo" />
          <span className="sr-only">Velocity</span>
          <span>LLM</span>
        </a>
        <nav className="nav__links" aria-label="Sections">
          {LINKS.map(([id, label]) => (
            <a key={id} href={`#${id}`} onClick={go(id)}>
              {label}
            </a>
          ))}
        </nav>
        <a className="nav__arena" href="#/lab">
          Open Arena
        </a>
        <a className="nav__repo" href={REPO} target="_blank" rel="noopener noreferrer" aria-label="VelocityLLM on GitHub">
          <Github width={20} height={20} />
        </a>
      </div>
      <div className="nav__progress" ref={bar} />
    </header>
  )
}
