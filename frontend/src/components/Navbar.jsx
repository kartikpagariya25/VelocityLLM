import { useEffect, useRef, useState } from 'react'
import { ScrollTrigger, scrollToId } from '../lib/motion'
import { List, X } from '@phosphor-icons/react'
import { Github } from './Icons'
import { REPO } from '../data/content'
import './Navbar.css'

const LINKS = [
  ['problem', 'Problem'],
  ['race', 'The race'],
  ['engine', 'Engine'],
  ['results', 'Results'],
  ['sweep', 'Load test'],
  ['team', 'Team'],
]

export default function Navbar() {
  const bar = useRef(null)
  const [hidden, setHidden] = useState(false)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const st = ScrollTrigger.create({
      start: 0,
      end: 'max',
      onUpdate: (self) => {
        bar.current.style.transform = `scaleX(${self.progress})`
        setHidden(self.direction === 1 && self.scroll() > 240)
        if (self.direction === 1) setOpen(false)
      },
    })
    return () => st.kill()
  }, [])

  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  const go = (id) => (e) => {
    e.preventDefault()
    setOpen(false)
    scrollToId(id)
  }

  return (
    <header className={`nav ${hidden && !open ? 'nav--hidden' : ''}`}>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <div className="nav__inner wrap">
        <a href="#top" className="nav__brand" onClick={go('top')}>
          Velocity<span>LLM</span>
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
        <button
          type="button"
          className="nav__toggle"
          aria-expanded={open}
          aria-controls="nav-sheet"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X size={22} weight="bold" /> : <List size={22} weight="bold" />}
        </button>
        <a className="nav__repo" href={REPO} target="_blank" rel="noopener noreferrer" aria-label="VelocityLLM on GitHub">
          <Github width={20} height={20} />
        </a>
      </div>
      <nav id="nav-sheet" className={`nav__sheet ${open ? 'nav__sheet--open' : ''}`} aria-label="Sections" aria-hidden={!open}>
        {LINKS.map(([id, label]) => (
          <a key={id} href={`#${id}`} onClick={go(id)} tabIndex={open ? 0 : -1}>
            {label}
          </a>
        ))}
      </nav>
      <div className="nav__progress" ref={bar} />
    </header>
  )
}
