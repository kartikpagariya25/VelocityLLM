import { useState } from 'react'
import { TEAM } from '../data/content'
import { useReveal } from '../lib/motion'
import { Github, Linkedin } from './Icons'
import './Team.css'

const initials = (name) =>
  name
    .replace(/\./g, '')
    .split(' ')
    .filter((p) => p.length > 1)
    .map((p) => p[0])
    .slice(0, 2)
    .join('')

function Avatar({ member }) {
  const [failed, setFailed] = useState(false)
  return (
    <div className="person__photo">
      {failed ? (
        <span>{initials(member.name)}</span>
      ) : (
        <img src={member.photo} alt={member.name} loading="lazy" onError={() => setFailed(true)} />
      )}
    </div>
  )
}

export default function Team() {
  const root = useReveal()
  return (
    <section className="team section" id="team" ref={root}>
      <div className="wrap">
        <p className="kicker" data-reveal>The team</p>
        <h2 className="h2" data-lines>Built by four, guided by one.</h2>
        <p className="lede" data-reveal>
          A Single Core Labs project at VIT Pune, mentored by Dr. Viomesh K. Singh.
        </p>
        <div className="team__grid">
          {TEAM.map((m) => (
            <article className="person" key={m.name} data-reveal>
              <Avatar member={m} />
              <h3>{m.name}</h3>
              <p>{m.role}</p>
              <div className="person__links">
                <a href={m.github} target="_blank" rel="noopener noreferrer" aria-label={`${m.name} on GitHub`}>
                  <Github width={20} height={20} />
                </a>
                <a href={m.linkedin} target="_blank" rel="noopener noreferrer" aria-label={`${m.name} on LinkedIn`}>
                  <Linkedin width={20} height={20} />
                </a>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  )
}
