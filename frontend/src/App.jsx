import { useEffect, useState } from 'react'
import { startSmoothScroll, ScrollTrigger } from './lib/motion'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Problem from './components/Problem'
import Race from './components/Race'
import HowItWorks from './components/HowItWorks'
import Benchmarks from './components/Benchmarks'
import ModelStrip from './components/ModelStrip'
import Team from './components/Team'
import Footer from './components/Footer'
import Lab from './lab/Lab'
import Boundary from './lab/Boundary'
import Phone from './live/Phone'

function Site() {
  useEffect(() => {
    const stop = startSmoothScroll()
    document.fonts.ready.then(() => ScrollTrigger.refresh())
    return stop
  }, [])

  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <Problem />
        <Race />
        <HowItWorks />
        <Benchmarks />
        <ModelStrip />
        <Team />
      </main>
      <Footer />
    </>
  )
}

const route = () => {
  const h = window.location.hash
  if (h.startsWith('#/join')) return 'join'
  if (h.startsWith('#/live')) return 'live'
  if (h.startsWith('#/lab')) return 'lab'
  return 'site'
}

export default function App() {
  const [view, setView] = useState(route)
  useEffect(() => {
    const on = () => {
      setView(route())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  if (view === 'site') return <Site />
  return <Boundary>{view === 'join' ? <Phone /> : <Lab key={view} session={view === 'live'} />}</Boundary>
}
