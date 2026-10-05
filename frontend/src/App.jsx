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

const inLab = () => window.location.hash.startsWith('#/lab')

export default function App() {
  const [lab, setLab] = useState(inLab)
  useEffect(() => {
    const on = () => {
      setLab(inLab())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return lab ? (
    <Boundary>
      <Lab />
    </Boundary>
  ) : (
    <Site />
  )
}
