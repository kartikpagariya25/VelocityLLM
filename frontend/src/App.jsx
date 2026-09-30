import { useEffect } from 'react'
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

export default function App() {
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
