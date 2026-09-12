import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Problem from './components/Problem';
import HowItWorks from './components/HowItWorks';
import Benchmarks from './components/Benchmarks';
import ModelStrip from './components/ModelStrip';
import TechStack from './components/TechStack';
import Team from './components/Team';
import Footer from './components/Footer';

function App() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <Problem />
        <HowItWorks />
        <Benchmarks />
        <ModelStrip />
        <TechStack />
        <Team />
      </main>
      <Footer />
    </>
  );
}

export default App;
