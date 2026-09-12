import { useRef, useEffect } from 'react';
import { motion, useInView } from 'framer-motion';
import SectionWrapper, { MotionDiv } from './SectionWrapper';
import AnimatedCounter from './AnimatedCounter';
import './Benchmarks.css';

const models = [
  {
    name: 'TinyLlama-1.1B',
    params: '1.1B',
    throughput: { value: -1.2, label: 'Equivalent' },
    latency: { value: -14.9, label: '-14.9%' },
    slaStatic: 100,
    slaDynamic: 100,
    color: '#FF9500',
  },
  {
    name: 'Llama-3.2-1B',
    params: '1B',
    throughput: { value: 14.7, label: '+14.7%' },
    latency: { value: -24.6, label: '-24.6%' },
    slaStatic: 80,
    slaDynamic: 100,
    color: '#FF7A00',
  },
  {
    name: 'StableLM-2-1.6B',
    params: '1.6B',
    throughput: { value: -1.3, label: 'Equivalent' },
    latency: { value: -32.9, label: '-32.9%' },
    slaStatic: 80,
    slaDynamic: 100,
    color: '#FFC300',
  },
  {
    name: 'Qwen2.5-1.5B',
    params: '1.5B',
    throughput: { value: 517.6, label: '+517.6%' },
    latency: { value: -91.2, label: '-91.2%' },
    slaStatic: 66.7,
    slaDynamic: 100,
    color: '#FF5500',
    highlight: true,
  },
];

function SLABar({ staticVal, dynamicVal, label, delay = 0 }) {
  return (
    <div className="bench__sla-row">
      <span className="bench__sla-label">{label}</span>
      <div className="bench__sla-bars">
        <div className="bench__sla-bar-wrapper">
          <span className="bench__sla-bar-label">Static</span>
          <div className="bench__sla-bar-track">
            <motion.div
              className="bench__sla-bar-fill bench__sla-bar-fill--static"
              initial={{ width: 0 }}
              whileInView={{ width: `${staticVal}%` }}
              viewport={{ once: true }}
              transition={{ duration: 1.2, delay: delay + 0.3, ease: [0.16, 1, 0.3, 1] }}
            />
          </div>
          <span className="bench__sla-bar-value">{staticVal}%</span>
        </div>
        <div className="bench__sla-bar-wrapper">
          <span className="bench__sla-bar-label">Dynamic</span>
          <div className="bench__sla-bar-track">
            <motion.div
              className="bench__sla-bar-fill bench__sla-bar-fill--dynamic"
              initial={{ width: 0 }}
              whileInView={{ width: `${dynamicVal}%` }}
              viewport={{ once: true }}
              transition={{ duration: 1.5, delay: delay + 0.5, ease: [0.16, 1, 0.3, 1] }}
            />
          </div>
          <span className="bench__sla-bar-value bench__sla-bar-value--glow">{dynamicVal}%</span>
        </div>
      </div>
    </div>
  );
}

export default function Benchmarks() {
  return (
    <SectionWrapper id="benchmarks" className="benchmarks">
      <div className="container">
        <MotionDiv className="benchmarks__header">
          <div className="section-label">Benchmark Results</div>
          <h2 className="section-title">
            Real numbers.<br />
            <span className="gradient-text">Not marketing claims.</span>
          </h2>
          <p className="section-subtitle">
            Tested across four independently trained language models on the same hardware
            (NVIDIA RTX 5050, 8 GB VRAM). Every number below comes from reproducible benchmarks.
          </p>
        </MotionDiv>

        {/* Headline stats */}
        <MotionDiv className="benchmarks__headlines">
          <div className="benchmarks__headline-card glass-card">
            <span className="benchmarks__headline-value gradient-text">
              <AnimatedCounter target={517.6} prefix="+" suffix="%" decimals={1} />
            </span>
            <span className="benchmarks__headline-label">Peak Throughput Gain</span>
            <span className="benchmarks__headline-model">Qwen2.5-1.5B</span>
          </div>
          <div className="benchmarks__headline-card glass-card">
            <span className="benchmarks__headline-value gradient-text">
              <AnimatedCounter target={91.2} prefix="-" suffix="%" decimals={1} />
            </span>
            <span className="benchmarks__headline-label">Best p99 Latency Drop</span>
            <span className="benchmarks__headline-model">Qwen2.5-1.5B</span>
          </div>
          <div className="benchmarks__headline-card glass-card">
            <span className="benchmarks__headline-value gradient-text">
              <AnimatedCounter target={100} suffix="%" decimals={0} />
            </span>
            <span className="benchmarks__headline-label">SLA Compliance</span>
            <span className="benchmarks__headline-model">Every Model Tested</span>
          </div>
        </MotionDiv>

        {/* Model cards */}
        <MotionDiv>
          <h3 className="benchmarks__section-title">Saturation Load Test — Per Model</h3>
        </MotionDiv>
        <div className="benchmarks__cards">
          {models.map((model, i) => (
            <MotionDiv
              key={model.name}
              className={`benchmarks__card glass-card ${model.highlight ? 'benchmarks__card--highlight' : ''}`}
              delay={i * 0.1}
            >
              <div className="benchmarks__card-header">
                <span className="benchmarks__card-name">{model.name}</span>
                <span className="benchmarks__card-params">{model.params} params</span>
              </div>
              <div className="benchmarks__card-stats">
                <div className="benchmarks__card-stat">
                  <span className="benchmarks__card-stat-label">Throughput</span>
                  <span className={`benchmarks__card-stat-value ${model.throughput.value > 0 ? 'text-positive' : ''}`}>
                    {model.throughput.label}
                  </span>
                </div>
                <div className="benchmarks__card-stat">
                  <span className="benchmarks__card-stat-label">p99 Latency</span>
                  <span className="benchmarks__card-stat-value text-positive">{model.latency.label}</span>
                </div>
              </div>
              {model.highlight && (
                <div className="benchmarks__card-badge">🏆 Best Result</div>
              )}
            </MotionDiv>
          ))}
        </div>

        {/* SLA Compliance Chart */}
        <MotionDiv className="benchmarks__sla-section">
          <h3 className="benchmarks__section-title">
            SLA Compliance — The Strongest Finding
          </h3>
          <p className="benchmarks__sla-desc">
            Under mixed realistic load, the dynamic scheduler achieved <strong>100% SLA compliance in every single model</strong>.
            The static baseline dropped as low as 66.7%.
          </p>
        </MotionDiv>

        <div className="benchmarks__sla-chart glass-card">
          {models.map((model, i) => (
            <SLABar
              key={model.name}
              staticVal={model.slaStatic}
              dynamicVal={model.slaDynamic}
              label={model.name}
              delay={i * 0.15}
            />
          ))}
        </div>

        {/* Insight callout */}
        <MotionDiv className="benchmarks__insight glass-card">
          <div className="benchmarks__insight-icon">💡</div>
          <div>
            <div className="benchmarks__insight-title">Resource-Pressure Pattern</div>
            <p className="benchmarks__insight-text">
              The magnitude of improvement scales directly with GPU memory pressure.
              When the system is relaxed, both policies perform equivalently.
              When constrained — the dynamic scheduler prevents severe latency spikes and delivers order-of-magnitude gains.
            </p>
          </div>
        </MotionDiv>
      </div>
    </SectionWrapper>
  );
}
