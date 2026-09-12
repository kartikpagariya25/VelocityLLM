import { motion } from 'framer-motion';
import SectionWrapper, { MotionDiv } from './SectionWrapper';
import './HowItWorks.css';

const steps = [
  {
    number: '01',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    ),
    title: 'Request Arrives',
    description: 'Client sends an inference request through the FastAPI gateway with automatic validation and correlation tracking.',
  },
  {
    number: '02',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="M9 12l2 2 4-4" />
      </svg>
    ),
    title: 'SLA-Aware Admission',
    description: 'Predicts whether the request can meet its latency target. Accepts, queues, or honestly rejects — never silently late.',
  },
  {
    number: '03',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
      </svg>
    ),
    title: 'Adaptive AIMD Tuning',
    description: 'Real-time concurrency controller driven by live GPU utilization and memory headroom — the same algorithm family as TCP congestion control.',
  },
  {
    number: '04',
    icon: (
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
      </svg>
    ),
    title: 'Served or Honestly Declined',
    description: 'Request completes within SLA — or gets an immediate, honest HTTP 429 rejection. Never silently degraded.',
  },
];

export default function HowItWorks() {
  return (
    <SectionWrapper id="how-it-works" className="how-it-works">
      <div className="container">
        <MotionDiv className="how-it-works__header">
          <div className="section-label">How It Works</div>
          <h2 className="section-title">
            Four steps to<br />
            <span className="gradient-text">guaranteed performance.</span>
          </h2>
          <p className="section-subtitle">
            Every request is intelligently routed through a pipeline designed for honesty over throughput theatre.
          </p>
        </MotionDiv>

        <div className="how-it-works__timeline">
          <div className="how-it-works__line" />
          {steps.map((step, i) => (
            <MotionDiv key={i} className="how-it-works__step" delay={i * 0.12}>
              <div className="how-it-works__step-number">{step.number}</div>
              <div className="how-it-works__step-icon glass-card">
                {step.icon}
              </div>
              <div className="how-it-works__step-content">
                <h3 className="how-it-works__step-title">{step.title}</h3>
                <p className="how-it-works__step-desc">{step.description}</p>
              </div>
              {i < steps.length - 1 && (
                <div className="how-it-works__connector">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--color-primary)" strokeWidth="2" strokeLinecap="round">
                    <path d="M12 5v14M19 12l-7 7-7-7" />
                  </svg>
                </div>
              )}
            </MotionDiv>
          ))}
        </div>
      </div>
    </SectionWrapper>
  );
}
