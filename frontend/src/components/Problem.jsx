import { motion } from 'framer-motion';
import SectionWrapper, { MotionDiv } from './SectionWrapper';
import './Problem.css';

export default function Problem() {
  return (
    <SectionWrapper id="problem" className="problem">
      <div className="container">
        <div className="problem__layout">
          <div className="problem__visual">
            <MotionDiv className="problem__gpu-grid">
              {/* Static side - underutilized */}
              <div className="problem__gpu-half problem__gpu-half--static">
                <div className="problem__gpu-label">Static Batching</div>
                <div className="problem__cores">
                  {[...Array(16)].map((_, i) => (
                    <div
                      key={i}
                      className={`problem__core ${i < 4 ? 'problem__core--active' : 'problem__core--idle'}`}
                    />
                  ))}
                </div>
                <div className="problem__utilization">
                  <div className="problem__util-bar">
                    <motion.div
                      className="problem__util-fill problem__util-fill--low"
                      initial={{ width: 0 }}
                      whileInView={{ width: '25%' }}
                      viewport={{ once: true }}
                      transition={{ duration: 1.2, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
                    />
                  </div>
                  <span className="problem__util-label">~25% GPU Utilization</span>
                </div>
              </div>

              {/* Arrow */}
              <div className="problem__arrow">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--color-primary)" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </div>

              {/* Dynamic side - fully utilized */}
              <div className="problem__gpu-half problem__gpu-half--dynamic">
                <div className="problem__gpu-label problem__gpu-label--glow">Dynamic Batching</div>
                <div className="problem__cores">
                  {[...Array(16)].map((_, i) => (
                    <div
                      key={i}
                      className="problem__core problem__core--active problem__core--glow"
                      style={{ animationDelay: `${i * 0.1}s` }}
                    />
                  ))}
                </div>
                <div className="problem__utilization">
                  <div className="problem__util-bar">
                    <motion.div
                      className="problem__util-fill problem__util-fill--high"
                      initial={{ width: 0 }}
                      whileInView={{ width: '95%' }}
                      viewport={{ once: true }}
                      transition={{ duration: 1.5, delay: 0.8, ease: [0.16, 1, 0.3, 1] }}
                    />
                  </div>
                  <span className="problem__util-label">~95% GPU Utilization</span>
                </div>
              </div>
            </MotionDiv>
          </div>

          <div className="problem__content">
            <MotionDiv>
              <div className="section-label">The Problem</div>
            </MotionDiv>
            <MotionDiv>
              <h2 className="section-title">
                Static batching<br />
                <span className="gradient-text">wastes your GPU.</span>
              </h2>
            </MotionDiv>
            <MotionDiv>
              <p className="problem__desc">
                Traditional LLM serving uses fixed batch sizes that under-utilize GPU throughput
                under variable request loads — and silently breaks latency promises when traffic spikes.
              </p>
            </MotionDiv>
            <MotionDiv>
              <div className="problem__callout glass-card">
                <div className="problem__callout-icon">⚠️</div>
                <div>
                  <div className="problem__callout-title">The Silent Failure</div>
                  <p className="problem__callout-text">
                    Static batching silently breaks its latency promise on up to
                    <strong> 1 in 3 requests</strong> — with no warning to the caller.
                  </p>
                </div>
              </div>
            </MotionDiv>
          </div>
        </div>
      </div>
    </SectionWrapper>
  );
}
