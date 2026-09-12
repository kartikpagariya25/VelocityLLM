import { motion } from 'framer-motion';
import './Hero.css';

export default function Hero() {
  return (
    <section className="hero" id="hero">
      {/* Background glow effects */}
      <div className="hero__glow hero__glow--1" />
      <div className="hero__glow hero__glow--2" />
      <div className="hero__grid-bg" />

      <div className="hero__content container">
        <div className="hero__text">
          <motion.div
            className="hero__badge"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <span className="hero__badge-dot" />
            Open Source · Production Ready
          </motion.div>

          <motion.h1
            className="hero__title"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            Velocity<span className="gradient-text">LLM</span>
          </motion.h1>

          <motion.p
            className="hero__subtitle"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            Intelligent GPU scheduling that keeps every promise.
            SLA-aware dynamic batching proven to boost throughput
            up to <strong>517%</strong> while guaranteeing <strong>100% SLA compliance</strong> — across every model tested.
          </motion.p>

          <motion.div
            className="hero__actions"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.8, ease: [0.16, 1, 0.3, 1] }}
          >
            <a
              href="https://github.com/kartikpagariya25/VelocityLLM"
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              View on GitHub
            </a>
            <a href="#benchmarks" className="btn btn-secondary">
              See the Benchmarks
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M7 17l9.2-9.2M17 17V7H7" />
              </svg>
            </a>
          </motion.div>

          <motion.div
            className="hero__stats"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 1.0, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="hero__stat">
              <span className="hero__stat-value gradient-text">+517%</span>
              <span className="hero__stat-label">Peak Throughput Gain</span>
            </div>
            <div className="hero__stat-divider" />
            <div className="hero__stat">
              <span className="hero__stat-value gradient-text">-91%</span>
              <span className="hero__stat-label">p99 Latency Reduction</span>
            </div>
            <div className="hero__stat-divider" />
            <div className="hero__stat">
              <span className="hero__stat-value gradient-text">100%</span>
              <span className="hero__stat-label">SLA Compliance</span>
            </div>
          </motion.div>
        </div>

        <motion.div
          className="hero__3d"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.2, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <video
            className="hero__3d-video"
            poster="/videos/gpu-module-poster.jpg"
            autoPlay
            loop
            muted
            playsInline
            disablePictureInPicture
            preload="auto"
            aria-label="Animated render of an AI accelerator module, cycling between idle and fully active"
          >
            <source src="/videos/gpu-module-loop.webm" type="video/webm" />
            <source src="/videos/gpu-module-loop.mp4" type="video/mp4" />
          </video>
          <div className="hero__3d-label hero__3d-label--static">
            <span className="hero__3d-label-dot hero__3d-label-dot--dim" />
            Idle — Underutilized
          </div>
          <div className="hero__3d-label hero__3d-label--dynamic">
            <span className="hero__3d-label-dot hero__3d-label-dot--glow" />
            Active — Fully Utilized
          </div>
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        className="hero__scroll"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 0.8 }}
      >
        <div className="hero__scroll-line" />
        <span>Scroll to explore</span>
      </motion.div>
    </section>
  );
}
