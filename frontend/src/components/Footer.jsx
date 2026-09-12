import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer__top">
          <div className="footer__brand">
            <div className="footer__logo">
              <span className="footer__logo-icon">⚡</span>
              <span className="footer__logo-text">Velocity<span className="gradient-text">LLM</span></span>
            </div>
            <p className="footer__desc">
              A dynamic batching engine for LLM inference serving.<br />
              Proven to raise throughput and guarantee latency SLAs.
            </p>
          </div>
          <div className="footer__links">
            <div className="footer__links-group">
              <h4 className="footer__links-title">Project</h4>
              <a href="https://github.com/kartikpagariya25/VelocityLLM" target="_blank" rel="noopener noreferrer">Source Code</a>
              <a href="https://github.com/kartikpagariya25/VelocityLLM/issues" target="_blank" rel="noopener noreferrer">Issue Tracker</a>
              <a href="#benchmarks">Benchmarks</a>
            </div>
          </div>
        </div>
        <div className="footer__bottom">
          <p className="footer__copy">
            Built as an industry-guided academic project under Dr. Viomesh K. Singh.
          </p>
          <p className="footer__license">MIT License</p>
        </div>
      </div>
    </footer>
  );
}
