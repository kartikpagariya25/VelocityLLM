import SectionWrapper, { MotionDiv } from './SectionWrapper';
import './TechStack.css';

const techStack = [
  { name: 'Python', icon: '🐍' },
  { name: 'vLLM', icon: '⚡' },
  { name: 'FastAPI', icon: '🚀' },
  { name: 'CUDA', icon: '🟩' },
  { name: 'PagedAttention', icon: '🧠' },
  { name: 'AIMD Control', icon: '⚙️' },
];

export default function TechStack() {
  return (
    <SectionWrapper id="tech-stack" className="tech-stack">
      <div className="container">
        <MotionDiv className="tech-stack__header">
          <h2 className="tech-stack__title">Powered by modern infra.</h2>
        </MotionDiv>
        <MotionDiv className="tech-stack__badges" delay={0.2}>
          {techStack.map((tech) => (
            <div key={tech.name} className="tech-stack__badge glass-card">
              <span className="tech-stack__icon">{tech.icon}</span>
              <span className="tech-stack__name">{tech.name}</span>
            </div>
          ))}
        </MotionDiv>
      </div>
    </SectionWrapper>
  );
}
