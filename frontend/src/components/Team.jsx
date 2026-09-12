import SectionWrapper, { MotionDiv } from './SectionWrapper';
import './Team.css';

const teamMembers = [
  {
    name: 'Kartik R. Pagariya',
    role: 'Scheduler architecture, infra, benchmarking',
    github: 'https://github.com/kartikpagariya25',
    linkedin: 'https://www.linkedin.com/in/kartikpagariya1911/',
    photo: 'https://media.licdn.com/dms/image/v2/D4E03AQHfIoFibBhVIA/profile-displayphoto-crop_800_800/B4EZqwcW3eKUAI-/0/1763896816275?e=1790208000&v=beta&t=qhM4SWwE-0WsnL9jFS8wPskQwHB_UivH7tPznjbzEDU'
  },
  {
    name: 'Vikrant K. Kadam',
    role: 'Core scheduler engine implementation (Phase 2 & 3)',
    github: 'https://github.com/VikrantKadam028',
    linkedin: 'https://linkedin.com/in/vikrantkadam028/',
    photo: 'https://media.licdn.com/dms/image/v2/D4D03AQESIR9c5L1XLA/profile-displayphoto-scale_400_400/B4DZ8E_5OrGwAk-/0/1782495287789?e=1790208000&v=beta&t=gWOA3iGPYT_C_8lEnUVOtZKKJ4c1XWFSx081JJhpzMk'
  },
  {
    name: 'Pranali D. Yelavikar',
    role: 'Testing & robustness validation',
    github: 'https://github.com/pranaliyelavikar14',
    linkedin: 'https://www.linkedin.com/in/pranali-yelavikar-2b3178383/',
    photo: 'https://media.licdn.com/dms/image/v2/D4D03AQGPWm01WRkKtw/profile-displayphoto-crop_800_800/B4DZ9fRdCqJwAM-/0/1784009840442?e=1790208000&v=beta&t=YwO7HJSxg9IFq6xJLJxRXCbbBmGdkC82PCmapKP_23I'
  },
  {
    name: 'Aditya D. Dengale',
    role: 'Documentation & benchmarking support',
    github: 'https://github.com/DevXDividends',
    linkedin: 'https://www.linkedin.com/in/adityadengale/',
    photo: 'https://media.licdn.com/dms/image/v2/D4D03AQEn0yj5zoQwHg/profile-displayphoto-crop_800_800/B4DZjHJmYJH0AM-/0/1755687840857?e=1790208000&v=beta&t=PuSgkFXGYMNt2eb6GCukzn8AR1XSTAHT_V_wdap0-YQ'
  }
];

export default function Team() {
  return (
    <SectionWrapper id="team" className="team">
      <div className="container">
        <MotionDiv className="team__header">
          <div className="section-label">The Team</div>
          <h2 className="section-title">Built by engineers.</h2>
          <p className="section-subtitle">
            An industry-guided academic project mentored by <strong>Dr. Viomesh K. Singh</strong>.
          </p>
        </MotionDiv>

        <div className="team__grid">
          {teamMembers.map((member, i) => (
            <MotionDiv key={member.name} className="team__card glass-card" delay={i * 0.1}>
              <div className="team__photo-wrapper">
                <img src={member.photo} alt={member.name} className="team__photo" loading="lazy" />
                <div className="team__photo-overlay" />
              </div>
              <div className="team__info">
                <h3 className="team__name">{member.name}</h3>
                <p className="team__role">{member.role}</p>
                <div className="team__links">
                  <a href={member.github} target="_blank" rel="noopener noreferrer" className="team__link" aria-label={`${member.name}'s GitHub`}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                  </a>
                  <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="team__link" aria-label={`${member.name}'s LinkedIn`}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
                    </svg>
                  </a>
                </div>
              </div>
            </MotionDiv>
          ))}
        </div>
      </div>
    </SectionWrapper>
  );
}
