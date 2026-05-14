import useInView from '../hooks/useInView'
import './SkillBars.css'

const featured = [
  { name: 'C / C++ / CUDA',      level: 88 },
  { name: 'Python / ML',         level: 82 },
  { name: 'React / Web',         level: 78 },
  { name: 'Linux / Systems',     level: 85 },
  { name: 'Java',                level: 72 },
  { name: 'SQL / Databases',     level: 68 },
  { name: 'Docker / AWS / CI',   level: 70 },
  { name: 'OpenMP / MPI',        level: 75 },
]

function Bar({ name, level, delay }) {
  const [ref, inView] = useInView()
  return (
    <div ref={ref} className="skill-bar-row">
      <div className="skill-bar-label">
        <span className="skill-bar-name">{name}</span>
        <span className="skill-bar-pct">{level}%</span>
      </div>
      <div className="skill-bar-track">
        <div
          className="skill-bar-fill"
          style={{
            width: inView ? `${level}%` : '0%',
            transitionDelay: `${delay}ms`,
          }}
        />
      </div>
    </div>
  )
}

export default function SkillBars() {
  return (
    <div className="skill-bars-section">
      {featured.map((s, i) => (
        <Bar key={s.name} name={s.name} level={s.level} delay={i * 80} />
      ))}
    </div>
  )
}
