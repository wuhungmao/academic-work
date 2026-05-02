import timeline from '../data/timeline'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import './Timeline.css'

const typeConfig = {
  education:   { icon: '🎓', color: '#79c0ff', label: 'Education' },
  work:        { icon: '💼', color: '#f0f6fc', label: 'Work' },
  project:     { icon: '⚙️',  color: '#c9d1d9', label: 'Project' },
  publication: { icon: '📄', color: '#e3b341', label: 'Publication' },
}

export default function Timeline() {
  return (
    <main className="timeline-page">
      <TerminalWindow title="wu@portfolio: ~/timeline">
        <Prompt path="~/timeline" command="git log --oneline --all" />

        <div className="legend">
          {Object.entries(typeConfig).map(([key, { icon, color, label }]) => (
            <span key={key} className="legend-item">
              <span className="legend-dot" style={{ background: color }} />
              {icon} {label}
            </span>
          ))}
        </div>

        <div className="timeline">
          {timeline.map((item, i) => {
            const cfg = typeConfig[item.type]
            return (
              <div key={i} className={`tl-entry ${i % 2 === 0 ? 'left' : 'right'}`}>
                <div className="tl-content" style={{ borderColor: cfg.color }}>
                  <div className="tl-date">{item.date}</div>
                  <div className="tl-icon">{cfg.icon}</div>
                  <h3 className="tl-title" style={{ color: cfg.color }}>{item.title}</h3>
                  <p className="tl-desc">{item.description}</p>
                </div>
                <div className="tl-dot" style={{ background: cfg.color }} />
              </div>
            )
          })}
          <div className="tl-line" />
        </div>
      </TerminalWindow>
    </main>
  )
}
