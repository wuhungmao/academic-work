import { useState } from 'react'
import projects, { categories } from '../data/projects'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import './Projects.css'

export default function Projects() {
  const [active, setActive] = useState('All')

  const filtered = active === 'All' ? projects : projects.filter(p => p.category === active)

  return (
    <main className="projects-page">
      <TerminalWindow title="wu@portfolio: ~/projects">
        <Prompt path="~/projects" command={`ls ${active === 'All' ? '' : `--filter="${active}"`}`} />

        <div className="filter-bar">
          {categories.map(cat => (
            <button
              key={cat}
              className={`filter-btn ${active === cat ? 'active' : ''}`}
              onClick={() => setActive(cat)}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="projects-grid">
          {filtered.map(p => (
            <div key={p.id} className="project-card">
              <div className="project-header">
                <h2 className="project-name">
                  {p.link ? (
                    <a href={p.link} target="_blank" rel="noreferrer" className="project-name-link">
                      {p.name}
                    </a>
                  ) : p.name}
                </h2>
                <div className="project-meta">
                  <span className="project-course">{p.category}</span>
                  <span className="project-date">{p.date}</span>
                </div>
              </div>

              <div className="project-body">
                <div className="project-ps-row">
                  <span className="ps-label problem-label">Problem</span>
                  <p className="ps-text">{p.problem}</p>
                </div>
                <div className="project-ps-row">
                  <span className="ps-label solution-label">Solution</span>
                  <p className="ps-text">{p.solution}</p>
                </div>
              </div>

              <div className="project-tags">
                {p.tags.map(t => <span key={t} className="project-tag">{t}</span>)}
              </div>

              <div className="project-links">
                {p.link && (
                  <a href={p.link} target="_blank" rel="noreferrer" className="project-link code-link">
                    ⎇ view code
                  </a>
                )}
                {p.pdf && (
                  <a href={p.pdf} target="_blank" rel="noreferrer" className="project-link paper-link">
                    ↗ read paper
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      </TerminalWindow>
    </main>
  )
}
