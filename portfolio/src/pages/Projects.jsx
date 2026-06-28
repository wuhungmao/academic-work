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
                <div>
                  <h2 className="project-name">{p.name}</h2>
                  <div className="project-meta">
                    <span className="project-course">{p.category}</span>
                    <span className="project-date">{p.date}</span>
                  </div>
                </div>
              </div>
              <p className="project-desc">{p.description}</p>
              <div className="project-tags">
                {p.tags.map(t => <span key={t} className="project-tag">{t}</span>)}
              </div>
              <div className="project-links">
                {p.link && (
                  <a href={p.link} target="_blank" rel="noreferrer" className="project-link">
                    view code →
                  </a>
                )}
                {p.pdf && (
                  <a href={p.pdf} target="_blank" rel="noreferrer" className="project-link">
                    read report →
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
