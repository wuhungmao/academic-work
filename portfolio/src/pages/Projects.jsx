import projects from '../data/projects'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import './Projects.css'

export default function Projects() {
  return (
    <main className="projects-page">
      <TerminalWindow title="wu@portfolio: ~/projects">
        <Prompt path="~/projects" command="ls -la" />
        <div className="projects-grid">
          {projects.map(p => (
            <div key={p.id} className="project-card">
              <div className="project-header">
                <span className="project-icon">⚙</span>
                <div>
                  <h2 className="project-name">{p.name}</h2>
                  <span className="project-course">{p.course}</span>
                </div>
              </div>
              <p className="project-desc">{p.description}</p>
              <div className="project-tags">
                {p.tags.map(t => (
                  <span key={t} className="project-tag">{t}</span>
                ))}
              </div>
              <a href={p.pdf} target="_blank" rel="noreferrer" className="project-link">
                cat report.pdf →
              </a>
            </div>
          ))}
        </div>
      </TerminalWindow>
    </main>
  )
}
