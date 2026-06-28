import projects from '../../data/projects.js'
import publications from '../../data/publications.js'
import timeline from '../../data/timeline.js'
import hobbies from '../../data/hobbies.js'
import './ToolResult.css'

const skills = {
  Languages: ['Python', 'C', 'C++', 'Java', 'SQL', 'JavaScript', 'HTML/CSS', 'PHP'],
  Frameworks: ['React', 'Node.js', 'jQuery', 'Django', 'Express.js', 'Vue3'],
  Tools: ['Linux', 'Git', 'Docker', 'AWS', 'Jenkins', 'Jira', 'Bazel', 'Gerrit'],
  Databases: ['PostgreSQL', 'SQLite'],
  'HPC & Profiling': ['CUDA', 'OpenMP', 'MPI', 'Nsight Systems', 'LTTng', 'Valgrind'],
}

function ProjectsResult() {
  const featured = projects.filter(p =>
    ['gphy2', 'gemini', 'ai-video-detection', 'jetbrains-lca', 'file-system'].includes(p.id)
  )
  return (
    <div className="tool-result">
      <div className="tool-result-header">Projects</div>
      <div className="tr-projects">
        {featured.map(p => (
          <div key={p.id} className="tr-project-card">
            <div className="tr-project-top">
              <span className="tr-project-name">{p.name}</span>
              <span className="tr-project-date">{p.date}</span>
            </div>
            <p className="tr-project-desc">{p.description}</p>
            <div className="tr-tags">
              {p.tags.slice(0, 4).map(t => <span key={t} className="tr-tag">{t}</span>)}
            </div>
            {p.link && (
              <a href={p.link} target="_blank" rel="noreferrer" className="tr-link">
                View on GitHub →
              </a>
            )}
          </div>
        ))}
      </div>
      <a href="#/projects" className="tr-see-all">See all {projects.length} projects →</a>
    </div>
  )
}

function SkillsResult() {
  return (
    <div className="tool-result">
      <div className="tool-result-header">Skills</div>
      <div className="tr-skills">
        {Object.entries(skills).map(([cat, items]) => (
          <div key={cat} className="tr-skill-group">
            <span className="tr-skill-cat">{cat}</span>
            <div className="tr-skill-tags">
              {items.map(s => <span key={s} className="tr-tag">{s}</span>)}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ExperienceResult() {
  return (
    <div className="tool-result">
      <div className="tool-result-header">Work Experience</div>
      <div className="tr-exp-card">
        <div className="tr-exp-header">
          <div>
            <div className="tr-exp-role">Physical Layer (vDU) Developer Intern</div>
            <div className="tr-exp-company">Ericsson · Ottawa, Ontario</div>
          </div>
          <span className="tr-exp-date">May 2024 – Sep 2025</span>
        </div>
        <ul className="tr-exp-bullets">
          <li>
            <strong>Gphy2.0</strong> — Offloaded LDPC encoding from Intel accelerator to latest Nvidia GPU using CUDA;
            profiled and tuned kernels with Nsight Systems & Nsight Compute
          </li>
          <li>
            <strong>Gemini (5G Digital Twin)</strong> — Built Vue3 constellation diagram UI, multi-threaded CUDA unit
            tests under ~35 μs timing constraint, and SQLite + LTTng histogram pipeline for channel quality analysis
          </li>
        </ul>
        <div className="tr-tags" style={{ marginTop: 10 }}>
          {['CUDA', 'C++', 'Vue3', 'SQLite', 'LTTng', 'Nsight', 'Docker', 'Jenkins'].map(t => (
            <span key={t} className="tr-tag">{t}</span>
          ))}
        </div>
      </div>
    </div>
  )
}

function PublicationsResult() {
  return (
    <div className="tool-result">
      <div className="tool-result-header">Publications</div>
      <div className="tr-pubs">
        {publications.map(p => (
          <div key={p.id} className="tr-pub-card">
            <div className="tr-pub-title">{p.title}</div>
            <div className="tr-pub-authors">{p.authors}</div>
            <div className="tr-pub-venue">{p.venue} · {p.date}</div>
            <div className="tr-tags" style={{ marginTop: 8 }}>
              {p.tags.map(t => <span key={t} className="tr-tag">{t}</span>)}
            </div>
            {p.pdf && (
              <a href={p.pdf} target="_blank" rel="noreferrer" className="tr-link">
                Read PDF →
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function TimelineResult() {
  const typeIcon = { education: '🎓', work: '💼', project: '⚙️', publication: '📄' }
  return (
    <div className="tool-result">
      <div className="tool-result-header">Timeline</div>
      <div className="tr-timeline">
        {timeline.map((item, i) => (
          <div key={i} className="tr-timeline-item">
            <div className="tr-timeline-dot">{typeIcon[item.type] || '•'}</div>
            <div className="tr-timeline-content">
              <div className="tr-timeline-date">{item.date}</div>
              <div className="tr-timeline-title">{item.title}</div>
              <div className="tr-timeline-desc">{item.description}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ContactResult() {
  return (
    <div className="tool-result">
      <div className="tool-result-header">Contact</div>
      <div className="tr-contact">
        <div className="tr-contact-row">
          <span className="tr-contact-key">email</span>
          <a href="mailto:hongmao.wu@mail.utoronto.ca" className="tr-contact-val">
            hongmao.wu@mail.utoronto.ca
          </a>
        </div>
        <div className="tr-contact-row">
          <span className="tr-contact-key">linkedin</span>
          <a href="https://www.linkedin.com/in/hungmao-wu/" target="_blank" rel="noreferrer" className="tr-contact-val">
            linkedin.com/in/hungmao-wu
          </a>
        </div>
        <div className="tr-contact-row">
          <span className="tr-contact-key">github</span>
          <a href="https://github.com/wuhungmao" target="_blank" rel="noreferrer" className="tr-contact-val">
            github.com/wuhungmao
          </a>
        </div>
        <div className="tr-contact-row">
          <span className="tr-contact-key">resume</span>
          <a href="/academic-work/resume.pdf" target="_blank" rel="noreferrer" className="tr-contact-val">
            resume.pdf ↓
          </a>
        </div>
      </div>
    </div>
  )
}

function HobbiesResult() {
  return (
    <div className="tool-result">
      <div className="tool-result-header">Hobbies</div>
      <div className="tr-hobbies">
        {hobbies.map(h => (
          <div key={h.id} className="tr-hobby-card">
            <span className="tr-hobby-icon">{h.icon}</span>
            <div>
              <div className="tr-hobby-name">{h.name}</div>
              <div className="tr-hobby-summary">{h.summary}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ToolResult({ toolName }) {
  switch (toolName) {
    case 'getProjects':     return <ProjectsResult />
    case 'getSkills':       return <SkillsResult />
    case 'getExperience':   return <ExperienceResult />
    case 'getPublications': return <PublicationsResult />
    case 'getTimeline':     return <TimelineResult />
    case 'getContact':      return <ContactResult />
    case 'getHobbies':      return <HobbiesResult />
    default:                return null
  }
}
