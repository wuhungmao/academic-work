import publications from '../data/publications'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import './Publications.css'

const statusLabel = { unpublished: 'Unpublished Manuscript', published: 'Published' }
const statusColor = { unpublished: '#e3b341', published: '#39d353' }

export default function Publications() {
  return (
    <main className="pubs-page">
      <TerminalWindow title="wu@portfolio: ~/publications">
        <Prompt path="~/publications" command="ls -la *.pdf" />
        <div className="pubs-list">
          {publications.map((p, i) => (
            <div key={p.id} className="pub-card">
              <div className="pub-index">[{i + 1}]</div>
              <div className="pub-body">
                <h2 className="pub-title">"{p.title}"</h2>
                <p className="pub-authors">{p.authors}</p>
                <p className="pub-venue">{p.venue} · {p.date}</p>
                <span
                  className="pub-status"
                  style={{ borderColor: statusColor[p.status], color: statusColor[p.status] }}
                >
                  {statusLabel[p.status]}
                </span>
                <p className="pub-desc">{p.description}</p>
                <div className="pub-tags">
                  {p.tags.map(t => <span key={t} className="pub-tag">{t}</span>)}
                </div>
                <a href={p.pdf} target="_blank" rel="noreferrer" className="pub-link">
                  read paper →
                </a>
              </div>
            </div>
          ))}
        </div>
      </TerminalWindow>
    </main>
  )
}
