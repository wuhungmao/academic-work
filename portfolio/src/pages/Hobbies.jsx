import { Link } from 'react-router-dom'
import hobbies from '../data/hobbies'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import './Hobbies.css'

export default function Hobbies() {
  return (
    <main className="hobbies-page">
      <TerminalWindow title="wu@portfolio: ~/hobbies">
        <Prompt path="~/hobbies" command="ls -la" />
        <div className="hobbies-grid">
          {hobbies.map(h => (
            <Link key={h.id} to={`/hobbies/${h.id}`} className="hobby-card">
              <span className="hobby-icon">{h.icon}</span>
              <div>
                <div className="hobby-name">{h.name}</div>
                <div className="hobby-summary">{h.summary}</div>
              </div>
              <span className="hobby-arrow">→</span>
            </Link>
          ))}
        </div>
      </TerminalWindow>
    </main>
  )
}
