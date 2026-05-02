import { useParams, Link, Navigate } from 'react-router-dom'
import hobbies from '../data/hobbies'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import PhotoAlbum from '../components/PhotoAlbum'
import './HobbyPage.css'

const BASE = '/academic-work/images'

export default function HobbyPage() {
  const { id } = useParams()
  const hobby = hobbies.find(h => h.id === id)

  if (!hobby) return <Navigate to="/hobbies" replace />

  const coverSrc = hobby.coverImage ? `${BASE}/${hobby.coverImage}` : null

  return (
    <main className="hobby-page">
      <Link to="/hobbies" className="back-link">← cd ../hobbies</Link>
      <TerminalWindow title={`wu@portfolio: ~/hobbies/${id}`}>
        <Prompt path="~/hobbies" command={`cat ${id}.md`} />

        <div className="hobby-content">
          <div className="hobby-header">
            <span className="hobby-icon-lg">{hobby.icon}</span>
            <h1 className="hobby-title">{hobby.name}</h1>
          </div>

          {coverSrc && (
            <img src={coverSrc} alt={hobby.name} className="hobby-cover" />
          )}

          <p className="hobby-description">{hobby.description}</p>

          {hobby.games && hobby.games.length > 0 && (
            <div className="games-list">
              <p className="games-label"># favourite games</p>
              {hobby.games.map(g => (
                <div key={g.title} className="game-row">
                  <span className="game-title">{g.title}</span>
                  <span className="game-genre">{g.genre}</span>
                  {g.studio && <span className="game-studio">{g.studio}</span>}
                </div>
              ))}
            </div>
          )}

          <PhotoAlbum photos={hobby.photos} />
        </div>
      </TerminalWindow>
    </main>
  )
}
