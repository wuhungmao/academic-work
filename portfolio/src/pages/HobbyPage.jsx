import { useParams, Link, Navigate } from 'react-router-dom'
import hobbies from '../data/hobbies'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import PhotoAlbum from '../components/PhotoAlbum'
import './HobbyPage.css'

const BASE = '/academic-work/Personal%20website/Hobbies%20pics'

export default function HobbyPage() {
  const { id } = useParams()
  const hobby = hobbies.find(h => h.id === id)

  if (!hobby) return <Navigate to="/hobbies" replace />

  const coverSrc = hobby.coverImage
    ? `${BASE}/${encodeURIComponent(hobby.coverImage).replace(/%2F/g, '/')}`
    : null

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

          {hobby.youtubeEmbed && (
            <iframe
              className="hobby-video"
              src={hobby.youtubeEmbed}
              title="YouTube video"
              frameBorder="0"
              allowFullScreen
            />
          )}

          <p className="hobby-description">{hobby.description}</p>

          <PhotoAlbum photos={hobby.photos} />
        </div>
      </TerminalWindow>
    </main>
  )
}
