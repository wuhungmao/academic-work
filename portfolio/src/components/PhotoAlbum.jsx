import { useState } from 'react'
import './PhotoAlbum.css'

const BASE = '/academic-work/Personal%20website/Hobbies%20pics'

export default function PhotoAlbum({ photos }) {
  const [open, setOpen] = useState(false)
  const [lightbox, setLightbox] = useState(null)

  if (!photos || photos.length === 0) return null

  return (
    <div className="album">
      <button className="album-toggle" onClick={() => setOpen(o => !o)}>
        <span className="album-toggle-icon">{open ? '▼' : '▶'}</span>
        {open ? 'Close Album' : `Open Album (${photos.length} photos)`}
      </button>

      {open && (
        <div className="album-grid">
          {photos.map((photo, i) => (
            <div key={i} className="album-photo" onClick={() => setLightbox(photo)}>
              <img
                src={`${BASE}/${encodeURIComponent(photo).replace(/%2F/g, '/')}`}
                alt={`Photo ${i + 1}`}
                loading="lazy"
              />
            </div>
          ))}
        </div>
      )}

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <img
            src={`${BASE}/${encodeURIComponent(lightbox).replace(/%2F/g, '/')}`}
            alt="Full size"
          />
          <button className="lightbox-close">✕</button>
        </div>
      )}
    </div>
  )
}
