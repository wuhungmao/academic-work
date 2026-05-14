import { useEffect, useState } from 'react'
import './GitHubActivity.css'

const LEVELS = ['#161b22', '#0e4429', '#006d32', '#26a641', '#39d353']

function getTooltip(date, count) {
  if (count === 0) return `No contributions on ${date}`
  return `${count} contribution${count > 1 ? 's' : ''} on ${date}`
}

export default function GitHubActivity({ username = 'wuhungmao' }) {
  const [weeks, setWeeks] = useState([])
  const [total, setTotal] = useState(null)
  const [error, setError] = useState(false)
  const [tooltip, setTooltip] = useState(null)

  useEffect(() => {
    fetch(`https://github-contributions-api.jogruber.de/v4/${username}?y=last`)
      .then(r => r.json())
      .then(data => {
        const days = data.contributions
        const yearTotal = days.reduce((s, d) => s + d.count, 0)
        setTotal(yearTotal)
        // group into weeks
        const w = []
        for (let i = 0; i < days.length; i += 7) w.push(days.slice(i, i + 7))
        setWeeks(w)
      })
      .catch(() => setError(true))
  }, [username])

  if (error) return null

  return (
    <div className="gh-activity">
      <div className="gh-header">
        <span className="gh-title">$ cat github-activity.log</span>
        {total !== null && (
          <span className="gh-total">{total} contributions in the last year</span>
        )}
      </div>

      <div className="gh-grid-wrap">
        {weeks.length === 0 ? (
          <div className="gh-loading">loading contribution data...</div>
        ) : (
          <div className="gh-grid">
            {weeks.map((week, wi) => (
              <div key={wi} className="gh-week">
                {week.map((day, di) => (
                  <div
                    key={di}
                    className="gh-day"
                    style={{ background: LEVELS[day.level] }}
                    onMouseEnter={e => setTooltip({ text: getTooltip(day.date, day.count), x: e.clientX, y: e.clientY })}
                    onMouseLeave={() => setTooltip(null)}
                  />
                ))}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="gh-legend">
        <span className="gh-legend-label">Less</span>
        {LEVELS.map((c, i) => <div key={i} className="gh-legend-box" style={{ background: c }} />)}
        <span className="gh-legend-label">More</span>
      </div>

      {tooltip && (
        <div
          className="gh-tooltip"
          style={{ left: tooltip.x + 12, top: tooltip.y - 36, position: 'fixed' }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  )
}
