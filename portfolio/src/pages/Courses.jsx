import { Link } from 'react-router-dom'
import courses from '../data/courses'
import TerminalWindow from '../components/TerminalWindow'
import './Courses.css'

function DifficultyBar({ value }) {
  if (value === null) return <span className="ls-na">—</span>
  const pct = (value / 10) * 100
  const color = value >= 7 ? '#f85149' : value >= 5 ? '#e3b341' : '#c9d1d9'
  return (
    <div className="diff-bar-wrap">
      <div className="diff-bar">
        <div className="diff-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="diff-label" style={{ color }}>{value}/10</span>
    </div>
  )
}

export default function Courses() {
  return (
    <main className="courses-page">
      <TerminalWindow title="wu@portfolio: ~/courses">
        <div className="ls-header">
          <span className="ls-col">name</span>
          <span className="ls-col">title</span>
          <span className="ls-col">grade</span>
          <span className="ls-col">difficulty</span>
        </div>
        <div className="ls-list">
          {courses.map(course => (
            <Link key={course.id} to={`/courses/${course.id}`} className="ls-row">
              <span className="ls-name">{course.name}</span>
              <span className="ls-title">{course.title}</span>
              <span className="ls-grade">{course.grade !== null ? `${course.grade}%` : '—'}</span>
              <DifficultyBar value={course.difficulty} />
            </Link>
          ))}
        </div>
      </TerminalWindow>
    </main>
  )
}
