import { useParams, Link, Navigate } from 'react-router-dom'
import courses from '../data/courses'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import './CoursePage.css'

export default function CoursePage() {
  const { id } = useParams()
  const course = courses.find(c => c.id === id)

  if (!course) return <Navigate to="/courses" replace />

  const gradeColor = course.grade >= 80 ? '#39d353' : course.grade >= 70 ? '#e3b341' : '#f85149'
  const diffColor = course.difficulty >= 7 ? '#f85149' : course.difficulty >= 5 ? '#e3b341' : '#39d353'

  return (
    <main className="course-page">
      <Link to="/courses" className="back-link">← cd ../courses</Link>
      <TerminalWindow title={`wu@portfolio: ~/courses/${id}`}>
        <Prompt path={`~/courses`} command={`cat ${id}.md`} />

        <div className="course-content">
          <div className="course-header">
            <h1 className="course-name">{course.name}</h1>
            <p className="course-title">{course.title}</p>
          </div>

          <div className="course-stats">
            <div className="stat-box">
              <span className="stat-label">// grade</span>
              <span className="stat-value" style={{ color: gradeColor }}>{course.grade}/100</span>
            </div>
            <div className="stat-box">
              <span className="stat-label">// difficulty</span>
              <span className="stat-value" style={{ color: diffColor }}>{course.difficulty}/10</span>
            </div>
          </div>

          <div className="course-reflection">
            <p className="reflect-label">/* reflection */</p>
            <p className="reflect-text">{course.reflection}</p>
          </div>
        </div>
      </TerminalWindow>
    </main>
  )
}
