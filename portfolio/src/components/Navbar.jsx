import { NavLink, useLocation } from 'react-router-dom'
import './Navbar.css'

const links = [
  { to: '/', label: '~/home' },
  { to: '/timeline', label: '~/timeline' },
  { to: '/courses', label: '~/courses' },
  { to: '/projects', label: '~/projects' },
  { to: '/publications', label: '~/publications' },
  { to: '/hobbies', label: '~/hobbies' },
]

export default function Navbar() {
  const location = useLocation()

  return (
    <nav className="navbar">
      <span className="navbar-prompt">wuhungmao@portfolio</span>
      <span className="navbar-sep">:</span>
      <div className="navbar-links">
        {links.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}
          >
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
