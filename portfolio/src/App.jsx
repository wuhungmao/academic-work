import { HashRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Courses from './pages/Courses'
import CoursePage from './pages/CoursePage'
import Projects from './pages/Projects'
import Hobbies from './pages/Hobbies'
import HobbyPage from './pages/HobbyPage'

export default function App() {
  return (
    <HashRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/courses" element={<Courses />} />
        <Route path="/courses/:id" element={<CoursePage />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/hobbies" element={<Hobbies />} />
        <Route path="/hobbies/:id" element={<HobbyPage />} />
      </Routes>
    </HashRouter>
  )
}
