import { HashRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Chat from './pages/Chat'
import Courses from './pages/Courses'
import CoursePage from './pages/CoursePage'
import Projects from './pages/Projects'
import Hobbies from './pages/Hobbies'
import HobbyPage from './pages/HobbyPage'
import Publications from './pages/Publications'
import Timeline from './pages/Timeline'

export default function App() {
  return (
    <HashRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/courses" element={<Courses />} />
        <Route path="/courses/:id" element={<CoursePage />} />
        <Route path="/projects" element={<Projects />} />
        <Route path="/hobbies" element={<Hobbies />} />
        <Route path="/hobbies/:id" element={<HobbyPage />} />
        <Route path="/publications" element={<Publications />} />
        <Route path="/timeline" element={<Timeline />} />
      </Routes>
    </HashRouter>
  )
}
