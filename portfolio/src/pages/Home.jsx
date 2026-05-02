import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import './Home.css'

const skills = ['Python', 'Java', 'C', 'C++', 'JavaScript', 'React', 'HTML/CSS', 'SQL', 'Git', 'CUDA', 'OpenMP', 'MPI', 'MIPS Assembly', 'Shell Scripting']

function TypeWriter({ text, speed = 40, onDone }) {
  const [displayed, setDisplayed] = useState('')

  useEffect(() => {
    let i = 0
    const timer = setInterval(() => {
      setDisplayed(text.slice(0, i + 1))
      i++
      if (i >= text.length) {
        clearInterval(timer)
        onDone?.()
      }
    }, speed)
    return () => clearInterval(timer)
  }, [text])

  return <span>{displayed}<span className="cursor">▋</span></span>
}

export default function Home() {
  const [step, setStep] = useState(0)

  return (
    <main className="home">
      <div className="home-grid">
        <div className="home-left">
          <TerminalWindow title="wu@portfolio: ~">
            <Prompt path="~" command="whoami" />
            {step >= 0 && (
              <div className="tw-output">
                <TypeWriter text="Wu Hung Mao" speed={60} onDone={() => setStep(1)} />
              </div>
            )}

            {step >= 1 && <Prompt path="~" command="cat about.txt" />}
            {step >= 1 && (
              <div className="tw-output">
                <TypeWriter
                  text="Software Developer | University of Toronto Mississauga"
                  speed={25}
                  onDone={() => setStep(2)}
                />
              </div>
            )}

            {step >= 2 && <Prompt path="~" command="cat bio.txt" />}
            {step >= 2 && (
              <div className="tw-output bio-text">
                <p>I'm a Computer Science student at UTM, admitted in 2021. My interests span systems programming, parallel computing, AI, and robotics.</p>
                <p style={{ marginTop: 8 }}>I use AI tools like ChatGPT and Copilot daily — but I've learned that blindly trusting them creates bugs nobody catches. A good developer knows when to trust the tool and when to question it.</p>
              </div>
            )}

            {step >= 2 && <Prompt path="~" command="ls skills/" />}
            {step >= 2 && (
              <div className="skills-grid">
                {skills.map(s => (
                  <span key={s} className="skill-tag">{s}</span>
                ))}
              </div>
            )}

            {step >= 2 && <Prompt path="~" command="cat contact.json" />}
            {step >= 2 && (
              <div className="contact-block">
                <span className="json-brace">{'{'}</span>
                <div className="json-row"><span className="json-key">"email"</span><span className="json-colon">:</span> <a href="mailto:wuhungmao.marvinwu@gmail.com" className="json-val">"wuhungmao.marvinwu@gmail.com"</a></div>
                <div className="json-row"><span className="json-key">"linkedin"</span><span className="json-colon">:</span> <a href="https://www.linkedin.com/in/hungmao-wu/" target="_blank" rel="noreferrer" className="json-val">"linkedin.com/in/hungmao-wu"</a></div>
                <div className="json-row"><span className="json-key">"github"</span><span className="json-colon">:</span> <a href="https://github.com/wuhungmao" target="_blank" rel="noreferrer" className="json-val">"github.com/wuhungmao"</a></div>
                <span className="json-brace">{'}'}</span>
              </div>
            )}
          </TerminalWindow>
        </div>

        <div className="home-right">
          <div className="quick-nav">
            <p className="quick-nav-title"># quick navigation</p>
            <Link to="/courses" className="quick-card">
              <span className="quick-icon">📚</span>
              <div>
                <div className="quick-name">Courses</div>
                <div className="quick-desc">15 courses reviewed</div>
              </div>
              <span className="quick-arrow">→</span>
            </Link>
            <Link to="/projects" className="quick-card">
              <span className="quick-icon">⚙️</span>
              <div>
                <div className="quick-name">Projects</div>
                <div className="quick-desc">4 school projects</div>
              </div>
              <span className="quick-arrow">→</span>
            </Link>
            <Link to="/hobbies" className="quick-card">
              <span className="quick-icon">🎯</span>
              <div>
                <div className="quick-name">Hobbies</div>
                <div className="quick-desc">Hiking, military, cooking & more</div>
              </div>
              <span className="quick-arrow">→</span>
            </Link>
          </div>
        </div>
      </div>
    </main>
  )
}
