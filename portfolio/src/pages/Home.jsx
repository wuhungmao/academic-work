import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import TerminalWindow, { Prompt } from '../components/TerminalWindow'
import SkillBars from '../components/SkillBars'
import GitHubActivity from '../components/GitHubActivity'
import InteractiveTerminal from '../components/InteractiveTerminal'
import './Home.css'

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
      {/* ── Main grid ────────────────────────────────────── */}
      <div className="home-grid">
        <div className="home-left">
          <TerminalWindow title="wu@portfolio: ~">
            <Prompt path="~" command="whoami" />
            {step >= 0 && (
              <div className="tw-output">
                <TypeWriter text="Marvin Wu (Wu Hung Mao)" speed={50} onDone={() => setStep(1)} />
              </div>
            )}

            {step >= 1 && <Prompt path="~" command="cat about.txt" />}
            {step >= 1 && (
              <div className="tw-output">
                <TypeWriter
                  text="CS Specialist @ UofT Mississauga | Former Physical Layer (vDU) Developer Intern @ Ericsson | C++, CUDA & Machine Learning"
                  speed={18}
                  onDone={() => setStep(2)}
                />
              </div>
            )}

            {step >= 2 && <Prompt path="~" command="cat bio.txt" />}
            {step >= 2 && (
              <div className="tw-output bio-text">
                <p>
                  Honours BSc Computer Science Specialist at the University of Toronto Mississauga
                  (GPA 3.45/4.0, graduating August 2026 with PEY CO-OP).
                </p>
                <p style={{ marginTop: 8 }}>
                  Most recently a <strong>Physical Layer (vDU) Developer Intern at Ericsson</strong> (May 2024 – Sep 2025),
                  where I offloaded LDPC encoding onto Nvidia GPUs using CUDA, profiled latency with Nsight Systems and
                  Nsight Compute, and built a digital twin for Radio/RF/UE simulation including a constellation diagram
                  (Vue3), multi-threaded CUDA unit tests, and an SQLite + LTTng channel quality analysis pipeline.
                </p>
                <p style={{ marginTop: 8 }}>
                  I work across the full stack — from CUDA kernels and OS-level systems programming to React frontends
                  and REST APIs. Currently focused on AI/ML: led a 5-person team building an ensemble deep learning
                  system for AI-generated video detection (EfficientNet, MesoNet, XceptionNet, AASIST), and contributed
                  to the JetBrains LCA benchmark for evaluating coding agents like Claude Code and Codex.
                </p>
              </div>
            )}

            {step >= 2 && <Prompt path="~" command="cat experience.txt" />}
            {step >= 2 && (
              <div className="experience-block">
                <div className="exp-item">
                  <div className="exp-header">
                    <span className="exp-role">Physical Layer (vDU) Developer Intern</span>
                    <span className="exp-date">May 2024 – Sep 2025</span>
                  </div>
                  <div className="exp-company">Ericsson · Ottawa, Ontario</div>
                  <ul className="exp-bullets">
                    <li>Offloaded LDPC encoding onto latest Nvidia GPU (Gphy2.0) using CUDA; profiled with Nsight Systems & Compute</li>
                    <li>Built digital twin (Gemini) for Radio, RF Channel & UE simulation using Vue3 and the Ericsson Design System</li>
                    <li>Wrote multi-threaded CUDA unit tests for a real-time system under ~35 μs timing constraint</li>
                    <li>Developed SQLite + LTTng histogram pipeline for channel quality analysis</li>
                  </ul>
                </div>
              </div>
            )}

            {step >= 2 && <Prompt path="~" command="ls skills/" />}
            {step >= 2 && <SkillBars />}

            {step >= 2 && <Prompt path="~" command="cat github-activity.log" />}
            {step >= 2 && <GitHubActivity username="wuhungmao" />}

            {step >= 2 && <Prompt path="~" command="cat contact.json" />}
            {step >= 2 && (
              <div className="contact-block">
                <span className="json-brace">{'{'}</span>
                <div className="json-row"><span className="json-key">"email"</span><span className="json-colon">:</span> <a href="mailto:hongmao.wu@mail.utoronto.ca" className="json-val">"hongmao.wu@mail.utoronto.ca"</a></div>
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
                <div className="quick-desc">20+ projects across AI, systems & web</div>
              </div>
              <span className="quick-arrow">→</span>
            </Link>
            <Link to="/publications" className="quick-card">
              <span className="quick-icon">📄</span>
              <div>
                <div className="quick-name">Publications</div>
                <div className="quick-desc">2 co-authored research papers</div>
              </div>
              <span className="quick-arrow">→</span>
            </Link>
            <Link to="/timeline" className="quick-card">
              <span className="quick-icon">📅</span>
              <div>
                <div className="quick-name">Timeline</div>
                <div className="quick-desc">Education, internship & milestones</div>
              </div>
              <span className="quick-arrow">→</span>
            </Link>
            <Link to="/hobbies" className="quick-card">
              <span className="quick-icon">🎯</span>
              <div>
                <div className="quick-name">Hobbies</div>
                <div className="quick-desc">Hiking, gaming, cooking & more</div>
              </div>
              <span className="quick-arrow">→</span>
            </Link>
            <a
              href="/academic-work/resume.pdf"
              target="_blank"
              rel="noreferrer"
              className="quick-card resume-btn"
            >
              <span className="quick-icon">⬇</span>
              <div>
                <div className="quick-name">Download CV</div>
                <div className="quick-desc">resume.pdf</div>
              </div>
            </a>
          </div>
        </div>
      </div>

      {/* ── Interactive terminal ──────────────────────────── */}
      <div className="home-terminal-section">
        <TerminalWindow title="wu@portfolio: ~ (interactive shell)">
          <Prompt path="~" command="bash --interactive" />
          <InteractiveTerminal />
        </TerminalWindow>
      </div>
    </main>
  )
}
