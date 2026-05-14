import { useState, useRef, useEffect } from 'react'
import projects from '../data/projects'
import './InteractiveTerminal.css'

const HELP_TEXT = `Available commands:
  whoami          Who am I?
  cat bio.txt     Full biography
  cat about.txt   One-line summary
  cat contact.json Contact information
  ls projects/    List all projects
  ls skills/      List all skills
  ssh ericsson    Ericsson internship details
  git log         Recent project history
  pwd             Current directory
  date            Current date & time
  clear           Clear terminal
  sudo hire-me    ¯\\_(ツ)_/¯`

const SKILLS = {
  Languages:     ['Python', 'C', 'C++', 'Java', 'SQL', 'JavaScript', 'HTML/CSS', 'PHP'],
  Frameworks:    ['React', 'Node.js', 'jQuery', 'Django', 'Express.js', 'Vue3'],
  Tools:         ['Linux', 'Git', 'Docker', 'AWS', 'Jenkins', 'Jira', 'Bazel', 'Gerrit'],
  Databases:     ['PostgreSQL', 'SQLite'],
  'HPC & GPU':   ['CUDA', 'OpenMP', 'MPI', 'Nsight Systems', 'LTTng', 'Valgrind'],
}

function processCommand(raw) {
  const cmd = raw.trim().toLowerCase()

  if (cmd === 'help') return HELP_TEXT

  if (cmd === 'whoami')
    return 'Marvin Wu (Wu Hung Mao)\nCS Specialist @ University of Toronto Mississauga\nFormer Physical Layer (vDU) Developer Intern @ Ericsson'

  if (cmd === 'pwd')
    return '/home/wuhungmao/portfolio'

  if (cmd === 'date')
    return new Date().toString()

  if (cmd === 'cat about.txt')
    return 'CS Specialist @ UofT Mississauga | Former Physical Layer (vDU) Developer Intern @ Ericsson | C++, CUDA & Machine Learning'

  if (cmd === 'cat bio.txt')
    return `Honours BSc Computer Science Specialist at UofT Mississauga (GPA 3.45/4.0, graduating Aug 2026).

Most recently a Physical Layer (vDU) Developer Intern at Ericsson (May 2024 – Sep 2025), where I offloaded LDPC encoding onto Nvidia GPUs using CUDA, profiled with Nsight Systems & Compute, and built a digital twin for Radio/RF/UE simulation.

I work across the full stack — from CUDA kernels and systems programming to React frontends. Currently focused on AI/ML: led a 5-person team building an ensemble deep learning system for AI-generated video detection.`

  if (cmd === 'cat contact.json')
    return `{
  "email":    "hongmao.wu@mail.utoronto.ca",
  "linkedin": "linkedin.com/in/hungmao-wu",
  "github":   "github.com/wuhungmao",
  "website":  "wuhungmao.github.io/academic-work"
}`

  if (cmd === 'ls projects/' || cmd === 'ls projects')
    return projects.map((p, i) => `  [${String(i + 1).padStart(2, '0')}] ${p.name}  (${p.category})`).join('\n')

  if (cmd === 'ls skills/' || cmd === 'ls skills')
    return Object.entries(SKILLS)
      .map(([cat, items]) => `  ${cat.padEnd(14)} ${items.join(', ')}`)
      .join('\n')

  if (cmd === 'ssh ericsson')
    return `Connecting to ericsson.com...
> Physical Layer (vDU) Developer Intern  |  May 2024 – Sep 2025  |  Ottawa, ON

  [Gphy2.0]  Feb 2025 – Sep 2025
    • Offloaded LDPC encoding onto Nvidia GPU (CUDA / Bazel rules_cuda)
    • Profiled with Nsight Systems & Nsight Compute across multiple servers
    • Improved SM occupancy via kernel tuning

  [Gemini]   May 2024 – Feb 2025
    • Digital twin for Radio, RF Channel & UE simulation (Vue3 + Ericsson DS)
    • Multi-threaded CUDA unit tests under a strict ~35 μs timing constraint
    • SQLite + LTTng histogram pipeline for channel quality analysis`

  if (cmd === 'git log')
    return projects
      .slice(0, 8)
      .map((p, i) => {
        const hash = Math.random().toString(16).slice(2, 9)
        return `  ${hash}  ${p.date.padEnd(18)}  ${p.name}`
      })
      .join('\n')

  if (cmd === 'clear') return '__CLEAR__'

  if (cmd === 'sudo hire-me')
    return `[sudo] password for recruiter: ••••••••
Granting access...

  ✓  Strong CUDA & systems programming background
  ✓  16-month Ericsson internship (real-world 5G vDU stack)
  ✓  Full-stack capability (React → C++ → CUDA)
  ✓  2 co-authored research papers
  ✓  Active open-source contributor

  → Contact: hongmao.wu@mail.utoronto.ca`

  if (cmd === '') return ''

  return `bash: ${raw}: command not found  (type 'help' for available commands)`
}

export default function InteractiveTerminal() {
  const [history, setHistory] = useState([
    { type: 'output', text: "Type 'help' to see available commands." }
  ])
  const [input, setInput] = useState('')
  const [cmdHistory, setCmdHistory] = useState([])
  const [histIdx, setHistIdx] = useState(-1)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history])

  function submit() {
    const cmd = input
    const output = processCommand(cmd)

    if (output === '__CLEAR__') {
      setHistory([])
    } else {
      setHistory(h => [
        ...h,
        { type: 'input', text: cmd },
        ...(output ? [{ type: 'output', text: output }] : []),
      ])
    }

    if (cmd.trim()) setCmdHistory(h => [cmd, ...h])
    setInput('')
    setHistIdx(-1)
  }

  function handleKey(e) {
    if (e.key === 'Enter') {
      submit()
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      const next = Math.min(histIdx + 1, cmdHistory.length - 1)
      setHistIdx(next)
      setInput(cmdHistory[next] ?? '')
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      const next = histIdx - 1
      if (next < 0) { setHistIdx(-1); setInput('') }
      else { setHistIdx(next); setInput(cmdHistory[next]) }
    }
  }

  return (
    <div className="iterm-wrap" onClick={() => inputRef.current?.focus()}>
      <div className="iterm-body">
        {history.map((line, i) => (
          <div key={i} className={`iterm-line iterm-${line.type}`}>
            {line.type === 'input' && (
              <span className="iterm-prompt">wu@portfolio:~$ </span>
            )}
            <pre className="iterm-text">{line.text}</pre>
          </div>
        ))}

        <div className="iterm-input-row">
          <span className="iterm-prompt">wu@portfolio:~$ </span>
          <input
            ref={inputRef}
            className="iterm-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            spellCheck={false}
            autoComplete="off"
            autoCapitalize="off"
          />
        </div>
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
