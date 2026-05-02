import './TerminalWindow.css'

export default function TerminalWindow({ title = 'bash', children }) {
  return (
    <div className="terminal">
      <div className="terminal-bar">
        <span className="dot red" />
        <span className="dot yellow" />
        <span className="dot green" />
        <span className="terminal-title">{title}</span>
      </div>
      <div className="terminal-body">{children}</div>
    </div>
  )
}

export function Prompt({ path = '~', command, output }) {
  return (
    <div className="prompt-block">
      <div className="prompt-line">
        <span className="prompt-user">wu@portfolio</span>
        <span className="prompt-colon">:</span>
        <span className="prompt-path">{path}</span>
        <span className="prompt-dollar">$</span>
        <span className="prompt-cmd">{command}</span>
      </div>
      {output && <div className="prompt-output">{output}</div>}
    </div>
  )
}
