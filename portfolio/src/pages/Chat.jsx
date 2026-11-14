import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport } from 'ai'
import { useRef, useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import ToolResult from '../components/chat/ToolResult'
import './Chat.css'

// The chat backend (api/chat.js) runs only on Vercel. GitHub Pages serves static
// files, so the copy there sends messages to the Vercel deployment instead.
const CHAT_API = import.meta.env.BASE_URL === '/'
  ? '/api/chat'
  : 'https://portfolio-mauve-eta-79.vercel.app/api/chat'
const transport = new DefaultChatTransport({ api: CHAT_API })

const QUICK_ACTIONS = [
  { label: 'Projects',     query: "What projects have you built?" },
  { label: 'Experience',   query: "Tell me about your Ericsson internship." },
  { label: 'Skills',       query: "What's your tech stack?" },
  { label: 'Publications', query: "Do you have any research papers?" },
  { label: 'Contact',      query: "How can I reach you?" },
]

export default function Chat() {
  const navigate = useNavigate()
  const bottomRef = useRef(null)
  const [draft, setDraft] = useState('')

  const { messages, sendMessage, regenerate, status, error } = useChat({
    transport,
  })

  const isLoading = status === 'submitted' || status === 'streaming'
  const hasMessages = messages.length > 0

  // The model scales to zero when idle, so the first reply can take a minute.
  const [slowStart, setSlowStart] = useState(false)
  useEffect(() => {
    if (status !== 'submitted') return
    const timer = setTimeout(() => setSlowStart(true), 8000)
    return () => clearTimeout(timer)
  }, [status])

  useEffect(() => {
    if (hasMessages) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, hasMessages])

  function submit(text) {
    const q = text.trim()
    if (!q || isLoading) return
    setDraft('')
    setSlowStart(false)
    sendMessage({ text: q })
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit(draft)
    }
  }

  function renderMessage(msg) {
    const parts = msg.parts ?? []
    if (parts.length === 0) {
      const text = typeof msg.content === 'string' ? msg.content : ''
      return text ? <p className="chat-msg-text">{text}</p> : null
    }
    return parts.map((part, j) => {
      if (part.type === 'text') {
        return <p key={j} className="chat-msg-text">{part.text}</p>
      }
      // Static tools: type is 'tool-{toolName}' (e.g. 'tool-getProjects')
      // Dynamic tools: type is 'dynamic-tool' with a toolName property
      if (part.type.startsWith('tool-') || part.type === 'dynamic-tool') {
        if (part.state !== 'output-available') return null
        const toolName = part.type === 'dynamic-tool' ? part.toolName : part.type.slice(5)
        return <ToolResult key={j} toolName={toolName} output={part.output} />
      }
      return null
    })
  }

  return (
    <div className="chat-page">
      <button className="chat-back" onClick={() => navigate('/')}>
        ← back
      </button>

      {!hasMessages ? (
        <div className="chat-landing">
          <div className="chat-landing-title">Chat with Marvin</div>
          <p className="chat-landing-sub">Ask me about my projects, skills, experience, or anything else.</p>

          <div className="chat-input-wrap chat-input-wrap--landing">
            <input
              className="chat-input"
              placeholder="Ask me anything..."
              value={draft}
              onChange={e => setDraft(e.target.value)}
              onKeyDown={handleKey}
              autoFocus
            />
            <button
              className="chat-send-btn"
              onClick={() => submit(draft)}
              disabled={!draft.trim() || isLoading}
              aria-label="Send"
            >
              →
            </button>
          </div>

          <div className="chat-quick-actions">
            {QUICK_ACTIONS.map(a => (
              <button key={a.label} className="chat-quick-btn" onClick={() => submit(a.query)}>
                {a.label}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-conversation">
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={msg.id ?? i} className={`chat-msg chat-msg--${msg.role}`}>
                <span className="chat-msg-label">
                  {msg.role === 'user' ? 'you' : 'marvin'}
                </span>
                <div className="chat-msg-body">
                  {renderMessage(msg)}
                </div>
              </div>
            ))}

            {error && !isLoading && (
              <div className="chat-msg chat-msg--assistant">
                <span className="chat-msg-label">marvin</span>
                <div className="chat-msg-body">
                  <p className="chat-msg-text chat-error">
                    Sorry, I couldn&apos;t answer that just now.{' '}
                    <button className="chat-retry-btn" onClick={() => { setSlowStart(false); regenerate() }}>Try again</button>
                  </p>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="chat-msg chat-msg--assistant">
                <span className="chat-msg-label">marvin</span>
                <div className="chat-msg-body">
                  <span className="chat-typing">
                    <span /><span /><span />
                  </span>
                  {slowStart && status === 'submitted' && (
                    <p className="chat-waking">Waking up the model — the first reply can take a minute.</p>
                  )}
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="chat-input-bar">
            <div className="chat-input-wrap">
              <input
                className="chat-input"
                placeholder="Ask a follow-up..."
                value={draft}
                onChange={e => setDraft(e.target.value)}
                onKeyDown={handleKey}
              />
              <button
                className="chat-send-btn"
                onClick={() => submit(draft)}
                disabled={!draft.trim() || isLoading}
                aria-label="Send"
              >
                →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
