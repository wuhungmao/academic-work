import { useChat } from '@ai-sdk/react'
import { useRef, useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import ToolResult from '../components/chat/ToolResult'
import './Chat.css'

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
  const inputRef = useRef(null)
  const [draft, setDraft] = useState('')

  const { messages, append, isLoading } = useChat({
    api: '/api/chat',
  })

  const hasMessages = messages.length > 0

  useEffect(() => {
    if (hasMessages) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, hasMessages])

  function submit(text) {
    const q = text.trim()
    if (!q || isLoading) return
    setDraft('')
    append({ role: 'user', content: q })
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit(draft)
    }
  }

  return (
    <div className="chat-page">
      {/* Back button */}
      <button className="chat-back" onClick={() => navigate('/')}>
        ← back
      </button>

      {!hasMessages ? (
        /* ── Landing state: centered search bar ── */
        <div className="chat-landing">
          <div className="chat-landing-title">Chat with Marvin</div>
          <p className="chat-landing-sub">Ask me about my projects, skills, experience, or anything else.</p>

          <div className="chat-input-wrap chat-input-wrap--landing">
            <input
              ref={inputRef}
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
        /* ── Conversation state ── */
        <div className="chat-conversation">
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={msg.id ?? i} className={`chat-msg chat-msg--${msg.role}`}>
                <span className="chat-msg-label">
                  {msg.role === 'user' ? 'you' : 'marvin'}
                </span>
                <div className="chat-msg-body">
                  {msg.parts ? (
                    msg.parts.map((part, j) => {
                      if (part.type === 'text') {
                        return <p key={j} className="chat-msg-text">{part.text}</p>
                      }
                      if (
                        part.type === 'tool-invocation' &&
                        part.toolInvocation?.state === 'result'
                      ) {
                        return (
                          <ToolResult
                            key={j}
                            toolName={part.toolInvocation.toolName}
                          />
                        )
                      }
                      return null
                    })
                  ) : (
                    <p className="chat-msg-text">{msg.content}</p>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="chat-msg chat-msg--assistant">
                <span className="chat-msg-label">marvin</span>
                <div className="chat-msg-body">
                  <span className="chat-typing">
                    <span /><span /><span />
                  </span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Sticky input bar */}
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
