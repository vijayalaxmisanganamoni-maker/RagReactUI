import { useEffect, useRef, useState } from 'react'

const DOMAINS = {
  customer_support: 'Customer Support',
  biomedical: 'Biomedical Research',
  general_knowledge: 'General Knowledge',
  legal: 'Legal',
  finance: 'Finance',
}

const SUGGESTIONS = {
  customer_support: [
    'What is Uconnect and what can I do with it?',
    'How do I connect my TV to a Wi-Fi network?',
    'How do I turn on the fog lights?',
  ],
  biomedical: [
    'How does COVID-19 spread between people?',
    'What treatments have been studied for sepsis?',
  ],
  general_knowledge: [
    'Who was the first person to walk on the moon?',
    'Why is the sky blue?',
  ],
  legal: [
    'Under what conditions can this agreement be terminated?',
    'What license rights are granted under the agreement?',
  ],
  finance: [
    "How did the company's revenue change year over year?",
    'What were the main operating expenses?',
  ],
}

function Sources({ contexts }) {
  if (!contexts?.length) return null
  return (
    <details className="sources">
      <summary>Sources ({contexts.length})</summary>
      {contexts.map((c, i) => (
        <div className="source" key={i}>
          <div className="source-head">
            <span className="badge">[{i + 1}] {c.source}</span>
            <span className="scores">
              retrieval {c.retrieval_score?.toFixed(3)}
              {c.rerank_score != null && <> · rerank {c.rerank_score.toFixed(3)}</>}
            </span>
          </div>
          <p>{c.text.length > 500 ? c.text.slice(0, 500) + '…' : c.text}</p>
        </div>
      ))}
    </details>
  )
}

export default function App() {
  const [domain, setDomain] = useState('customer_support')
  const [histories, setHistories] = useState(
    Object.fromEntries(Object.keys(DOMAINS).map((d) => [d, []])),
  )
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [health, setHealth] = useState(null)
  const endRef = useRef(null)

  const messages = histories[domain]

  useEffect(() => {
    fetch('/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const push = (d, msg) =>
    setHistories((h) => ({ ...h, [d]: [...h[d], msg] }))

  async function ask(question) {
    const q = question.trim()
    if (!q || loading) return
    setInput('')
    push(domain, { role: 'user', content: q })
    setLoading(true)
    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, domain }),
      })
      if (!res.ok) throw new Error(`API error ${res.status}`)
      const data = await res.json()
      push(domain, {
        role: 'assistant',
        content: data.answer,
        contexts: data.contexts,
        latency: data.latency_s,
        queryType: data.query_type,
      })
    } catch (e) {
      push(domain, {
        role: 'assistant',
        content: `Something went wrong: ${e.message}. Is the FastAPI server running?`,
        error: true,
      })
    } finally {
      setLoading(false)
    }
  }

  const chunkCount = health?.index_chunks_per_domain?.[domain]

  return (
    <div className="app">
      <header>
        <div className="brand">
          <h1>RAGBench RAG Assistant</h1>
          <span className="subtitle">
            Retrieval-Augmented Generation · {DOMAINS[domain]}
            {typeof chunkCount === 'number' &&
              ` · ${chunkCount.toLocaleString()} indexed chunks`}
            {health?.llm_provider && ` · LLM: ${health.llm_provider}`}
          </span>
        </div>
        <nav className="tabs">
          {Object.entries(DOMAINS).map(([key, label]) => (
            <button
              key={key}
              className={key === domain ? 'tab active' : 'tab'}
              onClick={() => setDomain(key)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      <div className="body">
        <aside className="sidebar">
          <h2>Ask a {DOMAINS[domain]} question</h2>
          <p className="hint">or try one of these:</p>
          {(SUGGESTIONS[domain] || []).map((s) => (
            <button key={s} className="chip" onClick={() => ask(s)}>
              {s}
            </button>
          ))}
        </aside>

        <div className="chat">
          <main>
            {messages.length === 0 && (
              <div className="empty">
                <p>Your {DOMAINS[domain]} conversation will appear here.</p>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}${m.error ? ' error' : ''}`}>
                <div className="bubble">
                  <p className="content">{m.content}</p>
                  <Sources contexts={m.contexts} />
                  {m.latency != null && (
                    <span className="meta">
                      {m.queryType} · {m.latency}s
                    </span>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="msg assistant">
                <div className="bubble typing">Retrieving and generating…</div>
              </div>
            )}
            <div ref={endRef} />
          </main>

          <footer>
            <form
              onSubmit={(e) => {
                e.preventDefault()
                ask(input)
              }}
            >
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={`Ask a ${DOMAINS[domain]} question…`}
                disabled={loading}
              />
              <button type="submit" disabled={loading || !input.trim()}>
                Send
              </button>
            </form>
          </footer>
        </div>
      </div>
    </div>
  )
}
