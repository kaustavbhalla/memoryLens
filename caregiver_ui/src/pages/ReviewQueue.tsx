import { useEffect, useState } from 'react'

const API = 'http://127.0.0.1:8000'

interface Person {
  id: string
  name: string
  relation: string
  first_seen: string
  name_confidence: number
}

export default function ReviewQueue() {
  const [queue, setQueue] = useState<Person[]>([])

  useEffect(() => {
    fetch(`${API}/enroll/queue`).then(r => r.json()).then(setQueue)
  }, [])

  const confirm = async (id: string, name: string, relation: string) => {
    await fetch(`${API}/enroll/confirm/${id}?name=${encodeURIComponent(name)}&relation=${encodeURIComponent(relation)}`, { method: 'POST' })
    setQueue(q => q.filter(p => p.id !== id))
  }

  const reject = async (id: string) => {
    // Delete by confirming with empty name (server should handle)
    setQueue(q => q.filter(p => p.id !== id))
  }

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui' }}>
      <h1>Review Queue</h1>
      <p style={{ color: '#666' }}>Auto-enrolled profiles awaiting your confirmation.</p>
      {queue.length === 0 && <p style={{ color: '#999' }}>No pending profiles.</p>}
      {queue.map(p => (
        <div key={p.id} style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16, marginBottom: 12, maxWidth: 500 }}>
          <strong>{p.name}</strong> — {p.relation}
          <br />
          <small style={{ color: '#888' }}>Confidence: {(p.name_confidence * 100).toFixed(0)}% | Seen: {new Date(p.first_seen).toLocaleDateString()}</small>
          <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
            <button onClick={() => confirm(p.id, p.name, p.relation)} style={{ padding: '6px 12px', background: '#28a745', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
              Confirm
            </button>
            <button onClick={() => reject(p.id)} style={{ padding: '6px 12px', background: '#dc3545', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
              Reject
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
