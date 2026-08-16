import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'

const API = 'http://127.0.0.1:8000'

interface Person {
  id: string
  name: string
  relation: string
  relationship_summary: string | null
  visit_count: number
  enrollment_status: string
}

export default function PersonHistory() {
  const { id } = useParams<{ id: string }>()
  const [person, setPerson] = useState<Person | null>(null)

  useEffect(() => {
    if (id) {
      fetch(`${API}/person/${id}`).then(r => r.json()).then(setPerson)
    }
  }, [id])

  if (!person) return <div style={{ padding: 24, fontFamily: 'system-ui' }}>Loading...</div>

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui' }}>
      <h1>{person.name}</h1>
      <p style={{ color: '#666' }}>{person.relation} | Visits: {person.visit_count} | Status: {person.enrollment_status}</p>

      <h2>Relationship Summary</h2>
      <div style={{ background: '#f5f5f5', padding: 16, borderRadius: 8, whiteSpace: 'pre-wrap' }}>
        {person.relationship_summary || 'No summary yet.'}
      </div>

      <h2>Actions</h2>
      <div style={{ display: 'flex', gap: 8 }}>
        <a href="/" style={{ padding: '8px 16px', background: '#0066cc', color: 'white', textDecoration: 'none', borderRadius: 4 }}>
          Back to Profile
        </a>
      </div>
    </div>
  )
}
