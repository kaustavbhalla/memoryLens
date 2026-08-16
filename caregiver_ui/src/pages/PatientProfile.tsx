import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

const API = 'http://127.0.0.1:8000'

interface Person {
  id: string
  name: string
  relation: string
  enrollment_status: string
  last_seen: string
}

export default function PatientProfile() {
  const [persons, setPersons] = useState<Person[]>([])
  const [profile, setProfile] = useState<Record<string, string>>({})
  const [editKey, setEditKey] = useState('')
  const [editVal, setEditVal] = useState('')

  useEffect(() => {
    fetch(`${API}/persons`).then(r => r.json()).then(setPersons)
  }, [])

  const updateProfile = async () => {
    if (!editKey || !editVal) return
    await fetch(`${API}/patient/profile?key=${encodeURIComponent(editKey)}&value=${encodeURIComponent(editVal)}`, { method: 'POST' })
    setProfile(p => ({ ...p, [editKey]: editVal }))
    setEditKey('')
    setEditVal('')
  }

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui' }}>
      <h1>MemoryLens Caregiver Portal</h1>

      <nav style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
        <Link to="/enroll">Enroll Person</Link>
        <Link to="/review">Review Queue</Link>
      </nav>

      <h2>Known People</h2>
      {persons.length === 0 && <p style={{ color: '#999' }}>No people enrolled yet.</p>}
      {persons.map(p => (
        <Link key={p.id} to={`/person/${p.id}`} style={{ display: 'block', border: '1px solid #ddd', borderRadius: 8, padding: 12, marginBottom: 8, textDecoration: 'none', color: 'inherit' }}>
          <strong>{p.name}</strong> — {p.relation}
          <br />
          <small style={{ color: '#888' }}>{p.enrollment_status} | Last seen: {new Date(p.last_seen).toLocaleDateString()}</small>
        </Link>
      ))}

      <h2 style={{ marginTop: 32 }}>Patient Profile</h2>
      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <input placeholder="Key (e.g. name, hometown)" value={editKey} onChange={e => setEditKey(e.target.value)} style={{ padding: 8, flex: 1 }} />
        <input placeholder="Value" value={editVal} onChange={e => setEditVal(e.target.value)} style={{ padding: 8, flex: 2 }} />
        <button onClick={updateProfile} style={{ padding: '8px 16px', background: '#0066cc', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
          Save
        </button>
      </div>
    </div>
  )
}
