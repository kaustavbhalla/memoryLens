import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

const API = 'http://127.0.0.1:8000'

const PROFILE_FIELDS = [
  { key: 'name', label: 'Full Name', placeholder: 'e.g. Ramesh Kumar' },
  { key: 'age', label: 'Age', placeholder: 'e.g. 72' },
  { key: 'hometown', label: 'Hometown', placeholder: 'e.g. Kolkata, India' },
  { key: 'occupation', label: 'Former Occupation', placeholder: 'e.g. school teacher' },
  { key: 'hobbies', label: 'Hobbies & Interests', placeholder: 'e.g. gardening, chess, classical music' },
  { key: 'diagnosis', label: 'Diagnosis', placeholder: 'e.g. early-stage Alzheimer\'s' },
  { key: 'notes', label: 'Caregiver Notes', placeholder: 'e.g. prefers tea over coffee, remembers childhood best' },
]

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
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    fetch(`${API}/persons`).then(r => r.json()).then(setPersons)
    fetch(`${API}/patient/profile`).then(r => r.json()).then(setProfile)
  }, [])

  const updateField = async (key: string, value: string) => {
    setProfile(p => ({ ...p, [key]: value }))
  }

  const saveAll = async () => {
    setSaving(true)
    for (const field of PROFILE_FIELDS) {
      const val = profile[field.key] || ''
      if (val) {
        await fetch(`${API}/patient/profile?key=${encodeURIComponent(field.key)}&value=${encodeURIComponent(val)}`, { method: 'POST' })
      }
    }
    setSaving(false)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui', maxWidth: 800, margin: '0 auto' }}>
      <h1 style={{ marginBottom: 8 }}>MemoryLens Caregiver Portal</h1>
      <p style={{ color: '#666', marginTop: 0 }}>Set up the patient's identity and manage enrolled people.</p>

      <nav style={{ display: 'flex', gap: 12, marginBottom: 32 }}>
        <Link to="/enroll" style={{ padding: '8px 16px', background: '#0066cc', color: 'white', textDecoration: 'none', borderRadius: 4 }}>Enroll Person</Link>
        <Link to="/review" style={{ padding: '8px 16px', background: '#666', color: 'white', textDecoration: 'none', borderRadius: 4 }}>Review Queue</Link>
      </nav>

      {/* ── Patient Profile ─────────────────────────────────── */}
      <div style={{ background: '#f8f9fa', borderRadius: 8, padding: 24, marginBottom: 32 }}>
        <h2 style={{ marginTop: 0, marginBottom: 4 }}>Patient Identity</h2>
        <p style={{ color: '#666', marginTop: 0, fontSize: 14 }}>
          This powers the "Who Am I?" feature — the system uses this to help the patient recall their own identity.
        </p>

        <div style={{ display: 'grid', gap: 16 }}>
          {PROFILE_FIELDS.map(f => (
            <div key={f.key}>
              <label style={{ display: 'block', fontWeight: 600, marginBottom: 4, fontSize: 14 }}>{f.label}</label>
              <input
                value={profile[f.key] || ''}
                onChange={e => updateField(f.key, e.target.value)}
                placeholder={f.placeholder}
                style={{ width: '100%', padding: '10px 12px', border: '1px solid #ddd', borderRadius: 4, fontSize: 14, boxSizing: 'border-box' }}
              />
            </div>
          ))}
        </div>

        <button
          onClick={saveAll}
          disabled={saving}
          style={{
            marginTop: 16,
            padding: '10px 24px',
            background: saved ? '#28a745' : '#0066cc',
            color: 'white',
            border: 'none',
            borderRadius: 4,
            cursor: saving ? 'wait' : 'pointer',
            fontSize: 14,
          }}
        >
          {saving ? 'Saving…' : saved ? 'Saved!' : 'Save Profile'}
        </button>
      </div>

      {/* ── Known People ────────────────────────────────────── */}
      <div>
        <h2 style={{ marginBottom: 8 }}>Known People</h2>
        <p style={{ color: '#666', marginTop: 0, fontSize: 14 }}>
          Enrolled people the system recognizes. Click to view conversation history.
        </p>
        {persons.length === 0 && <p style={{ color: '#999' }}>No people enrolled yet. Use "Enroll Person" to add someone.</p>}
        {persons.map(p => (
          <Link
            key={p.id}
            to={`/person/${p.id}`}
            style={{
              display: 'block',
              border: '1px solid #ddd',
              borderRadius: 8,
              padding: 12,
              marginBottom: 8,
              textDecoration: 'none',
              color: 'inherit',
              background: 'white',
            }}
          >
            <strong>{p.name}</strong> — {p.relation}
            <br />
            <small style={{ color: '#888' }}>
              {p.enrollment_status} | Last seen: {new Date(p.last_seen).toLocaleDateString()}
            </small>
          </Link>
        ))}
      </div>
    </div>
  )
}
