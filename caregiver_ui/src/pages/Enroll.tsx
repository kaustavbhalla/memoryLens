const API = 'http://127.0.0.1:8000'

export default function Enroll() {
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const form = new FormData(e.currentTarget)
    const name = form.get('name') as string
    const relation = form.get('relation') as string

    // Capture from webcam for enrollment
    const video = document.createElement('video')
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true })
    video.play()
    await new Promise(r => setTimeout(r, 1000))

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')!.drawImage(video, 0, 0)
    const image = canvas.toDataURL('image/jpeg').split(',')[1]
    video.srcObject!.getTracks().forEach(t => t.stop())

    const resp = await fetch(`${API}/enroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, relation, image }),
    })
    const data = await resp.json()
    if (data.error) {
      alert(`Error: ${data.error}`)
    } else {
      alert(`Enrolled ${data.name} (${data.relation})`)
      ;(e.target as HTMLFormElement).reset()
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui' }}>
      <h1>Enroll New Person</h1>
      <p style={{ color: '#666' }}>Capture a photo from the webcam and assign a name and relation.</p>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12, maxWidth: 400 }}>
        <input name="name" placeholder="Name" required style={{ padding: 8 }} />
        <input name="relation" placeholder="Relation (daughter, doctor, friend)" required style={{ padding: 8 }} />
        <button type="submit" style={{ padding: 10, background: '#0066cc', color: 'white', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
          Capture & Enroll
        </button>
      </form>
    </div>
  )
}
