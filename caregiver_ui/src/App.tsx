import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Enroll from './pages/Enroll'
import ReviewQueue from './pages/ReviewQueue'
import PersonHistory from './pages/PersonHistory'
import PatientProfile from './pages/PatientProfile'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PatientProfile />} />
        <Route path="/enroll" element={<Enroll />} />
        <Route path="/review" element={<ReviewQueue />} />
        <Route path="/person/:id" element={<PersonHistory />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
