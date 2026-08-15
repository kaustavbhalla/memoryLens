# MemoryLens

A wearable cognitive prosthetic for dementia & Alzheimer's patients — using multimodal identity recognition, conversational AI, and real-time memory recall.

---

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    CAPTURE (Webcam + Mic)                            │
│   [Webcam]  [USB Mic]  ──►  client/webcam_client.py                 │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP (base64 frames + audio chunks)
┌────────────────────────────▼────────────────────────────────────────┐
│                    LOCAL PROCESSING SERVER (FastAPI)                 │
│                                                                     │
│    Vision Pipeline              Audio Pipeline                      │
│    YOLOv8n-face + DeepFace      WhisperX (faster-whisper +         │
│    (ArcFace embeddings)          pyannote diarization +             │
│            │                     silero-vad)                        │
│            └────────────┬──────────────┘                            │
│                         │                                           │
│                Identity Fusion Engine                               │
│                "Face A + Voice A = Person X"                        │
│                         │                                           │
│            ┌────────────┼────────────────┐                          │
│            │            │                │                          │
│    Known face?   Unknown face?   Confusion phrase                   │
│            │            │          in STT stream?                   │
│            ▼            ▼                ▼                          │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐           │
│  │   MODE 1    │ │   MODE 3     │ │       MODE 2        │           │
│  │DETERMINISTIC│ │AUTO-ENROLL   │ │  AGENTIC (LangGraph)│           │
│  │             │ │AGENT         │ │  RECALL AGENT       │           │
│  │FAISS→SQLite │ │(LangGraph)   │ │                     │           │
│  │→ HUD card   │ │              │ │  Tools:             │           │
│  │             │ │Tools:        │ │  · get_person_profile│          │
│  │~20ms        │ │·extract_name │ │  · search_rag       │           │
│  │NO LLM       │ │·infer_relation││  · get_social_context│          │
│  │             │ │·check_quality│ │  · narrow_unknown   │           │
│  └──────┬──────┘ │·create_draft │ │  · get_atomic_facts │           │
│         │        │·store_embed  │ │  · get_patient_profile│         │
│         │        │·flag_review  │ │  · generate_narration│          │
│         │        │              │ └──────────┬───────────┘          │
│         │        │~3–5s         │            │                      │
│         │        │Runs parallel │  Claude decides tool sequence     │
│         │        │to capture    │  ~1.5–3s                          │
│         │        └──────┬───────┘            │                      │
│         └───────────────┼────────────────────┘                      │
│                         │                                           │
│                 HUD Renderer → Webcam Overlay (OpenCV)              │
│                                                                     │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ SESSION END ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─        │
│                                                                     │
│                 BACKGROUND PIPELINE (no latency constraint)         │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │  Memory Consolidation Agent (LangGraph)                  │      │
│   │  · Extract facts + inter-person relations from transcript│      │
│   │  · Compress new episode into rolling relationship summary│      │
│   │  · Resolve fact conflicts (mark outdated facts)          │      │
│   │  · Write → SQLite + LanceDB + NetworkX                   │      │
│   └──────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API server | FastAPI (Python) | Async, fast, typed |
| Face detection | YOLOv8n-face | Fastest YOLO variant, ~5ms CPU |
| Face recognition | DeepFace (ArcFace backbone) | Better accuracy than `face_recognition` lib |
| Speech-to-text + Diarization | **WhisperX** | Unified pipeline: faster-whisper + wav2vec2 alignment + pyannote diarization + silero-vad |
| Speaker embeddings | pyannote (via WhisperX) | Per-turn voice prints for identity matching |
| Biometric store | FAISS (flat L2 index) | Sub-10ms ANN, fully in-memory |
| Structured store | SQLite (WAL mode) via SQLModel | Zero infra, fast reads |
| Vector store | LanceDB | Embedded, fast filtered search, Arrow-backed |
| Relationship graph | NetworkX + pickle | In-RAM, trivially small at this scale |
| LLM | Claude claude-sonnet-4-6 (API) | Best instruction following, structured output |
| HUD renderer | OpenCV overlay (webcam feed) | No extra hardware — displays on webcam window |
| Caregiver UI | React + Vite | Fast to scaffold |

### Audio Pipeline: WhisperX

WhisperX replaces separate Whisper + pyannote integration with a single unified pipeline:

1. **faster-whisper** backend — 70x realtime transcription with CTranslate2 quantization (GPU)
2. **silero-vad** — Voice Activity Detection reduces hallucination on silence/noise
3. **wav2vec2 forced alignment** — word-level timestamps for precise speaker attribution
4. **pyannote diarization** — speaker segmentation + labels (SPEAKER_00, SPEAKER_01, ...)
5. **`assign_word_speakers()`** — one call maps each word to a speaker

**GPU acceleration**: Uses CUDA (RTX 3050) for WhisperX inference. Falls back to CPU with `int8` quantization if no GPU available.

---

## Operational Modes

| Feature | Mode 1: Deterministic | Mode 2: Agentic Recall | Mode 3: Auto-Enroll |
|---|---|---|---|
| **Trigger** | Known face recognized | Confusion phrase in STT | Unknown face detected |
| **Architecture** | FAISS → SQLite → HUD | LangGraph + Claude | LangGraph |
| **Latency** | ~20ms (no LLM) | 1.5–3.0s | ~3–5s (background) |
| **Output** | Context card on webcam | Recall narration on webcam | Provisional profile queued |

---

## Build Plan

- [x] Phase 1: Bug fixes + config + unified memory store
- [x] Phase 2: FastAPI server + vision pipeline + Mode 1
- [x] Phase 3: Webcam overlay client
- [x] Phase 4: Mode 2 — LangGraph recall agent
- [x] Phase 5: Mode 3 — Auto-enrollment agent
- [ ] Phase 6: Audio pipeline (WhisperX + fusion engine)
- [ ] Phase 7: Background consolidation agent + session manager
- [ ] Phase 8: Caregiver UI
- [ ] Phase 9: Polish

---

## Running

### Server
```bash
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

### Webcam Client
```bash
uv run python -m client.webcam_client http://127.0.0.1:8000
```

### Controls
- `e` — enroll the currently visible person
- `r` — trigger recall (enter a confusion phrase)
- `q` — quit
