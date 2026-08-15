# memorylens

Memory assistance system for dementia patients using multimodal identity recognition and conversational AI.

# 👓 Smart Glasses Identity & Recall Architecture

A wearable AI system built on Raspberry Pi, routing audio-visual streams to a local processing server for real-time identity resolution, agentic recall, and post-session memory consolidation.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                       GLASSES / WEARABLE                            │
│   [Pi Camera NoIR]  [USB Mic]  [SSD1306 OLED]  [Pi Zero 2W]         │
└────────────────────────────┬────────────────────────────────────────┘
                             │ WiFi stream (video + audio)
┌────────────────────────────▼────────────────────────────────────────┐
│                    LOCAL PROCESSING SERVER                          │
│                                                                     │
│    Vision Pipeline              Audio Pipeline                      │
│    YOLO + DeepFace              pyannote + Whisper                  │
│            │                             │                          │
│            └────────────┬──────────────┘                          │
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
│                 HUD Renderer → OLED / phone                         │
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

## 🔬 Component Breakdown

### 1. Hardware Layer (Wearable)

The edge device captures real-time environment data and displays context back to the user.

- **Microcontroller**: Pi Zero 2W
- **Vision**: Pi Camera NoIR
- **Audio**: USB Microphone
- **Display**: SSD1306 OLED

### 2. Local Processing Server (Real-Time)

Data is streamed via WiFi to a local server for heavy lifting.

- **Vision Pipeline**: Utilizes **YOLO** (object/face detection) + **DeepFace** (facial recognition/embedding).
- **Audio Pipeline**: Utilizes **pyannote** (speaker diarization) + **Whisper** (Speech-to-Text).
- **Identity Fusion Engine**: Merges sensory inputs to create a unified identity profile.

---

## ⚡ Operational Modes

| Feature          | Mode 1: Deterministic     | Mode 2: Agentic Recall           | Mode 3: Auto-Enroll         |
| :--------------- | :------------------------ | :------------------------------- | :-------------------------- |
| **Trigger**      | Known face recognized     | Confusion phrase in STT stream   | Unknown face detected       |
| **Architecture** | FAISS → SQLite → HUD      | LangGraph + Claude               | LangGraph                   |
| **Latency**      | **~20ms** (Ultra-fast)    | ~1.5–3.0s                        | ~3–5s (Runs parallel)       |
| **LLM Usage**    | None                      | Yes (Decides tool sequence)      | Yes (Agentic drafting)      |
| **Output**       | Instant HUD Identity Card | Contextual memory / facts on HUD | Background profile creation |

---

## 🧠 Background Pipeline (Post-Session)

When the active session ends, the **Memory Consolidation Agent** organizes and permanently stores the session data.

1.  **Extraction**: Pulls new facts and inter-person relations from the transcripts.
2.  **Compression**: Summarizes the new episode into a rolling relationship summary.
3.  **Conflict Resolution**: Identifies conflicting information and marks outdated facts.
4.  **Database Writes**: Pushes structured knowledge to SQLite, LanceDB, and NetworkX.
