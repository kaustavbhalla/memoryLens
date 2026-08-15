"""FastAPI app entrypoint + lifespan."""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.memory.store import memory
from server.pipelines.vision import VisionPipeline
from server.pipelines.audio import AudioPipeline
from server.pipelines.fusion import IdentityFusionEngine
from server.pipelines.trigger import TriggerDetector
from server.mode1.deterministic import DeterministicPipeline
from server.hud.renderer import HUDCard, EmptyCard, RecallCard, UnknownCard

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("memorylens")

# ── Globals ───────────────────────────────────────────────────────────

vision = VisionPipeline()
audio_pipeline = AudioPipeline()
fusion = IdentityFusionEngine()
trigger = TriggerDetector()
mode1 = DeterministicPipeline()

# Session transcript buffer (list of speaker turns)
_session_transcript: list[dict] = []

# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading memory stores…")
    memory.load()
    log.info("Loading vision pipeline (YOLO + DeepFace)…")
    await vision.load()
    log.info("Loading audio pipeline (WhisperX)…")
    try:
        await audio_pipeline.load()
    except Exception as e:
        log.warning(f"Audio pipeline failed to load (will retry): {e}")
    log.info("MemoryLens server ready")
    yield
    memory.save()
    log.info("MemoryLens server shut down")

# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="MemoryLens",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────

class FramePayload(BaseModel):
    image: str  # base64-encoded JPEG

class AudioPayload(BaseModel):
    audio: str  # base64-encoded PCM (int16, 16kHz, mono)
    sample_rate: int = 16000

class EnrollPayload(BaseModel):
    name: str
    relation: str
    image: str  # base64-encoded JPEG

class RecallPayload(BaseModel):
    trigger_phrase: str
    confirmed_person_id: str | None = None
    session_context: str = ""

class SessionStartPayload(BaseModel):
    person_id: str

class SessionEndPayload(BaseModel):
    person_id: str
    transcript: str = ""

# ── Helpers ───────────────────────────────────────────────────────────

def decode_image(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ── Real-time routes ──────────────────────────────────────────────────

@app.post("/frame")
async def process_frame(payload: FramePayload) -> dict:
    """Mode 1: known face → deterministic HUD card via fusion engine."""
    frame = decode_image(payload.image)
    faces = vision.detect_faces(frame)
    if not faces:
        return EmptyCard().to_dict()

    # Process all faces through fusion engine
    for face in faces:
        embedding = VisionPipeline.extract_embedding(face["crop"])
        if embedding is not None:
            fusion.process_face(embedding)

    # Get best confirmed match from fusion engine
    best_person = fusion.get_best_match()
    if best_person is None:
        return EmptyCard().to_dict()

    # Look up person directly (fusion already matched via FAISS)
    person = memory.db.get_person(best_person.person_id)
    if person is None:
        return EmptyCard().to_dict()

    recent = memory.db.get_recent_conversations(best_person.person_id, limit=2)
    facts = memory.db.get_top_facts(best_person.person_id, limit=5)
    from server.hud.renderer import build_person_card
    return build_person_card(person, recent, facts).to_dict()


@app.post("/audio")
async def process_audio(payload: AudioPayload) -> dict:
    """
    Process an audio chunk: WhisperX STT + diarization.
    Returns speaker-labeled transcript segments.
    Also checks for confusion triggers (Mode 2).
    """
    import base64
    audio_data = base64.b64decode(payload.audio)
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        result = audio_pipeline.process_chunk(audio_array, payload.sample_rate)
    except Exception as e:
        log.error(f"Audio processing failed: {e}")
        return {"error": str(e), "segments": [], "full_text": ""}

    # Add to session transcript
    for turn in result.speaker_turns:
        _session_transcript.append(turn)

    # Check for confusion triggers
    trigger_detected = trigger.check(result.full_text)

    return {
        "segments": result.speaker_turns,
        "full_text": result.full_text,
        "language": result.language,
        "trigger_detected": trigger_detected,
        "speaker_texts": result.speaker_texts,
    }


@app.post("/enroll")
async def enroll_person(payload: EnrollPayload) -> dict:
    """Manual enrollment (caregiver action)."""
    from server.memory.structured import Person, EnrollmentStatus, utcnow
    import uuid

    frame = decode_image(payload.image)
    faces = vision.detect_faces(frame)
    if not faces:
        return {"error": "No face detected in image"}

    best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
    embedding = VisionPipeline.extract_embedding(best["crop"])
    if embedding is None:
        return {"error": "Could not extract face embedding"}

    person_id = str(uuid.uuid4())
    memory.biometric.add_face(person_id, embedding)
    memory.biometric.save()

    person = Person(
        id=person_id,
        name=payload.name,
        relation=payload.relation,
        enrollment_status=EnrollmentStatus.CONFIRMED,
        name_confidence=1.0,
        relation_confidence=1.0,
    )
    memory.db.save_person(person)

    # Add to graph
    memory.graph.add_person(person_id, payload.name, payload.relation)
    memory.graph.save()

    return {"person_id": person_id, "name": payload.name, "relation": payload.relation}


@app.post("/recall")
async def trigger_recall(payload: RecallPayload) -> dict:
    """Mode 2: confusion phrase → LangGraph recall agent."""
    from server.mode2.agent import run_recall_agent
    try:
        narration = await run_recall_agent(
            trigger_phrase=payload.trigger_phrase,
            confirmed_person_id=payload.confirmed_person_id,
            session_context=payload.session_context,
        )
        return RecallCard(narration=narration).to_dict()
    except Exception as e:
        log.error(f"Recall agent error: {e}")
        return RecallCard(narration="I'm having trouble remembering right now.").to_dict()


@app.post("/session/start")
async def start_session(payload: SessionStartPayload) -> dict:
    person = memory.db.get_person(payload.person_id)
    if person:
        person.mark_seen()
        memory.db.save_person(person)
        return {"status": "ok", "person": person.name}
    return {"error": "Person not found"}


@app.post("/session/end")
async def end_session(payload: SessionEndPayload) -> dict:
    return {"status": "ok"}


# ── Memory / profile reads ────────────────────────────────────────────

@app.get("/person/{person_id}")
async def get_person(person_id: str) -> dict:
    person = memory.db.get_person(person_id)
    if not person:
        return {"error": "Not found"}
    return {
        "id": person.id,
        "name": person.name,
        "relation": person.relation,
        "relationship_summary": person.relationship_summary,
        "visit_count": person.visit_count,
        "enrollment_status": person.enrollment_status.value,
    }


@app.get("/persons")
async def list_persons() -> list[dict]:
    persons = memory.db.get_all_persons(include_unconfirmed=True)
    return [
        {
            "id": p.id,
            "name": p.display_name,
            "relation": p.relation,
            "enrollment_status": p.enrollment_status.value,
            "last_seen": p.last_seen,
        }
        for p in persons
    ]


@app.get("/enroll/queue")
async def get_enrollment_queue() -> list[dict]:
    persons = memory.db.get_enrollment_queue()
    return [
        {"id": p.id, "name": p.name, "relation": p.relation,
         "first_seen": p.first_seen, "name_confidence": p.name_confidence}
        for p in persons
    ]


@app.post("/enroll/confirm/{person_id}")
async def confirm_enrollment(
    person_id: str, name: str, relation: str
) -> dict:
    memory.db.confirm_enrollment(person_id, name, relation)
    return {"status": "confirmed"}
