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
from server.mode1.deterministic import DeterministicPipeline
from server.hud.renderer import HUDCard, EmptyCard, RecallCard

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("memorylens")

# ── Globals ───────────────────────────────────────────────────────────

vision = VisionPipeline()
mode1 = DeterministicPipeline()

# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading memory stores…")
    memory.load()
    log.info("Loading vision pipeline (YOLO + DeepFace)…")
    await vision.load()
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
    """Mode 1: known face → deterministic HUD card."""
    frame = decode_image(payload.image)
    faces = vision.detect_faces(frame)
    if not faces:
        return EmptyCard().to_dict()

    # Use the largest face
    best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
    embedding = VisionPipeline.extract_embedding(best["crop"])
    if embedding is None:
        return EmptyCard().to_dict()

    card = await mode1.run(embedding)
    return card.to_dict()


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
    """Mode 2: confusion phrase → LangGraph recall agent (stub for now)."""
    # Phase 4 will implement the LangGraph agent
    return RecallCard(narration="Recall agent not yet implemented.").to_dict()


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
