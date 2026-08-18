"""FastAPI app entrypoint + lifespan."""

from __future__ import annotations

import base64
import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.memory.store import memory
from server.pipelines.vision import VisionPipeline
from server.pipelines.audio import AudioPipeline
from server.pipelines.fusion import IdentityFusionEngine
from server.pipelines.trigger import TriggerDetector
from server.mode1.deterministic import DeterministicPipeline
from server.session_manager import session_manager
from server.hud.renderer import HUDCard, EmptyCard, RecallCard, UnknownCard

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("memorylens")

# Suppress noisy uvicorn access logs for streaming endpoints
class _QuietStreamingFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "/frame" in msg or "/audio" in msg:
            return False
        return True

logging.getLogger("uvicorn.access").addFilter(_QuietStreamingFilter())

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
    import asyncio

    log.info("Loading memory stores…")
    memory.load()
    log.info("Loading vision pipeline (YOLO + DeepFace)…")
    await vision.load()
    log.info("Loading audio pipeline (WhisperX)…")
    try:
        await audio_pipeline.load()
    except Exception as e:
        log.warning(f"Audio pipeline failed to load (will retry): {e}")

    # Sync patient name from SQLite to NetworkX graph
    patient_name = memory.db.get_patient_name()
    if patient_name:
        memory.graph.set_patient_name(patient_name)
        log.info(f"Patient name loaded: {patient_name}")

    # Background task: check silence timeouts and auto-consolidate
    async def _silence_watcher():
        while True:
            await asyncio.sleep(30)
            for pid in session_manager.check_silence_timeouts():
                log.info(f"Silence timeout for {pid}, consolidating…")
                asyncio.create_task(session_manager.consolidate_and_end(pid))

    silence_task = asyncio.create_task(_silence_watcher())
    log.info("MemoryLens server ready")
    yield
    silence_task.cancel()
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


async def _run_auto_enroll(face_id: str, crop_path: str, context: str):
    """Fire Mode 3 auto-enrollment in background (fire-and-forget)."""
    try:
        from server.mode3.agent import run_auto_enroll
        await run_auto_enroll(
            unknown_face_id=face_id,
            face_crop_path=crop_path,
            session_context=context,
        )
    except Exception as e:
        log.warning(f"Auto-enroll failed: {e}")
    finally:
        try:
            os.remove(crop_path)
        except OSError:
            pass

# ── Real-time routes ──────────────────────────────────────────────────

@app.post("/frame")
async def process_frame(payload: FramePayload) -> dict:
    """Mode 1: known face -> deterministic HUD card via fusion engine."""
    # First-run check: patient profile required
    if not memory.db.has_patient_profile():
        return {"type": "error", "error": "Patient profile not configured. Please set up the patient profile in the caregiver UI first."}

    frame = decode_image(payload.image)
    faces = vision.detect_faces(frame)

    if not faces:
        return EmptyCard().to_dict()

    log.info(f"[FRAME] {len(faces)} face(s) detected")

    # Process all faces through fusion engine
    for face in faces:
        embedding = VisionPipeline.extract_embedding(face["crop"])
        if embedding is not None:
            result = fusion.process_face(embedding)
            if result:
                log.info(f"[FRAME] Face matched → {result}")
            else:
                log.info(f"[FRAME] Face embedding extracted, no match")

    # Get best confirmed match from fusion engine
    best_person = fusion.get_best_match()
    if best_person is None:
        log.info(f"[FRAME] No confirmed match → triggering Mode 3 (auto-enroll)")
        import asyncio, uuid, tempfile
        for face in faces:
            face_id = str(uuid.uuid4())
            crop_path = os.path.join(tempfile.gettempdir(), f"{face_id}.jpg")
            cv2.imwrite(crop_path, face["crop"])
            asyncio.create_task(_run_auto_enroll(face_id, crop_path, ""))
        return UnknownCard().to_dict()

    # Look up person directly (fusion already matched via FAISS)
    person = memory.db.get_person(best_person.person_id)
    if person is None:
        return EmptyCard().to_dict()

    log.info(f"[FRAME] Mode 1 → {person.name} (confidence: {best_person.fused_confidence:.2f})")

    # Auto-start session for this person
    person.mark_seen()
    memory.db.save_person(person)
    session_manager.start_session(person.id, person.name)

    recent = memory.db.get_recent_conversations(best_person.person_id, limit=2)
    facts = memory.db.get_top_facts(best_person.person_id, limit=5)
    from server.hud.renderer import build_person_card
    return build_person_card(person, recent, facts, memory.db.get_patient_name()).to_dict()


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
        log.error(f"[AUDIO] Processing failed: {e}")
        return {"error": str(e), "segments": [], "full_text": ""}

    # Log transcript
    if result.full_text.strip():
        log.info(f"[AUDIO] Transcript: \"{result.full_text[:80]}\"")

    # Add to session transcript
    for turn in result.speaker_turns:
        _session_transcript.append(turn)
        session_manager.add_transcript(turn.get("speaker", "UNKNOWN"), turn.get("text", ""))

    # Check for confusion triggers
    trigger_detected = trigger.check(result.full_text)
    is_patient = False
    if trigger_detected:
        # Identify speaker
        if result.speaker_segments:
            best_segment = max(result.speaker_segments, key=lambda s: s.end - s.start)
            if best_segment.embedding is not None:
                person_id, conf = memory.biometric.search_voice(best_segment.embedding)
                is_patient = (person_id == "patient")
        log.info(f"[AUDIO] Trigger detected is_patient={is_patient}")

    return {
        "segments": result.speaker_turns,
        "full_text": result.full_text,
        "language": result.language,
        "trigger_detected": trigger_detected,
        "is_patient": is_patient,
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

    log.info(f"[ENROLL] Enrolled {payload.name} ({payload.relation}) id={person_id[:8]}")
    return {"person_id": person_id, "name": payload.name, "relation": payload.relation}


@app.post("/recall")
async def trigger_recall(payload: RecallPayload) -> dict:
    """Mode 2: confusion phrase -> LangGraph recall agent."""
    from server.mode2.agent import run_recall_agent
    log.info(f"[RECALL] Trigger: \"{payload.trigger_phrase[:60]}\" person={payload.confirmed_person_id}")
    try:
        narration = await run_recall_agent(
            trigger_phrase=payload.trigger_phrase,
            confirmed_person_id=payload.confirmed_person_id,
            session_context=payload.session_context,
        )
        log.info(f"[RECALL] Narration: \"{narration[:80]}\"")
        return RecallCard(narration=narration).to_dict()
    except Exception as e:
        log.error(f"[RECALL] Agent error: {e}")
        return RecallCard(narration="I'm having trouble remembering right now.").to_dict()


@app.post("/session/start")
async def start_session(payload: SessionStartPayload) -> dict:
    person = memory.db.get_person(payload.person_id)
    if person:
        person.mark_seen()
        memory.db.save_person(person)
        session_manager.start_session(payload.person_id, person.name)
        log.info(f"[SESSION] Started for {person.name}")
        return {"status": "ok", "person": person.name}
    return {"error": "Person not found"}


@app.post("/session/end")
async def end_session(payload: SessionEndPayload) -> dict:
    import asyncio
    log.info(f"[SESSION] Ending for {payload.person_id} -> consolidation queued")
    asyncio.create_task(session_manager.consolidate_and_end(payload.person_id))
    return {"status": "consolidation_queued"}


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


@app.get("/patient/identity")
async def narrate_patient_identity() -> dict:
    """'Who Am I?' — generates identity narration for the patient."""
    from server.llm.narrator import narrate_patient_identity
    profile = memory.db.get_patient_profile()
    if not profile:
        return {"narration": "I don't have your profile yet. Please ask your caregiver to set it up."}

    # Get recent interactions (last 7 days)
    persons = memory.db.get_all_persons()
    recent = []
    for p in persons[:5]:
        convos = memory.db.get_recent_conversations(p.id, limit=1)
        if convos:
            recent.append({
                "person_name": p.name,
                "relation": p.relation,
                "summary": convos[0].summary,
                "days_ago": 0,
            })

    try:
        narration = await narrate_patient_identity(profile, recent)
        return RecallCard(narration=narration).to_dict()
    except Exception as e:
        log.error(f"Identity narration failed: {e}")
        return RecallCard(narration="I'm having trouble remembering right now.").to_dict()


@app.get("/patient/profile")
async def get_patient_profile() -> dict:
    return memory.db.get_patient_profile()


@app.post("/patient/profile")
async def update_patient_profile(key: str, value: str) -> dict:
    memory.db.update_patient_profile(key, value)
    # Sync name to NetworkX graph root node
    if key == "name":
        memory.graph.set_patient_name(value)
        memory.graph.save()
    return {"status": "ok"}


@app.get("/patient/check")
async def check_patient_profile() -> dict:
    """Check if patient profile is configured. Used for first-run detection."""
    has_profile = memory.db.has_patient_profile()
    return {"has_profile": has_profile, "name": memory.db.get_patient_name()}


@app.post("/patient/voice")
async def enroll_patient_voice(payload: dict) -> dict:
    """
    Enroll patient's voice embedding.
    Expects {"audio": "<base64_pcm>", "sample_rate": 16000}
    """
    import base64 as b64mod
    audio_data = b64mod.b64decode(payload["audio"])
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    sample_rate = payload.get("sample_rate", 16000)

    # Process audio to get speaker embedding
    try:
        result = audio_pipeline.process_chunk(audio_array, sample_rate)
    except Exception as e:
        log.error(f"[PATIENT] Voice processing failed: {e}")
        return {"error": str(e)}

    # Get the first speaker's embedding (patient should be alone)
    if not result.speaker_segments:
        return {"error": "No speech detected"}

    # Find the segment with the longest duration for best embedding
    best_segment = max(result.speaker_segments, key=lambda s: s.end - s.start)
    if best_segment.embedding is None:
        return {"error": "Could not extract voice embedding"}

    # Store as patient voice (use special patient ID)
    patient_id = "patient"
    memory.biometric.add_voice(patient_id, best_segment.embedding)
    memory.biometric.save()

    log.info(f"[PATIENT] Voice enrolled")
    return {"status": "ok", "message": "Patient voice enrolled"}


@app.post("/patient/voice/test")
async def test_patient_voice(payload: dict) -> dict:
    """
    Test if an audio chunk matches the patient's voice.
    Returns {"is_patient": bool, "confidence": float}
    """
    import base64 as b64mod
    audio_data = b64mod.b64decode(payload["audio"])
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    sample_rate = payload.get("sample_rate", 16000)

    try:
        result = audio_pipeline.process_chunk(audio_array, sample_rate)
    except Exception as e:
        return {"error": str(e), "is_patient": False, "confidence": 0.0}

    if not result.speaker_segments:
        return {"is_patient": False, "confidence": 0.0}

    best_segment = max(result.speaker_segments, key=lambda s: s.end - s.start)
    if best_segment.embedding is None:
        return {"is_patient": False, "confidence": 0.0}

    person_id, confidence = memory.biometric.search_voice(best_segment.embedding)
    is_patient = (person_id == "patient")

    log.info(f"[PATIENT] Voice test: is_patient={is_patient} confidence={confidence:.2f}")
    return {"is_patient": is_patient, "confidence": confidence}


# ── WebSocket endpoint ───────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Single WebSocket for real-time frame + audio streaming.
    Client sends JSON messages:
      {"type": "frame", "image": "<base64>"}
      {"type": "audio", "audio": "<base64>", "sample_rate": 16000}
    Server pushes JSON HUD updates:
      {"type": "person", ...}
      {"type": "unknown", ...}
      {"type": "recall", ...}
      {"type": "empty"}
    """
    import base64 as b64mod
    await ws.accept()
    log.info("[WS] Client connected")

    # First-run check
    if not memory.db.has_patient_profile():
        await ws.send_json({"type": "error", "error": "Patient profile not configured. Please set up the patient profile in the caregiver UI first."})
        await ws.close()
        return

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "frame":
                frame = decode_image(msg["image"])
                faces = vision.detect_faces(frame)

                if not faces:
                    await ws.send_json(EmptyCard().to_dict())
                    continue

                log.info(f"[WS:FRAME] {len(faces)} face(s)")

                for face in faces:
                    embedding = VisionPipeline.extract_embedding(face["crop"])
                    if embedding is not None:
                        result = fusion.process_face(embedding)
                        if result:
                            log.info(f"[WS:FRAME] Matched -> {result}")

                best_person = fusion.get_best_match()
                if best_person is None:
                    log.info(f"[WS:FRAME] No match -> Mode 3")
                    import uuid, tempfile
                    for face in faces:
                        face_id = str(uuid.uuid4())
                        crop_path = os.path.join(tempfile.gettempdir(), f"{face_id}.jpg")
                        cv2.imwrite(crop_path, face["crop"])
                        import asyncio
                        asyncio.create_task(_run_auto_enroll(face_id, crop_path, ""))
                    await ws.send_json(UnknownCard().to_dict())
                    continue

                person = memory.db.get_person(best_person.person_id)
                if person is None:
                    await ws.send_json(EmptyCard().to_dict())
                    continue

                log.info(f"[WS:FRAME] Mode 1 -> {person.name} ({best_person.fused_confidence:.2f})")
                person.mark_seen()
                memory.db.save_person(person)
                session_manager.start_session(person.id, person.name)

                recent = memory.db.get_recent_conversations(person.id, limit=2)
                facts = memory.db.get_top_facts(person.id, limit=5)
                from server.hud.renderer import build_person_card
                await ws.send_json(build_person_card(person, recent, facts, memory.db.get_patient_name()).to_dict())

            elif msg_type == "audio":
                audio_data = b64mod.b64decode(msg["audio"])
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                sample_rate = msg.get("sample_rate", 16000)

                try:
                    result = audio_pipeline.process_chunk(audio_array, sample_rate)
                except Exception as e:
                    log.error(f"[WS:AUDIO] Failed: {e}")
                    await ws.send_json({"type": "error", "error": str(e)})
                    continue

                if result.full_text.strip():
                    log.info(f"[WS:AUDIO] \"{result.full_text[:60]}\"")

                for turn in result.speaker_turns:
                    session_manager.add_transcript(turn.get("speaker", "UNKNOWN"), turn.get("text", ""))

                trigger_detected = trigger.check(result.full_text)
                if trigger_detected:
                    log.info(f"[WS:AUDIO] Trigger detected")

                # If trigger detected, identify speaker and route
                if trigger_detected:
                    # Identify who is speaking from the speaker segments
                    is_patient_speaking = False
                    if result.speaker_segments:
                        best_segment = max(result.speaker_segments, key=lambda s: s.end - s.start)
                        if best_segment.embedding is not None:
                            person_id, conf = memory.biometric.search_voice(best_segment.embedding)
                            if person_id == "patient":
                                is_patient_speaking = True
                                log.info(f"[WS:AUDIO] Speaker is PATIENT (conf={conf:.2f})")

                    active = list(session_manager.active.values())
                    session_person_id = active[-1].person_id if active else None

                    try:
                        from server.mode2.agent import run_recall_agent

                        if is_patient_speaking:
                            # Patient speaking -> "Who Am I?" recall
                            log.info(f"[WS:RECALL] Patient trigger -> Who Am I")
                            narration = await run_recall_agent(
                                trigger_phrase=result.full_text,
                                confirmed_person_id=session_person_id,
                                session_context=result.full_text,
                                recall_type="identity",
                            )
                        else:
                            # Other person speaking -> Mode 2 recall about that person
                            log.info(f"[WS:RECALL] Other trigger -> Mode 2")
                            narration = await run_recall_agent(
                                trigger_phrase=result.full_text,
                                confirmed_person_id=session_person_id,
                                session_context=result.full_text,
                                recall_type="person",
                            )

                        log.info(f"[WS:RECALL] \"{narration[:60]}\"")
                        await ws.send_json(RecallCard(narration=narration).to_dict())
                        continue
                    except Exception as e:
                        log.error(f"[WS:RECALL] Failed: {e}")

                # Send transcript update (no card change)
                await ws.send_json({
                    "type": "transcript",
                    "full_text": result.full_text,
                    "speaker_texts": result.speaker_texts,
                    "trigger_detected": trigger_detected,
                })

    except WebSocketDisconnect:
        log.info("[WS] Client disconnected")
    except Exception as e:
        log.error(f"[WS] Error: {e}")
