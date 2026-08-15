"""Tools for the Mode 3 auto-enrollment agent."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from server.memory.store import MemoryStore

_memory: "MemoryStore | None" = None


def bind_memory(store: "MemoryStore") -> None:
    global _memory
    _memory = store


def _mem() -> "MemoryStore":
    if _memory is None:
        raise RuntimeError("MemoryStore not bound")
    return _memory


@tool
def extract_name_from_context(session_context: str, speaker_label: str) -> dict:
    """
    Analyze conversation context to extract the name used for the unknown speaker.
    Returns: {name: str | None, confidence: float, evidence: str}
    Confidence: 1.0 = self-introduced, 0.9 = addressed multiple times,
                0.7 = addressed once, 0.4 = inferred, 0.0 = not found.
    """
    import anthropic
    from server.config import ANTHROPIC_MODEL

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Extract the name being used for the unknown speaker labeled {speaker_label}.

Conversation:
{session_context}

Return ONLY valid JSON:
{{"name": "Arjun" or null, "confidence": 0.0-1.0, "evidence": "quote from transcript"}}"""
        }],
    )
    import json
    return json.loads(msg.content[0].text)


@tool
def infer_relation_from_context(
    unknown_speaker_label: str,
    session_context: str,
    patient_name: str,
) -> dict:
    """
    Infer the relationship between the unknown person and the patient.
    Returns: {relation: str, confidence: float, evidence: str}
    Be conservative — return "unknown" with low confidence rather than guessing.
    """
    import anthropic, json
    from server.config import ANTHROPIC_MODEL

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Infer the relationship between the unknown speaker ({unknown_speaker_label})
and the patient ({patient_name}) from this conversation.

Conversation:
{session_context}

Return ONLY valid JSON:
{{"relation": "daughter", "confidence": 0.0-1.0, "evidence": "patient called them beta"}}"""
        }],
    )
    return json.loads(msg.content[0].text)


@tool
def check_embedding_quality(
    face_crop_path: str,
    audio_segment_seconds: float,
) -> dict:
    """
    Check whether biometric data is sufficient for enrollment.
    Face: frontal, unblurred, >= 80x80px. Audio: >= 8 seconds.
    Returns: {face_ok, voice_ok, face_reason, voice_reason}
    """
    import cv2
    from server.config import AUTO_ENROLL_MIN_FACE_PX, AUTO_ENROLL_MIN_AUDIO_S

    face_ok = True
    voice_ok = audio_segment_seconds >= AUTO_ENROLL_MIN_AUDIO_S
    face_reason = "sufficient"
    voice_reason = "sufficient" if voice_ok else f"too short: {audio_segment_seconds:.1f}s"

    try:
        img = cv2.imread(face_crop_path)
        if img is None:
            face_ok = False
            face_reason = "could not read image"
        else:
            h, w = img.shape[:2]
            if h < AUTO_ENROLL_MIN_FACE_PX or w < AUTO_ENROLL_MIN_FACE_PX:
                face_ok = False
                face_reason = f"too small: {w}x{h}px"
    except Exception as e:
        face_ok = False
        face_reason = str(e)

    return {
        "face_ok": face_ok,
        "voice_ok": voice_ok,
        "face_reason": face_reason,
        "voice_reason": voice_reason,
    }


@tool
def create_provisional_profile(
    name: str,
    name_confidence: float,
    relation: str,
    relation_confidence: float,
    face_crop_path: str,
) -> dict:
    """
    Create a new Person with enrollment_status=AUTO.
    Extracts and stores face embedding into FAISS.
    Flags profile for caregiver review.
    """
    from server.memory.structured import Person, EnrollmentStatus
    from server.pipelines.vision import VisionPipeline
    import cv2

    person_id = str(uuid.uuid4())

    # Extract face embedding
    img = cv2.imread(face_crop_path)
    if img is None:
        return {"success": False, "reason": "could not read face crop"}

    embedding = VisionPipeline.extract_embedding(img)
    if embedding is None:
        return {"success": False, "reason": "embedding extraction failed"}

    _mem().biometric.add_face(person_id, embedding)
    _mem().biometric.save()

    person = Person(
        id=person_id,
        name=name if name_confidence >= 0.5 else "Unknown Person",
        relation=relation,
        relation_confidence=relation_confidence,
        name_confidence=name_confidence,
        enrollment_status=EnrollmentStatus.AUTO,
        visit_count=1,
    )
    _mem().db.save_person(person)

    return {
        "success": True,
        "person_id": person_id,
        "enrollment_status": "auto",
        "hud_display_name": name if name_confidence >= 0.75 else "Someone new",
    }


AUTO_ENROLL_TOOLS = [
    extract_name_from_context,
    infer_relation_from_context,
    check_embedding_quality,
    create_provisional_profile,
]
