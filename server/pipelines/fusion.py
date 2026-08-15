"""Identity fusion engine — combines face + voice to confirm identity."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from server.config import (
    FACE_WEIGHT,
    VOICE_WEIGHT,
    FUSION_CONFIRMATION_THRESHOLD,
    FACE_RECOGNITION_THRESHOLD,
    VOICE_RECOGNITION_THRESHOLD,
)
from server.memory.store import memory

log = logging.getLogger("memorylens.fusion")


@dataclass
class ActivePerson:
    person_id: str
    name: str
    face_confidence: float = 0.0
    voice_confidence: float = 0.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_confirmed: datetime = field(default_factory=datetime.now)

    @property
    def fused_confidence(self) -> float:
        if self.face_confidence > 0 and self.voice_confidence > 0:
            return FACE_WEIGHT * self.face_confidence + VOICE_WEIGHT * self.voice_confidence
        elif self.face_confidence > 0:
            return self.face_confidence * 0.85
        elif self.voice_confidence > 0:
            return self.voice_confidence * 0.7
        return 0.0

    @property
    def is_confirmed(self) -> bool:
        return self.fused_confidence >= FUSION_CONFIRMATION_THRESHOLD


class IdentityFusionEngine:
    """
    Maintains the set of people currently present in the scene.
    Fuses face + voice signals to produce confirmed identity.

    Face match alone → tentative identity (0.85x penalty)
    Voice match alone → tentative identity (0.7x penalty)
    Both match → high confidence (weighted sum)
    """

    def __init__(self):
        self.active_persons: dict[str, ActivePerson] = {}

    def process_face(self, face_embedding: np.ndarray) -> str | None:
        """Match a face embedding against FAISS store. Returns person_id or None."""
        person_id, confidence = memory.biometric.search_face(face_embedding)
        if person_id is None:
            return None

        if person_id in self.active_persons:
            self.active_persons[person_id].face_confidence = confidence
            self.active_persons[person_id].last_confirmed = datetime.now()
        else:
            person = memory.db.get_person(person_id)
            self.active_persons[person_id] = ActivePerson(
                person_id=person_id,
                name=person.name if person else "Unknown",
                face_confidence=confidence,
            )

        if confidence >= FACE_RECOGNITION_THRESHOLD:
            return person_id
        return None

    def process_voice(self, voice_embedding: np.ndarray) -> str | None:
        """Match a voice embedding against FAISS store. Returns person_id or None."""
        person_id, confidence = memory.biometric.search_voice(voice_embedding)
        if person_id is None:
            return None

        if person_id in self.active_persons:
            self.active_persons[person_id].voice_confidence = confidence
            self.active_persons[person_id].last_confirmed = datetime.now()
        else:
            person = memory.db.get_person(person_id)
            self.active_persons[person_id] = ActivePerson(
                person_id=person_id,
                name=person.name if person else "Unknown",
                voice_confidence=confidence,
            )

        if confidence >= VOICE_RECOGNITION_THRESHOLD:
            return person_id
        return None

    def get_confirmed_persons(self) -> list[ActivePerson]:
        return [p for p in self.active_persons.values() if p.is_confirmed]

    def get_best_match(self) -> ActivePerson | None:
        """Return the person with highest fused confidence."""
        confirmed = self.get_confirmed_persons()
        if not confirmed:
            return None
        return max(confirmed, key=lambda p: p.fused_confidence)

    def get_unknown_faces(self, face_embeddings: list[np.ndarray]) -> list[np.ndarray]:
        """Return face embeddings that did NOT match any known person."""
        unknown = []
        for emb in face_embeddings:
            person_id, _ = memory.biometric.search_face(emb)
            if person_id is None:
                unknown.append(emb)
        return unknown

    def prune_stale(self, max_age_s: float = 300.0):
        """Remove persons not seen for max_age_s seconds."""
        now = datetime.now()
        stale = [
            pid for pid, p in self.active_persons.items()
            if (now - p.last_confirmed).total_seconds() > max_age_s
        ]
        for pid in stale:
            del self.active_persons[pid]

    def reset(self):
        self.active_persons.clear()
