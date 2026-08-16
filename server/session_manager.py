"""Session lifecycle manager — tracks active sessions, fires consolidation on end."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from server.memory.store import memory
from server.llm.extractor import extract_session_data

log = logging.getLogger("memorylens.session")


@dataclass
class ActiveSession:
    person_id: str
    person_name: str
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    transcript_parts: list[str] = field(default_factory=list)

    def add_transcript(self, text: str):
        self.transcript_parts.append(text)
        self.last_activity = time.time()

    @property
    def full_transcript(self) -> str:
        return " ".join(self.transcript_parts)

    @property
    def duration_s(self) -> float:
        return time.time() - self.started_at


class SessionManager:
    """
    Tracks active sessions per person.
    Fires consolidation when:
      - session/end is called explicitly
      - person is not seen for > 5 minutes (silence timeout)
    """

    def __init__(self, silence_timeout_s: float = 300.0):
        self.active: dict[str, ActiveSession] = {}
        self.silence_timeout_s = silence_timeout_s

    def start_session(self, person_id: str, person_name: str) -> None:
        if person_id in self.active:
            self.active[person_id].last_activity = time.time()
            return
        self.active[person_id] = ActiveSession(person_id=person_id, person_name=person_name)
        log.info(f"Session started: {person_name}")

    def add_transcript(self, person_id: str, text: str) -> None:
        if person_id in self.active:
            self.active[person_id].add_transcript(text)

    def end_session(self, person_id: str) -> ActiveSession | None:
        session = self.active.pop(person_id, None)
        if session:
            log.info(f"Session ended: {session.person_name} ({session.duration_s:.0f}s)")
        return session

    def check_silence_timeouts(self) -> list[str]:
        """Return person_ids of sessions that timed out."""
        now = time.time()
        timed_out = []
        for pid, session in list(self.active.items()):
            if now - session.last_activity > self.silence_timeout_s:
                timed_out.append(pid)
        return timed_out

    async def consolidate_and_end(self, person_id: str) -> None:
        """Extract data from session and fire consolidation agent."""
        session = self.end_session(person_id)
        if not session:
            return

        transcript = session.full_transcript.strip()
        if not transcript:
            log.info(f"No transcript for {session.person_name}, skipping consolidation")
            return

        # Get known people for extraction context
        known_people = [
            {"id": p.id, "name": p.name}
            for p in memory.db.get_all_persons(include_unconfirmed=True)
        ]

        try:
            # Extract facts from transcript
            extracted = await extract_session_data(transcript, known_people)
        except Exception as e:
            log.error(f"Extraction failed for {session.person_name}: {e}")
            return

        # Fire consolidation agent in background
        from server.background.consolidation_agent import run_consolidation
        asyncio.create_task(run_consolidation(
            person_id=person_id,
            person_name=session.person_name,
            new_transcript=transcript,
            new_episode_summary=extracted.get("episode_summary", ""),
            emotional_tone=extracted.get("emotional_tone", "neutral"),
            key_topics=extracted.get("key_topics", []),
            new_facts=extracted.get("atomic_facts", []),
            inter_person_relations=extracted.get("inter_person_relations", []),
        ))


# Module-level singleton
session_manager = SessionManager()
