"""HUD card builder — returns structured JSON for webcam overlay."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from server.memory.structured import Person, Conversation, AtomicFact
from server.config import HUD_PERSON_CARD_MAX_CHARS


@dataclass
class PersonCard:
    type: Literal["person"] = "person"
    name: str = ""
    relation: str = ""
    last_seen: str = ""
    summary: str = ""
    enrollment_status: str = "confirmed"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "relation": self.relation,
            "last_seen": self.last_seen,
            "summary": self.summary,
            "enrollment_status": self.enrollment_status,
        }


@dataclass
class RecallCard:
    type: Literal["recall"] = "recall"
    narration: str = ""

    def to_dict(self) -> dict:
        return {"type": self.type, "narration": self.narration}


@dataclass
class UnknownCard:
    type: Literal["unknown"] = "unknown"
    label: str = "Someone new"

    def to_dict(self) -> dict:
        return {"type": self.type, "label": self.label}


@dataclass
class EmptyCard:
    type: Literal["empty"] = "empty"

    def to_dict(self) -> dict:
        return {"type": self.type}


HUDCard = PersonCard | RecallCard | UnknownCard | EmptyCard


def _format_relative_time(iso_str: str) -> str:
    """Convert ISO timestamp to '3 days ago', 'just now', etc."""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        days = seconds // 86400
        if days == 1:
            return "yesterday"
        if days < 30:
            return f"{days}d ago"
        months = days // 30
        return f"{months}mo ago"
    except Exception:
        return "recently"


def build_person_card(
    person: Person,
    conversations: list[Conversation],
    facts: list[AtomicFact],
    patient_name: str = "",
) -> PersonCard:
    """Build a compact HUD card from database records."""
    last_seen = _format_relative_time(person.last_seen)

    if person.relationship_summary:
        summary = person.relationship_summary
    elif conversations:
        summary = conversations[0].summary
    else:
        summary = "No recent conversations"

    if len(summary) > HUD_PERSON_CARD_MAX_CHARS:
        summary = summary[:HUD_PERSON_CARD_MAX_CHARS] + "..."

    # Use patient-relative phrasing ("your daughter" instead of just "daughter")
    relation = person.relation
    if relation and patient_name:
        # Add "your" prefix if not already present
        if not relation.lower().startswith("your "):
            relation = f"your {relation}"

    return PersonCard(
        name=person.display_name,
        relation=relation,
        last_seen=last_seen,
        summary=summary,
        enrollment_status=person.enrollment_status.value,
    )
