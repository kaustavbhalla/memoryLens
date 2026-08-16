"""Consolidation agent tools."""

from __future__ import annotations

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
def get_existing_relationship_summary(person_id: str) -> dict:
    """
    Retrieve the current relationship_summary for a person.
    Returns: {summary: str | None, last_updated: str | None, episode_count: int}
    If None, this is the first session — no prior summary exists.
    """
    person = _mem().db.get_person(person_id)
    if not person:
        return {"error": f"Person {person_id} not found"}
    convos = _mem().db.get_recent_conversations(person_id, limit=100)
    return {
        "summary": person.relationship_summary,
        "last_updated": person.relationship_summary_updated,
        "episode_count": len(convos),
    }


@tool
def get_all_active_facts(person_id: str) -> list[dict]:
    """
    Retrieve all non-outdated atomic facts for a person.
    Returns: [{id, fact_text, confidence, timestamp}]
    """
    facts = _mem().db.get_all_active_facts(person_id)
    return [
        {"id": f.id, "fact": f.fact_text, "confidence": f.confidence, "timestamp": f.timestamp}
        for f in facts
    ]


@tool
def mark_fact_outdated(fact_id: str, superseded_by_text: str) -> dict:
    """
    Mark an existing atomic fact as outdated and record what superseded it.
    Returns: {success: bool}
    """
    from server.memory.structured import AtomicFact, utcnow
    _mem().db.mark_fact_outdated(fact_id, superseded_by_text)
    return {"success": True}


@tool
def store_episode_and_chunks(
    person_id: str,
    conversation_id: str,
    episode_summary: str,
    emotional_tone: str,
    key_topics: list[str],
    new_facts: list[dict],
) -> dict:
    """
    Write the new conversation episode to SQLite and chunk the transcript
    into LanceDB for future RAG.
    new_facts: [{fact_text, confidence}] — facts extracted from new session only.
    Returns: {episode_id: str, facts_stored: int}
    """
    from server.memory.structured import Conversation, AtomicFact, EmotionalTone

    # Create conversation record
    tone_map = {"warm": EmotionalTone.WARM, "neutral": EmotionalTone.NEUTRAL, "stressful": EmotionalTone.STRESSFUL}
    conv = Conversation(
        id=conversation_id,
        person_id=person_id,
        summary=episode_summary,
        emotional_tone=tone_map.get(emotional_tone, EmotionalTone.NEUTRAL),
    )
    conv.set_topics(key_topics)
    _mem().db.save_conversation(conv)

    # Store atomic facts
    for f in new_facts:
        fact = AtomicFact(
            person_id=person_id,
            fact_text=f.get("fact", ""),
            confidence=f.get("confidence", 0.8),
            source_conversation_id=conversation_id,
        )
        _mem().db.save_fact(fact)

    return {"episode_id": conversation_id, "facts_stored": len(new_facts)}


@tool
def write_relationship_summary(person_id: str, new_summary: str) -> dict:
    """
    Overwrite Person.relationship_summary with the consolidated summary.
    Call this LAST, after all fact conflicts are resolved.
    The summary should be 4-6 sentences covering:
    - Who this person is to the patient
    - The arc of the relationship
    - The most recent interaction highlights
    - Any important upcoming events or commitments
    Returns: {success: bool, updated_at: str}
    """
    _mem().db.update_relationship_summary(person_id, new_summary)
    return {"success": True}


CONSOLIDATION_TOOLS = [
    get_existing_relationship_summary,
    get_all_active_facts,
    mark_fact_outdated,
    store_episode_and_chunks,
    write_relationship_summary,
]
