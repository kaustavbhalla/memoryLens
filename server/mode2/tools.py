"""Tools for the Mode 2 recall agent."""

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
        raise RuntimeError("MemoryStore not bound — call bind_memory() first")
    return _memory


@tool
def get_person_profile(person_id: str) -> dict:
    """
    Retrieve the full profile of a person from the structured store.
    Returns name, relation to patient, last seen date, visit count,
    emotional baseline, and caregiver notes.
    Use this first when you know who you're recalling.
    """
    person = _mem().db.get_person(person_id)
    if not person:
        return {"error": f"Person {person_id} not found"}
    return {
        "id": person.id,
        "name": person.name,
        "relation": person.relation,
        "last_seen": person.last_seen,
        "visit_count": person.visit_count,
        "emotional_baseline": person.emotional_baseline,
        "notes": person.notes,
        "relationship_summary": person.relationship_summary,
    }


@tool
def search_conversation_rag(query: str, person_id: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search over past conversation chunks for a specific person.
    Use when the patient asks about what was discussed, or needs specific
    details from past interactions. Do NOT use for simple who-is-this queries.
    """
    results = _mem().vector.search(query=query, person_id=person_id, top_k=top_k)
    return [{k: v for k, v in r.items() if k != "vector"} for r in results]


@tool
def get_atomic_facts(person_id: str) -> list[dict]:
    """
    Retrieve all stored atomic facts about a person, ordered by confidence.
    Examples: "works at a hospital in Bangalore", "has two children".
    Use when you need concrete details to anchor the patient's memory.
    """
    facts = _mem().db.get_all_active_facts(person_id)
    return [
        {"id": f.id, "fact": f.fact_text, "confidence": f.confidence, "timestamp": f.timestamp}
        for f in facts
    ]


@tool
def get_social_context(person_id: str) -> dict:
    """
    Query the relationship graph for who this person knows and how.
    Returns connections to other people in the patient's life.
    Use when profile + facts alone feel insufficient.
    """
    return _mem().graph.get_social_context_for_recall(person_id)


@tool
def narrow_unknown_face(co_present_person_ids: list[str]) -> list[str]:
    """
    When an unrecognized face is present alongside known people, use the
    relationship graph to return candidate person_ids connected within 2 hops.
    Use ONLY when there is an unidentified face AND known people are co-present.
    """
    return _mem().graph.narrow_unknown_face(co_present_person_ids)


@tool
def get_patient_profile() -> dict:
    """
    Retrieve the patient's own identity profile: name, former occupation,
    hometown, hobbies, important life events.
    Use when the patient is confused about their own identity.
    """
    return _mem().db.get_patient_profile()


@tool
def generate_narration(
    person_name: str,
    relation_to_patient: str,
    key_facts: list[str],
    social_anchors: list[str],
    conversation_highlights: list[str],
    patient_name: str,
) -> str:
    """
    Generate the final warm narration to display on the HUD.
    Call this LAST, after you have gathered sufficient context.
    Returns 2-3 sentence narration string, ready for HUD display.
    """
    import anthropic
    from server.config import ANTHROPIC_MODEL, ANTHROPIC_MAX_TOKENS_RECALL

    client = anthropic.Anthropic()
    facts_str = "\n".join(f"- {f}" for f in key_facts) if key_facts else "- None available"
    anchors_str = "\n".join(f"- {a}" for a in social_anchors) if social_anchors else "- None available"
    highlights_str = "\n".join(f"- {h}" for h in conversation_highlights) if conversation_highlights else "- None available"

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS_RECALL,
        messages=[{
            "role": "user",
            "content": f"""You are a memory assistant for a dementia patient named {patient_name}.
Generate a SHORT, warm 2-3 sentence narration helping them remember the person
in front of them. Use only the facts provided. Simple language. Present tense.
Do NOT mention memory loss, dementia, or that you are an AI.
Use social anchors to make connections feel familiar, not clinical.

Person: {person_name} ({relation_to_patient})
Facts:
{facts_str}
Social anchors:
{anchors_str}
Conversation highlights:
{highlights_str}"""
        }],
    )
    return msg.content[0].text.strip()


RECALL_TOOLS = [
    get_person_profile,
    search_conversation_rag,
    get_atomic_facts,
    get_social_context,
    narrow_unknown_face,
    get_patient_profile,
    generate_narration,
]
