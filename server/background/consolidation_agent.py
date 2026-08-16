"""LangGraph memory consolidation agent."""

from __future__ import annotations

import uuid
import logging

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

from server.config import ANTHROPIC_MODEL, ANTHROPIC_MAX_TOKENS_RECALL, CONSOLIDATION_MAX_AGENT_STEPS
from server.background.tools import CONSOLIDATION_TOOLS, bind_memory
from server.background.prompts import CONSOLIDATION_SYSTEM_PROMPT
from server.memory.store import memory

log = logging.getLogger("memorylens.consolidation")

bind_memory(memory)

_llm = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=ANTHROPIC_MAX_TOKENS_RECALL)

consolidation_agent = create_react_agent(
    model=_llm,
    tools=CONSOLIDATION_TOOLS,
    prompt=CONSOLIDATION_SYSTEM_PROMPT,
)


def _chunk_transcript(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split transcript into fixed-size word chunks for LanceDB."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


async def run_consolidation(
    person_id: str,
    person_name: str,
    new_transcript: str,
    new_episode_summary: str,
    emotional_tone: str,
    key_topics: list[str],
    new_facts: list[dict],
    inter_person_relations: list[dict] | None = None,
) -> None:
    """
    Entry point for post-session consolidation.
    Runs as asyncio background task — fire and forget.
    """
    conversation_id = str(uuid.uuid4())
    chunks = _chunk_transcript(new_transcript)

    log.info(f"Consolidating session with {person_name} ({len(chunks)} chunks, {len(new_facts)} facts)")

    try:
        await consolidation_agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"""Consolidate the new session with {person_name} (person_id: {person_id}).

Pre-extracted episode summary: {new_episode_summary}
Emotional tone: {emotional_tone}
Key topics: {key_topics}
New facts extracted: {new_facts}
Conversation ID for storage: {conversation_id}

Perform consolidation."""
            }]
        }, config={"recursion_limit": CONSOLIDATION_MAX_AGENT_STEPS})
        log.info(f"Consolidation complete for {person_name}")
    except Exception as e:
        log.error(f"Consolidation failed for {person_name}: {e}")

    # Write inter-person relations to graph (outside agent — direct call)
    if inter_person_relations:
        for rel in inter_person_relations:
            a_id = _resolve_name_to_id(rel.get("person_a_name", ""))
            b_id = _resolve_name_to_id(rel.get("person_b_name", ""))
            if a_id and b_id:
                memory.graph.add_inter_person_edge(
                    a_id, b_id,
                    rel.get("relation", "knows"),
                    rel.get("confidence", 0.8),
                    "extracted",
                )
        memory.graph.save()


def _resolve_name_to_id(name: str) -> str | None:
    """Resolve a person name to their ID from the database."""
    persons = memory.db.get_all_persons(include_unconfirmed=True)
    name_lower = name.lower().strip()
    for p in persons:
        if p.name.lower() == name_lower:
            return p.id
    return None
