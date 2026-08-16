"""Fact extraction + inter-person relation extraction from transcripts."""

from __future__ import annotations

import json
import logging

import anthropic

from server.config import ANTHROPIC_MODEL, ANTHROPIC_MAX_TOKENS_EXTRACT

log = logging.getLogger("memorylens.extractor")

FACT_EXTRACTION_PROMPT = """You are analyzing a conversation transcript involving a dementia patient.
Extract structured information that will help the patient's memory system.

Return ONLY valid JSON:
{{
  "episode_summary": "2-3 sentence summary of what a patient should remember",
  "emotional_tone": "warm|neutral|stressful",
  "key_topics": ["topic1", "topic2"],
  "atomic_facts": [
    {{"fact": "Sarah works at a hospital in Bangalore", "confidence": 0.95}},
    {{"fact": "Sarah is bringing biryani next Sunday", "confidence": 0.8}}
  ],
  "patient_updates": [
    {{"key": "favorite_food", "value": "chai and samosas"}}
  ],
  "inter_person_relations": [
    {{
      "person_a_name": "Sarah",
      "person_b_name": "Dr. Mehta",
      "relation": "works_with",
      "confidence": 0.85,
      "evidence": "Sarah said 'my colleague Dr. Mehta at the hospital'"
    }}
  ]
}}

Relation types: "knows", "related_to", "works_with", "mentioned_together"
Only include inter_person_relations with clear evidence.

Conversation (PATIENT = patient, other labels = visitor speakers):
{transcript}

Known people in this patient's life:
{known_people}"""


async def extract_session_data(
    transcript: str,
    known_people: list[dict],
) -> dict:
    """Single Claude call extracts everything needed to update all stores."""
    client = anthropic.Anthropic()
    known_str = "\n".join(f"- {p['name']} (id: {p['id']})" for p in known_people)

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS_EXTRACT,
        messages=[{
            "role": "user",
            "content": FACT_EXTRACTION_PROMPT.format(
                transcript=transcript,
                known_people=known_str,
            )
        }],
    )

    try:
        return json.loads(msg.content[0].text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        text = msg.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())
