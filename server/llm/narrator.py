"""'Who Am I?' identity narration — fixed single call, no agent needed."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from server.config import OPENCODE_API_KEY, OPENCODE_BASE_URL, OPENCODE_MODEL, OPENCODE_MAX_TOKENS_NARRATE

log = logging.getLogger("memorylens.narrator")

IDENTITY_PROMPT = """You are helping a dementia patient remember who they are.
Write a gentle 4-5 sentence narration in second person ("Your name is...").
Start with their name and something they love. End with who visited recently.
Do NOT mention memory loss. Write as if reminding a friend.

Patient profile: {profile}
Recent interactions (last 7 days): {recent}"""


async def narrate_patient_identity(profile: dict, recent: list[dict]) -> str:
    client = OpenAI(api_key=OPENCODE_API_KEY, base_url=OPENCODE_BASE_URL)

    recent_text = "\n".join(
        f"- {r.get('person_name', 'Someone')} ({r.get('relation', 'unknown')}): "
        f"{r.get('summary', '')} [{r.get('days_ago', '?')}d ago]"
        for r in recent
    ) if recent else "- No recent interactions"

    msg = client.chat.completions.create(
        model=OPENCODE_MODEL,
        max_tokens=OPENCODE_MAX_TOKENS_NARRATE,
        messages=[{
            "role": "user",
            "content": IDENTITY_PROMPT.format(
                profile=json.dumps(profile, indent=2),
                recent=recent_text,
            )
        }],
    )
    return msg.choices[0].message.content.strip()
