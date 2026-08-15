"""LangGraph auto-enrollment agent definition."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

from server.config import ANTHROPIC_MODEL, ANTHROPIC_MAX_TOKENS_ENROLL, AUTO_ENROLL_MAX_AGENT_STEPS
from server.mode3.tools import AUTO_ENROLL_TOOLS, bind_memory
from server.mode3.prompts import AUTO_ENROLL_SYSTEM_PROMPT
from server.memory.store import memory

bind_memory(memory)

_llm = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=ANTHROPIC_MAX_TOKENS_ENROLL)

auto_enroll_agent = create_react_agent(
    model=_llm,
    tools=AUTO_ENROLL_TOOLS,
    prompt=AUTO_ENROLL_SYSTEM_PROMPT,
)


async def run_auto_enroll(
    unknown_face_id: str,
    face_crop_path: str,
    session_context: str,
    speaker_label: str = "SPEAKER_01",
    patient_name: str = "Patient",
) -> dict:
    """
    Entry point for Mode 3. Runs as a background task.
    Returns enrollment result dict.
    """
    result = await auto_enroll_agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": f"""Unknown person detected.
Face ID: {unknown_face_id}
Face crop: {face_crop_path}
Patient name: {patient_name}
Speaker label: {speaker_label}

Recent conversation context:
{session_context}

Attempt to create a provisional profile."""
        }]
    }, config={"recursion_limit": AUTO_ENROLL_MAX_AGENT_STEPS})
    return result
