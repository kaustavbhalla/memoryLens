"""LangGraph recall agent definition."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

from server.config import ANTHROPIC_MODEL, ANTHROPIC_MAX_TOKENS_RECALL
from server.mode2.tools import RECALL_TOOLS, bind_memory
from server.mode2.prompts import RECALL_AGENT_SYSTEM_PROMPT
from server.memory.store import memory

# Bind memory store to tools
bind_memory(memory)

_llm = ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=ANTHROPIC_MAX_TOKENS_RECALL)

recall_agent = create_react_agent(
    model=_llm,
    tools=RECALL_TOOLS,
    prompt=RECALL_AGENT_SYSTEM_PROMPT,
)


async def run_recall_agent(
    trigger_phrase: str,
    confirmed_person_id: str | None = None,
    unknown_face_present: bool = False,
    co_present_ids: list[str] | None = None,
    session_context: str = "",
) -> str:
    """
    Entry point for Mode 2. Returns the narration string for HUD display.
    """
    user_message = f"""The patient just said: "{trigger_phrase}"

Current scene context:
- Confirmed person present: {confirmed_person_id or 'None'}
- Unknown face also present: {unknown_face_present}
- Other confirmed people present: {co_present_ids or []}
- Recent conversation: {session_context}

Retrieve the necessary memory and generate a narration to help the patient."""

    result = await recall_agent.ainvoke({
        "messages": [{"role": "user", "content": user_message}]
    })
    return result["messages"][-1].content
