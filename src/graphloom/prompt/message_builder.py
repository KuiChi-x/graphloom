"""
llm_message_builder.py

Single source of truth for assembling the per-turn prompt messages sent to the
agent LLM. Both `ai_node` (at invocation time) and `context_compaction_node`
(for token estimation) must go through this function so the token budget and
the real payload stay in sync.
"""
from datetime import datetime
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from graphloom.model.state import AgentState
from graphloom.prompt.stack import PromptStack
from graphloom.prompt.context_renderer import (
    build_prompt_context,
    build_user_request_str,
)


async def build_llm_messages(state: AgentState, prompt_stack: PromptStack) -> List[BaseMessage]:
    messages: List[BaseMessage] = []

    system_message: str = await prompt_stack.build_system_messages()
    messages.append(SystemMessage(content=system_message))

    current_time = datetime.now().isoformat()
    prompt_context = build_prompt_context(
        state,
        current_time=current_time,
        todo_contents=state.get("todo_contents") or "",
    )
    messages.append(HumanMessage(content=prompt_context))

    observer_message_parts = list(state.get("observer_message_parts", []) or [])
    if observer_message_parts:
        messages.extend(observer_message_parts)

    user_request = build_user_request_str(state)
    attach_message_parts = list(state.get("attach_message_parts") or [])
    messages.append(
        HumanMessage(
            content=[
                {"type": "text", "text": user_request},
                *attach_message_parts,
            ]
        )
    )
    return messages
