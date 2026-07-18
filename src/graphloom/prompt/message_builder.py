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
from graphloom.prompt.context_renderer import (
    build_prompt_context,
    build_user_request_str,
    render_past_steps,
)
from graphloom.prompt.stack import PromptStack


async def build_llm_messages(state: AgentState, prompt_stack: PromptStack) -> List[BaseMessage]:
    messages: List[BaseMessage] = []

    system_message: str = await prompt_stack.build_system_messages()
    messages.append(SystemMessage(content=system_message))

    past_steps = list(state.get("past_steps", []) or [])
    history_blocks = render_past_steps(past_steps)
    if len(history_blocks) > 1:
        messages.append(HumanMessage(content=[
            *({"type": "text", "text": block} for block in history_blocks[:-1]),
        ]))
        messages.append(HumanMessage(content="</agent_history>"))
    else:
        messages.append(HumanMessage(content=history_blocks[0]))

    conversation = list(state.get("conversation", []) or [])
    if conversation and isinstance(conversation[-1], HumanMessage):
        conversation = conversation[:-1]
    messages.extend(conversation)

    current_hour = datetime.now().replace(minute=0, second=0, microsecond=0).isoformat()
    prompt_context = build_prompt_context(
        state,
        current_time=current_hour,
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
