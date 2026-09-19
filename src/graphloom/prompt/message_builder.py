"""message_builder.py

Single source of truth for the per-turn LLM payload. Both `ai_node` (at
invocation time) and `context_compaction_node` (for token estimation) go
through this function so the budget and the real payload stay in sync.

Layout — everything before the tail must be byte-stable so the provider's
prefix cache keeps matching:

    SystemMessage      static: system prompt + skills
    *state.messages    native history, append-only (thinking + signature
                       intact, so the model reads its own prior reasoning
                       instead of re-deriving it from a paraphrase). The user's
                       request lives here, at its position — it is never
                       restated or copied elsewhere. Cache breakpoints are
                       placed on turn boundaries here, because this is the part
                       that grows; the system block alone is not where the win
                       is.
    HumanMessage       volatile tail: env (OS/shell/cwd/clock), todo,
                       manifests, delivery status

The tail is last on purpose, and it is a *user* turn because the protocol has
no other role for framework-supplied context. On the Anthropic wire it merges
into the same `user` turn as the trailing tool_result blocks, so the request
ends `[tool_result..., text]` — one user turn, not two — which is exactly the
shape Anthropic's own tool-use docs show for injected per-turn context. Putting
volatile text last also keeps it out of the cached prefix: refreshed clock and
manifests at the front would invalidate the whole history every single turn.
"""
from datetime import datetime
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from graphloom.model.state import AgentState
from graphloom.prompt.context_renderer import (
    apply_history_breakpoints,
    build_prompt_context,
    cache_block,
)
from graphloom.prompt.stack import PromptStack


async def build_llm_messages(
    state: AgentState,
    prompt_stack: PromptStack,
    llm: BaseChatModel | None = None,
) -> List[BaseMessage]:
    messages: List[BaseMessage] = [
        SystemMessage(content=[cache_block(
            await prompt_stack.build_system_messages(),
            llm,
        )]),
        *apply_history_breakpoints(list(state.get("messages", []) or []), llm),
    ]

    tail = build_prompt_context(
        state,
        current_time=datetime.now().isoformat(),
        todo_contents=state.get("todo_contents") or "",
    )

    messages.extend(list(state.get("observer_message_parts", []) or []))
    messages.append(HumanMessage(content=[{"type": "text", "text": tail}]))
    return messages
