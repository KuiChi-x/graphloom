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

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from graphloom.model.state import AgentState
from graphloom.prompt.context_renderer import (
    apply_history_breakpoints,
    build_prompt_context,
    cache_block,
)
from graphloom.prompt.stack import PromptStack


def _repair_orphan_tool_calls(messages: List[BaseMessage]) -> List[BaseMessage]:
    """孤儿对账：给没收到回答的 tool_call 就地补一条错误 ToolMessage。

    暂停/崩溃可以留下"带 tool_calls 的 AIMessage 已进 checkpoint、ToolMessages
    还没写"的历史（ToolMessages 是 tool 节点整轮跑完才一次性返回的）。Responses /
    Anthropic 协议都要求每个 function_call 有配对 output，这种历史会被网关确定性
    地 400（No tool output found for function call ...），此后每轮回放都如此。

    只修补发给 LLM 的 payload，不回写 state；补位就放在所属 AIMessage 的真实
    回答之前，保证 output 不会掉出它所属的轮。
    """
    answered = {
        str(message.tool_call_id)
        for message in messages
        if isinstance(message, ToolMessage)
    }
    repaired: List[BaseMessage] = []
    patched = 0
    for message in messages:
        repaired.append(message)
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            call_id = str(call.get("id") or "")
            if not call_id or call_id in answered:
                continue
            repaired.append(ToolMessage(
                content="[graphloom] 该工具调用在上次运行中被中断，未执行，没有结果。如需该结果请重新调用。",
                tool_call_id=call_id,
                name=str(call.get("name") or ""),
                status="error",
            ))
            answered.add(call_id)
            patched += 1
    if patched:
        logging.warning(
            "[message_builder] 历史中存在 %d 个无配对的 tool_call，已在 payload 里合成错误结果修补",
            patched,
        )
    return repaired


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
        *apply_history_breakpoints(
            _repair_orphan_tool_calls(list(state.get("messages", []) or [])), llm
        ),
    ]

    tail = build_prompt_context(
        state,
        current_time=datetime.now().isoformat(),
        todo_contents=state.get("todo_contents") or "",
    )

    messages.extend(list(state.get("observer_message_parts", []) or []))
    messages.append(HumanMessage(content=[{"type": "text", "text": tail}]))
    return messages
