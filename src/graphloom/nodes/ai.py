"""
ai_node.py — minimal shared AI node for all expert agents.

Streams the bound LLM, merges the response, and records a pending step when the
LLM calls tools. Token streaming to any external sink is the host's concern
(wire LangGraph callbacks via RunnableConfig) — this node emits nothing.
"""
import logging
import time
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from graphloom.events import emit_step
from graphloom.model.state import AgentState
from graphloom.model.timeline import build_turn, step_id_for
from graphloom.nodes.interrupt_guard import raise_if_cancelled
from graphloom.prompt.message_builder import build_llm_messages
from graphloom.prompt.stack import PromptStack
from graphloom.util.model_info import provider_of, supports_explicit_cache_breakpoints


def _chunk_parts(message: Any) -> tuple[str, str]:
    """Read text and reasoning from LangChain's standard content blocks."""
    text: List[str] = []
    reasoning: List[str] = []
    for block in message.content_blocks:
        if block.get("type") == "text":
            text.append(str(block.get("text") or ""))
        elif block.get("type") == "reasoning":
            reasoning.append(str(block.get("reasoning") or ""))
    return "".join(text), "".join(reasoning)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.warning(
        f"[ai_node] LLM call failed (attempt {retry_state.attempt_number}), retrying: {retry_state.outcome.exception()!r}"
    ),
    reraise=True,
)
async def _astream_with_retry(
    llm,
    messages,
    config: RunnableConfig,
    *,
    agent_name: str = "main",
    session_id: str = "default",
    step_index: int = 0,
):
    """流式调用 LLM,合并出完整 AIMessage 返回。

    返回 (merged_ai_message, reasoning_total):merged 含 content 全文,
    reasoning_total 是累计的 think 全文,供上层持久化进 timeline(刷新后可还原)。

    Each chunk is also published via event_emitter (if injected) as an
    "ai_delta" event so the host can stream token/reasoning to its UI in
    real-time — this is the primary streaming path, not a side-effect of
    astream_events. Identity fields (agent/session/step) are included so the
    host can route tokens before step_planned arrives.
    """
    merged = None
    reasoning_seen = ""

    async for chunk in llm.astream(messages, config=config):
        merged = chunk if merged is None else merged + chunk

        content_delta, piece = _chunk_parts(chunk)
        reasoning_delta = ""
        if piece:
            if piece.startswith(reasoning_seen) and len(piece) > len(reasoning_seen):
                reasoning_delta = piece[len(reasoning_seen):]
                reasoning_seen = piece
            elif piece == reasoning_seen:
                pass
            else:
                reasoning_delta = piece
                reasoning_seen += piece

        # Publish every chunk to the host via the observer emitter so the
        # UI shows live token / reasoning streaming. This is the design-
        # contract path (not the astream_events side-channel).
        if content_delta or reasoning_delta:
            await emit_step(config, "ai_delta", {
                "content": content_delta,
                "reasoning": reasoning_delta,
                "agent_name": agent_name,
                "session_id": session_id,
                "step_index": step_index,
            })

    return merged, reasoning_seen


def create_ai_node(
    *,
    prompt_stack: PromptStack,
    tools: List[object],
    llm: BaseChatModel,
    tool_filter: Optional[Callable] = None,
):
    # llm is required — the framework never reaches into a host singleton.
    _all_tools = list(tools)
    provider = provider_of(llm)

    _static_llm = llm.bind_tools(_all_tools)
    _cache: Dict[str, Any] = {"hidden": frozenset(), "llm": _static_llm}

    async def ai_node(state: AgentState, config: RunnableConfig) -> Dict[str, object]:
        raise_if_cancelled(config)

        if tool_filter:
            hidden: FrozenSet[str] = frozenset(tool_filter(state, config) or ())
            if hidden != _cache["hidden"]:
                filtered = [t for t in _all_tools if t.name not in hidden]
                _cache["llm"] = llm.bind_tools(filtered)
                _cache["hidden"] = hidden
            llm_to_use = _cache["llm"]
        else:
            llm_to_use = _static_llm

        agent_name = str(state.get("current_agent_name") or "main")
        session_id = str(state.get("session_id") or "default")
        if provider == "openai":
            # Routing affinity hint: keep every turn of this session on the
            # machine already holding its growing prefix. Scoped per agent too,
            # since subagents in the same session run different system prompts
            # and therefore are different prefixes.
            bind_kwargs: Dict[str, Any] = {
                "prompt_cache_key": f"graphloom:{agent_name}:{session_id}"
            }
            if supports_explicit_cache_breakpoints(llm):
                bind_kwargs["prompt_cache_options"] = {"mode": "explicit"}
            llm_to_use = llm_to_use.bind(**bind_kwargs)
        # Tokens stream before the step is planned; use the upcoming 1-based index.
        step_index = int(state.get("step_counter") or 0) + 1

        messages = await build_llm_messages(state, prompt_stack, llm)

        response, reasoning_text = await _astream_with_retry(
            llm_to_use,
            messages,
            config,
            agent_name=agent_name,
            session_id=session_id,
            step_index=step_index,
        )
        # The merged AIMessage carries thinking + signature + tool_calls
        # verbatim; handing it straight to the channel is what lets the model
        # read its own prior reasoning next turn instead of re-deriving it.
        updates: Dict[str, object] = {"messages": [response]}
        if not response.tool_calls:
            return updates

        next_counter = int(state.get("step_counter") or 0) + 1
        turn = build_turn(
            step_id=step_id_for(agent_name, session_id, next_counter),
            ai_message=response,
            status="pending_tool",
        )
        updates["timeline"] = [turn]
        updates["step_counter"] = next_counter
        # Publish step_planned so observers (host WS bridge) can show the step
        # title + think before the tool chip lands. Framework stays decoupled —
        # emit_step is a no-op when no emitter was injected.
        await emit_step(config, "step_planned", {
            "step_id": turn["step_id"],
            "step_index": next_counter,
            "agent_name": agent_name,
            "session_id": session_id,
            "last_step_review": turn["last_step_review"],
            "working_notes": turn["working_notes"],
            "next_action": turn["next_action"],
            "think": turn["think"],
            "content": turn["content"],
            "tool_calls": [
                {"tool_name": call["tool_name"], "tool_args": call["tool_args"]}
                for call in turn["tool_calls"]
            ],
        })
        return updates

    return ai_node
