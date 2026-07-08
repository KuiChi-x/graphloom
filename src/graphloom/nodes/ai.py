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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from graphloom.model.state import AgentState
from graphloom.nodes.history import _filter_thought_args
from graphloom.nodes.interrupt_guard import raise_if_cancelled
from graphloom.prompt.stack import PromptStack
from graphloom.prompt.message_builder import build_llm_messages


def _planned_step_id(state: AgentState, counter: int) -> str:
    current_agent_name = str(state.get("current_agent_name") or "main")
    session_id = str(state.get("session_id") or "default")
    return f"{current_agent_name}:{session_id}:step:{counter}"


def _chunk_text(content: Any) -> str:
    """从 AIMessageChunk.content 取纯文本增量(content 可能是 str 或 v1 的块列表)。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return ""


def _build_pending_step(
    state: AgentState,
    tool_calls: List[Dict[str, Any]],
    counter: int,
    *,
    think: str = "",
    content: str = "",
) -> Dict[str, Any]:
    last_step_review = ""
    working_notes = ""
    next_action = ""
    tool_calls_data: List[Dict[str, Any]] = []
    action_lines: List[str] = []

    for tool_call in tool_calls:
        raw_args = dict(tool_call.get("args") or {})
        last_step_review = str(raw_args.get("last_step_review") or last_step_review).strip()
        working_notes = str(raw_args.get("working_notes") or working_notes).strip()
        next_action = str(raw_args.get("next_action") or next_action).strip()
        tool_name = str(tool_call.get("name") or "")
        filtered_args = _filter_thought_args(raw_args)
        tool_calls_data.append({
            "tool_name": tool_name,
            "tool_args": filtered_args,
            "result": "",
            "has_error": False,
        })
        action_lines.append(
            f"Planned tool {tool_name} with args:{filtered_args}; result pending."
        )

    return {
        "step_id": _planned_step_id(state, counter),
        "status": "pending_tool",
        "last_step_review": last_step_review,
        "working_notes": working_notes,
        "next_action": next_action,
        # think(reasoning 全文)与 content(LLM 正文全文)持久化进 step,
        # 刷新后前端可从 past_steps 还原,不再依赖一次性的实时 token 流。
        "think": think,
        "content": content,
        "action_results": "\n".join(action_lines),
        "tool_calls": tool_calls_data,
        "timestamp": int(time.time() * 1000),
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.warning(
        f"[ai_node] LLM call failed (attempt {retry_state.attempt_number}), retrying: {retry_state.outcome.exception()!r}"
    ),
    reraise=True,
)
async def _astream_with_retry(llm, messages, config: RunnableConfig):
    """流式调用 LLM,合并出完整 AIMessage 返回。

    返回 (merged_ai_message, reasoning_total):merged 含 content 全文,
    reasoning_total 是累计的 think 全文,供上层持久化进 past_steps(刷新后可还原)。

    整轮包在 @retry 里,半截失败重新流式。
    """
    merged = None
    reasoning_seen = ""

    async for chunk in llm.astream(messages, config=config):
        merged = chunk if merged is None else merged + chunk

        # reasoning 增量:ReasoningChatOpenAI 每个 chunk 的 additional_kwargs.reasoning_content
        piece = str((getattr(chunk, "additional_kwargs", {}) or {}).get("reasoning_content") or "")
        if piece:
            if piece.startswith(reasoning_seen) and len(piece) > len(reasoning_seen):
                reasoning_seen = piece
            elif piece == reasoning_seen:
                continue
            else:
                reasoning_seen += piece

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

        messages = await build_llm_messages(state, prompt_stack)

        response, reasoning_text = await _astream_with_retry(llm_to_use, messages, config)
        updates: Dict[str, object] = {"latest_ai_message": response}
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        if tool_calls:
            next_counter = int(state.get("step_counter") or 0) + 1
            content_text = _chunk_text(getattr(response, "content", ""))
            pending_step = _build_pending_step(
                state, tool_calls, next_counter,
                think=reasoning_text, content=content_text,
            )
            updates["past_steps"] = [pending_step]
            updates["step_counter"] = next_counter
        return updates

    return ai_node
