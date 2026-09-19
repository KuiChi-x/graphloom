"""timeline.py — the UI/persistence projection of a turn.

`messages` is what the LLM sees; `timeline` is what the UI replays after a
refresh. Everything here is derivable from the messages except wall-clock
timestamps and the step id, which is exactly why the projection exists.

Entry shape (unchanged from the past_steps contract the frontend already
reads, so `stepToTurnBlock` keeps working):

    step_id, status, think, content, tool_calls[], timestamp,
    completed_timestamp, last_step_review, working_notes, next_action
"""
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage

# Thought fields ride in every tool call's args as the agent's chain-of-thought
# channel; they are projected onto the turn and stripped from the shown args.
THOUGHT_FIELDS = frozenset({
    "last_step_review",
    "working_notes",
    "next_action",
    "session_id",
    "runtime_context",
})


def visible_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (args or {}).items() if k not in THOUGHT_FIELDS}


def step_id_for(agent_name: str, session_id: str, counter: int) -> str:
    return f"{agent_name}:{session_id}:step:{counter}"


def turn_index_of(step_id: str) -> int:
    try:
        return int(str(step_id).rsplit(":step:", 1)[-1])
    except (TypeError, ValueError):
        return 0


def _thoughts(ai_message: AIMessage) -> Dict[str, str]:
    """Last non-empty value wins across the turn's tool calls."""
    found = {"last_step_review": "", "working_notes": "", "next_action": ""}
    for call in ai_message.tool_calls or []:
        for field in found:
            value = str((call.get("args") or {}).get(field) or "").strip()
            if value:
                found[field] = value
    return found


def _parts(ai_message: AIMessage) -> Dict[str, str]:
    text: List[str] = []
    reasoning: List[str] = []
    for block in ai_message.content_blocks:
        kind = block.get("type")
        if kind == "text":
            text.append(str(block.get("text") or ""))
        elif kind == "reasoning":
            reasoning.append(str(block.get("reasoning") or ""))
        elif kind == "non_standard":
            # `content_blocks` only classifies a raw thinking block as
            # `reasoning` when the message carries provider metadata
            # (response_metadata["model_provider"]). Messages rebuilt from a
            # checkpoint, or produced by a gateway that drops it, arrive as
            # `non_standard` — without this the UI's think panel goes silently
            # empty for the whole session.
            raw = block.get("value")
            if isinstance(raw, dict) and raw.get("type") == "thinking":
                reasoning.append(str(raw.get("thinking") or ""))
    return {"content": "".join(text), "think": "".join(reasoning)}


def build_turn(
    *,
    step_id: str,
    ai_message: AIMessage,
    tool_messages: Optional[List[ToolMessage]] = None,
    status: str = "completed",
    timestamp: Optional[int] = None,
) -> Dict[str, Any]:
    """Project one turn. Called twice per turn: at plan time (status
    `pending_tool`, no results yet) and at completion (results filled in).
    Both writes carry the same `step_id`, so the reducer replaces in place."""
    results = {
        str(message.tool_call_id): message
        for message in (tool_messages or [])
    }
    tool_calls: List[Dict[str, Any]] = []
    for call in ai_message.tool_calls or []:
        call_id = str(call.get("id") or "")
        answer = results.get(call_id)
        tool_calls.append({
            "call_id": call_id,
            "tool_name": str(call.get("name") or ""),
            "tool_args": visible_args(call.get("args") or {}),
            "result": "" if answer is None else str(answer.content or ""),
            "has_error": bool(answer is not None and answer.status == "error"),
        })

    now = int(time.time() * 1000)
    entry = {
        "step_id": step_id,
        "status": status,
        **_parts(ai_message),
        **_thoughts(ai_message),
        "tool_calls": tool_calls,
        "timestamp": int(timestamp or now),
    }
    if status == "completed":
        entry["completed_timestamp"] = now
    return entry


def merge_turns(
    existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Replace-by-step_id, else append. Lets the completion write update the
    entry the plan write created without duplicating the turn."""
    combined = list(existing or [])
    index = {
        str(item.get("step_id")): position
        for position, item in enumerate(combined)
        if isinstance(item, dict) and item.get("step_id")
    }
    for item in incoming or []:
        key = str(item.get("step_id") or "")
        if key and key in index:
            previous = combined[index[key]]
            # Keep the turn's original start time — the completion write carries
            # its own `timestamp` and must not move when the step began.
            combined[index[key]] = {
                **previous, **item,
                "timestamp": previous.get("timestamp", item.get("timestamp")),
            }
            continue
        if key:
            index[key] = len(combined)
        combined.append(item)
    return combined
