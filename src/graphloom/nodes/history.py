import time
from typing import Any, Dict, List, Set

from graphloom.events import emit_step
from graphloom.util.message_utils import get_last_ai_message
from graphloom.model.state import AgentState

THOUGHT_FIELDS: Set[str] = {
    "last_step_review",
    "working_notes",
    "next_action",
    "session_id",
    "runtime_context",
}


def _filter_thought_args(args: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in args.items() if k not in THOUGHT_FIELDS}


def _latest_pending_step(past_steps: List[Dict[str, Any]], step_id: str = "") -> Dict[str, Any]:
    for step in reversed(past_steps):
        if not isinstance(step, dict):
            continue
        if step_id and str(step.get("step_id") or "") == step_id:
            return dict(step)
        if not step_id and step.get("status") == "pending_tool":
            return dict(step)
    return {}


def create_history_node():
    async def history_node(state: AgentState, config: Dict[str, Any] = None) -> Dict[str, Any]:
        last_ai_message = state.get("latest_ai_message") or get_last_ai_message(list(state.get("messages", []) or []))
        if not last_ai_message:
            return {}

        tool_result_history = list(state.get("tool_result_history", []) or [])
        if not tool_result_history:
            return {}

        result_step_id = str(tool_result_history[0].get("step_id") or "")
        past_steps_existing = list(state.get("past_steps", []) or [])
        pending_step = _latest_pending_step(past_steps_existing, result_step_id)

        last_step_review = str(pending_step.get("last_step_review") or "").strip()
        working_notes = str(pending_step.get("working_notes") or "").strip()
        next_action = str(pending_step.get("next_action") or "").strip()
        action_results = ''
        tool_calls_data: List[Dict[str, Any]] = []
        for tool_result in tool_result_history:
            tool_args = tool_result.get("tool_args") or {}
            last_step_review = str(tool_args.get("last_step_review") or last_step_review).strip()
            working_notes = str(tool_args.get("working_notes") or working_notes).strip()
            next_action = str(tool_args.get("next_action") or next_action).strip()
            tool_name = tool_result["tool_name"]
            result = tool_result["result"]
            filtered_args = _filter_thought_args(tool_args)

            action_results += f"Executed tool {tool_name} with args:{filtered_args}, tool return: {result}.\n"

            tool_calls_data.append({
                "tool_name": tool_name,
                "tool_args": filtered_args,
                "result": result,
                "has_error": bool(tool_result.get("has_error")),
            })

        step_payload = {
            "step_id": result_step_id or str(pending_step.get("step_id") or ""),
            "status": "completed",
            "last_step_review": last_step_review,
            "working_notes": working_notes,
            "next_action": next_action,
            # 从 pending step 透传 think/content(ai_node 已写入),完成态保留以便刷新还原。
            "think": str(pending_step.get("think") or ""),
            "content": str(pending_step.get("content") or ""),
            "action_results": action_results,
            "tool_calls": tool_calls_data,
            "timestamp": int(pending_step.get("timestamp") or int(time.time() * 1000)),
            "completed_timestamp": int(time.time() * 1000),
        }
        # Publish step_done so observers can close out the turn (timeline end,
        # [STEP_DONE] sentinel, persist full step). No-op without an emitter.
        await emit_step(config or {}, "step_done", {
            "step_id": step_payload["step_id"],
            "agent_name": str(state.get("current_agent_name") or "main"),
            "session_id": str(state.get("session_id") or "default"),
            "last_step_review": last_step_review,
            "working_notes": working_notes,
            "next_action": next_action,
            "think": step_payload["think"],
            "content": step_payload["content"],
            "tool_calls": tool_calls_data,
        })
        return {
            "past_steps": [step_payload],
            "tool_result_history": [],
        }

    return history_node
