import time
from typing import Any, Dict, List, Set

from graphloom.util.message_utils import get_last_ai_message
from graphloom.model.state import AgentState

THOUGHT_FIELDS: Set[str] = {
    "evaluation_previous_goal",
    "memory",
    "next_goal",
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
    async def history_node(state: AgentState) -> Dict[str, Any]:
        last_ai_message = state.get("latest_ai_message") or get_last_ai_message(list(state.get("messages", []) or []))
        if not last_ai_message:
            return {}

        tool_result_history = list(state.get("tool_result_history", []) or [])
        if not tool_result_history:
            return {}

        result_step_id = str(tool_result_history[0].get("step_id") or "")
        past_steps_existing = list(state.get("past_steps", []) or [])
        pending_step = _latest_pending_step(past_steps_existing, result_step_id)

        evaluation_previous_goal = str(pending_step.get("evaluation_previous_goal") or "").strip()
        memory = str(pending_step.get("memory") or "").strip()
        next_goal = str(pending_step.get("next_goal") or "").strip()
        action_results = ''
        tool_calls_data: List[Dict[str, Any]] = []
        for tool_result in tool_result_history:
            tool_args = tool_result.get("tool_args") or {}
            evaluation_previous_goal = str(tool_args.get("evaluation_previous_goal") or evaluation_previous_goal).strip()
            memory = str(tool_args.get("memory") or memory).strip()
            next_goal = str(tool_args.get("next_goal") or next_goal).strip()
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
            "evaluation_previous_goal": evaluation_previous_goal,
            "memory": memory,
            "next_goal": next_goal,
            # 从 pending step 透传 think/content(ai_node 已写入),完成态保留以便刷新还原。
            "think": str(pending_step.get("think") or ""),
            "content": str(pending_step.get("content") or ""),
            "action_results": action_results,
            "tool_calls": tool_calls_data,
            "timestamp": int(pending_step.get("timestamp") or int(time.time() * 1000)),
            "completed_timestamp": int(time.time() * 1000),
        }
        return {
            "past_steps": [step_payload],
            "tool_result_history": [],
        }

    return history_node
