import json
from typing import Any, Dict, List

from graphloom.model.state import AgentState
from graphloom.util.session_store import session_store


def _compress_past_steps(past_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compressed = []
    for index, step in enumerate(past_steps):
        if index == 0:
            compressed.append(dict(step))
            continue
        last = compressed[-1]
        current_action_results = str(step.get("action_results", "")).strip()

        last_action_results = str(last.get("action_results", "")).strip()
        if current_action_results == last_action_results:
            last["repeatCount"] = last.get("repeatCount", 1) + 1
        else:
            compressed.append(dict(step))

    return compressed


def render_past_steps(past_steps: List[Dict[str, Any]]) -> str:
    if not past_steps:
        return "<agent_history>\n    New task, no operation history yet.\n</agent_history>"

    compressed = _compress_past_steps(past_steps)
    lines = ["<agent_history>"]
    for idx, step in enumerate(compressed):
        repeat_count = step.get("repeatCount", 1)
        repeat_info = (
            f" [WARNING: This action has been repeated {repeat_count} times consecutively]"
            if repeat_count > 1 else ""
        )
        compacted_count = int(step.get("compacted_step_count") or 0)
        summary_prefix = (
            f" [SUMMARY of {compacted_count} prior steps]" if compacted_count > 0 else ""
        )
        evaluation_previous_goal = str(step.get("evaluation_previous_goal") or "").strip()
        memory = str(step.get("memory") or "").strip()
        next_goal = str(step.get("next_goal") or "").strip()
        action_results = str(step.get("action_results") or "").strip()
        status = str(step.get("status") or "completed").strip()

        lines.append(f"<step_{idx + 1}>{summary_prefix}")
        if status and status != "completed":
            lines.append(f"Status: {status}")
        lines.append(f"Evaluation of Previous Step: {evaluation_previous_goal}")
        lines.append(f"Memory: {memory}")
        lines.append(f"Next Goal: {next_goal}{repeat_info}")

        # Compacted summary steps carry no real action_results; skip the
        # empty line so the archival block stays clean.
        if compacted_count <= 0:
            if status and status != "completed" and not action_results:
                action_results = "Tool execution did not finish yet; this step may have been interrupted."
            lines.append(f"Action Results: {action_results}")
        lines.append(f"</step_{idx + 1}>")
    lines.append("</agent_history>")
    return "\n".join(lines)


def _json_block(tag: str, value: Any) -> str:
    return f"<{tag}>\n{json.dumps(value, ensure_ascii=False, indent=2)}\n</{tag}>"


def render_delivery_status(session_id: str) -> str:
    delivery_status = session_store.get(session_id, "delivery_status", {})
    if not delivery_status:
        return ""

    lines = ["<delivery_status>"]
    for name, entry in delivery_status.items():
        status = entry.get("status", "UNKNOWN")
        path = entry.get("path", "")
        summary = entry.get("summary", "")
        fatal_gaps = entry.get("fatal_gaps", [])
        recommended_rework = entry.get("recommended_rework", [])

        if (fatal_gaps or recommended_rework) and status == "REJECTED":
            lines.append(f'<artifact path="{path}" status="{status}" summary="{summary}">')
            if fatal_gaps:
                lines.append("<fatal_gaps>")
                for i, gap in enumerate(fatal_gaps, 1):
                    lines.append(f"{i}. {gap}")
                lines.append("</fatal_gaps>")
            if recommended_rework:
                lines.append("<recommended_rework>")
                for i, rework in enumerate(recommended_rework, 1):
                    lines.append(f"{i}. {rework}")
                lines.append("</recommended_rework>")
            lines.append("</artifact>")
        else:
            lines.append(f'<artifact path="{path}" status="{status}" summary="{summary}" />')

    lines.append("</delivery_status>")
    return "\n".join(lines)


def render_todo_contents(todo_contents: str) -> str:
    """Render the agent's todo/note text. Empty input → no section."""
    content = (todo_contents or "").strip()
    if not content:
        return ""
    return f"<todo_contents>\n{content}\n</todo_contents>"


def build_user_request_str(state: AgentState) -> str:
    return f"<user_request>\n{state.get('input_query', '')}\n</user_request>"


def build_prompt_context(state: AgentState, current_time: str = "", todo_contents: str = "") -> str:
    session_id = str(state.get("session_id") or "default")
    sections = [
        render_past_steps(list(state.get("past_steps", []) or [])),
        f"<environment>\nCurrent time: {current_time}\n</environment>" if current_time else "",
        render_todo_contents(todo_contents),
        render_delivery_status(session_id),
        _json_block("input_artifact_manifest", list(state.get("input_artifact_manifest", []) or [])),
        _json_block("current_delivery_manifest", list(state.get("current_delivery_manifest", []) or [])),
        _json_block("approved_artifact_manifest", list(state.get("approved_artifact_manifest", []) or [])),
    ]
    return "\n\n".join(section for section in sections if section)
