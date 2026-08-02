import json
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from graphloom.model.state import AgentState
from graphloom.util.session_store import session_store


def _cache_marker(llm: BaseChatModel | None) -> Dict[str, Any]:
    if llm is not None and llm.get_lc_namespace()[-1] == "anthropic":
        return {"cache_control": {"type": "ephemeral"}}
    return {}


def _cache_block(text: str, llm: BaseChatModel | None = None) -> Dict[str, Any]:
    return {"type": "text", "text": text, **_cache_marker(llm)}


def build_past_steps_message(
    past_steps: List[Dict[str, Any]],
    llm: BaseChatModel | None = None,
) -> HumanMessage:
    if not past_steps:
        return HumanMessage(
            content=[_cache_block(
                "<agent_history>\n    New task, no operation history yet.\n</agent_history>",
                llm,
            )]
        )

    blocks = ["<agent_history>"]
    for idx, step in enumerate(past_steps):
        step_lines: List[str] = []
        repeat_count = step.get("repeatCount", 1)
        repeat_info = (
            f" [WARNING: This action has been repeated {repeat_count} times consecutively]"
            if repeat_count > 1 else ""
        )
        compacted_count = int(step.get("compacted_step_count") or 0)
        summary_prefix = (
            f" [SUMMARY of {compacted_count} prior steps]" if compacted_count > 0 else ""
        )
        last_step_review = str(step.get("last_step_review") or "").strip()
        working_notes = str(step.get("working_notes") or "").strip()
        next_action = str(step.get("next_action") or "").strip()
        action_results = str(step.get("action_results") or "").strip()
        status = str(step.get("status") or "completed").strip()

        step_lines.append(f"<step_{idx + 1}>{summary_prefix}")
        if status and status != "completed":
            step_lines.append(f"Status: {status}")
        step_lines.append(f"Last Step Review: {last_step_review}")
        step_lines.append(f"Notes: {working_notes}")
        step_lines.append(f"Next Action: {next_action}{repeat_info}")

        # Compacted summary steps carry no real action_results; skip the
        # empty line so the archival block stays clean.
        if compacted_count <= 0:
            if status and status != "completed" and not action_results:
                action_results = "Tool execution did not finish yet; this step may have been interrupted."
            step_lines.append(f"Action Results: {action_results}")
        step_lines.append(f"</step_{idx + 1}>")
        blocks.append("\n".join(step_lines))

    content = [{"type": "text", "text": block} for block in blocks]
    content[-1].update(_cache_marker(llm))
    content.append({"type": "text", "text": "</agent_history>"})
    return HumanMessage(content=content)


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
        f"<environment>\nCurrent time: {current_time}\n</environment>" if current_time else "",
        render_todo_contents(todo_contents),
        render_delivery_status(session_id),
        _json_block("input_artifact_manifest", list(state.get("input_artifact_manifest", []) or [])),
        _json_block("current_delivery_manifest", list(state.get("current_delivery_manifest", []) or [])),
        _json_block("approved_artifact_manifest", list(state.get("approved_artifact_manifest", []) or [])),
    ]
    return "\n\n".join(section for section in sections if section)
