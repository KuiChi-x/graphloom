import json
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from graphloom.model.state import AgentState
from graphloom.util.model_info import provider_of, supports_explicit_cache_breakpoints
from graphloom.util.session_store import session_store


def _provider(llm: BaseChatModel | None) -> str:
    """该模型说哪种缓存断点方言，判不出来返回 ""。"""
    if provider_of(llm) == "anthropic":
        return "anthropic"
    if supports_explicit_cache_breakpoints(llm):
        return "openai"
    return ""


def _cache_marker(llm: BaseChatModel | None) -> Dict[str, Any]:
    provider = _provider(llm)
    if provider == "anthropic":
        return {"cache_control": {"type": "ephemeral"}}
    if provider == "openai":
        return {"prompt_cache_breakpoint": {"mode": "explicit"}}
    return {}


def _cache_block(text: str, llm: BaseChatModel | None = None) -> Dict[str, Any]:
    return {"type": "text", "text": text, **_cache_marker(llm)}


# OpenAI 读取时最多考虑 50 个断点，超过就接不上。留一个给 system。
_OPENAI_MAX_BREAKPOINTS = 49


def _history_breakpoints(step_count: int, provider: str) -> set:
    """哪些步骤要打缓存断点。两家的写入名额都是每请求 4 个，但读取规则完全不同,
    所以位置策略也不同。

    - Anthropic: 回溯窗口是 20 个块 —— 从断点往前最多查 20 个位置，找不到就停。
      所以钉最新两步：一个写在增长边缘，另一个留在上轮写入处兜底，间距永远是 1,
      不可能超窗。实测 46% -> 88%。
    - OpenAI /responses: 没有块距限制，但读取只看最近 50 个断点。写入名额虽然是
      4 个，已写入的旧断点不重写也照样能读，所以断点集合应该只增不减 —— 每步都
      钉，实测 40 步 95%（单调爬到 97%，无锯齿）。
      反过来，删掉任何一个旧断点都会让整条链作废且再也接不回来：之前只留 3 个
      断点时每次换锚点那轮就掉到 3%，40 步总命中 69%；到 50 上限后开始丢最老的
      断点，第 51 步起永久锁死在 1%。
    """
    if step_count <= 0:
        return set()
    if provider == "anthropic":
        return {i for i in (step_count - 2, step_count - 1) if i >= 0}
    if provider == "openai":
        # 每步一个断点，逼近 50 上限时步长翻倍。翻倍那一轮会掉一次（~4%）,
        # 下一轮就回到 98%；200 步的会话里只发生 3 次（第 50、99、197 步）。
        # 步长必须只增不减，否则旧断点被丢弃，缓存链断掉且无法恢复。
        stride = 1
        while (step_count + stride - 1) // stride > _OPENAI_MAX_BREAKPOINTS - 1:
            stride *= 2
        return {0} | {i for i in range(stride - 1, step_count, stride)}
    return set()


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
    # blocks[0] is the "<agent_history>" opener, so step N sits at content[N].
    marker = _cache_marker(llm)
    if marker:
        for step_idx in _history_breakpoints(len(past_steps), _provider(llm)):
            content[step_idx + 1].update(marker)
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
