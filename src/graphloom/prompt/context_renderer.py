import json
import os
import platform
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage

from graphloom.model.state import AgentState
from graphloom.util.model_info import (
    is_real_anthropic_model,
    supports_explicit_cache_breakpoints,
)
from graphloom.util.message_utils import turns
from graphloom.util.session_store import session_store


def provider_dialect(llm: BaseChatModel | None) -> str:
    """该模型说哪种缓存断点方言，判不出来返回 ""。

    判的是**模型**而不是客户端库：网关会把百炼 qwen / kimi / glm 挂在 Anthropic
    协议线上并套 ``claude-`` 前缀，但它们的缓存规则跟 Anthropic 不兼容。
    """
    if is_real_anthropic_model(llm):
        return "anthropic"
    if supports_explicit_cache_breakpoints(llm):
        return "openai"
    return ""


def cache_marker(llm: BaseChatModel | None) -> Dict[str, Any]:
    dialect = provider_dialect(llm)
    if dialect == "anthropic":
        return {"cache_control": {"type": "ephemeral"}}
    if dialect == "openai":
        return {"prompt_cache_breakpoint": {"mode": "explicit"}}
    return {}


def cache_block(text: str, llm: BaseChatModel | None = None) -> Dict[str, Any]:
    return {"type": "text", "text": text, **cache_marker(llm)}


# OpenAI 读取时最多考虑 50 个断点，超过就接不上。留一个给 system。
_OPENAI_MAX_BREAKPOINTS = 49


def _history_breakpoints(turn_count: int, dialect: str) -> set:
    """哪些轮次要打缓存断点。两家的写入名额都是每请求 4 个，但读取规则完全不同,
    所以位置策略也不同。

    - Anthropic: 回溯窗口是 20 个块 —— 从断点往前最多查 20 个位置，找不到就停。
      所以钉最新两轮：一个写在增长边缘，另一个留在上轮写入处兜底，间距永远是 1,
      不可能超窗。实测 46% -> 88%。
    - OpenAI /responses: 没有块距限制，但读取只看最近 50 个断点。写入名额虽然是
      4 个，已写入的旧断点不重写也照样能读，所以断点集合应该只增不减 —— 每轮都
      钉，实测 40 轮 95%（单调爬到 97%，无锯齿）。
      反过来，删掉任何一个旧断点都会让整条链作废且再也接不回来：之前只留 3 个
      断点时每次换锚点那轮就掉到 3%，40 轮总命中 69%；到 50 上限后开始丢最老的
      断点，第 51 轮起永久锁死在 1%。

    轮次口径从 past_steps 换成了原生消息的 turn 分组（见 message_utils.turns）,
    位置策略不变：断点永远落在某一轮的最后一个块上，也就是该轮的 tool_result
    结尾处 —— 那里是历史的稳定前缀边界。
    """
    if turn_count <= 0:
        return set()
    if dialect == "anthropic":
        return {i for i in (turn_count - 2, turn_count - 1) if i >= 0}
    if dialect == "openai":
        # 每轮一个断点，逼近 50 上限时步长翻倍。翻倍那一轮会掉一次（~4%）,
        # 下一轮就回到 98%；200 轮的会话里只发生 3 次（第 50、99、197 轮）。
        # 步长必须只增不减，否则旧断点被丢弃，缓存链断掉且无法恢复。
        stride = 1
        while (turn_count + stride - 1) // stride > _OPENAI_MAX_BREAKPOINTS - 1:
            stride *= 2
        return {0} | {i for i in range(stride - 1, turn_count, stride)}
    return set()


def _mark_last_block(message: BaseMessage, marker: Dict[str, Any]) -> BaseMessage:
    """Copy `message` with `marker` on its final content block.

    Never mutates the input: these messages live in the checkpointed channel,
    and a marker written into them would be persisted and replayed forever.

    A ToolMessage's string content is promoted to a one-element block list,
    which the Anthropic adapter hoists onto the `tool_result` block itself.
    An AIMessage is marked on its last text block; a turn whose AIMessage has
    no text block (pure tool call) is skipped rather than marked on `thinking`,
    which must stay byte-identical to what the provider signed.
    """
    content = message.content
    blocks: List[Dict[str, Any]]
    if isinstance(content, str):
        if not content.strip():
            return message
        blocks = [{"type": "text", "text": content}]
    else:
        blocks = [
            dict(block) if isinstance(block, dict) else {"type": "text", "text": str(block)}
            for block in content or []
        ]
    if not blocks:
        return message

    target = None
    if isinstance(message, ToolMessage):
        target = len(blocks) - 1
    else:
        for index in range(len(blocks) - 1, -1, -1):
            if blocks[index].get("type") == "text":
                target = index
                break
    if target is None:
        return message

    blocks[target] = {**blocks[target], **marker}
    return message.model_copy(update={"content": blocks})


def apply_history_breakpoints(
    messages: List[BaseMessage],
    llm: BaseChatModel | None = None,
) -> List[BaseMessage]:
    """Place cache breakpoints on turn boundaries in the native history.

    The system prompt carries its own breakpoint; this pins the *growing* part
    of the prefix, which is where the win is — without it only the static
    system block is cached and every replayed turn is re-read at full price.
    """
    marker = cache_marker(llm)
    if not marker:
        return messages

    groups = turns(messages)
    if not groups:
        return messages

    # Map each turn to the index of its last message in the flat list.
    boundaries: List[int] = []
    position = 0
    for group in groups:
        position += len(group)
        boundaries.append(position - 1)

    marked = list(messages)
    for turn_index in _history_breakpoints(len(groups), provider_dialect(llm)):
        flat_index = boundaries[turn_index]
        marked[flat_index] = _mark_last_block(marked[flat_index], marker)
    return marked


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
        fatal_gaps = entry.get("fatal_gaps") or []
        recommended_rework = entry.get("recommended_rework") or []

        if fatal_gaps or recommended_rework:
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



def render_environment(current_time: str = "") -> str:
    """Host facts the agent needs before it writes a shell command or a path.

    Without the OS the model guesses, and on Windows it reliably guesses wrong:
    `ls`/`rm -rf` instead of PowerShell, forward slashes, `/tmp` paths. The
    shell name is included because "Windows" alone still leaves cmd vs
    PowerShell vs the Git-Bash case ambiguous.
    """
    lines = []
    if current_time:
        lines.append(f"Current time: {current_time}")
    lines.extend([
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
        # f"Default shell: {_default_shell()}",
        # f"Path separator: {os.sep!r}",
        # f"Working directory: {os.getcwd()}",
        # f"Python: {platform.python_version()}",
    ])
    return "<environment>\n" + "\n".join(lines) + "\n</environment>"


def _default_shell() -> str:
    if platform.system() == "Windows":
        comspec = os.environ.get("COMSPEC", "")
        if os.environ.get("MSYSTEM") or "bash" in os.environ.get("SHELL", "").lower():
            return "bash (Git Bash / MSYS)"
        return "powershell" if "powershell" in comspec.lower() else "cmd.exe"
    return os.environ.get("SHELL") or "sh"


def build_prompt_context(state: AgentState, current_time: str = "", todo_contents: str = "") -> str:
    session_id = str(state.get("session_id") or "default")
    sections = [
        render_environment(current_time),
        render_todo_contents(todo_contents),
        render_delivery_status(session_id),
        _json_block("input_artifact_manifest", list(state.get("input_artifact_manifest", []) or [])),
        _json_block("current_delivery_manifest", list(state.get("current_delivery_manifest", []) or [])),
        _json_block("approved_artifact_manifest", list(state.get("approved_artifact_manifest", []) or [])),
    ]
    return "\n\n".join(section for section in sections if section)
