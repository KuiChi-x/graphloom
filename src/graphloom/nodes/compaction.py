"""compaction.py

A LangGraph node between `tool` and the next turn. When the estimated payload
crosses COMPACT_TRIGGER_RATIO, the oldest turns are folded into one archival
HumanMessage and the whole `messages` channel is replaced with
`[request, summary, *recent_turns]`.

Folding happens on TURN boundaries (`message_utils.turns`), never inside one:
a `tool_use` block separated from its `tool_result`, or a turn stripped of the
thinking block whose signature the provider validates, is a malformed request.
That is why the summary is a plain HumanMessage rather than a synthetic
AIMessage — archived reasoning is described, not impersonated.
"""
import logging
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from graphloom.config import (
    COMPACT_EMERGENCY_TRUNC_CHARS,
    COMPACT_KEEP_RECENT_STEPS,
    COMPACT_MAX_RETRY,
    COMPACT_TARGET_RATIO,
    COMPACT_TRIGGER_RATIO,
    MODEL_CONTEXT_WINDOW,
)
from graphloom.model.base_tool_input import StandardThoughtInput
from graphloom.model.state import AgentState
from graphloom.model.timeline import visible_args
from graphloom.prompt.message_builder import build_llm_messages
from graphloom.prompt.stack import PromptStack
from graphloom.util.message_utils import text_of, turns
from graphloom.util.structured_output import structured_output_llm
from graphloom.util.token_counter import count_messages_tokens

logger = logging.getLogger(__name__)


# Per-field share of the total char budget. Must sum to 1.0.
# `working_notes` is the primary archival surface; the other two frame it.
_FIELD_BUDGET_SHARE = {
    "last_step_review": 0.08,
    "working_notes": 0.85,
    "next_action": 0.07,
}


COMPACTION_SYSTEM_PROMPT = """You are a LOSSLESS ARCHIVER for a long-running agent.

You are given the earliest turns of an agent run that must be folded into a
single condensed archive so the agent can keep working without losing the
thread. The recent kept turns and todo.md are NOT shown to you because they are
still kept verbatim downstream — do not try to repeat them.

CORE PRINCIPLE — LOSSLESS ARCHIVAL
Your job is NOT to summarize loosely. Your job is to ARCHIVE every concrete
fact the agent may still need. The following MUST be preserved verbatim
(copy them out, do not paraphrase, do not merge, do not drop):
  * numbers, IDs, tokens, API keys, session IDs, hashes
  * URLs, endpoints, file paths, artifact paths
  * selectors (XPath / CSS), parameter names, request/response field names
  * credentials, cookies, headers that were needed
  * exact error messages and the action that produced them
  * decisions already made, and approaches already ruled out (with the reason)

FIELD BUDGETS (characters, approximate)
  last_step_review: {eval_budget}
  working_notes:    {working_notes_budget}
  next_action:      {next_action_budget}

WRITING RULES
- `last_step_review`: the run's trajectory so far and where it currently stands.
- `working_notes`: the archive. Dense, structured, fact-first. This is where
  every durable detail above must land.
- `next_action`: the immediate next action implied by the turns you were given.
- Write in the same language as the input turns.
- If you are forced to choose between brevity and preserving a concrete
  fact, ALWAYS preserve the fact.
"""


def _max_output_tokens() -> int:
    """Output budget mirrors the target ratio: the compacted payload should
    occupy roughly `target_ratio` of the window, so the LLM is allowed to
    emit up to that many tokens. Provider-side caps will clamp as needed.
    """
    return max(2000, int(MODEL_CONTEXT_WINDOW * COMPACT_TARGET_RATIO))


def _total_char_budget() -> int:
    # Rough total char budget ~ target_ratio of the window (4 chars/token).
    return max(2000, int(MODEL_CONTEXT_WINDOW * COMPACT_TARGET_RATIO * 4))


def _field_budgets(total: int) -> Dict[str, int]:
    return {field: max(100, int(total * share)) for field, share in _FIELD_BUDGET_SHARE.items()}


def _token_budget() -> int:
    return int(MODEL_CONTEXT_WINDOW * COMPACT_TRIGGER_RATIO)


def render_turns(groups: List[List[BaseMessage]]) -> str:
    """Flatten turns to text for the summarizer. Reasoning is included: it is
    where the agent recorded why it ruled things out."""
    lines: List[str] = []
    for index, group in enumerate(groups, start=1):
        lines.append(f'<turn index="{index}">')
        for message in group:
            if isinstance(message, AIMessage):
                for block in message.content_blocks:
                    kind = block.get("type")
                    if kind == "reasoning":
                        lines.append(f"reasoning: {block.get('reasoning') or ''}")
                    elif kind == "text":
                        lines.append(f"said: {block.get('text') or ''}")
                for call in message.tool_calls or []:
                    lines.append(
                        f"called {call.get('name')} with "
                        f"{visible_args(call.get('args') or {})}"
                    )
            elif isinstance(message, ToolMessage):
                marker = "error" if message.status == "error" else "result"
                lines.append(f"{message.name} {marker}: {text_of(message)}")
            else:
                lines.append(f"user: {text_of(message)}")
        lines.append("</turn>")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …[truncated]"


def _archive_message(fields: Dict[str, Any], folded: int, budgets: Dict[str, int]) -> HumanMessage:
    """The compacted turns, as one archival HumanMessage."""
    capped = {
        field: _truncate(str(fields.get(field) or ""), limit)
        for field, limit in budgets.items()
    }
    return HumanMessage(content=(
        f'<archived_history turns="{folded}">\n'
        f"Trajectory so far: {capped['last_step_review']}\n\n"
        f"Durable facts and decisions:\n{capped['working_notes']}\n\n"
        f"Next action implied: {capped['next_action']}\n"
        "</archived_history>"
    ))


async def _summarize(
    groups: List[List[BaseMessage]],
    budgets: Dict[str, int],
    llm: BaseChatModel,
) -> Dict[str, Any]:
    # max_tokens is bound BEFORE the structured wrapper so the provider sees it.
    structured_llm = structured_output_llm(
        llm, StandardThoughtInput, max_tokens=_max_output_tokens()
    )
    system = COMPACTION_SYSTEM_PROMPT.format(
        eval_budget=budgets["last_step_review"],
        working_notes_budget=budgets["working_notes"],
        next_action_budget=budgets["next_action"],
    )
    result: StandardThoughtInput = await structured_llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=(
            f"<folded_turn_count>{len(groups)}</folded_turn_count>\n"
            f"<old_turns>\n{render_turns(groups)}\n</old_turns>"
        )),
    ])
    return result.model_dump()


def _emergency_truncate(groups: List[List[BaseMessage]]) -> List[List[BaseMessage]]:
    """Last resort when retries are exhausted: shrink the single largest tool
    result among the kept turns. Tool output is the only safely editable part —
    reasoning blocks carry signatures and must stay byte-exact."""
    candidates = [
        (len(text_of(message)), group_index, message_index)
        for group_index, group in enumerate(groups)
        for message_index, message in enumerate(group)
        if isinstance(message, ToolMessage)
    ]
    if not candidates:
        return groups
    size, group_index, message_index = max(candidates)
    if size <= COMPACT_EMERGENCY_TRUNC_CHARS:
        return groups
    target = groups[group_index][message_index]
    trimmed = target.model_copy(update={
        "content": text_of(target)[:COMPACT_EMERGENCY_TRUNC_CHARS].rstrip()
        + " …[emergency-truncated]"
    })
    logger.warning(
        "[compaction] emergency truncation applied to %s result in turn %d (was %d chars)",
        target.name, group_index + 1, size,
    )
    groups = [list(group) for group in groups]
    groups[group_index][message_index] = trimmed
    return groups


async def _estimate_state_tokens(state: AgentState, prompt_stack: PromptStack) -> int:
    """Token count over the EXACT payload ai_node will send to the LLM.

    Going through `build_llm_messages` keeps the gate and the real request in
    lockstep — anything that doesn't reach the LLM won't be counted, and
    anything that will (system prompt, observer parts, attachments) is.
    """
    return count_messages_tokens(await build_llm_messages(state, prompt_stack))


def _replace_channel(messages: List[BaseMessage]) -> List[BaseMessage]:
    """`add_messages` clears the channel on RemoveMessage(REMOVE_ALL_MESSAGES),
    then appends what follows — an atomic swap in one update."""
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]


def create_context_compaction_node(prompt_stack: PromptStack, llm: BaseChatModel):
    async def context_compaction_node(state: AgentState) -> Dict[str, Any]:
        groups = turns(state.get("messages", []))
        # groups[0] is the originating request; it is never folded away.
        if len(groups) - 1 <= COMPACT_KEEP_RECENT_STEPS:
            return {}
        if await _estimate_state_tokens(state, prompt_stack) < _token_budget():
            return {}

        opening, rest = groups[0], groups[1:]
        recent = rest[-COMPACT_KEEP_RECENT_STEPS:]
        old = rest[:-COMPACT_KEEP_RECENT_STEPS]

        budgets = _field_budgets(_total_char_budget())
        logger.info(
            "[compaction] triggered: %d turns, folding %d -> 1, keeping last %d",
            len(rest), len(old), len(recent),
        )

        archive = None
        for attempt in range(1, COMPACT_MAX_RETRY + 1):
            try:
                fields = await _summarize(old, budgets, llm)
            except Exception as exc:  # noqa: BLE001
                logger.exception("[compaction] summarize failed on attempt %d: %s", attempt, exc)
                # Fall back to the raw render so the channel still shrinks.
                fields = {
                    "last_step_review": "Compaction fallback: summarizer failed.",
                    "working_notes": render_turns(old),
                    "next_action": "",
                }
            archive = _archive_message(fields, len(old), budgets)

            kept = [message for group in recent for message in group]
            projected = {**state, "messages": [*opening, archive, *kept]}
            post_tokens = await _estimate_state_tokens(projected, prompt_stack)
            if post_tokens < _token_budget():
                return {"messages": _replace_channel(projected["messages"])}

            logger.warning(
                "[compaction] attempt %d still over budget (%d / %d), tightening",
                attempt, post_tokens, _token_budget(),
            )
            budgets = {field: max(100, limit // 2) for field, limit in budgets.items()}

        # Exhausted retries — degrade the kept window and emit anyway.
        recent = _emergency_truncate(recent)
        logger.error("[compaction] max retries reached; emitting best-effort compaction")
        kept = [message for group in recent for message in group]
        return {"messages": _replace_channel([*opening, archive, *kept])}

    return context_compaction_node


def create_context_compaction_node(prompt_stack: PromptStack, llm: BaseChatModel):
    async def context_compaction_node(state: AgentState) -> Dict[str, Any]:
        groups = turns(state.get("messages", []))
        # groups[0] is the originating request; it is never folded away.
        if len(groups) - 1 <= COMPACT_KEEP_RECENT_STEPS:
            return {}
        if await _estimate_state_tokens(state, prompt_stack) < _token_budget():
            return {}

        opening, body = groups[0], groups[1:]
        recent = body[-COMPACT_KEEP_RECENT_STEPS:]
        old = body[:-COMPACT_KEEP_RECENT_STEPS]
        budgets = _field_budgets(_total_char_budget())

        logger.info(
            "[compaction] triggered: %d turns, folding %d -> 1, keeping last %d",
            len(body), len(old), len(recent),
        )

        archive = HumanMessage(content="")
        for attempt in range(1, COMPACT_MAX_RETRY + 1):
            try:
                fields = await _summarize(old, budgets, llm)
            except Exception as exc:  # noqa: BLE001
                logger.exception("[compaction] summarize failed on attempt %d: %s", attempt, exc)
                # Fall back to the raw rendering so the channel still shrinks
                # instead of growing unbounded.
                fields = {
                    "last_step_review": "Compaction fallback: summarizer failed.",
                    "working_notes": render_turns(old),
                    "next_action": "",
                }

            archive = _archive_message(fields, len(old), budgets)
            kept = [*opening, archive, *[m for group in recent for m in group]]
            post_tokens = await _estimate_state_tokens({**state, "messages": kept}, prompt_stack)
            if post_tokens < _token_budget():
                return {"messages": _replace_channel(kept)}

            logger.warning(
                "[compaction] attempt %d still over budget (%d / %d), tightening",
                attempt, post_tokens, _token_budget(),
            )
            budgets = {field: max(100, limit // 2) for field, limit in budgets.items()}

        recent = _emergency_truncate(recent)
        logger.error(
            "[compaction] max retries reached; emitting best-effort compaction with emergency truncation"
        )
        return {"messages": _replace_channel(
            [*opening, archive, *[m for group in recent for m in group]]
        )}

    return context_compaction_node
