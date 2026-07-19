"""
context_compaction_node.py

A LangGraph node inserted between `history` and the next turn. When the
estimated context token usage crosses COMPACT_TRIGGER_RATIO, this node folds
past_steps[:-KEEP_RECENT] into a single summary step (shape-compatible with
the regular past_step dict) via a structured LLM call, and asks the reducer
to REPLACE state["past_steps"] with [summary, *recent].

The compacted step reuses the StandardThoughtInput schema
(last_step_review / memory / next_action). The action_results field is
deliberately left empty — the summary step is not a real action, and all
durable facts are archived into `working_notes`.
"""
import logging
import time
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from graphloom.config import (
    COMPACT_EMERGENCY_TRUNC_CHARS,
    COMPACT_KEEP_RECENT_STEPS,
    COMPACT_MAX_RETRY,
    COMPACT_SENTINEL_KEY,
    COMPACT_TARGET_RATIO,
    COMPACT_TRIGGER_RATIO,
    MODEL_CONTEXT_WINDOW,
)
from graphloom.model.base_tool_input import StandardThoughtInput
from graphloom.model.state import AgentState
from graphloom.prompt.message_builder import build_llm_messages
from graphloom.prompt.stack import PromptStack
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

You are given the earliest past_steps of an agent run that must be folded
into a single condensed past_step so the agent can keep working without
losing the thread. The recent kept steps and todo.md are NOT shown to you
because they are still kept verbatim downstream — do not try to repeat them.

CORE PRINCIPLE — LOSSLESS ARCHIVAL
Your job is NOT to summarize loosely. Your job is to ARCHIVE every concrete
fact the agent may still need. The following MUST be preserved verbatim
(copy them out, do not paraphrase, do not merge, do not drop):
  * numbers, IDs, tokens, API keys, session IDs, hashes
  * URLs, endpoints, file paths, artifact ids
  * user's explicit instructions, clarifications, constraints, preferences
  * decisions taken and the reasoning the user gave for them
  * error messages, exception types, and how each was resolved
  * data-schema hints: field names, enum values, parameter shapes
  * named entities: product names, brand names, property names, person names

Things you MAY compress (but still keep pointers to):
  * redundant intermediate reasoning that was superseded
  * repeated failed attempts — keep the first failure + final outcome
  * verbose tool payloads — keep the lesson/pointer, drop the raw dump

OUTPUT SCHEMA (StandardThoughtInput)
Produce exactly these three fields. A fourth field `action_results` exists
in the schema but you MUST leave it as an empty string — this compacted step
is not a real action, and flowing narrative belongs in `working_notes` instead.

- last_step_review
  AIM around {eval_budget} characters.
  An overall stage assessment across ALL compacted steps: what worked,
  what failed, what remains uncertain. Aggregate, do not enumerate.

- working_notes
  AIM around {working_notes_budget} characters. This is the primary archive.
  Use a dense bulleted list. One concrete fact per bullet. Include
  every item from the "preserved verbatim" list above that appeared in
  the input. It is better to overshoot the aim than to drop facts.
  Group bullets by topic (e.g. "APIs & keys", "URLs visited",
  "User decisions", "Discovered constraints", "Errors & resolutions").

- next_action
  AIM around {next_action_budget} characters.
  The immediate next goal implied by the compacted history. If the agent
  was mid-step, state exactly where to resume (which URL, which tool,
  which parameter).

HARD RULES
- Do not invent facts; only compress what is present in the input.
- Do not duplicate content that obviously belongs in todo.md (pending tasks).
- Write in the same language as the input steps.
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


def _render_old_steps_for_summarizer(old_steps: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, step in enumerate(old_steps, start=1):
        lines.append(f"<step index=\"{idx}\">")
        lines.append(f"last_step_review: {step.get('last_step_review', '')}")
        lines.append(f"working_notes: {step.get('working_notes', '')}")
        lines.append(f"next_action: {step.get('next_action', '')}")
        ar = str(step.get("action_results", ""))
        lines.append(f"action_results: {ar}")
        lines.append("</step>")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …[truncated]"


def _enforce_field_budgets(step: Dict[str, Any], budgets: Dict[str, int]) -> Dict[str, Any]:
    """Hard-cap each thought field to its budget. Acts as a safety net only;
    the prompt is what drives the LLM to fill the fields."""
    out = dict(step)
    for field, limit in budgets.items():
        val = str(out.get(field, "") or "")
        if len(val) > limit:
            out[field] = _truncate(val, limit)
    return out


def _apply_emergency_truncation(recent: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not recent:
        return recent
    # Find the single longest action_results and truncate it.
    target_idx = max(
        range(len(recent)),
        key=lambda i: len(str(recent[i].get("action_results") or "")),
    )
    target = dict(recent[target_idx])
    ar = str(target.get("action_results") or "")
    if len(ar) > COMPACT_EMERGENCY_TRUNC_CHARS:
        target["action_results"] = ar[:COMPACT_EMERGENCY_TRUNC_CHARS].rstrip() + " …[emergency-truncated]"
        recent = list(recent)
        recent[target_idx] = target
        logger.warning(
            "[compaction] emergency truncation applied to recent step index %d (was %d chars)",
            target_idx,
            len(ar),
        )
    return recent


async def _summarize_old_steps(
    old_steps: List[Dict[str, Any]],
    budgets: Dict[str, int],
    llm: BaseChatModel,
) -> Dict[str, Any]:
    # bind BEFORE with_structured_output so the provider sees max_tokens.
    structured_llm = (
        llm.bind(max_tokens=_max_output_tokens())
        .with_structured_output(StandardThoughtInput, method="function_calling")
    )

    system = COMPACTION_SYSTEM_PROMPT.format(
        eval_budget=budgets["last_step_review"],
        working_notes_budget=budgets["working_notes"],
        next_action_budget=budgets["next_action"],
    )
    user_payload = (
        f"<compacted_step_count>{len(old_steps)}</compacted_step_count>\n"
        f"<old_past_steps>\n{_render_old_steps_for_summarizer(old_steps)}\n</old_past_steps>"
    )
    messages = [SystemMessage(content=system), HumanMessage(content=user_payload)]

    result: StandardThoughtInput = await structured_llm.ainvoke(messages)
    step = result.model_dump()
    # Compacted step is not a real action; keep action_results empty so the
    # renderer can suppress it, and stamp the authoritative step count.
    step["action_results"] = ""
    step["compacted_step_count"] = len(old_steps)
    return _enforce_field_budgets(step, budgets)


def _token_budget() -> int:
    return int(MODEL_CONTEXT_WINDOW * COMPACT_TRIGGER_RATIO)


async def _estimate_state_tokens(state: AgentState, prompt_stack: PromptStack) -> int:
    """Token count over the EXACT message payload ai_node will send to the LLM.

    Going through `build_llm_messages` keeps the gate and the real request in
    lockstep — anything that doesn't reach the LLM won't be counted, and
    anything that will (system prompt, observer parts, attachments) is.
    """
    messages = await build_llm_messages(state, prompt_stack)
    return count_messages_tokens(messages)


async def _should_trigger(state: AgentState, past_steps: List[Dict[str, Any]], prompt_stack: PromptStack) -> bool:
    if len(past_steps) <= COMPACT_KEEP_RECENT_STEPS:
        return False
    return await _estimate_state_tokens(state, prompt_stack) >= _token_budget()


def create_context_compaction_node(prompt_stack: PromptStack, llm: BaseChatModel):
    async def context_compaction_node(state: AgentState) -> Dict[str, Any]:
        past_steps: List[Dict[str, Any]] = list(state.get("past_steps", []) or [])
        if not await _should_trigger(state, past_steps, prompt_stack):
            return {}

        total_budget = _total_char_budget()
        budgets = _field_budgets(total_budget)
        recent = past_steps[-COMPACT_KEEP_RECENT_STEPS:]
        old = past_steps[:-COMPACT_KEEP_RECENT_STEPS]

        logger.info(
            "[compaction] triggered: %d past_steps, folding %d -> 1, keeping last %d (total_budget=%d chars)",
            len(past_steps),
            len(old),
            len(recent),
            total_budget,
        )

        attempt = 0
        compacted_step: Dict[str, Any] = {}
        current_agent_name = str(state.get("current_agent_name") or "main")
        session_id = str(state.get("session_id") or "default")
        compacted_step_id = f"{current_agent_name}:{session_id}:compacted:{int(time.time() * 1000)}"
        while attempt < COMPACT_MAX_RETRY:
            attempt += 1
            try:
                compacted_step = await _summarize_old_steps(old, budgets, llm)
            except Exception as exc:  # noqa: BLE001
                logger.exception("[compaction] summarize failed on attempt %d: %s", attempt, exc)
                # Fallback: synthesize a minimal step from raw fields so we
                # still shrink the channel instead of leaving it unbounded.
                fallback = {
                    "last_step_review": "Compaction fallback: summarizer failed.",
                    "working_notes": _render_old_steps_for_summarizer(old),
                    "next_action": old[-1].get("next_action", "") if old else "",
                    "action_results": "",
                    "compacted_step_count": len(old),
                }
                compacted_step = _enforce_field_budgets(fallback, budgets)

            compacted_step["step_id"] = compacted_step_id
            compacted_step.setdefault("status", "compacted")

            projected_state = dict(state)
            projected_state["past_steps"] = [compacted_step, *recent]
            post_tokens = await _estimate_state_tokens(projected_state, prompt_stack)

            if post_tokens < _token_budget():
                sentinel = {COMPACT_SENTINEL_KEY: True}
                return {"past_steps": [sentinel, compacted_step, *recent]}

            logger.warning(
                "[compaction] attempt %d still over budget (%d / %d), tightening",
                attempt,
                post_tokens,
                _token_budget(),
            )
            # Tighten budgets for the next attempt (halve each field's cap).
            budgets = {field: max(100, limit // 2) for field, limit in budgets.items()}

        # Exhausted retries — emergency degrade the kept window and emit anyway.
        recent = _apply_emergency_truncation(recent)
        sentinel = {COMPACT_SENTINEL_KEY: True}
        logger.error(
            "[compaction] max retries reached; emitting best-effort compaction with emergency truncation"
        )
        return {"past_steps": [sentinel, compacted_step, *recent]}

    return context_compaction_node
