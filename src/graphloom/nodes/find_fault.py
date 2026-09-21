import json
import logging
import os
from typing import Any, Dict, List

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from graphloom.model.state import AgentState
from graphloom.prompt.context_renderer import _json_block
from graphloom.prompt.find_fault_system_prompt import COMMON_FIND_FAULT_SYSTEM_PROMPT
from graphloom.util.session_store import session_store
from graphloom.util.structured_output import structured_output_llm

logger = logging.getLogger(__name__)


class GenericFindFaultOutput(BaseModel):
    is_acceptable: bool = Field(description="Whether the delivered artifacts are acceptable.")
    decisive_assessment: str = Field(description="Short final assessment.")
    fatal_gaps: List[str] = Field(default_factory=list,
                                  description="Blocking data completeness issues ONLY: missing fields, missing data, or wrong-element mappings. Never include xpath format or quality concerns.")
    suspicious_claims: List[str] = Field(default_factory=list, description="Claims that look unsupported by evidence.")
    missing_proof: List[str] = Field(default_factory=list, description="Missing evidence or missing deliverables.")
    recommended_rework: List[str] = Field(default_factory=list,
                                          description="Specific missing fields or data to add. Never recommend changing XPaths.")
    confidence: float = Field(default=0.0, description="Confidence score between 0 and 1.")


# Artifacts exceeding this threshold get a metadata stub instead of full content.
_MAX_ARTIFACT_BYTES = 50_000  # ~50 KB

# Roles whose full content is always useful for auditing (even if large).
_ALWAYS_INLINE_ROLES = {"crawler_entry", "readme", "signature_runtime"}

# File types that are raw dumps — never inline their full content.
_DUMP_TYPES = {"wasm", "bin"}


def _should_inline(item: Dict[str, Any], size_bytes: int) -> bool:
    """Decide whether to include full content vs. a metadata stub."""
    if item.get("is_binary"):
        return False
    if item.get("type") in _DUMP_TYPES:
        return False
    if item.get("role") in _ALWAYS_INLINE_ROLES:
        return True
    # Large JS/text files that aren't core deliverables get stubbed.
    if size_bytes > _MAX_ARTIFACT_BYTES:
        return False
    return True


def _read_artifacts(manifest: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for item in manifest:
        path = os.path.abspath(str(item.get("path") or "").strip())
        if not path or not os.path.exists(path):
            continue

        try:
            size_bytes = os.path.getsize(path)
        except OSError:
            size_bytes = 0

        if not _should_inline(item, size_bytes):
            summary = str(item.get("summary") or "").strip()
            stub = (
                f"<artifact_stub>\n"
                f"path: {path}\n"
                f"type: {item.get('type', 'unknown')}\n"
                f"role: {item.get('role', 'supporting')}\n"
                f"size_bytes: {size_bytes}\n"
                f"tags: {item.get('tags', [])}\n"
            )
            if summary:
                stub += f"summary: {summary}\n"
            stub += "</artifact_stub>"
            chunks.append(f"[Artifact] {path}\n{stub}")
            continue

        try:
            with open(path, "rb") as handle:
                raw = handle.read()
        except OSError:
            continue

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = (
                "<binary artifact omitted>\n"
                f"path: {path}\n"
                f"size_bytes: {size_bytes or len(raw)}\n"
                f"head_hex: {raw[:32].hex()}"
            )

        chunks.append(f"[Artifact] {path}\n{content}")
    return "\n\n".join(chunks)


def build_find_fault_context_str(state: AgentState) -> str | None:
    current_delivery_manifest = list(state.get("current_delivery_manifest", []) or [])
    # `skip_audit` filters out CONTENT reads only (e.g. bundled sandbox engine,
    # dumped obfuscated risk-control JS — no value in auditing their content).
    # The manifest JSON block below still shows every entry so the auditor
    # knows which runtime dependencies shipped with the delivery.
    readable_manifest = [
        item for item in current_delivery_manifest
        if "skip_audit" not in (item.get("tags") or [])
    ]
    artifact_text = _read_artifacts(readable_manifest)
    if not artifact_text.strip():
        return None

    input_artifact_manifest = list(state.get("input_artifact_manifest", []) or [])
    input_text = _read_artifacts(input_artifact_manifest)

    sections = [
        _json_block("input_artifact_manifest", input_artifact_manifest),
    ]

    if input_text.strip():
        sections.append(f"<input_artifact_contents>\n{input_text}\n</input_artifact_contents>")

    sections.extend([
        _json_block("current_delivery_manifest", current_delivery_manifest),
        f"<delivered_artifact_contents>\n{artifact_text}\n</delivered_artifact_contents>"
    ])

    return "\n\n".join(section for section in sections if section)


def _for_review(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Replay the run's history into a *different* model's request.

    Reasoning blocks are provider-signed and only valid on the request that
    produced them, so sending them to the reviewer — often another model or
    another provider entirely — risks a signature rejection. The reasoning text
    is what the audit actually needs, so it is flattened into an ordinary text
    block: content preserved, signature contract dropped.

    tool_calls stay attached, and their ToolMessages follow in the list, so the
    reviewer sees which calls produced which results.
    """
    converted: List[BaseMessage] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            converted.append(message)
            continue
        parts: List[Dict[str, Any]] = []
        for block in message.content_blocks:
            kind = block.get("type")
            if kind == "reasoning":
                text = str(block.get("reasoning") or "").strip()
                if text:
                    parts.append({"type": "text", "text": f"[reasoning]\n{text}"})
            elif kind == "text":
                text = str(block.get("text") or "").strip()
                if text:
                    parts.append({"type": "text", "text": text})
        converted.append(AIMessage(
            id=message.id,
            content=parts or "",
            tool_calls=list(message.tool_calls or []),
        ))
    return converted


def review_verdict(feedback: str) -> HumanMessage:
    """Reviewer feedback re-enters the loop as a framework-authored user turn.

    It cannot be a ToolMessage: there is no tool_call_id for it to answer, and
    a ToolMessage that answers nothing is a malformed request next turn.

    Public so host-defined reviewer nodes (custom_find_fault) can report a
    rejection the same way the builtin does::

        return {"messages": [review_verdict(msg)],
                "end_tag": False, "current_delivery_manifest": []}
    """
    return HumanMessage(content=f"[Find-Fault Review]\n{feedback}")


_verdict_message = review_verdict


# 结构化输出失败 = 模型这一轮没按 schema 调用工具。这是随机的，同一份 prompt 再来
# 一次常常就调用了，所以值得重试；网络/限流那层由 SDK 自己重试，这里不重复管。
# 重试用尽后照旧往上抛，不把"没审出来"降级成"通过"。
_AUDIT_ATTEMPTS = 2


@retry(
    stop=stop_after_attempt(_AUDIT_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((OutputParserException, ValidationError)),
    before_sleep=lambda retry_state: logger.warning(
        "[find_fault] structured output failed (attempt %d/%d), retrying: %r",
        retry_state.attempt_number,
        _AUDIT_ATTEMPTS,
        retry_state.outcome.exception(),
    ),
    reraise=True,
)
async def _audit_with_retry(
    structured_llm: Any, messages: List[BaseMessage]
) -> GenericFindFaultOutput:
    return await structured_llm.ainvoke(messages)


def create_find_fault_node(system_prompt: str, llm: BaseChatModel):
    async def find_fault_node(state: AgentState) -> Dict[str, Any]:
        current_delivery_manifest = list(state.get("current_delivery_manifest", []) or [])
        if not current_delivery_manifest:
            # Empty manifest = text-only delivery. Nothing to audit; let the
            # graph proceed to finish with end_tag preserved.
            return {}

        combined_system_prompt = (
            f"{COMMON_FIND_FAULT_SYSTEM_PROMPT}\n\n{system_prompt}"
        )
        messages: List[BaseMessage] = [SystemMessage(content=combined_system_prompt)]

        # 1. The run itself, verbatim.
        #
        # The reviewer needs the whole conversation, not a summary of it and not
        # the latest message: on a continued thread the newest request is often
        # "继续", which says nothing about what the delivery had to satisfy. The
        # requirements were stated earlier, refined across turns, and the agent's
        # own reasoning recorded which readings it settled on — so the audit
        # replays the native history and judges the artifacts against all of it.
        messages.extend(_for_review(list(state.get("messages", []) or [])))

        # 2. Context
        prompt_context = build_find_fault_context_str(state)
        if not prompt_context:
            feedback = "Find-fault rejected the delivery because the artifact content could not be read."
            return {
                "end_tag": False,
                "current_delivery_manifest": [],
                "approved_artifact_manifest": [],
                "messages": [_verdict_message(feedback)],
            }
        messages.append(HumanMessage(content=prompt_context))

        # 3. Observer messages
        observer_message_parts = list(state.get("observer_message_parts", []) or [])
        if observer_message_parts:
            messages.extend(observer_message_parts)

        # 4. The audit instruction, last so it is what the reviewer acts on.
        messages.append(HumanMessage(content=(
            "Please formally evaluate the delivered artifacts based on your rules.\n"
            "Judge them against everything the user asked for across this entire "
            "conversation, not only the most recent message."
        )))

        structured_llm = structured_output_llm(llm, GenericFindFaultOutput)

        result = await _audit_with_retry(structured_llm, messages)
        validation = result.model_dump()
        logging.info(f"Find-fault validation result: {json.dumps(validation, indent=2, ensure_ascii=False)}")

        accepted = bool(validation.get("is_acceptable"))

        # Build actionable feedback from all relevant fields
        feedback_parts = []
        assessment = str(validation.get("decisive_assessment") or "").strip()
        if assessment:
            feedback_parts.append(assessment)

        fatal_gaps = validation.get("fatal_gaps") or []
        if fatal_gaps and not accepted:
            feedback_parts.append("\n[Fatal Gaps]")
            for i, gap in enumerate(fatal_gaps, 1):
                feedback_parts.append(f"  {i}. {gap}")

        recommended_rework = validation.get("recommended_rework") or []
        if recommended_rework and not accepted:
            feedback_parts.append("\n[Recommended Rework]")
            for i, rework in enumerate(recommended_rework, 1):
                feedback_parts.append(f"  {i}. {rework}")

        feedback = "\n".join(feedback_parts) if feedback_parts else "Find-fault completed."

        # --- Update delivery_status in session store ---
        session_id = str(state.get("session_id") or "default")
        delivery_status = session_store.get(session_id, "delivery_status", {})
        for item in current_delivery_manifest:
            # Skip artifacts flagged as not-audited (e.g. raw runtime dumps).
            if "skip_audit" in (item.get("tags") or []):
                continue
            artifact_name = os.path.basename(str(item.get("path") or ""))
            if not artifact_name:
                continue
            if accepted:
                entry = delivery_status.get(artifact_name)
                if entry:
                    entry["status"] = "ACCEPTED"
                    entry["fatal_gaps"] = []
                    entry["recommended_rework"] = []
            else:
                entry = delivery_status.get(artifact_name, {"path": os.path.abspath(str(item.get("path") or ""))})
                entry["status"] = "REJECTED"
                entry["fatal_gaps"] = list(fatal_gaps)
                entry["recommended_rework"] = list(recommended_rework)
                delivery_status[artifact_name] = entry
        session_store.set(session_id, "delivery_status", delivery_status)

        return {
            "end_tag": False,
            "current_delivery_manifest": current_delivery_manifest if accepted else [],
            "approved_artifact_manifest": current_delivery_manifest if accepted else [],
            # Accepted deliveries end the run; only a rejection needs to be fed
            # back for rework.
            "messages": [] if accepted else [_verdict_message(feedback)],
        }

    return find_fault_node
