import json
import logging
import os
from typing import Any, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from graphloom.model.state import AgentState
from graphloom.prompt.context_renderer import _json_block, build_user_request_str, render_past_steps
from graphloom.prompt.find_fault_system_prompt import COMMON_FIND_FAULT_SYSTEM_PROMPT
from graphloom.util.session_store import session_store


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

        history_blocks = render_past_steps(list(state.get("past_steps", []) or []))
        if len(history_blocks) > 1:
            messages.append(HumanMessage(content=[
                *({"type": "text", "text": block} for block in history_blocks[:-1]),
            ]))
            messages.append(HumanMessage(content="</agent_history>"))
        else:
            messages.append(HumanMessage(content=history_blocks[0]))

        # 2. Context (SystemMessage)
        prompt_context = build_find_fault_context_str(state)
        if not prompt_context:
            feedback = "Find-fault rejected the delivery because the artifact content could not be read."
            step = len(state.get("past_steps", []) or []) + 1
            return {
                "end_tag": False,
                "current_delivery_manifest": [],
                "approved_artifact_manifest": [],
                "tool_result_history": [
                    {
                        "step": step,
                        "tool_name": "find_fault",
                        "tool_args": {},
                        "result": feedback,
                        "has_error": True,
                    }
                ],
            }
        messages.append(SystemMessage(content=prompt_context))

        # 3. Observer messages
        observer_message_parts = list(state.get("observer_message_parts", []) or [])
        if observer_message_parts:
            messages.extend(observer_message_parts)

        # 4. User Request / HumanMessage
        user_request = build_user_request_str(state)
        human_trigger = f"{user_request}\n\nPlease formally evaluate the delivered artifacts based on your rules."

        messages.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": human_trigger},
                ]
            )
        )

        structured_llm = llm.with_structured_output(
            GenericFindFaultOutput,
            method="json_schema",
            strict=False,
        )

        result = await structured_llm.ainvoke(messages)
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
        step = len(state.get("past_steps", []) or []) + 1

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
            "tool_result_history": [
                {
                    "step": step,
                    "tool_name": "find_fault",
                    "tool_args": {},
                    "result": feedback,
                    "has_error": not accepted,
                }
            ],
        }

    return find_fault_node
