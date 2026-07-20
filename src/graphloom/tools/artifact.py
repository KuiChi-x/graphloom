import csv
import io
import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import Field
from langchain_core.tools import tool

from graphloom.model.base_tool_input import StandardThoughtInput
from graphloom.model.artifact_manifest import normalize_artifact_manifest_entry
from graphloom.util.session_store import session_store


def build_artifact_metadata(
    artifact_path: str,
    *,
    producer: str = "",
    role: str = "",
    summary: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    metadata = normalize_artifact_manifest_entry(
        {
            "path": artifact_path,
            "producer": producer,
            "role": role,
            "summary": summary,
            "tags": list(tags or []),
        }
    )
    return metadata


class WriteArtifactInput(StandardThoughtInput):
    artifact_name: str = Field(description="Relative file name. Name it after the business content, e.g. 'booking_hotel_list_blueprint.md', 'agoda_flight_search_crawler.py'. Avoid generic names like 'blueprint.md' or 'results.csv'.")
    content: str = Field(description=(
        "The artifact content to save. "
        "Before composing this content, you MUST review two sources:\n"
        "  1. <agent_history> — scan the Memory field of each step for accumulated structured data "
        "(XPaths, API endpoints, request parameters, field mappings, code snippets).\n"
        "  2. <agent_history> — check recent tool results for the latest raw data.\n"
        "Then compose the artifact by incorporating this evidence. "
        "CRITICAL: All structured data MUST be copied EXACTLY as it appears in Memory or tool results. "
        "NEVER abbreviate, truncate ancestor segments, or simplify. "
        "WRONG: '#login-btn'  RIGHT: 'div.header-nav > ul.menu-list > li:nth-of-type(3) > #login-btn'. "
        "WRONG: '/api/search'  RIGHT: 'https://www.example.com/api/v2/search?category=electronics&page=1'."
    ))
    append: bool = Field(default=False, description=(
        "True = append content to the end of the file (use for incremental data collection). "
        "False = overwrite the file (default)."
    ))
    summary: str = Field(default="", description="One sentence describing what is in this file, about 300-500 chars.")
    tags: List[str] = Field(default_factory=list, description="Short labels for filtering, e.g. ['site:booking.com', 'kind:blueprint'].")


def _session_base_dir(runtime_context: Dict[str, Any]) -> str:
    """Resolve the artifact workspace root. The host injects it via
    runtime_context["artifact_base_dir"]; falls back to the
    GRAPHLOOM_ARTIFACT_BASE_DIR env var, then cwd."""
    return str(
        runtime_context.get("artifact_base_dir")
        or os.environ.get("GRAPHLOOM_ARTIFACT_BASE_DIR")
        or "."
    )


def _get_session_dir(session_id: str, runtime_context: Dict[str, Any]) -> str:
    explicit = str(runtime_context.get("artifact_dir") or "").strip()
    path = explicit or os.path.join(_session_base_dir(runtime_context), session_id)
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_csv(raw: str) -> str:
    """Re-parse LLM-generated CSV through Python's csv module to fix common issues.

    Handles: unquoted fields with commas, unescaped quotes, double-escaped
    JSON tool-call output (literal \\n and \\"), blank rows, and trailing whitespace.
    """
    text = raw.strip()
    if not text:
        return text

    # Detect double-escaped LLM output: no real newlines but has literal \n sequences
    if "\n" not in text and "\\n" in text:
        text = text.replace('\\"', '"').replace("\\n", "\n")

    try:
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    except csv.Error:
        return raw  # fall back to original if parsing fails

    if not rows:
        return raw

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerows(rows)
    return out.getvalue().rstrip("\r\n")


def _is_csv_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() == ".csv"


def _validate_write_path(filepath: str, session_dir: str) -> Optional[str]:
    """Return error message if filepath escapes session_dir, else None."""
    real_file = os.path.realpath(filepath)
    real_dir = os.path.realpath(session_dir)
    if not real_file.startswith(real_dir + os.sep) and real_file != real_dir:
        return (
            f"Write denied: path '{os.path.basename(filepath)}' resolves outside your workspace. "
            f"You can only write to files in your own session directory."
        )
    return None


@tool("write_artifact", args_schema=WriteArtifactInput)
async def write_artifact(
    artifact_name: str,
    content: str,
    append: bool = False,
    summary: str = "",
    tags: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Save data or analysis results as an artifact file. Use descriptive file names.
    Before writing, review the Memory fields in <agent_history> and recent <tool_result_history> to collect all structured data (XPaths, URLs, API endpoints, parameters, code snippets). Copy them EXACTLY into the artifact — never truncate or simplify.
    Set append=True to add content incrementally (e.g., collecting data across multiple pages)."""
    return await _write_artifact_impl(
        artifact_name=artifact_name,
        content=content,
        append=append,
        summary=summary,
        tags=tags,
        **kwargs,
    )


async def _write_artifact_impl(
    artifact_name: str,
    content: str,
    append: bool = False,
    summary: str = "",
    tags: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """Core artifact-writing logic. Shared by the generic `write_artifact` tool
    and by structured wrappers that register under the same public name but
    compose their own markdown first.
    """
    session_id = kwargs.get("session_id", "default")
    runtime_context = dict(kwargs.get("runtime_context") or {})
    session_dir = _get_session_dir(session_id, runtime_context)
    filepath = os.path.join(session_dir, artifact_name)

    write_error = _validate_write_path(filepath, session_dir)
    if write_error:
        return write_error

    try:
        # Normalize CSV content to fix common LLM formatting issues
        write_content = content
        if _is_csv_file(artifact_name):
            new_content = _normalize_csv(content)
            if append and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                existing_header: Optional[List[str]] = None
                try:
                    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
                        for row in csv.reader(f):
                            if any(cell.strip() for cell in row):
                                existing_header = row
                                break
                except (OSError, csv.Error):
                    existing_header = None

                if existing_header is not None and new_content:
                    try:
                        new_rows = list(csv.reader(io.StringIO(new_content)))
                    except csv.Error:
                        new_rows = []
                    if new_rows and new_rows[0] == existing_header:
                        out = io.StringIO()
                        csv.writer(out).writerows(new_rows[1:])
                        new_content = out.getvalue().rstrip("\r\n")

                # Ensure existing file ends with a newline so the first new row
                # doesn't get glued onto the previous last row.
                try:
                    with open(filepath, "rb") as f:
                        f.seek(-1, os.SEEK_END)
                        last_byte = f.read(1)
                except OSError:
                    last_byte = b"\n"
                prefix = "" if last_byte in (b"\n", b"\r") else "\n"
                write_content = (prefix + new_content) if new_content else ""
                mode = "a"
            else:
                write_content = new_content
                mode = "w"
        else:
            mode = "a" if append else "w"

        open_kwargs = {"encoding": "utf-8"}
        if _is_csv_file(artifact_name):
            open_kwargs["newline"] = ""
            # Use UTF-8 BOM for new CSV files so Excel detects encoding correctly
            if mode == "w":
                open_kwargs["encoding"] = "utf-8-sig"
        # Skip the write if CSV append was fully dedup'd away (nothing new to add).
        if write_content or mode != "a" or not _is_csv_file(artifact_name):
            with open(filepath, mode, **open_kwargs) as f:
                f.write(write_content)
        action = "appended to" if append else "saved"
        logging.info(f"[ArtifactTools] Artifact {action}: {filepath}")

        delivery_status = session_store.get(session_id, "delivery_status", {})
        author = str(runtime_context.get("current_agent_name") or kwargs.get("current_agent_name") or "").strip()
        delivery_status[artifact_name] = {
            "path": os.path.abspath(filepath),
            "status": "DRAFT",
            "fatal_gaps": [],
            "recommended_rework": [],
            "summary": str(summary or "").strip(),
            "tags": [str(t).strip() for t in (tags or []) if str(t).strip()],
            "producer": author,
        }
        session_store.set(session_id, "delivery_status", delivery_status)

        return (
            f"Artifact {action} successfully: {filepath}\n\n"
            f"[SYSTEM NOTE] Do not rewrite this artifact unless you need to correct an error. "
            f"Use `patch_artifact` for targeted fixes instead of full rewrites. "
            f"If you need to create other deliverables, continue doing so. "
            f"If ALL required deliverables are complete, you MUST call `deliver_artifact` to submit them and finish your task."
        )
    except Exception as e:
        return f"Failed to save artifact: {str(e)}"


def _resolve_workspace_path(path: str, session_dir: str) -> str:
    normalized = str(path or "").strip()
    if not normalized:
        return ""
    if os.path.isabs(normalized):
        return os.path.abspath(normalized)
    return os.path.abspath(os.path.join(session_dir, normalized))


class ReadArtifactInput(StandardThoughtInput):
    artifact_path: str = Field(description="Absolute path of the artifact to read. Use paths from <artifact_manifest> or from write_artifact's return value.")


@tool("read_artifact", args_schema=ReadArtifactInput)
async def read_artifact(artifact_path: str, **kwargs) -> str:
    """Read an artifact file when metadata alone is insufficient. Use this to inspect the actual contents before reusing, reviewing, or delivering an artifact."""
    session_id = kwargs.get("session_id", "default")
    runtime_context = dict(kwargs.get("runtime_context") or {})
    session_dir = _get_session_dir(session_id, runtime_context)
    resolved_path = _resolve_workspace_path(artifact_path, session_dir)

    if not os.path.exists(resolved_path):
        return f"Error: file not found at {resolved_path} (input: {artifact_path})"

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"File {resolved_path} contents:\n\n{content}"
    except Exception as e:
        return f"Failed to read artifact: {str(e)}"


class DeliverArtifactInput(StandardThoughtInput):
    artifact_paths: List[str] = Field(description="List of absolute paths to deliver. Use paths from write_artifact's return value or from <approved_artifact_manifest>.")
    final_reply: str = Field(
        description=(
            "Final message to the user. "
            "ONLY report data you directly observed during this session. "
            "Do NOT use training knowledge to fill gaps — if information was not found, say so explicitly. "
            "Do NOT claim completion of steps from compacted_memory or prior session summaries "
            "unless you explicitly verified them yourself. "
            "If uncertain whether a prior step completed, say so explicitly."
            "When referencing artifacts, write the ABSOLUTE path (same value as in `artifact_paths`)."
        ),
    )


class PatchArtifactInput(StandardThoughtInput):
    artifact_name: str = Field(description="Relative file name of the artifact to patch (e.g., 'blueprint.md'). Must already exist in your session directory. Do NOT use absolute paths.")
    old_str: str = Field(description="The exact text to find and replace. Must match the file content exactly (including whitespace and newlines).")
    new_str: str = Field(description="The replacement text. Must be different from old_str.")


@tool("patch_artifact", args_schema=PatchArtifactInput)
async def patch_artifact(artifact_name: str, old_str: str, new_str: str, **kwargs) -> str:
    """Make a targeted edit to an existing artifact file by replacing a specific text fragment.
    Use this instead of rewriting the entire file when fixing bugs or updating specific sections.
    The old_str must match exactly one location in the file."""
    session_id = kwargs.get("session_id", "default")
    runtime_context = dict(kwargs.get("runtime_context") or {})
    session_dir = _get_session_dir(session_id, runtime_context)
    filepath = os.path.join(session_dir, artifact_name)

    write_error = _validate_write_path(filepath, session_dir)
    if write_error:
        return write_error

    if not os.path.exists(filepath):
        return f"Error: artifact '{artifact_name}' not found. Use write_artifact to create it first."

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if old_str == new_str:
            return "Error: old_str and new_str are identical. No change needed."

        occurrences = content.count(old_str)
        if occurrences == 0:
            preview = content[:500] + ("..." if len(content) > 500 else "")
            return (
                f"Error: old_str not found in '{artifact_name}'. "
                f"Make sure the text matches exactly (including whitespace).\n"
                f"File preview:\n{preview}"
            )

        updated = content.replace(old_str, new_str)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(updated)

        logging.info(f"[ArtifactTools] Artifact patched: {filepath} ({occurrences} replacement(s))")
        return (
            f"Artifact patched successfully: {filepath} ({occurrences} replacement(s) made).\n"
            f"Use `read_artifact` if you need to verify the full updated content."
        )
    except Exception as e:
        return f"Failed to patch artifact: {str(e)}"


@tool("deliver_artifact", args_schema=DeliverArtifactInput)
async def deliver_artifact(artifact_paths: List[str], final_reply: str = "", **kwargs) -> Dict[str, Any]:
    """
    Final delivery tool: Submits all completed artifact files to the human user and ends your current workflow.

    CRITICAL USAGE RULES:
    1. BATCH DELIVERY: Only call this tool ONCE. If your task requires multiple artifacts, finish writing ALL of them first before calling this tool. Pass all filenames together in the `artifact_paths` list.
    2. PREREQUISITES: Every file in `artifact_paths` MUST have already been successfully saved using `write_artifact`.
    3. END OF TASK: This is the ONLY way to complete your task. Never call this if you still have remaining work to do.
    """
    runtime_context = dict(kwargs.get("runtime_context") or {})
    session_id = runtime_context.get("session_id") or kwargs.get("session_id", "default")
    session_dir = _get_session_dir(session_id, runtime_context)
    producer = str(runtime_context.get("current_agent_name") or kwargs.get("current_agent_name") or "main").strip()

    valid_paths = []
    missing_files = []

    for path in artifact_paths:
        full_path = _resolve_workspace_path(path, session_dir)
        if os.path.exists(full_path):
            valid_paths.append(full_path)
        else:
            missing_files.append(path)

    if missing_files:
        return {
            "delivery_error": (
                "Delivery failed: the following files were not found: "
                f"{', '.join(missing_files)}. Make sure to first save them using write_artifact "
                "or have them generated by system verification."
            )
        }

    delivery_status = session_store.get(session_id, "delivery_status", {}) or {}

    # Auto-promote every runtime mount registered in this session
    # (sandbox engine / dumped JS/WASM mounted for replay).
    auto_added_paths: List[str] = []
    seen = {os.path.abspath(p) for p in valid_paths}
    for entry in delivery_status.values():
        if "kind:runtime_mount" not in (entry.get("tags") or []):
            continue
        mount_abs = os.path.abspath(str(entry.get("path") or "").strip())
        if not mount_abs or mount_abs in seen:
            continue
        if not os.path.exists(mount_abs):
            continue
        auto_added_paths.append(mount_abs)
        seen.add(mount_abs)

    delivered_paths = valid_paths + auto_added_paths

    def _metadata_for(path: str) -> Dict[str, Any]:
        entry = delivery_status.get(os.path.basename(path)) or {}
        return build_artifact_metadata(
            path,
            producer=str(entry.get("producer") or producer),
            summary=str(entry.get("summary") or ""),
            tags=list(entry.get("tags") or []),
        )

    return {
        "status": "success",
        "end_tag": True,
        "current_delivery_manifest": [_metadata_for(path) for path in delivered_paths],
        "approved_artifact_manifest": [],
        "delivered_artifact_paths": delivered_paths,
        "final_reply": str(final_reply or "").strip(),
    }
