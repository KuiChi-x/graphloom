import os
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ArtifactManifestEntry(BaseModel):
    path: str = Field(default="", description="Absolute artifact path.")
    filename: str = Field(default="", description="Basename of the artifact path.")
    type: str = Field(default="unknown", description="Artifact type such as py, js, wasm, md, txt, json, or bin.")
    role: str = Field(default="supporting", description="Artifact role, for example crawler_entry, signature_runtime, or readme.")
    producer: str = Field(default="", description="Producer name for traceability.")
    size_bytes: int = Field(default=0, description="Artifact file size.")
    created_at: str = Field(default="", description="ISO timestamp from filesystem metadata.")
    updated_at: str = Field(default="", description="ISO timestamp from filesystem metadata.")
    is_binary: bool = Field(default=False, description="Whether the artifact is binary.")
    summary: str = Field(default="")
    tags: List[str] = Field(default_factory=list)


_TEXT_LIKE_EXTENSIONS = {"py", "js", "ts", "json", "md", "txt", "yaml", "yml", "html", "css", "xml", "csv", "sh"}


def _iso_from_ts(timestamp: float) -> str:
    try:
        return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")
    except Exception:
        return ""


def _artifact_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ext or "unknown"


def _is_binary(path: str, artifact_type: str) -> bool:
    if artifact_type in _TEXT_LIKE_EXTENSIONS:
        return False
    if artifact_type in {"wasm", "png", "jpg", "jpeg", "gif", "ico", "pdf", "zip", "bin"}:
        return True
    try:
        with open(path, "rb") as handle:
            sample = handle.read(2048)
    except OSError:
        return False
    return b"\x00" in sample


def normalize_artifact_manifest_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(entry or {})
    path = str(data.get("path") or "").strip()
    filename = str(data.get("filename") or "").strip()

    if path:
        resolved_path = os.path.abspath(path)
        data["path"] = resolved_path
        if not filename:
            data["filename"] = os.path.basename(resolved_path)
        if not data.get("type"):
            data["type"] = _artifact_type(resolved_path)
        if os.path.exists(resolved_path):
            try:
                stat_result = os.stat(resolved_path)
                data.setdefault("size_bytes", int(stat_result.st_size))
                data.setdefault("created_at", _iso_from_ts(stat_result.st_ctime))
                data.setdefault("updated_at", _iso_from_ts(stat_result.st_mtime))
            except OSError:
                pass
            data.setdefault("is_binary", _is_binary(resolved_path, str(data.get("type") or "unknown")))

    return ArtifactManifestEntry(**data).model_dump()


def merge_artifact_manifest(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    left = list(left or [])
    right = list(right or [])

    merged: Dict[str, Dict[str, Any]] = {}
    for raw_item in left + right:
        if not isinstance(raw_item, dict):
            continue
        item = normalize_artifact_manifest_entry(raw_item)
        path_value = str(item.get("path") or "").strip()
        if not path_value:
            continue

        previous = merged.get(path_value)
        if previous is None:
            merged[path_value] = item
            continue

        prev_updated = str(previous.get("updated_at") or "")
        curr_updated = str(item.get("updated_at") or "")
        prev_created = str(previous.get("created_at") or "")
        curr_created = str(item.get("created_at") or "")
        if (curr_updated, curr_created, path_value) >= (prev_updated, prev_created, path_value):
            merged[path_value] = item

    return sorted(
        merged.values(),
        key=lambda item: (
            str(item.get("updated_at") or ""),
            str(item.get("created_at") or ""),
            str(item.get("path") or ""),
        ),
        reverse=True,
    )


def replace_artifact_manifest(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    right = list(right or [])

    replaced: List[Dict[str, Any]] = []
    for raw_item in right:
        if not isinstance(raw_item, dict):
            continue
        path_value = str(raw_item.get("path") or "").strip()
        if not path_value:
            continue
        replaced.append(normalize_artifact_manifest_entry(raw_item))

    return sorted(
        replaced,
        key=lambda item: (
            str(item.get("updated_at") or ""),
            str(item.get("created_at") or ""),
            str(item.get("path") or ""),
        ),
        reverse=True,
    )
