"""
token_counter.py

tiktoken-based token estimation for context budget gating. Used only by the
context compaction gate — not a general-purpose utility.
"""
import json
from typing import Any, Iterable, List, Optional

import tiktoken
from langchain_core.messages import BaseMessage

from graphloom.config import COMPACT_TOKENIZER_ENCODING


_ENCODER: Optional["tiktoken.Encoding"] = None

# Rough heuristic for a single image part. Providers differ:
#   - OpenAI GPT-4o high-detail: 85 + 170 * tiles (≈ 765 for 1024x1024)
#   - Anthropic Claude:          ≈ (w * h) / 750, capped around 1600
# We take a single conservative upper-bound to avoid under-counting. The base64
# payload itself MUST NOT be tokenized — it would overcount by orders of magnitude.
_IMAGE_TOKEN_ESTIMATE = 1500


def _get_encoder() -> "tiktoken.Encoding":
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding(COMPACT_TOKENIZER_ENCODING)
    return _ENCODER


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def _count_part_tokens(part: Any) -> int:
    """Count tokens for a single message-content part.

    Handles the two shapes LangChain multimodal messages use:
      - plain text part: {"type": "text", "text": "..."}
      - image part:      {"type": "image_url", "image_url": {...}} or {"type": "image", ...}
    Unknown dict shapes fall back to JSON-stringified text counting.
    """
    if isinstance(part, str):
        return count_tokens(part)
    if isinstance(part, dict):
        ptype = part.get("type")
        if ptype == "text":
            return count_tokens(str(part.get("text") or ""))
        if ptype in ("image_url", "image", "input_image"):
            return _IMAGE_TOKEN_ESTIMATE
        # Unknown structured part — count its serialized form.
        try:
            return count_tokens(json.dumps(part, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            return count_tokens(str(part))
    return count_tokens(str(part))


def count_message_tokens(message: BaseMessage) -> int:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return sum(_count_part_tokens(p) for p in content)
    if isinstance(content, str):
        return count_tokens(content)
    return _count_part_tokens(content)


def count_messages_tokens(messages: Iterable[BaseMessage]) -> int:
    return sum(count_message_tokens(m) for m in messages)
