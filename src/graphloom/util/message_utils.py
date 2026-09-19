"""Readers over the native `messages` channel.

The channel is the single source of truth, so anything that used to live in a
dedicated state field (the user's request, the latest AI message, the pending
tool results) is recovered from it here instead of being stored twice.
"""
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

MessageList = Sequence[BaseMessage] | None


def get_last_ai_message(messages: MessageList) -> Optional[AIMessage]:
    for message in reversed(list(messages or [])):
        if isinstance(message, AIMessage):
            return message
    return None


def text_of(message: Any) -> str:
    """Flatten a message's content to plain text, dropping non-text blocks."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    parts: List[str] = []
    for part in content or []:
        if isinstance(part, dict):
            if part.get("type") in ("text", None):
                parts.append(str(part.get("text") or ""))
        else:
            parts.append(str(part))
    return "".join(parts).strip()


def last_user_text(messages: MessageList) -> str:
    """Text of the most recent HumanMessage. For labelling a run in the event
    log — the model reads the message itself, in place, not this."""
    for message in reversed(list(messages or [])):
        if isinstance(message, HumanMessage):
            return text_of(message)
    return ""


def turns(messages: MessageList) -> List[List[BaseMessage]]:
    """Group messages into replayable turns.

    A turn starts at an AIMessage and runs until the next one, so it carries
    that AIMessage plus the ToolMessages answering it. Anything before the
    first AIMessage (the originating request) is its own leading group.

    Compaction cuts only on these boundaries: splitting a tool_use from its
    tool_result, or dropping a thinking block from a turn that carries one,
    fails the Anthropic-protocol signature check.
    """
    grouped: List[List[BaseMessage]] = []
    for message in list(messages or []):
        if isinstance(message, AIMessage) or not grouped:
            grouped.append([message])
        else:
            grouped[-1].append(message)
    return grouped


def tool_call_map(message: Optional[AIMessage]) -> Dict[str, Dict[str, Any]]:
    """call_id -> tool_call, for pairing ToolMessages back to their request."""
    if message is None:
        return {}
    return {
        str(call.get("id") or ""): call
        for call in (message.tool_calls or [])
    }
