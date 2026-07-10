"""Observer pattern for graphloom step events.

graphloom's nodes build rich structured steps (step_id, titles, think,
tool_calls, results) but publish nothing. This module defines the minimal
emission contract: a node calls ``emit(event_type, payload)`` on whatever the
host injected via ``configurable["runtime_context"]["event_emitter"]``.

The framework stays fully decoupled:
- It never references WebSocket / HTTP / any transport.
- It makes no assumption a listener exists — ``get_emitter`` returns None when
  none was injected, and ``emit_step`` is a no-op then. Nodes guard with it.
- Hosts (reverseloom, or any other deployment) supply their own async callable
  to bridge events to whatever sink they use (WS push, SSE, log, test recorder).

Event types:
    step_planned  — ai_node built a pending step (titles + think + content)
    tool_start    — a tool invocation is about to run
    tool_end      — a tool invocation finished (result + has_error)
    step_done     — history_node closed out a completed step (full payload)
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

# A host-supplied async emitter: emit(event_type: str, payload: dict) -> None.
StepEmitter = Callable[[str, Dict[str, Any]], Awaitable[None]]


def get_emitter(config: Dict[str, Any]) -> Optional[StepEmitter]:
    """Return the host-injected emitter, or None (nodes then skip emitting).

    The emitter lives at the top level of ``configurable`` (NOT inside
    ``runtime_context``), because runtime_context is copied into tool args and
    thus into checkpointed state — a function/callable there is not msgpack-
    serializable. configurable itself is never serialized.
    """
    configurable = config.get("configurable") or {}
    emitter = configurable.get("event_emitter")
    return emitter if callable(emitter) else None


async def emit_step(config: Dict[str, Any], event_type: str, payload: Dict[str, Any]) -> None:
    """Publish an event to the host emitter if one was injected; no-op otherwise."""
    emitter = get_emitter(config)
    if emitter is None:
        return
    try:
        await emitter(event_type, dict(payload))
    except Exception:  # a broken listener must never break the graph run
        import logging
        logging.warning(f"[graphloom] event_emitter raised on {event_type}", exc_info=True)
