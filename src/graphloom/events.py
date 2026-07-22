"""Observer pattern for graphloom step events.

graphloom's nodes build rich structured steps (step_id, titles, think,
tool_calls, results) but publish nothing by themselves. This module defines
the emission contract: a node calls ``emit(event_type, payload)`` on whatever
the host injected via ``configurable["event_emitter"]``.

The framework stays fully decoupled:
- It never references WebSocket / HTTP / any transport.
- It makes no assumption a listener exists — ``get_emitter`` returns None when
  none was injected, and ``emit_step`` is a no-op then.
- Hosts supply either a plain async callable, or subclass ``BaseEventEmitter``
  and override the hooks they care about.

Event types (stable contract):
    ai_delta        — streaming token/reasoning chunk from the LLM
    step_planned    — ai_node built a pending step (titles + think + content)
    tool_start      — a tool invocation is about to run
    tool_end        — a tool invocation finished (result + has_error)
    step_done       — history_node closed out a completed step (full payload)
    subagent_state  — dispatch lifecycle: running | done | error | skipped
    subagent_reply  — sub-agent final_reply text

Inject::

    from graphloom import BaseEventEmitter, build_agent_graph

    class MyEmitter(BaseEventEmitter):
        async def on_ai_delta(self, payload):
            print(payload.get("reasoning") or "", payload.get("content") or "")

        async def on_step_planned(self, payload):
            print("step", payload.get("step_index"), payload.get("next_action"))

    graph = build_agent_graph(...)
    await graph.ainvoke(
        {"input_query": "..."},
        config={"configurable": {"event_emitter": MyEmitter()}},
    )
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

# A host-supplied async emitter: emit(event_type: str, payload: dict) -> None.
StepEmitter = Callable[[str, Dict[str, Any]], Awaitable[None]]

# Known event type names — hosts may still receive unknown types via on_unknown.
EVENT_AI_DELTA = "ai_delta"
EVENT_STEP_PLANNED = "step_planned"
EVENT_TOOL_START = "tool_start"
EVENT_TOOL_END = "tool_end"
EVENT_STEP_DONE = "step_done"
EVENT_SUBAGENT_STATE = "subagent_state"
EVENT_SUBAGENT_REPLY = "subagent_reply"


class BaseEventEmitter:
    """Default, subclassable host adapter for graphloom events.

    Override only the hooks you need. The instance is callable so it can be
    injected as ``configurable["event_emitter"]`` without wrapping.

    Payload fields vary by event; common ones include::

        agent_name, session_id, step_index, step_id
        content, reasoning                       # ai_delta
        last_step_review, working_notes, next_action, tool_calls  # step_*
        tool_name, tool_args, result, has_error, call_id  # tool_*
        status, dispatch_id, child_session_id, parent_session_id, task_id, title, task, error  # subagent_state
        text                                     # subagent_reply
    """

    async def __call__(self, event_type: str, payload: Dict[str, Any]) -> None:
        data = dict(payload or {})
        handler = {
            EVENT_AI_DELTA: self.on_ai_delta,
            EVENT_STEP_PLANNED: self.on_step_planned,
            EVENT_TOOL_START: self.on_tool_start,
            EVENT_TOOL_END: self.on_tool_end,
            EVENT_STEP_DONE: self.on_step_done,
            EVENT_SUBAGENT_STATE: self.on_subagent_state,
            EVENT_SUBAGENT_REPLY: self.on_subagent_reply,
        }.get(str(event_type or ""))
        if handler is None:
            await self.on_unknown(str(event_type or ""), data)
            return
        await handler(data)

    # ---- hooks (override in host subclasses) --------------------------------

    async def on_ai_delta(self, payload: Dict[str, Any]) -> None:
        """Streaming LLM chunk. Keys: content, reasoning, agent_name, session_id, step_index."""

    async def on_step_planned(self, payload: Dict[str, Any]) -> None:
        """A tool-calling step was planned. Keys: step_id, step_index, next_action, tool_calls, …"""

    async def on_tool_start(self, payload: Dict[str, Any]) -> None:
        """Tool about to run. Keys: tool_name, tool_args, step_id, step_index, …"""

    async def on_tool_end(self, payload: Dict[str, Any]) -> None:
        """Tool finished. Keys: tool_name, tool_args, result, has_error, …"""

    async def on_step_done(self, payload: Dict[str, Any]) -> None:
        """Step closed after tools. Keys: step_id, last_step_review, working_notes, next_action, tool_calls, …"""

    async def on_subagent_state(self, payload: Dict[str, Any]) -> None:
        """Sub-agent lifecycle. Keys: status (running|done|error|skipped), child_session_id, …"""

    async def on_subagent_reply(self, payload: Dict[str, Any]) -> None:
        """Sub-agent final text. Keys: text, child_session_id, agent_name, …"""

    async def on_unknown(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Future/unknown event types land here. Default: ignore."""


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
