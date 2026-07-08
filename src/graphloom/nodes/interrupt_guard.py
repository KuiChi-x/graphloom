"""
interrupt_guard — cooperative pause checkpoint.

Call raise_if_cancelled(config) at node entry points (ai_node / tool_node).
If configurable.cancel_event is set, raise GraphInterrupt so LangGraph stops at
the nearest checkpoint, leaving full state intact for a subsequent
ainvoke(None, config) to resume.
"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphInterrupt


def raise_if_cancelled(config: RunnableConfig | None) -> None:
    if not config:
        return
    configurable = config.get("configurable") or {}
    cancel_event = configurable.get("cancel_event")
    if isinstance(cancel_event, asyncio.Event) and cancel_event.is_set():
        # LangGraph treats GraphInterrupt as a pause signal; the latest
        # checkpoint remains intact so a subsequent ainvoke(None, config)
        # will resume execution from this point.
        raise GraphInterrupt(("paused_by_user",))
