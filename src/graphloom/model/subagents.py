from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class SubAgentRunContext:
    parent_session_id: str
    child_session_id: str
    step: Any
    result: Any = None
    error: Optional[BaseException] = None
    user_id: str = ""


@dataclass(frozen=True)
class SubAgentSpec:
    agent_name: str
    description: str
    factory: Callable[[], Any]
    cleanup_hook: Optional[Callable[[SubAgentRunContext], Any]] = None
