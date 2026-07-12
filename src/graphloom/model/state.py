from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from graphloom.config import COMPACT_SENTINEL_KEY
from graphloom.model.artifact_manifest import merge_artifact_manifest, replace_artifact_manifest


def add_past_steps(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if left is None:
        left = []
    if right is None:
        right = []
    # Sentinel-aware replace: when the compaction node wants to swap the whole
    # channel, it returns [{"__compact_replace__": True}, compacted, *recent].
    # All other callers keep append semantics.
    if right and isinstance(right[0], dict) and right[0].get(COMPACT_SENTINEL_KEY) is True:
        return list(right[1:])

    combined = list(left)
    step_index = {
        str(step.get("step_id")): idx
        for idx, step in enumerate(combined)
        if isinstance(step, dict) and step.get("step_id")
    }
    for step in right:
        if isinstance(step, dict) and step.get("step_id"):
            step_id = str(step.get("step_id"))
            if step_id in step_index:
                combined[step_index[step_id]] = step
                continue
            step_index[step_id] = len(combined)
        combined.append(step)
    return combined


def keep_max_step_counter(left: Any, right: Any) -> int:
    """Reducer for `step_counter`. Always keep the highest value seen so the
    counter is strictly monotonic across the graph — including across
    compaction runs that shrink `past_steps`. Using a dedicated counter
    (instead of `len(past_steps)`) is what prevents step_id collisions when
    compaction folds earlier steps into a summary."""
    try:
        left_int = int(left or 0)
    except (TypeError, ValueError):
        left_int = 0
    try:
        right_int = int(right or 0)
    except (TypeError, ValueError):
        right_int = 0
    return max(left_int, right_int)


_TOOL_HISTORY_MAX = 8


def add_tool_results(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if left is None:
        left = []
    if right is None:
        right = []
    combined = left + right
    return combined[-_TOOL_HISTORY_MAX:]


def append_items(left: List[Any], right: List[Any]) -> List[Any]:
    return list(left or []) + list(right or [])


class AgentState(TypedDict):
    current_agent_name: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    conversation: Annotated[Sequence[BaseMessage], add_messages]
    events: Annotated[List[Dict[str, Any]], append_items]
    input_query: str
    attach_message_parts: Optional[List[Dict[str, Any]]]
    session_id: Optional[str]
    latest_ai_message: Optional[AIMessage]

    past_steps: Annotated[List[Dict[str, Any]], add_past_steps]
    step_counter: Annotated[int, keep_max_step_counter]
    tool_result_history: List[Dict[str, Any]]
    observer_message_parts: Optional[List[HumanMessage]]

    input_artifact_manifest: Annotated[List[Dict[str, Any]], replace_artifact_manifest]
    current_delivery_manifest: Annotated[List[Dict[str, Any]], replace_artifact_manifest]
    approved_artifact_manifest: Annotated[List[Dict[str, Any]], merge_artifact_manifest]

    final_reply: Optional[str]
    agent_status: Optional[str]

    end_tag: bool


def build_initial_agent_state(**overrides: Any) -> AgentState:
    state: AgentState = {
        "current_agent_name": "main",
        "messages": [],
        "conversation": [],
        "events": [],
        "input_query": "",
        "attach_message_parts": None,
        "session_id": None,
        "latest_ai_message": None,
        "past_steps": [],
        "step_counter": 0,
        "tool_result_history": [],
        "observer_message_parts": None,
        "input_artifact_manifest": [],
        "current_delivery_manifest": [],
        "approved_artifact_manifest": [],
        "final_reply": "",
        "agent_status": "running",
        "end_tag": False,
    }
    state.update(overrides)
    return state
