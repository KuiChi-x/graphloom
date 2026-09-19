from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from graphloom.model.artifact_manifest import merge_artifact_manifest, replace_artifact_manifest
from graphloom.model.timeline import merge_turns


def keep_max_step_counter(left: Any, right: Any) -> int:
    """Reducer for `step_counter`. Always keep the highest value seen so the
    counter is strictly monotonic across the graph — including across
    compaction runs that shrink `messages`. Turn identity (used by the UI to
    group think/tool chips) is keyed on this counter, never on list length."""
    try:
        left_int = int(left or 0)
    except (TypeError, ValueError):
        left_int = 0
    try:
        right_int = int(right or 0)
    except (TypeError, ValueError):
        right_int = 0
    return max(left_int, right_int)


def append_items(left: List[Any], right: List[Any]) -> List[Any]:
    return list(left or []) + list(right or [])


class AgentState(TypedDict):
    current_agent_name: str

    # The single source of truth for what the LLM sees. Native LangChain
    # messages only: HumanMessage / AIMessage (thinking + signature + tool
    # calls intact) / ToolMessage. `add_messages` dedupes by id, so nodes
    # return the messages they produced and the channel accumulates them.
    # Compaction replaces the whole channel via RemoveMessage(REMOVE_ALL).
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # UI/persistence projection of `messages`, written once per turn at plan
    # time and updated in place on completion. Never sent to the LLM — it
    # exists because wall-clock timestamps and step ids are not recoverable
    # from messages alone.
    timeline: Annotated[List[Dict[str, Any]], merge_turns]
    step_counter: Annotated[int, keep_max_step_counter]

    events: Annotated[List[Dict[str, Any]], append_items]
    session_id: Optional[str]

    # Per-turn volatile tail, rebuilt by the observer every turn and appended
    # after the stable message prefix so the cache prefix stays byte-stable.
    observer_message_parts: Optional[List[HumanMessage]]

    input_artifact_manifest: Annotated[List[Dict[str, Any]], replace_artifact_manifest]
    current_delivery_manifest: Annotated[List[Dict[str, Any]], replace_artifact_manifest]
    approved_artifact_manifest: Annotated[List[Dict[str, Any]], merge_artifact_manifest]

    final_reply: Optional[str]
    agent_status: Optional[str]

    end_tag: bool
