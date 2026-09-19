from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from graphloom.model.state import AgentState
from graphloom.util.message_utils import last_user_text


def create_start_node():
    async def start_node(state: AgentState, config: RunnableConfig | None = None) -> Dict[str, Any]:
        """Reset per-run state and publish the opening user event.

        The request itself arrives as a HumanMessage in `messages`, appended by
        the caller like any other message — attachments are extra content parts
        on it. There is no separate `input_query` channel.
        """
        configurable = dict((config or {}).get("configurable") or {})
        updates: Dict[str, Any] = {
            "current_agent_name": state.get("current_agent_name") or "main",
            "final_reply": "",
            "agent_status": "running",
            "end_tag": False,
            "session_id": str(configurable.get("thread_id") or "default"),
            "observer_message_parts": None,
            "current_delivery_manifest": [],
        }
        request = last_user_text(state.get("messages", []))
        if request:
            updates["events"] = [{
                "id": f"user-{uuid4().hex}",
                "type": "message",
                "role": "user",
                "content": request,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }]
        return updates

    return start_node
