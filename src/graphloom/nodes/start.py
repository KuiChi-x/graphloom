from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from graphloom.model.state import AgentState


def create_start_node():
    async def start_node(state: AgentState, config: RunnableConfig | None = None) -> Dict[str, Any]:
        query = str(state.get("input_query") or "").strip()
        configurable = dict((config or {}).get("configurable") or {})
        session_id = str(configurable.get("thread_id") or "default")
        return {
            "current_agent_name": state.get("current_agent_name") or "main",
            "final_reply": "",
            "agent_status": "running",
            "end_tag": False,
            "latest_ai_message": None,
            "session_id": session_id,
            "tool_result_history": [],
            "observer_message_parts": None,
            "current_delivery_manifest": [],
            "conversation": [HumanMessage(content=query)] if query else [],
            "events": [{
                "id": f"user-{uuid4().hex}",
                "type": "message",
                "role": "user",
                "content": query,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }] if query else [],
        }

    return start_node
