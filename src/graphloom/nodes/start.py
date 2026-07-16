from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from langchain_core.messages import HumanMessage

from graphloom.model.state import AgentState


def create_start_node():
    async def start_node(state: AgentState) -> Dict[str, Any]:
        query = str(state.get("input_query") or "").strip()
        return {
            "current_agent_name": state.get("current_agent_name") or "main",
            "final_reply": "",
            "agent_status": "running",
            "end_tag": False,
            "latest_ai_message": None,
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
