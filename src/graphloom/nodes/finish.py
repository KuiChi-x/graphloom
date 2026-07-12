from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from langchain_core.messages import AIMessage

from graphloom.model.state import AgentState


def create_finish_node():
    async def finish_node(state: AgentState) -> Dict[str, Any]:
        approved_manifest = list(state.get("current_delivery_manifest", []) or [])
        if not approved_manifest:
            approved_manifest = list(state.get("approved_artifact_manifest", []) or [])

        updates: Dict[str, Any] = {
            "end_tag": True,
            "agent_status": "done",
            "approved_artifact_manifest": approved_manifest,
        }
        final_reply = str(state.get("final_reply") or "").strip()
        if final_reply:
            assistant_id = f"assistant-{uuid4().hex}"
            updates["conversation"] = [AIMessage(id=assistant_id, content=final_reply)]
            updates["events"] = [{
                "id": assistant_id,
                "type": "message",
                "role": "assistant",
                "content": final_reply,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }]
        return updates

    return finish_node
