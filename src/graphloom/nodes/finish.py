from typing import Any, Dict

from graphloom.model.state import AgentState


def create_finish_node():
    async def finish_node(state: AgentState) -> Dict[str, Any]:
        approved_manifest = list(state.get("current_delivery_manifest", []) or [])
        if not approved_manifest:
            approved_manifest = list(state.get("approved_artifact_manifest", []) or [])

        return {
            "end_tag": True,
            "agent_status": "done",
            "approved_artifact_manifest": approved_manifest,
        }

    return finish_node
