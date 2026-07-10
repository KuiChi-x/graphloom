import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from graphloom.nodes.history import THOUGHT_FIELDS, _filter_thought_args
from graphloom.nodes.interrupt_guard import raise_if_cancelled
from graphloom.events import emit_step
from graphloom.util.message_utils import get_last_ai_message
from graphloom.model.state import AgentState

_ERROR_RE = re.compile(r"Traceback \(most recent call last\)|Error:|Exception:|FAILED", re.IGNORECASE)


def _contains_error(text: str) -> bool:
    return bool(_ERROR_RE.search(text))


def _make_tool_history_entry(step: int, tool_name: str, tool_args: Dict[str, Any], result: str, has_error: bool) -> \
        Dict[str, Any]:
    return {
        "step": step,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "result": result,
        "has_error": has_error,
    }


def _current_step_id(state: AgentState) -> str:
    for step in reversed(list(state.get("past_steps", []) or [])):
        if isinstance(step, dict) and step.get("status") == "pending_tool" and step.get("step_id"):
            return str(step.get("step_id"))
    current_agent_name = str(state.get("current_agent_name") or "main")
    session_id = str(state.get("session_id") or "default")
    return f"{current_agent_name}:{session_id}:step:{max(0, len(state.get('past_steps', []) or []) - 1)}"


def _current_step_number(state: AgentState) -> int:
    past_steps = list(state.get("past_steps", []) or [])
    for index in range(len(past_steps) - 1, -1, -1):
        step = past_steps[index]
        if isinstance(step, dict) and step.get("status") == "pending_tool":
            return index
    return len(past_steps)


def _build_runtime_context(
        *,
        session_id: str,
        user_id: str,
        enforce_approved_subset: bool,
        current_agent_name: str,
        host_context: Dict[str, Any],
) -> Dict[str, Any]:
    ctx = {
        "session_id": session_id,
        "user_id": user_id,
        "enforce_approved_subset": enforce_approved_subset,
        "current_agent_name": current_agent_name,
    }
    # Host injects whatever its tools need (ws_handler, cancel_event, …) via
    # configurable["runtime_context"]; the framework stays agnostic to it.
    ctx.update(host_context or {})
    return ctx


def _inject_hidden_args(
        state: AgentState,
        raw_args: Dict[str, Any],
        tool_name: str,
        session_id: str,
        runtime_context: Dict[str, Any],
) -> Dict[str, Any]:
    args = dict(raw_args)
    args["session_id"] = session_id
    args["runtime_context"] = dict(runtime_context)
    for field in THOUGHT_FIELDS - {"session_id", "runtime_context"}:
        args.setdefault(field, "")
    if tool_name == "dispatch_subagents":
        args.setdefault("input_query", state.get("input_query", ""))
        args.setdefault("input_artifact_manifest", list(state.get("input_artifact_manifest", []) or []))
        args.setdefault("approved_artifact_manifest", list(state.get("approved_artifact_manifest", []) or []))
    elif tool_name == "deliver_artifact":
        args.setdefault("approved_artifact_manifest", list(state.get("approved_artifact_manifest", []) or []))
    return args


_HARD_TRUNCATE_LIMIT = 50000


def _stringify_tool_result(result: Any) -> str:
    """Convert a tool result to string, hard-truncating at _HARD_TRUNCATE_LIMIT chars."""
    try:
        if isinstance(result, (dict, list)):
            text = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            text = str(result)
    except Exception:
        text = str(result)

    if len(text) > _HARD_TRUNCATE_LIMIT:
        text = text[:_HARD_TRUNCATE_LIMIT] + f"\n... [truncated, original length: {len(text)} chars]"
    return text


def _merge_state_patch(target: Dict[str, Any], patch: Dict[str, Any]) -> None:
    if not patch:
        return
    merge_keys = {"approved_artifact_manifest", "messages"}
    for key, value in patch.items():
        if key in merge_keys and isinstance(value, list):
            existing = list(target.get(key, []) or [])
            target[key] = existing + list(value)
            continue
        target[key] = value


def create_tool_node(tools: List[Any], allow_direct_reply: bool = False):
    tool_map = {tool.name: tool for tool in tools}

    async def tool_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        raise_if_cancelled(config)
        last_ai_message = state.get("latest_ai_message") or get_last_ai_message(list(state.get("messages", []) or []))
        session_id = str(state.get("session_id") or "default")
        configurable = config.get("configurable", {})
        user_id = str(configurable.get("user_id") or session_id or "default")
        host_context = dict(configurable.get("runtime_context") or {})
        current_step = _current_step_number(state)
        current_step_id = _current_step_id(state)
        current_agent_name = str(state.get("current_agent_name") or "main")

        if not last_ai_message:
            return {"end_tag": False}

        if (not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls) and last_ai_message.content:
            if allow_direct_reply:
                content = last_ai_message.content
                if isinstance(content, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    ).strip()
                else:
                    text = str(content or "").strip()
                return {
                    "end_tag": True,
                    "final_reply": text,
                }
            # Otherwise, force the agent to use tools explicitly (artifact-first mode).
            message = (
                "The assistant replied without any tool call. Use tools explicitly: "
                "call request_user_interaction when user review is needed, or call deliver_artifact to finish."
            )
            return {
                "end_tag": False,
                "tool_result_history": [
                    _make_tool_history_entry(current_step, "tool_node_guard", {}, message, True)
                ],
                "messages": [HumanMessage(content="[Framework Reminder]\n" + message)],
            }

        tool_calls = list(last_ai_message.tool_calls or [])
        tool_map_has_subagents = "dispatch_subagents" in tool_map

        new_history_entries: List[Dict[str, Any]] = []
        accumulated_state_patch: Dict[str, Any] = {}

        async def _run_tool_call(tool_call: Dict[str, Any], step_number: int) -> Dict[str, Any]:
            name = str(tool_call["name"])
            runtime_context = _build_runtime_context(
                session_id=session_id,
                user_id=user_id,
                enforce_approved_subset=(name == "deliver_artifact" and tool_map_has_subagents),
                current_agent_name=current_agent_name,
                host_context=host_context,
            )
            raw_args = _inject_hidden_args(
                state,
                dict(tool_call["args"]),
                name,
                session_id,
                runtime_context,
            )
            args_json = json.dumps(tool_call["args"], ensure_ascii=False, indent=2)
            logging.info(
                f"agent:{current_agent_name}, step_number:{step_number}, Executing tool {name} (session: {session_id}) with args:{args_json}")
            tool = tool_map.get(name)
            if not tool:
                return {
                    "step": step_number,
                    "step_id": current_step_id,
                    "tool_name": name,
                    "tool_args": raw_args,
                    "result": f"Tool not found: {name}",
                    "has_error": True,
                    "raw_result": None,
                }

            invoke_args = dict(raw_args)
            await emit_step(config, "tool_start", {
                "step_id": current_step_id,
                "step_index": step_number,
                "agent_name": current_agent_name,
                "session_id": session_id,
                "tool_name": name,
                "tool_args": _filter_thought_args(raw_args),
            })
            try:
                result = await tool.ainvoke(invoke_args)
                result_str = _stringify_tool_result(result)
                await emit_step(config, "tool_end", {
                    "step_id": current_step_id,
                    "step_index": step_number,
                    "agent_name": current_agent_name,
                    "session_id": session_id,
                    "tool_name": name,
                    "tool_args": _filter_thought_args(raw_args),
                    "result": result_str,
                    "has_error": _contains_error(result_str),
                })
                return {
                    "step": step_number,
                    "step_id": current_step_id,
                    "tool_name": name,
                    "tool_args": raw_args,
                    "result": result_str,
                    "has_error": _contains_error(result_str),
                    "raw_result": result,
                }
            except Exception as exc:
                err_msg = f"Tool {name} execution failed: {exc}"
                logging.error(err_msg)
                await emit_step(config, "tool_end", {
                    "step_id": current_step_id,
                    "step_index": step_number,
                    "agent_name": current_agent_name,
                    "session_id": session_id,
                    "tool_name": name,
                    "tool_args": _filter_thought_args(raw_args),
                    "result": err_msg,
                    "has_error": True,
                })
                return {
                    "step": step_number,
                    "step_id": current_step_id,
                    "tool_name": name,
                    "tool_args": raw_args,
                    "result": err_msg,
                    "has_error": True,
                    "raw_result": None,
                }

        if tool_calls:
            execution_results = []
            for index, tool_call in enumerate(tool_calls):
                result = await _run_tool_call(tool_call, current_step + index)
                execution_results.append(result)

            for item in execution_results:
                entry = _make_tool_history_entry(
                    int(item["step"]),
                    str(item["tool_name"]),
                    dict(item["tool_args"]),
                    str(item["result"]),
                    bool(item["has_error"]),
                )
                entry["step_id"] = str(item.get("step_id") or current_step_id)
                new_history_entries.append(entry)

                result = item["result"]
                logging.info(f"agent:{current_agent_name}, tool_call_end, result:{result}")
                if isinstance(item.get("raw_result"), dict):
                    _merge_state_patch(accumulated_state_patch, dict(item["raw_result"]))
            current_step += len(execution_results)

        updates: Dict[str, Any] = {
            "tool_result_history": new_history_entries,
            "end_tag": False,
        }
        if accumulated_state_patch:
            _merge_state_patch(updates, accumulated_state_patch)
        return updates

    return tool_node
