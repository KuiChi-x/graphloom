import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphBubbleUp

from graphloom.events import emit_step
from graphloom.model.state import AgentState
from graphloom.model.timeline import (
    THOUGHT_FIELDS,
    build_turn,
    step_id_for,
    turn_index_of,
    visible_args,
)
from graphloom.nodes.interrupt_guard import raise_if_cancelled
from graphloom.util.message_utils import get_last_ai_message, text_of

# 工具在返回值首行放这个标记自报成败。没有标记就算成功 —— 真正的失败走下面的
# except 分支或 report_outcome，不靠猜。
#
# 曾经这里是拿正则在正文里搜 "Error:|Traceback|FAILED"，但工具的返回体往往裹着
# 别人的内容：read_artifact 读一份写了 `except ImportError:` 的爬虫源码会被判红，
# 抓回来的网页正文提到 "Error:" 也会；反过来 run_shell 里 exit!=0 但输出没这些词
# （如 "can't open file"）又判成绿。裹别人内容的工具是多数，所以这个默认方向是错的。
_OUTCOME_PREFIX = "\x00graphloom-outcome:"


def report_outcome(text: str, *, failed: bool) -> str:
    """给工具用：在返回值里标上成败，别让框架去正文里猜。

    标记在首行、由框架摘掉，模型和前端都看不到它。只在工具自己知道失败了、但
    又不想抛异常时才需要（比如 run_shell 拿到非零退出码）。
    """
    return f"{_OUTCOME_PREFIX}{'error' if failed else 'ok'}\n{text}"


def _resolve_error(text: str) -> tuple[str, bool]:
    """摘掉自报标记，返回（正文, 是否出错）。没有标记就算成功。"""
    if not text.startswith(_OUTCOME_PREFIX):
        return text, False
    marker, _, rest = text.partition("\n")
    return rest, marker[len(_OUTCOME_PREFIX):].strip() == "error"


def _current_step_id(state: AgentState) -> str:
    """This turn's step id. `step_counter` is monotonic and survives
    compaction, so the id stays stable for the whole turn — ai_node already
    bumped it when it planned the tool calls."""
    return step_id_for(
        str(state.get("current_agent_name") or "main"),
        str(state.get("session_id") or "default"),
        int(state.get("step_counter") or 0),
    )


def _build_runtime_context(
        *,
        session_id: str,
        user_id: str,
        current_agent_name: str,
        host_context: Dict[str, Any],
) -> Dict[str, Any]:
    ctx = {
        "session_id": session_id,
        "user_id": user_id,
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


async def _emit_step_done(
        config: RunnableConfig,
        *,
        step_id: str,
        turn_index: int,
        agent_name: str,
        session_id: str,
        turn: Dict[str, Any],
) -> None:
    """Close out the UI turn block. No-op without an emitter."""
    await emit_step(config, "step_done", {
        "step_id": step_id,
        "step_index": turn_index,
        "agent_name": agent_name,
        "session_id": session_id,
        **{
            key: turn.get(key)
            for key in (
                "last_step_review", "working_notes", "next_action",
                "think", "content", "tool_calls", "completed_timestamp",
            )
        },
    })


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
        messages = list(state.get("messages", []) or [])
        last_ai_message = get_last_ai_message(messages)
        session_id = str(state.get("session_id") or "default")
        configurable = config.get("configurable", {})
        user_id = str(configurable.get("user_id") or session_id or "default")
        host_context = dict(configurable.get("runtime_context") or {})
        current_step_id = _current_step_id(state)
        current_agent_name = str(state.get("current_agent_name") or "main")

        if not last_ai_message:
            return {"end_tag": False}

        if not last_ai_message.tool_calls and last_ai_message.content:
            if allow_direct_reply:
                return {
                    "end_tag": True,
                    "final_reply": text_of(last_ai_message),
                }
            # Otherwise, force the agent to use tools explicitly (artifact-first mode).
            message = (
                "The assistant replied without any tool call. Use tools explicitly: "
                "call request_user_interaction when user review is needed, or call deliver_artifact to finish."
            )
            return {
                "end_tag": False,
                "messages": [HumanMessage(content="[Framework Reminder]\n" + message)],
            }

        tool_calls = list(last_ai_message.tool_calls or [])
        accumulated_state_patch: Dict[str, Any] = {}
        # Turn index must match ai_node's step_counter so a step's think, tool
        # chips, and result all land in one UI turn block.
        turn_index = turn_index_of(current_step_id)

        async def _run_tool_call(
                tool_call: Dict[str, Any],
                call_index: int,
        ) -> Dict[str, Any]:
            name = str(tool_call["name"])
            call_id = str(tool_call.get("id") or f"{current_step_id}:call:{call_index}")
            runtime_context = _build_runtime_context(
                session_id=session_id,
                user_id=user_id,
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
                f"agent:{current_agent_name}, step:{turn_index}, Executing tool {name} (session: {session_id}) with args:{args_json}")
            tool = tool_map.get(name)
            if not tool:
                return {
                    "call_id": call_id,
                    "tool_name": name,
                    "result": f"Tool not found: {name}",
                    "has_error": True,
                    "raw_result": None,
                }

            invoke_args = dict(raw_args)
            # Pass the graph config into the tool so builtins that need the
            # parent event_emitter / cancel_event (dispatch_subagents) receive it
            # by injection — never via a contextvar that async fan-out can drop.
            await emit_step(config, "tool_start", {
                "step_id": current_step_id,
                "call_id": call_id,
                "step_index": turn_index,
                "agent_name": current_agent_name,
                "session_id": session_id,
                "tool_name": name,
                "tool_args": visible_args(raw_args),
            })
            try:
                result = await tool.ainvoke(invoke_args, config=config)
                result_str, tool_failed = _resolve_error(_stringify_tool_result(result))
                await emit_step(config, "tool_end", {
                    "step_id": current_step_id,
                    "call_id": call_id,
                    "step_index": turn_index,
                    "agent_name": current_agent_name,
                    "session_id": session_id,
                    "tool_name": name,
                    "tool_args": visible_args(raw_args),
                    "result": result_str,
                    "has_error": tool_failed,
                })
                return {
                    "call_id": call_id,
                    "tool_name": name,
                    "result": result_str,
                    "has_error": tool_failed,
                    "raw_result": result_str if isinstance(result, str) else result,
                }
            except GraphBubbleUp:
                raise
            except Exception as exc:
                err_msg = f"Tool {name} execution failed: {exc}"
                logging.error(err_msg)
                await emit_step(config, "tool_end", {
                    "step_id": current_step_id,
                    "call_id": call_id,
                    "step_index": turn_index,
                    "agent_name": current_agent_name,
                    "session_id": session_id,
                    "tool_name": name,
                    "tool_args": visible_args(raw_args),
                    "result": err_msg,
                    "has_error": True,
                })
                return {
                    "call_id": call_id,
                    "tool_name": name,
                    "result": err_msg,
                    "has_error": True,
                    "raw_result": None,
                }

        tool_messages: List[ToolMessage] = []
        for index, tool_call in enumerate(tool_calls):
            item = await _run_tool_call(tool_call, index)
            logging.info(f"agent:{current_agent_name}, tool_call_end, result:{item['result']}")
            tool_messages.append(ToolMessage(
                content=str(item["result"]),
                tool_call_id=str(item["call_id"]),
                name=str(item["tool_name"]),
                status="error" if item["has_error"] else "success",
            ))
            if isinstance(item.get("raw_result"), dict):
                _merge_state_patch(accumulated_state_patch, dict(item["raw_result"]))

        updates: Dict[str, Any] = {
            "messages": tool_messages,
            # Same step_id as ai_node's plan-time write, so the reducer fills
            # results into that entry instead of appending a second turn.
            "timeline": [build_turn(
                step_id=current_step_id,
                ai_message=last_ai_message,
                tool_messages=tool_messages,
            )],
            "end_tag": False,
        }
        if accumulated_state_patch:
            _merge_state_patch(updates, accumulated_state_patch)
        await _emit_step_done(
            config,
            step_id=current_step_id,
            turn_index=turn_index,
            agent_name=current_agent_name,
            session_id=session_id,
            turn=updates["timeline"][0],
        )
        return updates

    return tool_node
