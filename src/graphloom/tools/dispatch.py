"""dispatch_subagents — optional builtin tool for multi-agent orchestration.

Groups run in ascending `group_id` order; same-group steps run in parallel.
The framework threads the parent graph's `checkpointer` into each sub-graph,
propagates parent configurable (event_emitter / cancel_event / runtime_context)
into children, and emits transport-agnostic lifecycle events via event_emitter:
  - subagent_state  (running | done | error | skipped)
  - subagent_reply  (final_reply text)

Hosts map those events to their own UI wire (WS codes, SSE, logs).
"""
import asyncio
import inspect
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from graphloom.config import SUBAGENT_MAX_CONCURRENCY
from graphloom.events import emit_step
from graphloom.model.artifact_manifest import merge_artifact_manifest
from graphloom.model.base_tool_input import PlannerThoughtInput
from graphloom.model.subagents import SubAgentRunContext, SubAgentSpec


def render_available_subagents(subagents: List[SubAgentSpec]) -> str:
    if not subagents:
        return ""

    lines = [
        "<subagent_runtime>",
        "- A `dispatch_subagents` tool is available in this graph because subagents are configured.",
        "- `dispatch_subagents` executes groups in ascending `group_id` order.",
        "- Steps with the same `group_id` run in parallel.",
        f"- HARD LIMIT: a single dispatch_subagents call may contain at most {SUBAGENT_MAX_CONCURRENCY} steps total "
        f"(summed across all group_ids). Calls that exceed this cap are rejected outright — the tool returns an error "
        f"and no step runs. If you have more than {SUBAGENT_MAX_CONCURRENCY} targets (e.g. 200 URLs), split them across "
        f"multiple dispatch_subagents calls of ≤{SUBAGENT_MAX_CONCURRENCY} steps each and issue them sequentially, "
        f"synthesizing results between batches.",
        "- Use the available subagents below as the only dispatch targets for this graph.",
        "</subagent_runtime>",
        "",
        "<available_subagents>",
    ]
    for item in subagents:
        lines.append(f"- agent_name: {item.agent_name}")
        lines.append(f"  description: {item.description}")
    lines.append("</available_subagents>")
    return "\n".join(lines)


def compose_system_prompt(base_prompt: str, subagents: List[SubAgentSpec]) -> str:
    subagent_block = render_available_subagents(subagents)
    if not subagent_block.strip():
        return base_prompt
    return base_prompt.rstrip() + "\n\n" + subagent_block


def _safe_suffix(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return normalized.strip("._") or "task"


def _build_child_session_id_from_parts(parent_session_id: str, agent_name: str, task_id: int | str) -> str:
    return f"{parent_session_id}_{_safe_suffix(agent_name)}_{_safe_suffix(str(task_id))}"


def _build_child_session_id(parent_session_id: str, spec: SubAgentSpec, step: "SubAgentTask") -> str:
    return _build_child_session_id_from_parts(parent_session_id, spec.agent_name, step.task_id)


class SubAgentTask(BaseModel):
    task_id: int = Field(
        description=(
            "Stable workstream identity. The framework keys all memory, "
            "session store, and checkpoints on (agent_name, task_id). "
            "Reuse the same task_id to continue or retry an existing "
            "workstream — including after an error; picking a new id "
            "discards its prior progress. Only mint a new id for "
            "genuinely new work."
        ),
    )
    title: str = Field(default="", description="Short internal task label.")
    agent_name: str = Field(description="Target subagent name.")
    instruction: str = Field(description="Self-contained instruction for the sub-agent. Include the concrete goal, what evidence to gather or produce, and what outcome should be delivered.")
    data_requirements: str = Field(default="", description="Specific data fields or entities to extract, if applicable.")
    constraints: str = Field(default="", description="Rules, limitations, or specific conditions to follow.")
    target_sites: List[str] = Field(default_factory=list, description="Sites or URLs this step is focused on. Keep this aligned with the user's visible targets whenever possible.")
    group_id: int = Field(default=0, description="Parallel group ID. Same group_id means steps may run in parallel; larger group_id means later dependent phases.")
    dependencies: List[int] = Field(default_factory=list, description="List of task_ids that must complete before this task can start.")
    expected_output: str = Field(default="", description="Clear description of the expected output or artifact from this task.")
    consumed_artifact_paths: List[str] = Field(
        default_factory=list,
        description="Absolute paths of upstream artifacts this task should consume. Leave empty to receive all upstream artifacts.",
    )
    timeout_seconds: int = Field(default=300, description="Maximum execution time allowed for this task in seconds.")
    max_retries: int = Field(default=1, description="Number of times to retry if the task fails.")
    status: str = Field(default="pending")


class DispatchSubagentsInput(PlannerThoughtInput):
    steps: List[SubAgentTask] = Field(description="Compact next-phase execution plan. Groups run in ascending order; same-group steps run in parallel. Do not use decorative micro-steps. Reuse the same task_id to continue or retry an existing sub-agent workstream — including after an error.")


def _render_input_manifest(manifest: List[Dict[str, Any]]) -> str:
    if not manifest:
        return "<input_artifact_manifest>\n[]\n</input_artifact_manifest>"
    return "<input_artifact_manifest>\n" + json.dumps(manifest, ensure_ascii=False, indent=2) + "\n</input_artifact_manifest>"


def _filter_manifest(manifest: List[Dict[str, Any]], consumed: List[str]) -> List[Dict[str, Any]]:
    if not consumed:
        return list(manifest)
    wanted = {os.path.abspath(p) for p in consumed if p and os.path.isabs(p)}
    return [item for item in manifest if os.path.abspath(str(item.get("path") or "")) in wanted]


def _render_target_sites(target_sites: List[str]) -> str:
    if not target_sites:
        return "<target_sites>\n[]\n</target_sites>"
    return "<target_sites>\n" + json.dumps(target_sites, ensure_ascii=False, indent=2) + "\n</target_sites>"


def _render_step_request(step: SubAgentTask) -> str:
    lines = [
        f"title: {step.title or f'Task {step.task_id}'}",
        f"task_id: {step.task_id}",
        f"instruction: {step.instruction}",
    ]
    if step.data_requirements:
        lines.append(f"data_requirements: {step.data_requirements}")
    if step.constraints:
        lines.append(f"constraints: {step.constraints}")
    if step.expected_output:
        lines.append(f"expected_output: {step.expected_output}")
    if step.target_sites:
        lines.append("target_sites: " + ", ".join(step.target_sites))
    return "\n".join(lines)


def _resolve_subagent(subagents: List[SubAgentSpec], step: SubAgentTask) -> SubAgentSpec:
    target_name = str(step.agent_name or "").strip()
    for item in subagents:
        if item.agent_name == target_name:
            return item
    known = ", ".join(item.agent_name for item in subagents)
    raise KeyError(f"Unknown plan step target `{target_name}`. Available subagents: {known}")


def _build_sub_state(
    *,
    session_id: str,
    step: SubAgentTask,
    spec: SubAgentSpec,
    input_artifact_manifest: List[Dict[str, Any]],
) -> Dict[str, Any]:
    child_session_id = _build_child_session_id(session_id, spec, step)
    step_request = _render_step_request(step)
    blocks = [
        f"<planned_step_request>\n{step_request}\n</planned_step_request>",
        f"<child_instruction>\n{step.instruction}\n</child_instruction>",
        f"<plan_step>\n{json.dumps(step.model_dump(), ensure_ascii=False, indent=2)}\n</plan_step>",
        _render_target_sites(list(step.target_sites or [])),
        _render_input_manifest(input_artifact_manifest),
    ]

    bootstrap_message = HumanMessage(content="\n\n".join(blocks))
    return {
        "messages": [bootstrap_message],
        "input_query": step_request,
        "session_id": child_session_id,
        "current_agent_name": spec.agent_name,
        "input_artifact_manifest": input_artifact_manifest,
    }



def _extract_final_reply(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    text = str(result.get("final_reply") or "").strip()
    if text:
        return text
    latest_ai = result.get("latest_ai_message")
    if latest_ai is None:
        return ""
    content = getattr(latest_ai, "content", "")
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    else:
        text = str(content or "").strip()
    return text


async def _run_cleanup_hook(spec: SubAgentSpec, run_context: SubAgentRunContext) -> None:
    if not spec.cleanup_hook:
        return

    cleanup_result = spec.cleanup_hook(run_context)
    if inspect.isawaitable(cleanup_result):
        await cleanup_result


async def _update_agent_status(agent: Any, config: Dict[str, Any], status: str) -> None:
    update_state = getattr(agent, "aupdate_state", None)
    if not update_state:
        return
    try:
        await update_state(config, {"agent_status": status})
    except Exception as exc:
        logging.debug("[dispatch_subagents] update agent_status=%s failed: %s", status, exc)


def create_dispatch_subagents_tool(subagents: List[SubAgentSpec], checkpointer: Any = None):
    @tool("dispatch_subagents", args_schema=DispatchSubagentsInput)
    async def dispatch_subagents(steps: List[Dict[str, Any]], config: RunnableConfig = None, **kwargs) -> Dict[str, Any]:
        """Dispatch a compact grouped next-phase plan to configured subagents. Use when delegation is genuinely useful; groups run sequentially and same-group steps run in parallel."""
        parsed_steps = [step if isinstance(step, SubAgentTask) else SubAgentTask(**step) for step in list(steps or [])]
        if not parsed_steps:
            return {"approved_artifact_manifest": []}

        # Hard cap — reject the whole call if the planner tried to schedule
        if len(parsed_steps) > SUBAGENT_MAX_CONCURRENCY:
            return {
                "dispatch_error": (
                    f"Too many steps in a single dispatch: {len(parsed_steps)} > "
                    f"max {SUBAGENT_MAX_CONCURRENCY}. Split the targets across "
                    f"multiple dispatch_subagents calls of ≤{SUBAGENT_MAX_CONCURRENCY} "
                    f"steps each and issue them sequentially."
                ),
                "approved_artifact_manifest": [],
            }

        session_id = str(kwargs.get("session_id") or "default")
        runtime_context = dict(kwargs.get("runtime_context") or {})
        user_id = str(runtime_context.get("user_id") or kwargs.get("user_id") or session_id or "default")

        # Parent graph config is injected by the tool runner (RunnableConfig),
        # not read from a contextvar — async fan-out (gather) can drop contextvars.
        parent_conf = dict((config or {}).get("configurable") or {})
        cancel_event = parent_conf.get("cancel_event") or runtime_context.get("cancel_event")
        parent_emitter = parent_conf.get("event_emitter")
        # Config object so emit_step finds the parent emitter for lifecycle events.
        parent_emit_config = {"configurable": parent_conf} if parent_conf else {}

        rolling_manifest = merge_artifact_manifest(
            list(kwargs.get("input_artifact_manifest", []) or []),
            list(kwargs.get("approved_artifact_manifest", []) or []),
        )
        grouped: Dict[int, List[SubAgentTask]] = defaultdict(list)
        for step in parsed_steps:
            grouped[int(step.group_id or 0)].append(step)

        dispatch_results: List[Dict[str, Any]] = []
        abort_remaining = False

        async def _emit_state(
            *,
            status: str,
            step: SubAgentTask,
            agent_name: str,
            group_id: int,
            child_session_id: str,
            error: str | None = None,
            artifacts: List[Dict[str, Any]] | None = None,
        ) -> None:
            data: Dict[str, Any] = {
                "status": status,
                "task_id": step.task_id,
                "workstream_id": str(step.task_id),
                "title": step.title or f"Task {step.task_id}",
                "task": step.instruction or "",
                "agent_name": agent_name,
                "group_id": group_id,
                "child_session_id": child_session_id,
                "parent_session_id": session_id,
                "session_id": child_session_id,
            }
            if error is not None:
                data["error"] = error
            if artifacts:
                data["artifacts"] = list(artifacts)
            await emit_step(parent_emit_config, "subagent_state", data)

        for group_id in sorted(grouped):
            group_steps = sorted(grouped[group_id], key=lambda item: int(item.task_id))

            if abort_remaining:
                for step in group_steps:
                    child_sid = _build_child_session_id_from_parts(
                        session_id, step.agent_name, step.task_id,
                    )
                    logging.warning(
                        "[dispatch_subagents] task_id=%s agent=%s SKIPPED: "
                        "upstream group failed",
                        step.task_id, step.agent_name,
                    )
                    await _emit_state(
                        status="skipped",
                        step=step,
                        agent_name=step.agent_name,
                        group_id=group_id,
                        child_session_id=child_sid,
                        error="Skipped: upstream group failed",
                    )
                    dispatch_results.append({
                        "task_id": step.task_id,
                        "workstream_id": str(step.task_id),
                        "group_id": group_id,
                        "agent_name": step.agent_name,
                        "child_session_id": child_sid,
                        "approved_artifact_manifest": [],
                        "error": "Skipped: upstream group failed",
                    })
                continue

            async def _run_step(step: SubAgentTask) -> Dict[str, Any]:
                spec = _resolve_subagent(subagents, step)
                # Thread the parent graph's checkpointer so sub-graphs persist state.
                try:
                    agent = spec.factory(checkpointer=checkpointer) if checkpointer is not None else spec.factory()
                except TypeError:
                    # Backward-compat: factory without checkpointer kwarg
                    agent = spec.factory()
                child_session_id = _build_child_session_id(session_id, spec, step)

                child_runtime_context = dict(runtime_context)
                child_runtime_context.update({
                    "user_id": user_id,
                    "session_id": child_session_id,
                    "current_agent_name": spec.agent_name,
                })
                # Child gets the same event_emitter as parent so think/tool events
                # stream; host emitters should seed/route by payload session_id.
                child_configurable: Dict[str, Any] = {
                    "thread_id": child_session_id,
                    "user_id": user_id,
                    "cancel_event": cancel_event,
                    "runtime_context": child_runtime_context,
                }
                if parent_emitter is not None:
                    child_configurable["event_emitter"] = parent_emitter
                # Pass through host-only handles if present (ws_handler, etc.).
                for key in ("ws_handler",):
                    if key in parent_conf:
                        child_configurable[key] = parent_conf[key]

                sub_config = {
                    "configurable": child_configurable,
                    "recursion_limit": 1000,
                }

                await _emit_state(
                    status="running",
                    step=step,
                    agent_name=spec.agent_name,
                    group_id=group_id,
                    child_session_id=child_session_id,
                )

                result = None
                error: BaseException | None = None
                try:
                    result = await agent.ainvoke(
                        _build_sub_state(
                            session_id=session_id,
                            step=step,
                            spec=spec,
                            input_artifact_manifest=_filter_manifest(
                                rolling_manifest, list(step.consumed_artifact_paths or [])
                            ),
                        ),
                        config=sub_config,
                        # Match host: async mid-run checkpoints for crash/HITL
                        # resume; host schedules keep_latest prune after the
                        # parent ainvoke returns so child intermediates are trimmed.
                        durability="async",
                    )
                except BaseException as exc:
                    error = exc
                    await _update_agent_status(agent, sub_config, "error")
                    await _emit_state(
                        status="error",
                        step=step,
                        agent_name=spec.agent_name,
                        group_id=group_id,
                        child_session_id=child_session_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    raise
                finally:
                    await _run_cleanup_hook(
                        spec,
                        SubAgentRunContext(
                            parent_session_id=session_id,
                            child_session_id=child_session_id,
                            step=step,
                            result=result,
                            error=error,
                            user_id=user_id,
                        ),
                    )
                approved_manifest = list(result.get("approved_artifact_manifest", []) or [])
                final_reply_text = _extract_final_reply(result)
                # Emit done first (carries this child's artifact manifest) so hosts
                # can render the sub-agent's artifact area, then the reply text.
                await _emit_state(
                    status="done",
                    step=step,
                    agent_name=spec.agent_name,
                    group_id=group_id,
                    child_session_id=child_session_id,
                    artifacts=approved_manifest,
                )
                if final_reply_text:
                    await emit_step(parent_emit_config, "subagent_reply", {
                        "task_id": step.task_id,
                        "workstream_id": str(step.task_id),
                        "agent_name": spec.agent_name,
                        "group_id": group_id,
                        "child_session_id": child_session_id,
                        "parent_session_id": session_id,
                        "session_id": child_session_id,
                        "text": final_reply_text,
                        "artifacts": approved_manifest,
                    })
                return {
                    "task_id": step.task_id,
                    "workstream_id": str(step.task_id),
                    "group_id": group_id,
                    "agent_name": spec.agent_name,
                    "child_session_id": child_session_id,
                    "approved_artifact_manifest": approved_manifest,
                    "final_reply": final_reply_text,
                }

            group_results = await asyncio.gather(
                *[_run_step(step) for step in group_steps],
                return_exceptions=True,
            )
            for step, item in zip(group_steps, group_results):
                if isinstance(item, BaseException):
                    abort_remaining = True
                    logging.error(
                        f"[dispatch_subagents] task_id={step.task_id} agent={step.agent_name} "
                        f"group={group_id} failed: {item!r}"
                    )
                    dispatch_results.append({
                        "task_id": step.task_id,
                        "workstream_id": str(step.task_id),
                        "group_id": group_id,
                        "agent_name": step.agent_name,
                        "child_session_id": _build_child_session_id_from_parts(session_id, step.agent_name, step.task_id),
                        "approved_artifact_manifest": [],
                        "error": f"{type(item).__name__}: {item}",
                    })
                    continue
                rolling_manifest = merge_artifact_manifest(
                    rolling_manifest,
                    list(item.get("approved_artifact_manifest", []) or []),
                )
                dispatch_results.append(item)

        return {
            "approved_artifact_manifest": rolling_manifest,
            "dispatch_results": dispatch_results,
        }

    return dispatch_subagents
