"""graph_builder.py — the generic agent-loop factory.

`build_agent_graph` assembles a standard ReAct-style loop:
    observer? → ai → tool → route → history → compaction → ai  (+ finish)

Everything transport- or business-specific (HITL, subagent dispatch, artifact
delivery over a wire) is a tool the caller supplies; the framework wires only
the loop and the structural nodes.
"""
from typing import Any, Callable, List, Optional

from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph

from graphloom.model.state import AgentState
from graphloom.model.subagents import SubAgentSpec
from graphloom.nodes.ai import create_ai_node
from graphloom.nodes.compaction import create_context_compaction_node
from graphloom.nodes.finish import create_finish_node
from graphloom.nodes.history import create_history_node
from graphloom.nodes.tool import create_tool_node
from graphloom.prompt.stack import create_prompt_stack
from graphloom.tools.artifact import (
    deliver_artifact,
    patch_artifact,
    read_artifact,
    write_artifact,
)

_BUILTIN_ARTIFACT_TOOLS = [write_artifact, read_artifact, patch_artifact, deliver_artifact]


def build_agent_graph(
    *,
    custom_system_prompt: str,
    tools: List[Any],
    llm: ChatLiteLLM,
    find_fault: Optional[Any] = None,
    custom_find_fault: Optional[Callable] = None,
    observer: Optional[Callable] = None,
    subagents: Optional[List[SubAgentSpec]] = None,
    checkpointer=None,
    tool_filter: Optional[Callable] = None,
    allow_direct_reply: bool = False,
    available_skills: Optional[List[str]] = None,
    skills_dir: Optional[str] = None,
):
    # Deduplicate user tools while preserving the FIRST occurrence (highest priority)
    unique_tools = []
    seen_names = set()
    for t in tools:
        t_name = getattr(t, "name", None)
        if t_name not in seen_names:
            unique_tools.append(t)
            if t_name:
                seen_names.add(t_name)

    # Auto-inject builtin artifact tools (deduplicated)
    builtin = [t for t in _BUILTIN_ARTIFACT_TOOLS if t.name not in seen_names]
    compiled_tools = unique_tools + builtin
    compiled_prompt = str(custom_system_prompt or "").strip()
    if not compiled_prompt:
        raise ValueError("custom_system_prompt must be provided.")
    compiled_subagents = list(subagents or [])
    if compiled_subagents:
        # Lazy import: dispatch_subagents is an optional builtin, only needed
        # when subagents are actually wired in.
        from graphloom.tools.dispatch import compose_system_prompt, create_dispatch_subagents_tool
        compiled_prompt = compose_system_prompt(compiled_prompt, compiled_subagents)
        compiled_tools.append(create_dispatch_subagents_tool(compiled_subagents, checkpointer))

    # Lazy import: find_fault is an optional review node.
    find_fault_node = None
    if isinstance(find_fault, str):
        from graphloom.nodes.find_fault import create_find_fault_node
        find_fault_node = create_find_fault_node(find_fault, llm)
    elif find_fault is not None:
        find_fault_node = find_fault

    prompt_stack = create_prompt_stack(
        custom_system_prompt=compiled_prompt,
        available_skills=available_skills,
        skills_dir=skills_dir,
    )
    ai_node = create_ai_node(prompt_stack=prompt_stack, tools=compiled_tools, llm=llm, tool_filter=tool_filter)
    workflow = StateGraph(AgentState)

    if observer:
        workflow.add_node("observer", observer)
    workflow.add_node("ai", ai_node)
    workflow.add_node("tool", create_tool_node(compiled_tools, allow_direct_reply=allow_direct_reply))
    if custom_find_fault:
        workflow.add_node("custom_find_fault", custom_find_fault)
    if find_fault_node:
        workflow.add_node("find_fault", find_fault_node)
    workflow.add_node("history", create_history_node())
    workflow.add_node("compaction", create_context_compaction_node(prompt_stack, llm))
    workflow.add_node("finish", create_finish_node())

    workflow.set_entry_point("observer" if observer else "ai")
    if observer:
        workflow.add_edge("observer", "ai")

    def route_after_tool(state: AgentState) -> str:
        # Once deliver_artifact has marked the run as ended (end_tag=True), we
        # always want to exit the loop — even if the manifest is empty (e.g.
        # pure-text answer with no artifacts). Find-fault nodes are still
        # consulted when a custom reviewer is wired in; empty manifests are
        # treated as a no-op by those reviewers.
        if state.get("end_tag"):
            # Pure-text direct replies have no delivery manifest and need no
            if not list(state.get("current_delivery_manifest", []) or []):
                return "finish"
            if custom_find_fault:
                return "custom_find_fault"
            if find_fault_node:
                return "find_fault"
            return "finish"
        return "history"

    def route_after_custom_find_fault(state: AgentState) -> str:
        if list(state.get("current_delivery_manifest", []) or []):
            if find_fault_node:
                return "find_fault"
            return "finish"
        return "history"

    def route_after_find_fault(state: AgentState) -> str:
        if list(state.get("current_delivery_manifest", []) or []):
            return "finish"
        return "history"

    workflow.add_edge("ai", "tool")

    tool_targets = {"history": "history", "finish": "finish"}
    if custom_find_fault:
        tool_targets["custom_find_fault"] = "custom_find_fault"
    if find_fault_node:
        tool_targets["find_fault"] = "find_fault"
    workflow.add_conditional_edges("tool", route_after_tool, tool_targets)

    if custom_find_fault:
        custom_targets = {"history": "history", "finish": "finish"}
        if find_fault_node:
            custom_targets["find_fault"] = "find_fault"
        workflow.add_conditional_edges("custom_find_fault", route_after_custom_find_fault, custom_targets)

    if find_fault_node:
        workflow.add_conditional_edges("find_fault", route_after_find_fault, {"history": "history", "finish": "finish"})

    workflow.add_edge("history", "compaction")
    workflow.add_edge("compaction", "observer" if observer else "ai")
    workflow.add_edge("finish", END)

    return workflow.compile(checkpointer=checkpointer)
