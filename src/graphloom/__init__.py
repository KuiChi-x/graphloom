"""graphloom — a minimal generic agent-loop framework on top of LangGraph.

`build_agent_graph` assembles a standard ReAct loop
(ai / tool / history / compaction / finish) with dependency-injected
llm, checkpointer, tools, and runtime_context. Everything transport- or
business-specific (HITL, subagent dispatch, artifact delivery over a wire)
is a tool the caller supplies; the framework wires only the loop.

Quick start::

    from graphloom import build_agent_graph, build_initial_agent_state
    graph = build_agent_graph(custom_system_prompt=..., tools=[...], llm=...)
    await graph.ainvoke(build_initial_agent_state(input_query=...))
"""
from graphloom.graph_builder import build_agent_graph
from graphloom.model.state import AgentState, build_initial_agent_state
from graphloom.model.subagents import SubAgentSpec, SubAgentRunContext
from graphloom.model.base_tool_input import StandardThoughtInput, PlannerThoughtInput

__all__ = [
    "build_agent_graph",
    "AgentState",
    "build_initial_agent_state",
    "SubAgentSpec",
    "SubAgentRunContext",
    "StandardThoughtInput",
    "PlannerThoughtInput",
]

__version__ = "0.1.0"
