"""graphloom — a minimal generic agent-loop framework on top of LangGraph.

`build_agent_graph` assembles a standard ReAct loop
(ai / tool / compaction / finish) with dependency-injected llm, checkpointer,
tools, and runtime_context. Everything transport- or business-specific (HITL,
subagent dispatch, artifact delivery over a wire) is a tool the caller
supplies; the framework wires only the loop.

State is native LangChain messages end to end: the model sees its own prior
`AIMessage`s — reasoning blocks, signatures and tool calls intact — rather than
a third-person paraphrase of them, so it continues its earlier reasoning
instead of rebuilding it every turn.

Quick start::

    from graphloom import build_agent_graph, BaseEventEmitter
    from langchain_core.messages import HumanMessage

    class Printer(BaseEventEmitter):
        async def on_ai_delta(self, payload):
            print(payload.get("content") or "", end="")

    graph = build_agent_graph(custom_system_prompt=..., tools=[...], llm=...)
    await graph.ainvoke(
        {"messages": [HumanMessage(content="...")]},
        config={"configurable": {"event_emitter": Printer()}},
    )
"""
from graphloom.events import BaseEventEmitter
from graphloom.graph_builder import build_agent_graph
from graphloom.model.base_tool_input import PlannerThoughtInput, StandardThoughtInput
from graphloom.model.state import AgentState
from graphloom.model.subagents import SubAgentRunContext, SubAgentSpec
from graphloom.nodes.tool import report_outcome

from graphloom.nodes.find_fault import review_verdict

__all__ = [
    "build_agent_graph",
    "report_outcome",
    "review_verdict",
    "AgentState",
    "SubAgentSpec",
    "SubAgentRunContext",
    "StandardThoughtInput",
    "PlannerThoughtInput",
    "BaseEventEmitter",
]

__version__ = "0.2.0"
