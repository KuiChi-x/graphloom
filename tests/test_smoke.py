"""Smoke test: build_agent_graph compiles and runs a minimal loop."""
import asyncio
from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM

from graphloom import build_agent_graph, build_initial_agent_state


class _FakeLLM:
    """Minimal BaseChatModel stand-in: bind_tools/bind return self, astream
    yields one chunk with content and no tool calls."""

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    async def astream(self, messages, config=None):
        yield AIMessageChunk(content="done")


@tool
def _dummy(query: str) -> str:
    """A dummy tool."""
    return "ok"


async def test_build_and_run_minimal():
    graph = build_agent_graph(
        custom_system_prompt="You are a test agent.",
        tools=[_dummy],
        llm=cast(ChatLiteLLM, _FakeLLM()),
        allow_direct_reply=True,
    )
    result = await graph.ainvoke(
        build_initial_agent_state(input_query="hi", session_id="s1")
    )
    assert result["agent_status"] == "done"
    assert result["end_tag"] is True
    assert result["final_reply"] == "done"

if __name__ == '__main__':
    load_dotenv()
    asyncio.run(test_build_and_run_minimal())