"""Smoke test: build_agent_graph compiles and runs a minimal loop."""
import asyncio
from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeCancelledError

from graphloom import build_agent_graph


class _FakeLLM:
    """Minimal BaseChatModel stand-in: bind_tools/bind return self, astream
    yields one chunk with content and no tool calls."""

    def __init__(self):
        self.calls = []

    def bind_tools(self, tools):
        return self

    def bind(self, **kwargs):
        return self

    async def astream(self, messages, config=None):
        self.calls.append(list(messages))
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
        {"input_query": "hi"}
    )
    assert result["agent_status"] == "done"
    assert result["end_tag"] is True
    assert result["final_reply"] == "done"


async def test_follow_up_request_uses_checkpoint_conversation_history():
    llm = _FakeLLM()
    checkpointer = InMemorySaver()
    graph = build_agent_graph(
        custom_system_prompt="You are a test agent.",
        tools=[_dummy],
        llm=cast(ChatLiteLLM, llm),
        allow_direct_reply=True,
        checkpointer=checkpointer,
    )
    await graph.ainvoke(
        {"input_query": "Only use GET requests"},
    )
    await graph.ainvoke(
        {"input_query": "Continue with authentication"},
    )

    second_prompt = "\n".join(str(message.content) for message in llm.calls[-1])
    assert "Only use GET requests" in second_prompt
    assert "Continue with authentication" in second_prompt

    snapshot = await graph.aget_state({"configurable": {"thread_id": "default", "checkpoint_ns": ""}})
    conversation = list(snapshot.values["conversation"])
    assert [message.type for message in conversation] == [
        "human",
        "ai",
        "human",
        "ai",
    ]
    assert [message.content for message in conversation] == [
        "Only use GET requests",
        "done",
        "Continue with authentication",
        "done",
    ]
    assert [event["type"] for event in snapshot.values["events"]] == [
        "message",
        "message",
        "message",
        "message",
    ]
    assert [event["role"] for event in snapshot.values["events"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]




async def test_cancelled_turn_keeps_user_message_for_next_request():
    class CancelOnceLLM(_FakeLLM):
        async def astream(self, messages, config=None):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                raise asyncio.CancelledError
            yield AIMessageChunk(content="done")

    llm = CancelOnceLLM()
    checkpointer = InMemorySaver()
    graph = build_agent_graph(
        custom_system_prompt="You are a test agent.",
        tools=[_dummy],
        llm=cast(ChatLiteLLM, llm),
        allow_direct_reply=True,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "cancelled-turn", "checkpoint_ns": ""}}

    try:
        await graph.ainvoke({"input_query": "Remember this before cancellation"}, config=config)
    except NodeCancelledError:
        pass

    snapshot = await graph.aget_state(config)
    assert [message.content for message in snapshot.values["conversation"]] == [
        "Remember this before cancellation"
    ]

    await graph.ainvoke({"input_query": "Continue now"}, config=config)
    second_prompt = "\n".join(str(message.content) for message in llm.calls[-1])
    assert second_prompt.count("Remember this before cancellation") == 1
    assert second_prompt.count("Continue now") == 1

if __name__ == '__main__':
    load_dotenv()
    asyncio.run(test_build_and_run_minimal())
