"""Smoke test: build_agent_graph compiles and runs a minimal loop."""
import asyncio
from typing import cast

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeCancelledError

from graphloom import build_agent_graph
from graphloom.nodes.ai import _astream_with_retry
from graphloom.nodes.find_fault import GenericFindFaultOutput, create_find_fault_node
from graphloom.nodes.tool import create_tool_node
from graphloom.util.message_utils import text_of


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
        llm=cast(BaseChatModel, _FakeLLM()),
        allow_direct_reply=True,
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]}
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
        llm=cast(BaseChatModel, llm),
        allow_direct_reply=True,
        checkpointer=checkpointer,
    )
    await graph.ainvoke(
        {"messages": [HumanMessage(content="Only use GET requests")]},
    )
    await graph.ainvoke(
        {"messages": [HumanMessage(content="Continue with authentication")]},
    )

    second_prompt = "\n".join(str(message.content) for message in llm.calls[-1])
    assert "Only use GET requests" in second_prompt
    assert "Continue with authentication" in second_prompt

    snapshot = await graph.aget_state({"configurable": {"thread_id": "default", "checkpoint_ns": ""}})
    conversation = list(snapshot.values["messages"])
    assert [message.type for message in conversation] == [
        "human",
        "ai",
        "human",
        "ai",
    ]
    assert [text_of(message) for message in conversation] == [
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


async def test_streams_standard_reasoning_content_blocks():
    cases = [
        AIMessageChunk(
            content=[{
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "OpenAI thought"}],
            }],
            response_metadata={"model_provider": "openai"},
        ),
        AIMessageChunk(
            content=[{"type": "thinking", "thinking": "Anthropic thought"}],
            response_metadata={"model_provider": "anthropic"},
        ),
    ]

    for chunk, expected in zip(cases, ("OpenAI thought", "Anthropic thought")):
        events = []

        class ChunkLLM:
            async def astream(self, _messages, config=None):
                yield chunk

        async def emit(event_type, payload):
            events.append((event_type, payload))

        _, reasoning = await _astream_with_retry(
            ChunkLLM(), [], {"configurable": {"event_emitter": emit}}
        )

        assert reasoning == expected
        assert len(events) == 1
        event_type, payload = events[0]
        assert event_type == "ai_delta"
        assert (payload["content"], payload["reasoning"]) == ("", expected)


async def test_same_step_same_tool_calls_keep_distinct_ids():
    events = []

    class Emitter:
        async def __call__(self, event_type, payload):
            events.append((event_type, payload))

    @tool
    def echo(value: str) -> str:
        """Return the supplied value."""
        return value

    node = create_tool_node([echo])
    state = {
        "current_agent_name": "browser",
        "session_id": "s1",
        # ai_node stamped this turn as step 1; tool_node must reuse that id.
        "step_counter": 1,
        "timeline": [{
            "step_id": "browser:s1:step:1",
            "status": "pending_tool",
        }],
        "messages": [AIMessage(content="", id="ai-1", tool_calls=[
            {"name": "echo", "args": {"value": "first"}, "id": "call-first", "type": "tool_call"},
            {"name": "echo", "args": {"value": "second"}, "id": "call-second", "type": "tool_call"},
        ])],
    }

    result = await node(state, {"configurable": {"event_emitter": Emitter()}})

    starts = [payload for event, payload in events if event == "tool_start"]
    ends = [payload for event, payload in events if event == "tool_end"]
    assert [payload["call_id"] for payload in starts] == ["call-first", "call-second"]
    assert [payload["call_id"] for payload in ends] == ["call-first", "call-second"]
    assert [m.tool_call_id for m in result["messages"]] == ["call-first", "call-second"]
    assert [m.content for m in result["messages"]] == ["first", "second"]
    assert all(isinstance(m, ToolMessage) for m in result["messages"])
    # One timeline entry, keyed to the planning write so it updates in place.
    assert [entry["step_id"] for entry in result["timeline"]] == ["browser:s1:step:1"]


async def test_find_fault_uses_function_calling_structured_output(tmp_path):
    artifact = tmp_path / "result.txt"
    artifact.write_text("hello", encoding="utf-8")
    manifest = [{"path": str(artifact), "type": "text"}]

    class StructuredLLM:
        def with_structured_output(self, schema, **kwargs):
            assert schema is GenericFindFaultOutput
            assert kwargs == {"method": "function_calling"}
            return self

        async def ainvoke(self, messages):
            roles = [message.type for message in messages]
            assert roles[0] == "system"
            assert "system" not in roles[1:]
            # The reviewer replays the native run, not a digest of it.
            assert messages[1].content == "Deliver hello."
            return GenericFindFaultOutput(
                is_acceptable=True,
                decisive_assessment="accepted",
                confidence=1.0,
            )

    node = create_find_fault_node("Review the artifact.", StructuredLLM())
    result = await node({
        "current_delivery_manifest": manifest,
        "input_artifact_manifest": [],

        "observer_message_parts": [],
        "messages": [HumanMessage(content="Deliver hello.")],
        "session_id": "structured-output-test",
    })

    assert result["current_delivery_manifest"] == manifest
    # Accepted: nothing is fed back into the loop.
    assert result["messages"] == []




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
        llm=cast(BaseChatModel, llm),
        allow_direct_reply=True,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "cancelled-turn", "checkpoint_ns": ""}}

    try:
        await graph.ainvoke({"messages": [HumanMessage(content="Remember this before cancellation")]}, config=config)
    except NodeCancelledError:
        pass

    snapshot = await graph.aget_state(config)
    assert [text_of(m) for m in snapshot.values["messages"]] == [
        "Remember this before cancellation"
    ]

    await graph.ainvoke({"messages": [HumanMessage(content="Continue now")]}, config=config)
    second_prompt = "\n".join(str(message.content) for message in llm.calls[-1])
    assert second_prompt.count("Remember this before cancellation") == 1
    assert second_prompt.count("Continue now") == 1

if __name__ == '__main__':
    load_dotenv()
    asyncio.run(test_build_and_run_minimal())
