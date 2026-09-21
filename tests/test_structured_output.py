"""结构化输出：只绑工具、不强制调用（思考模式的上游拒绝强制 tool_choice）。"""
from typing import Any, List

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from graphloom.util.structured_output import structured_output_llm


class RelevanceScore(BaseModel):
    score: float = Field(description="relevance 0..1")


class _FakeLLM(Runnable):
    """记录绑定参数，并把工具调用结果发回来。"""

    def __init__(self, *, calls_tool: bool = True):
        self.calls_tool = calls_tool
        self.bound_tools: List[Any] = []
        self.bind_tool_kwargs: dict = {}
        self.bind_kwargs: List[dict] = []

    def bind(self, **kwargs: Any) -> "_FakeLLM":
        self.bind_kwargs.append(kwargs)
        return self

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> "_FakeLLM":
        self.bound_tools = list(tools)
        self.bind_tool_kwargs = kwargs
        return self

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        if self.calls_tool:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": RelevanceScore.__name__,
                        "args": {"score": 0.75},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="score is probably 0.75")


def test_tool_choice_is_never_forced():
    """langchain 默认会把 schema 绑成强制调用，思考模式的上游会因此 400。"""
    llm = _FakeLLM()

    chain = structured_output_llm(llm, RelevanceScore)

    assert llm.bind_tool_kwargs == {}
    assert llm.bound_tools == [RelevanceScore]
    assert chain.invoke([]) == RelevanceScore(score=0.75)


def test_max_tokens_is_bound_before_the_tool_schema():
    llm = _FakeLLM()

    structured_output_llm(llm, RelevanceScore, max_tokens=4096)

    assert llm.bind_kwargs == [{"max_tokens": 4096}]


def test_a_missing_tool_call_fails_loudly():
    llm = _FakeLLM(calls_tool=False)

    with pytest.raises(OutputParserException, match="did not call RelevanceScore"):
        structured_output_llm(llm, RelevanceScore).invoke([])
