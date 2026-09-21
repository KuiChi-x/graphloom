"""结构化输出：绑工具，但不强制调用。

langchain 的 ``with_structured_output`` 三种 method 在这套网关上都不成立：

- ``function_calling``（默认）：内部是 ``bind_tools([schema], tool_choice="any")``
  （langchain_core/language_models/chat_models.py:2512）。思考模式的上游直接 400 ——
  实测 DeepSeek ``Thinking mode does not support this tool_choice``、Qwen
  ``tool_choice ... does not support being set to required or object``。这些模型
  **不传 thinking 也会思考**，所以“没开思考就能强制”并不成立。langchain 1.x 的
  ``ToolStrategy`` 内部同样是 ``tool_choice="any"``，一样 400。
- ``json_schema``（原生结构化输出）：anthropic 线发 ``output_config.format`` ——
  DeepSeek 返 200 但**静默忽略**（吐散文），opus-5 直接 400 ``Extra inputs are not
  permitted``；openai 线发 ``response_format``，被拒 ``does not support the
  requested response_format``。官方 anthropic 的 ``output_format`` + beta 头同样
  200 + 静默忽略。所以这条路实测不通。
- ``json_mode``：只有 openai 线有，且只保证“是合法 JSON”，schema 要自己写进 prompt。

三条协议线的唯一交集就是：绑上工具、让模型自己调用。这里只是把 ``tool_choice="any"``
从内置实现里拿掉，没调用工具就抛 OutputParserException，交给调用方重试。
"""
from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


def structured_output_llm(
    llm: Any, schema: Type[BaseModel], *, max_tokens: Optional[int] = None
) -> Runnable:
    """拿得到 ``schema`` 的 LLM：绑工具，但不强制模型调用。"""
    if max_tokens is not None:
        llm = llm.bind(max_tokens=max_tokens)

    parser = PydanticToolsParser(tools=[schema], first_tool_only=True)

    def _parse_or_raise(message: Any) -> Any:
        parsed = parser.invoke(message)
        if parsed is None:
            raise OutputParserException(f"Model did not call {schema.__name__}.")
        return parsed

    return llm.bind_tools([schema]) | RunnableLambda(_parse_or_raise)
