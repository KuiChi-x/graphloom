import re
from typing import Any

_GPT_VERSION_RE = re.compile(r"gpt-(\d+)(?:\.(\d+))?")

# 显式缓存断点从 GPT-5.6 才开始支持。更
_MIN_EXPLICIT_BREAKPOINT_MODEL = (5, 6)


def provider_of(llm: Any) -> str:
    """模型的厂商命名空间：``"anthropic"``、``"openai"`` 或 ``""``"""
    get_namespace = getattr(llm, "get_lc_namespace", None)
    if not callable(get_namespace):
        return ""
    try:
        namespace = get_namespace()
    except Exception:
        return ""
    return str(namespace[-1]) if namespace else ""


def model_name_of(llm: Any) -> str:
    return str(getattr(llm, "model_name", "") or "")


def supports_explicit_cache_breakpoints(llm: Any) -> bool:
    """模型是否接受 ``prompt_cache_breakpoint`` 内容块标记。"""
    if provider_of(llm) != "openai" or not getattr(llm, "use_responses_api", False):
        return False

    match = _GPT_VERSION_RE.search(model_name_of(llm).lower())
    if not match:
        # OpenAI 兼容网关后面挂的非 GPT 模型。按不支持处理：猜高了每个请求都挂,
        return False
    version = (int(match.group(1)), int(match.group(2) or 0))
    return version >= _MIN_EXPLICIT_BREAKPOINT_MODEL
