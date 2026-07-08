from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage


def get_last_ai_message(messages: list[BaseMessage] | tuple[BaseMessage, ...] | None) -> Optional[AIMessage]:
    if not messages:
        return None

    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            return message
    return None
