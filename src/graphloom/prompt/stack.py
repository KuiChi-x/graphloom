from graphloom.prompt.system_prompt import COMMON_AGENT_SYSTEM_PROMPT


class PromptStack:
    """Assembles the per-turn system prompt: custom (agent-specific) + common
    (framework default). Overridable entirely via custom_system_prompt."""

    def __init__(self, *, custom_system_prompt: str) -> None:
        self._custom_system_prompt = str(custom_system_prompt or "").strip()
        self._common_system_prompt = str(COMMON_AGENT_SYSTEM_PROMPT or "").strip()
        if not self._custom_system_prompt:
            raise ValueError("custom_system_prompt must be provided.")

    async def build_system_messages(self) -> str:
        system_prompt = self._custom_system_prompt
        if self._common_system_prompt:
            system_prompt += "\n\n" + self._common_system_prompt
        return system_prompt


def create_prompt_stack(*, custom_system_prompt: str) -> PromptStack:
    return PromptStack(custom_system_prompt=custom_system_prompt)
