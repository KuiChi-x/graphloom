from typing import List, Optional

from graphloom.prompt.system_prompt import COMMON_AGENT_SYSTEM_PROMPT
from graphloom.skills.loader import get_skills_prompt_section


class PromptStack:
    """Assembles the per-turn system prompt: custom (agent-specific) + common
    (framework default) + an optional skills section. Overridable entirely via
    custom_system_prompt."""

    def __init__(
        self,
        *,
        custom_system_prompt: str,
        available_skills: Optional[List[str]] = None,
        skills_dir: Optional[str] = None,
    ) -> None:
        self._custom_system_prompt = str(custom_system_prompt or "").strip()
        self._common_system_prompt = str(COMMON_AGENT_SYSTEM_PROMPT or "").strip()
        self._available_skills = available_skills
        self._skills_dir = skills_dir
        if not self._custom_system_prompt:
            raise ValueError("custom_system_prompt must be provided.")

    async def build_system_messages(self) -> str:
        system_prompt = self._custom_system_prompt
        if self._common_system_prompt:
            system_prompt += "\n\n" + self._common_system_prompt
        if self._available_skills:
            skills_prompt = get_skills_prompt_section(self._available_skills, self._skills_dir)
            if skills_prompt:
                system_prompt += "\n\n" + skills_prompt
        return system_prompt


def create_prompt_stack(
    *,
    custom_system_prompt: str,
    available_skills: Optional[List[str]] = None,
    skills_dir: Optional[str] = None,
) -> PromptStack:
    return PromptStack(
        custom_system_prompt=custom_system_prompt,
        available_skills=available_skills,
        skills_dir=skills_dir,
    )
