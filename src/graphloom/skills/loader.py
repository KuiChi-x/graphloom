"""Skill loading: progressive-disclosure skills à la Claude Code.

A "skill" is a directory containing a ``SKILL.md`` file whose YAML front-matter
declares a ``name`` and ``description``. The agent is shown only the name +
description + location up front; it reads the full skill file (and any
referenced resources) on demand via the ``read_artifact`` tool when a task
actually needs it. This keeps the system prompt small while giving the agent a
menu of deep, reusable workflows.

The framework is agnostic to where skills live: pass ``skills_dirs`` with one
or more skill-library roots. Nothing here is hard-coded to any application.
"""
import logging
import os
import re
from typing import List, Optional, Sequence, Tuple

import yaml

logger = logging.getLogger(__name__)


def parse_skill_file(skill_file: str) -> Optional[Tuple[str, str]]:
    """Parse a SKILL.md file and extract ``(name, description)`` from its YAML
    front-matter. Returns None if the file is missing/invalid."""
    if not os.path.exists(skill_file) or os.path.basename(skill_file) != "SKILL.md":
        return None

    try:
        # utf-8-sig tolerates a leading BOM (common from Windows editors) so the
        # front-matter fence still matches at the start of the file.
        with open(skill_file, "r", encoding="utf-8-sig") as f:
            content = f.read()

        front_matter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not front_matter_match:
            return None

        front_matter_text = front_matter_match.group(1)

        try:
            metadata = yaml.safe_load(front_matter_text)
        except yaml.YAMLError as exc:
            logger.error(f"Invalid YAML front-matter in {skill_file}: {exc}")
            return None

        if not isinstance(metadata, dict):
            logger.error(f"Front-matter in {skill_file} is not a YAML mapping")
            return None

        name = metadata.get("name")
        description = metadata.get("description")

        if not name or not isinstance(name, str):
            return None
        if not description or not isinstance(description, str):
            return None

        name = name.strip()
        description = description.strip()

        if not name or not description:
            return None

        return name, description

    except Exception as e:
        logger.exception(f"Unexpected error parsing skill file {skill_file}: {e}")
        return None


def _normalise_skill_dirs(skills_dirs: Optional[Sequence[str]] = None) -> List[str]:
    return [
        os.path.abspath(str(root))
        for root in (skills_dirs or [])
        if root and os.path.isdir(root)
    ]


def get_skills_prompt_section(
    available_skills: Optional[List[str]] = None,
    skills_dirs: Optional[Sequence[str]] = None,
) -> str:
    """Build the ``<skill_system>`` prompt block for discovered skills.

    Later roots override earlier roots with the same skill name.
    Pass ``["*"]`` to expose every valid discovered skill.
    """
    if not available_skills:
        return ""

    roots = _normalise_skill_dirs(skills_dirs)
    if not roots:
        return ""

    expose_all = "*" in available_skills
    selected = set(available_skills)
    resolved: dict[str, Tuple[str, str]] = {}

    for skills_root in roots:
        for root_dir, dir_names, file_names in os.walk(skills_root):
            dir_names[:] = sorted(name for name in dir_names if not name.startswith("."))
            if "SKILL.md" not in file_names:
                continue

            skill_file = os.path.join(root_dir, "SKILL.md")
            parsed = parse_skill_file(skill_file)
            if not parsed:
                continue

            name, description = parsed
            if expose_all or name in selected:
                resolved[name] = (description, skill_file)

    if not resolved:
        return ""

    skill_items = "\n".join(
        f"    <skill>\n        <name>{name}</name>\n        <description>{description}</description>\n        <location>{location}</location>\n    </skill>"
        for name, (description, location) in sorted(resolved.items())
    )
    skills_list = f"<available_skills>\n{skill_items}\n</available_skills>"

    return f"""<skill_system>
You have access to skills that provide optimized workflows for specific tasks. Each skill contains best practices, frameworks, and references to additional resources.

**Progressive Loading Pattern:**
1. When the task needs a skill, immediately call `read_artifact` on the skill's main file using the location attribute provided in the skill tag below.
2. Read and understand the skill's workflow and instructions.
3. The skill file may reference external resources (scripts, references) in the same folder.
4. Load referenced resources only when needed during execution using `read_artifact`.
5. Follow the skill's instructions precisely.

{skills_list}

</skill_system>"""
