from pathlib import Path

from graphloom.skills.loader import get_skills_prompt_section


def _write_skill(root: Path, folder: str, name: str, description: str) -> Path:
    skill_dir = root / folder
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n",
        encoding="utf-8",
    )
    return skill_file


def test_multiple_skill_directories_are_combined(tmp_path):
    builtins = tmp_path / "builtins"
    user = tmp_path / "user"
    _write_skill(builtins, "crawl", "web-crawl", "Built-in crawler")
    _write_skill(user, "reports", "reporting", "User reporting workflow")

    prompt = get_skills_prompt_section(["*"], skills_dirs=[str(builtins), str(user)])

    assert "<name>web-crawl</name>" in prompt
    assert "<name>reporting</name>" in prompt


def test_later_skill_directory_overrides_same_name(tmp_path):
    builtins = tmp_path / "builtins"
    user = tmp_path / "user"
    _write_skill(builtins, "crawl", "web-crawl", "Built-in crawler")
    user_skill = _write_skill(user, "crawl", "web-crawl", "Customized crawler")

    prompt = get_skills_prompt_section(["*"], skills_dirs=[str(builtins), str(user)])

    assert "Customized crawler" in prompt
    assert "Built-in crawler" not in prompt
    assert f"<location>{user_skill}</location>" in prompt
