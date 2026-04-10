"""Tests for dazi/skills.py — skill parsing, substitution, and registry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dazi.skills import (
    SkillError,
    SkillRegistry,
    _normalize_to_list,
    _parse_bool,
    parse_skill_file,
    substitute_arguments,
)

# ─────────────────────────────────────────────────────────
# _parse_bool
# ─────────────────────────────────────────────────────────


class TestParseBool:
    @pytest.mark.parametrize("value", [True, "true", "True", "TRUE", "1", "yes", "Yes"])
    def test_truthy_values(self, value):
        assert _parse_bool(value) is True

    @pytest.mark.parametrize("value", [False, "false", "False", "0", "no", "No"])
    def test_falsy_values(self, value):
        assert _parse_bool(value) is False

    def test_bool_true(self):
        assert _parse_bool(True) is True

    def test_bool_false(self):
        assert _parse_bool(False) is False


# ─────────────────────────────────────────────────────────
# substitute_arguments
# ─────────────────────────────────────────────────────────


class TestSubstituteArguments:
    def test_full_arguments(self):
        result = substitute_arguments("Hello $ARGUMENTS world", "my args", [])
        assert result == "Hello my args world"

    def test_indexed_arguments(self):
        result = substitute_arguments(
            "First: $ARGUMENTS[0], Second: $ARGUMENTS[1]", "hello world", []
        )
        assert result == "First: hello, Second: world"

    def test_shorthand_arguments(self):
        result = substitute_arguments("First: $1, Second: $2", "hello world", [])
        assert result == "First: hello, Second: world"

    def test_shorthand_zero_not_replaced(self):
        result = substitute_arguments("Zero: $0", "hello", [])
        assert "$0" in result

    def test_named_arguments(self):
        result = substitute_arguments("Hello $name", "Alice", ["name"])
        assert result == "Hello Alice"

    def test_no_placeholder_appends(self):
        result = substitute_arguments("Just a prompt", "extra args", [])
        assert result == "Just a prompt\n\nARGUMENTS: extra args"

    def test_no_placeholder_no_args(self):
        result = substitute_arguments("Just a prompt $1", "", [])
        # $1 with no tokens stays as empty or placeholder
        assert "ARGUMENTS:" not in result

    def test_indexed_out_of_bounds(self):
        result = substitute_arguments("$ARGUMENTS[5]", "hello", [])
        assert result == ""

    def test_mixed_placeholders(self):
        result = substitute_arguments(
            "$ARGUMENTS[0] is $1 and all: $ARGUMENTS",
            "hello world",
            [],
        )
        assert "hello" in result
        assert "hello world" in result


# ─────────────────────────────────────────────────────────
# parse_skill_file
# ─────────────────────────────────────────────────────────


class TestParseSkillFile:
    def test_with_frontmatter(self, tmp_path: Path):
        skill_dir = tmp_path / "myskill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            "---\n"
            'description: "Test skill"\n'
            'argument-hint: "[msg]"\n'
            "user-invocable: true\n"
            "---\n"
            "This is the prompt body.\n"
        )
        skill = parse_skill_file(skill_file)
        assert skill.name == "myskill"
        assert skill.description == "Test skill"
        assert skill.prompt == "This is the prompt body."
        assert skill.argument_hint == "[msg]"
        assert skill.user_invocable is True
        assert skill.source_path == skill_file

    def test_no_frontmatter(self, tmp_path: Path):
        skill_dir = tmp_path / "plain"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("Just a plain prompt.\n")
        skill = parse_skill_file(skill_file)
        assert skill.name == "plain"
        assert skill.description == ""
        assert skill.prompt == "Just a plain prompt."

    def test_invalid_path_raises(self):
        with pytest.raises(SkillError, match="Cannot read skill file"):
            parse_skill_file(Path("/nonexistent/path/SKILL.md"))


# ─────────────────────────────────────────────────────────
# SkillRegistry
# ─────────────────────────────────────────────────────────


class TestSkillRegistry:
    def test_load_bundled(self):
        registry = SkillRegistry()
        # Don't load from disk (no project root), but bundled skills should load
        # We patch discover_skills to return only bundled ones
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            count = registry.load_skills(project_root=Path("/nonexistent"))
            assert count >= 4  # commit, review, explain, summarize

    def test_get_existing(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            skill = registry.get("commit")
            assert skill is not None
            assert skill.name == "commit"

    def test_get_missing(self):
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_has_skill(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            assert registry.has_skill("commit") is True
            assert registry.has_skill("nonexistent") is False

    def test_expand_skill(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            expanded = registry.expand_skill("commit", "fix login bug")
            assert "fix login bug" in expanded

    def test_expand_missing_raises(self):
        registry = SkillRegistry()
        with pytest.raises(SkillError, match="not found"):
            registry.expand_skill("nonexistent")

    def test_list_all(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            all_skills = registry.list_all()
            names = {s.name for s in all_skills}
            assert "commit" in names
            assert "review" in names

    def test_list_user_invocable(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            invocable = registry.list_user_invocable()
            assert all(s.user_invocable for s in invocable)

    def test_reset(self):
        registry = SkillRegistry()
        with patch("dazi.skills.discover_skills") as mock_ds:
            from dazi.skills import _get_bundled_skills

            mock_ds.return_value = _get_bundled_skills()
            registry.load_skills()
            assert len(registry.list_all()) > 0
            registry.reset()
            assert registry.list_all() == []


# ─────────────────────────────────────────────────────────
# _normalize_to_list
# ─────────────────────────────────────────────────────────


class TestNormalizeToList:
    def test_none(self):
        assert _normalize_to_list(None) == []

    def test_string(self):
        assert _normalize_to_list("hello") == ["hello"]

    def test_list(self):
        assert _normalize_to_list(["a", "b"]) == ["a", "b"]

    def test_list_of_ints(self):
        assert _normalize_to_list([1, 2, 3]) == ["1", "2", "3"]
