"""Tests for dazi/llm.py — create_llm, _get_llm, _get_model_name, get_memory_content,
get_skills_content."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.helpers.mock_singletons import patch_singletons


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)


# ─────────────────────────────────────────────────────────
# create_llm
# ─────────────────────────────────────────────────────────


class TestCreateLlm:
    def test_basic_creation(self):
        from dazi.llm import create_llm

        llm = create_llm(model="gpt-4o", api_key="test-key")
        assert llm.model == "gpt-4o"

    def test_custom_params(self):
        from dazi.llm import create_llm

        llm = create_llm(model="gpt-3.5-turbo", temperature=0.5, api_key="test-key")
        assert llm.model == "gpt-3.5-turbo"
        assert llm.temperature == 0.5

    def test_custom_base_url(self):
        from dazi.llm import create_llm

        llm = create_llm(model="gpt-4o", api_key="test-key", base_url="https://custom.api.com")
        assert llm.openai_api_base == "https://custom.api.com"

    def test_custom_api_key(self):
        from dazi.llm import create_llm

        llm = create_llm(api_key="override-key")
        assert llm.openai_api_key.get_secret_value() == "override-key"


# ─────────────────────────────────────────────────────────
# _get_llm lazy init
# ─────────────────────────────────────────────────────────


class TestGetLlm:
    def test_lazy_initialization(self, monkeypatch):
        import dazi.llm as llm_mod

        # Reset the global
        monkeypatch.setattr(llm_mod, "_base_llm", None)
        monkeypatch.setattr("dazi.config.OPENAI_API_KEY", "test-key")
        monkeypatch.setattr("dazi.config.OPENAI_MODEL", "gpt-4o")
        monkeypatch.setattr("dazi.config.OPENAI_BASE_URL", "")

        # Mock settings_manager on the llm module (it imports from _singletons)
        sm = MagicMock()
        sm.get_model_name.return_value = "gpt-4o"
        sm.get_api_key.return_value = "test-key"
        sm.get_api_base_url.return_value = ""
        monkeypatch.setattr(llm_mod, "settings_manager", sm)

        result = llm_mod._get_llm()
        assert result is not None
        # Second call returns same instance
        result2 = llm_mod._get_llm()
        assert result is result2

        # Cleanup
        monkeypatch.setattr(llm_mod, "_base_llm", None)


# ─────────────────────────────────────────────────────────
# _get_model_name
# ─────────────────────────────────────────────────────────


class TestGetModelName:
    def test_gets_name_from_settings(self, monkeypatch):
        import dazi.llm as llm_mod

        sm = MagicMock()
        sm.get_model_name.return_value = "claude-3.5-sonnet"
        monkeypatch.setattr(llm_mod, "settings_manager", sm)

        result = llm_mod._get_model_name()
        assert result == "claude-3.5-sonnet"


# ─────────────────────────────────────────────────────────
# get_memory_content
# ─────────────────────────────────────────────────────────


class TestGetMemoryContent:
    def test_no_relevant_memories(self, monkeypatch):
        import dazi.llm as llm_mod

        ms = MagicMock()
        ms.find_relevant.return_value = []
        monkeypatch.setattr(llm_mod, "memory_store", ms)

        result = llm_mod.get_memory_content("test query")
        assert result == ""

    def test_with_relevant_memories(self, monkeypatch):
        import dazi.llm as llm_mod
        from dazi.memory import MemoryCategory

        mem = MagicMock()
        mem.category = MemoryCategory.USER
        mem.description = "User likes dark mode"
        mem.content = "The user prefers dark mode for the UI"

        ms = MagicMock()
        ms.find_relevant.return_value = [mem]
        monkeypatch.setattr(llm_mod, "memory_store", ms)

        result = llm_mod.get_memory_content("UI preferences")
        assert "dark mode" in result


# ─────────────────────────────────────────────────────────
# get_skills_content
# ─────────────────────────────────────────────────────────


class TestGetSkillsContent:
    def test_no_skills(self, monkeypatch):
        import dazi.llm as llm_mod

        sr = MagicMock()
        sr.list_user_invocable.return_value = []
        monkeypatch.setattr(llm_mod, "skill_registry", sr)

        result = llm_mod.get_skills_content()
        assert result == ""

    def test_with_skills(self, monkeypatch):
        import dazi.llm as llm_mod

        skill = MagicMock()
        skill.name = "commit"
        skill.description = "Create a git commit"
        skill.argument_hint = ""

        sr = MagicMock()
        sr.list_user_invocable.return_value = [skill]
        monkeypatch.setattr(llm_mod, "skill_registry", sr)

        result = llm_mod.get_skills_content()
        assert "commit" in result
        assert "git commit" in result
