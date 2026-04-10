"""Tests for dazi/registry.py — ALL_TOOL_META, PLAN_MODE_META, EXECUTE_MODE_META."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.mock_singletons import patch_singletons


# Apply singleton patches before importing registry
@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)


class TestAllToolMeta:
    def test_has_entries_for_core_tools(self):
        from dazi.registry import ALL_TOOL_META

        assert "file_reader" in ALL_TOOL_META
        assert "file_writer" in ALL_TOOL_META
        assert "shell_exec" in ALL_TOOL_META
        assert "calculator" in ALL_TOOL_META
        assert "plan_writer" in ALL_TOOL_META
        assert "sleep" in ALL_TOOL_META

    def test_has_entries_for_task_tools(self):
        from dazi.registry import ALL_TOOL_META

        assert "task_create" in ALL_TOOL_META
        assert "task_update" in ALL_TOOL_META
        assert "task_list" in ALL_TOOL_META
        assert "task_get" in ALL_TOOL_META

    def test_has_entries_for_team_and_messaging(self):
        from dazi.registry import ALL_TOOL_META

        assert "send_message" in ALL_TOOL_META
        assert "check_inbox" in ALL_TOOL_META
        assert "create_team" in ALL_TOOL_META
        assert "list_teams" in ALL_TOOL_META


class TestPlanModeSubset:
    def test_plan_tools_are_subset_of_execute_tools(self):
        from dazi.registry import EXECUTE_MODE_TOOLS, PLAN_MODE_TOOLS

        plan_names = {t.name for t in PLAN_MODE_TOOLS}
        execute_names = {t.name for t in EXECUTE_MODE_TOOLS}
        assert plan_names.issubset(execute_names)


class TestPlanToolSafety:
    def test_plan_tools_safety_constraints(self):
        from dazi.registry import PLAN_MODE_META

        # Plan mode excludes the most dangerous tools (file_writer, delete_team, etc.)
        # but may include some WRITE tools (run_background) for read-only exploration use
        # and DESTRUCTIVE tools (shell_exec) for read-only exploration commands
        assert "file_writer" not in PLAN_MODE_META
        assert "delete_team" not in PLAN_MODE_META
        assert "cancel_background" not in PLAN_MODE_META
        assert "request_permission" not in PLAN_MODE_META
        assert "create_worktree" not in PLAN_MODE_META
        assert "finish_worktree" not in PLAN_MODE_META


class TestToolNamesMatchKeys:
    def test_meta_dict_keys_match_tool_names(self):
        from dazi.registry import ALL_TOOL_META

        for key, meta in ALL_TOOL_META.items():
            assert meta.name == key, f"Key '{key}' does not match meta.name '{meta.name}'"
