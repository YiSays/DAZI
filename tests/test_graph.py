"""Tests for dazi/graph.py — _get_effective_rules, should_continue, has_allowed_tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from langchain_core.messages import AIMessage, HumanMessage

from tests.helpers.mock_singletons import patch_singletons


@pytest.fixture(autouse=True)
def _patch(monkeypatch, tmp_path: Path):
    patch_singletons(monkeypatch, tmp_path)


# ─────────────────────────────────────────────────────────
# _get_effective_rules
# ─────────────────────────────────────────────────────────


class TestGetEffectiveRules:
    def test_combines_settings_and_cli_rules(self, monkeypatch):
        import dazi.graph as graph_mod
        from dazi.permissions import PermissionBehavior, PermissionRule

        # Setup settings_manager mock to return a rule
        settings_rule = PermissionRule(
            behavior=PermissionBehavior.ALLOW,
            tool_name="file_reader",
            source="settings",
        )
        sm = MagicMock()
        sm.get_permission_rules.return_value = [settings_rule]
        monkeypatch.setattr(graph_mod, "settings_manager", sm)

        # Add a CLI rule
        cli_rule = PermissionRule(
            behavior=PermissionBehavior.DENY,
            tool_name="shell_exec",
            source="cli",
        )
        monkeypatch.setattr(graph_mod, "permission_rules", [cli_rule])

        result = graph_mod._get_effective_rules()
        assert len(result) == 2
        assert result[0].source == "settings"
        assert result[1].source == "cli"

    def test_empty_when_no_rules(self, monkeypatch):
        import dazi.graph as graph_mod

        sm = MagicMock()
        sm.get_permission_rules.return_value = []
        monkeypatch.setattr(graph_mod, "settings_manager", sm)
        monkeypatch.setattr(graph_mod, "permission_rules", [])

        result = graph_mod._get_effective_rules()
        assert result == []


# ─────────────────────────────────────────────────────────
# should_continue routing
# ─────────────────────────────────────────────────────────


class TestShouldContinue:
    def test_ai_message_with_tool_calls_routes_to_check_permissions(self):
        from dazi.graph import should_continue
        from langgraph.graph import END

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "file_reader", "args": {"file_path": "/tmp/x"}}],
        )
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}
        result = should_continue(state)
        assert result == "check_permissions"

    def test_ai_message_without_tool_calls_routes_to_end(self):
        from dazi.graph import should_continue
        from langgraph.graph import END

        ai_msg = AIMessage(content="Hello! How can I help?")
        state = {"messages": [HumanMessage(content="hi"), ai_msg]}
        result = should_continue(state)
        assert result == END


# ─────────────────────────────────────────────────────────
# has_allowed_tools routing
# ─────────────────────────────────────────────────────────


class TestHasAllowedTools:
    def test_allowed_tools_routes_to_execute(self):
        from dazi.graph import has_allowed_tools

        state = {
            "messages": [],
            "allowed_tool_ids": ["tc1", "tc2"],
        }
        result = has_allowed_tools(state)
        assert result == "execute_tools"

    def test_empty_allowed_tools_routes_to_call_llm(self):
        from dazi.graph import has_allowed_tools

        state = {
            "messages": [],
            "allowed_tool_ids": [],
        }
        result = has_allowed_tools(state)
        assert result == "call_llm"

    def test_missing_allowed_tools_routes_to_call_llm(self):
        from dazi.graph import has_allowed_tools

        state = {"messages": []}
        result = has_allowed_tools(state)
        assert result == "call_llm"


# ─────────────────────────────────────────────────────────
# _build_full_tool_lists
# ─────────────────────────────────────────────────────────


class TestBuildFullToolLists:
    def test_returns_plan_and_execute_lists(self, monkeypatch):
        import dazi.graph as graph_mod

        # Ensure mcp_manager has no tools (clean state)
        mm = MagicMock()
        mm.build_langchain_tools.return_value = []
        monkeypatch.setattr(graph_mod, "mcp_manager", mm)

        execute_tools, plan_tools = graph_mod._build_full_tool_lists()
        assert len(execute_tools) > 0
        assert len(plan_tools) > 0
        # Execute should have more tools than plan
        assert len(execute_tools) >= len(plan_tools)
