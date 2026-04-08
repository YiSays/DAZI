"""Tests for dazi.settings — DaziSettings, merge_settings, SettingsManager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dazi.settings import (
    DaziSettings,
    SettingsManager,
    SettingsSource,
    _dedupe_preserve_order,
    merge_settings,
)


# ─────────────────────────────────────────────────────────
# DaziSettings dataclass
# ─────────────────────────────────────────────────────────


class TestDaziSettingsDefaults:
    def test_model_default_none(self):
        s = DaziSettings()
        assert s.model is None

    def test_api_base_url_default_none(self):
        s = DaziSettings()
        assert s.api_base_url is None

    def test_default_mode_default(self):
        s = DaziSettings()
        assert s.default_mode == "default"

    def test_auto_compact_default_true(self):
        s = DaziSettings()
        assert s.auto_compact is True

    def test_auto_memory_default_true(self):
        s = DaziSettings()
        assert s.auto_memory is True

    def test_max_concurrent_tools_default(self):
        s = DaziSettings()
        assert s.max_concurrent_tools == 5

    def test_lists_default_empty(self):
        s = DaziSettings()
        assert s.allow_rules == []
        assert s.deny_rules == []

    def test_env_and_mcp_servers_default_empty(self):
        s = DaziSettings()
        assert s.env == {}
        assert s.mcp_servers == {}


class TestDaziSettingsToDict:
    def test_excludes_none_and_empty(self):
        s = DaziSettings()
        d = s.to_dict()
        assert "model" not in d
        assert "allow_rules" not in d
        assert "env" not in d
        assert "mcp_servers" not in d

    def test_camel_case_mcp_servers(self):
        s = DaziSettings(mcp_servers={"srv": {"command": "cmd"}})
        d = s.to_dict()
        assert "mcpServers" in d
        assert "mcp_servers" not in d

    def test_includes_set_values(self):
        s = DaziSettings(model="gpt-4", auto_compact=False)
        d = s.to_dict()
        assert d["model"] == "gpt-4"
        assert d["auto_compact"] is False


class TestDaziSettingsFromDict:
    def test_basic_from_dict(self):
        s = DaziSettings.from_dict({"model": "gpt-3.5", "default_mode": "plan"})
        assert s.model == "gpt-3.5"
        assert s.default_mode == "plan"

    def test_camel_case_mcp_servers(self):
        s = DaziSettings.from_dict({"mcpServers": {"s": {"command": "c"}}})
        assert s.mcp_servers == {"s": {"command": "c"}}

    def test_ignores_unknown_fields(self):
        s = DaziSettings.from_dict({"model": "x", "unknown_field": 42})
        assert s.model == "x"


# ─────────────────────────────────────────────────────────
# Merge algorithm
# ─────────────────────────────────────────────────────────


class TestMergeSettings:
    def test_primitives_override(self):
        base = DaziSettings(model="base", max_concurrent_tools=3)
        override = DaziSettings(model="override", max_concurrent_tools=10)
        merged = merge_settings(base, override)
        assert merged.model == "override"
        assert merged.max_concurrent_tools == 10

    def test_none_fallback(self):
        base = DaziSettings(api_base_url="https://base.com")
        override = DaziSettings()  # api_base_url=None
        merged = merge_settings(base, override)
        assert merged.api_base_url == "https://base.com"

    def test_none_fallback_api_key(self):
        base = DaziSettings(api_key="key-base")
        override = DaziSettings()
        merged = merge_settings(base, override)
        assert merged.api_key == "key-base"

    def test_list_concat_and_dedup(self):
        base = DaziSettings(allow_rules=["a", "b"])
        override = DaziSettings(allow_rules=["b", "c"])
        merged = merge_settings(base, override)
        assert merged.allow_rules == ["a", "b", "c"]

    def test_dict_merge(self):
        base = DaziSettings(env={"A": "1", "B": "2"})
        override = DaziSettings(env={"B": "override", "C": "3"})
        merged = merge_settings(base, override)
        assert merged.env == {"A": "1", "B": "override", "C": "3"}

    def test_mcp_servers_merge(self):
        base = DaziSettings(mcp_servers={"s1": {"command": "a"}})
        override = DaziSettings(mcp_servers={"s2": {"command": "b"}})
        merged = merge_settings(base, override)
        assert "s1" in merged.mcp_servers
        assert "s2" in merged.mcp_servers


class TestDedupePreserveOrder:
    def test_basic(self):
        assert _dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty(self):
        assert _dedupe_preserve_order([]) == []

    def test_no_duplicates(self):
        assert _dedupe_preserve_order(["x", "y", "z"]) == ["x", "y", "z"]


# ─────────────────────────────────────────────────────────
# SettingsManager
# ─────────────────────────────────────────────────────────


class TestSettingsManagerLoad:
    def test_load_defaults(self, mock_settings_manager):
        s = mock_settings_manager.settings
        # auto_compact is a boolean default that is unlikely to be overridden
        assert s.auto_compact is True
        assert isinstance(s.max_concurrent_tools, int)

    def test_user_only(self, mock_settings_manager):
        user_path = mock_settings_manager.user_path
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(json.dumps({"model": "user-model"}), encoding="utf-8")
        mock_settings_manager.reload()
        assert mock_settings_manager.settings.model == "user-model"

    def test_project_overrides_user(self, mock_settings_manager):
        user_path = mock_settings_manager.user_path
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(json.dumps({"model": "user-model"}), encoding="utf-8")

        proj_path = mock_settings_manager.project_path
        proj_path.parent.mkdir(parents=True, exist_ok=True)
        proj_path.write_text(json.dumps({"model": "project-model"}), encoding="utf-8")

        mock_settings_manager.reload()
        assert mock_settings_manager.settings.model == "project-model"

    def test_invalid_json_skipped(self, mock_settings_manager):
        user_path = mock_settings_manager.user_path
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text("{invalid json!!!", encoding="utf-8")
        mock_settings_manager.reload()
        # Should fall back to defaults
        assert mock_settings_manager.settings.auto_compact is True


class TestSettingsManagerSave:
    def test_save_user_settings(self, mock_settings_manager):
        s = DaziSettings(model="saved-model")
        mock_settings_manager.save_user_settings(s)
        assert mock_settings_manager.user_path.exists()
        data = json.loads(mock_settings_manager.user_path.read_text())
        assert data["model"] == "saved-model"

    def test_save_project_settings(self, mock_settings_manager):
        s = DaziSettings(model="proj-model")
        mock_settings_manager.save_project_settings(s)
        assert mock_settings_manager.project_path.exists()
        data = json.loads(mock_settings_manager.project_path.read_text())
        assert data["model"] == "proj-model"


class TestSettingsManagerMethods:
    def test_get_model_name(self, mock_settings_manager):
        user_path = mock_settings_manager.user_path
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_path.write_text(json.dumps({"model": "gpt-4o"}), encoding="utf-8")
        mock_settings_manager.reload()
        assert mock_settings_manager.get_model_name() == "gpt-4o"

    def test_source_map_default(self, mock_settings_manager):
        sm = mock_settings_manager.source_map
        # source_map tracks which layer set each field;
        # with no user/project files, all fields come from "default"
        assert isinstance(sm, dict)
        assert "model" in sm

    def test_get_permission_rules_empty(self, mock_settings_manager):
        rules = mock_settings_manager.get_permission_rules()
        assert isinstance(rules, list)
        assert len(rules) == 0
