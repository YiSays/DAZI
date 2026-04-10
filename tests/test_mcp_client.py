"""Tests for dazi/mcp_client.py — name normalization, config parsing, MCPManager initial state."""

from __future__ import annotations

from dazi.mcp_client import (
    MCPManager,
    MCPServerConfig,
    _build_mcp_tool_name,
    _normalize_name,
    _parse_mcp_tool_name,
)

# ─────────────────────────────────────────────────────────
# Name normalization
# ─────────────────────────────────────────────────────────


class TestNormalizeName:
    def test_lowercases(self):
        assert _normalize_name("FileSystem") == "filesystem"

    def test_replaces_special_chars(self):
        # Dots are NOT in [a-zA-Z0-9_-], so they get replaced with underscores
        assert _normalize_name("my-server.v2") == "my-server_v2"
        assert _normalize_name("hello world") == "hello_world"

    def test_strips_non_alphanumeric_except_underscore_dash(self):
        assert _normalize_name("a!@#b") == "a___b"


class TestBuildMcpToolName:
    def test_builds_qualified_name(self):
        result = _build_mcp_tool_name("filesystem", "read_file")
        assert result == "mcp__filesystem__read_file"

    def test_normalizes_names(self):
        result = _build_mcp_tool_name("My Server", "Some Tool")
        assert result == "mcp__my_server__some_tool"


class TestParseMcpToolName:
    def test_parses_qualified_name(self):
        server, tool = _parse_mcp_tool_name("mcp__filesystem__read_file")
        assert server == "filesystem"
        assert tool == "read_file"

    def test_raises_for_non_mcp_name(self):
        import pytest

        with pytest.raises(ValueError, match="Not an MCP tool name"):
            _parse_mcp_tool_name("regular_tool")

    def test_raises_for_invalid_format(self):
        import pytest

        with pytest.raises(ValueError, match="Invalid MCP tool name format"):
            _parse_mcp_tool_name("mcp__noserverpart")


# ─────────────────────────────────────────────────────────
# MCPServerConfig
# ─────────────────────────────────────────────────────────


class TestMCPServerConfig:
    def test_from_dict(self):
        data = {
            "command": "npx",
            "args": ["-y", "@mcp/server"],
            "env": {"KEY": "value"},
            "description": "Test server",
        }
        config = MCPServerConfig.from_dict("test", data)
        assert config.name == "test"
        assert config.command == "npx"
        assert config.args == ["-y", "@mcp/server"]
        assert config.env == {"KEY": "value"}
        assert config.description == "Test server"

    def test_from_dict_minimal(self):
        data = {"command": "node"}
        config = MCPServerConfig.from_dict("minimal", data)
        assert config.name == "minimal"
        assert config.command == "node"
        assert config.args == []
        assert config.env is None
        assert config.description == ""

    def test_to_dict(self):
        config = MCPServerConfig(
            name="test",
            command="npx",
            args=["-y", "@mcp/server"],
            description="My server",
        )
        d = config.to_dict()
        assert d["command"] == "npx"
        assert d["args"] == ["-y", "@mcp/server"]
        assert d["description"] == "My server"

    def test_to_dict_omits_empty_fields(self):
        config = MCPServerConfig(name="test", command="node")
        d = config.to_dict()
        assert "args" not in d
        assert "env" not in d
        assert "description" not in d


# ─────────────────────────────────────────────────────────
# MCPManager initial state
# ─────────────────────────────────────────────────────────


class TestMCPManagerInitialState:
    def test_no_servers(self):
        mgr = MCPManager()
        assert mgr.list_servers() == []

    def test_no_tools(self):
        mgr = MCPManager()
        assert mgr.get_tools() == []

    def test_get_tool_returns_none(self):
        mgr = MCPManager()
        assert mgr.get_tool("mcp__test__tool") is None

    def test_get_server_returns_none(self):
        mgr = MCPManager()
        assert mgr.get_server("nonexistent") is None
