"""Tests for dazi.base — ToolSafety enum, DaziTool dataclass."""

from dazi.base import DaziTool, ToolSafety


class TestToolSafety:
    def test_safe_value(self):
        assert ToolSafety.SAFE.value == "safe"

    def test_write_value(self):
        assert ToolSafety.WRITE.value == "write"

    def test_destructive_value(self):
        assert ToolSafety.DESTRUCTIVE.value == "destructive"

    def test_value_access(self):
        assert ToolSafety.SAFE.value == "safe"
        assert ToolSafety.WRITE.value == "write"
        assert ToolSafety.DESTRUCTIVE.value == "destructive"


class TestDaziTool:
    def test_defaults(self):
        tool = DaziTool(name="test", description="A test tool")
        assert tool.safety == ToolSafety.DESTRUCTIVE
        assert tool.enabled is True
        assert tool.is_concurrency_safe is False
        assert tool.is_read_only is False

    def test_safe_flags(self):
        tool = DaziTool(name="reader", description="Read", safety=ToolSafety.SAFE)
        assert tool.is_concurrency_safe is True
        assert tool.is_read_only is True

    def test_write_flags(self):
        tool = DaziTool(name="writer", description="Write", safety=ToolSafety.WRITE)
        assert tool.is_concurrency_safe is False
        assert tool.is_read_only is False

    def test_destructive_flags(self):
        tool = DaziTool(name="rm", description="Delete", safety=ToolSafety.DESTRUCTIVE)
        assert tool.is_concurrency_safe is False
        assert tool.is_read_only is False

    def test_custom_values(self):
        tool = DaziTool(
            name="my_tool",
            description="Custom",
            safety=ToolSafety.SAFE,
            enabled=False,
        )
        assert tool.name == "my_tool"
        assert tool.description == "Custom"
        assert tool.safety == ToolSafety.SAFE
        assert tool.enabled is False
