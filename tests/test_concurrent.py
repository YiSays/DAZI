"""Tests for dazi/concurrent.py — tool partitioning and concurrent execution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage

from dazi.base import DaziTool, ToolSafety
from dazi.concurrent import (
    execute_tool,
    execute_tools_concurrent,
    execute_tools_sync,
    partition_tool_calls,
)

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────


def _tool_meta(safe: bool = True) -> dict[str, DaziTool]:
    """Build a tool_meta dict with one safe or unsafe tool."""
    safety = ToolSafety.SAFE if safe else ToolSafety.WRITE
    meta = DaziTool(name="test_tool", description="test", safety=safety)
    return {meta.name: meta}


def _mock_tool(name: str = "test_tool", result: str = "ok") -> MagicMock:
    """Create a mock LangChain tool."""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=result)
    return tool


# ─────────────────────────────────────────────────────────
# partition_tool_calls
# ─────────────────────────────────────────────────────────


class TestPartitionToolCalls:
    def test_safe_tools_go_parallel(self):
        tc = {"name": "file_reader", "args": {"path": "/tmp/x"}, "id": "1"}
        meta = {"file_reader": DaziTool(name="file_reader", description="", safety=ToolSafety.SAFE)}
        batch = partition_tool_calls([tc], meta)
        assert len(batch.parallel) == 1
        assert len(batch.serial) == 0

    def test_write_tools_go_serial(self):
        tc = {"name": "file_writer", "args": {"path": "/tmp/x"}, "id": "1"}
        meta = {
            "file_writer": DaziTool(name="file_writer", description="", safety=ToolSafety.WRITE)
        }
        batch = partition_tool_calls([tc], meta)
        assert len(batch.parallel) == 0
        assert len(batch.serial) == 1

    def test_destructive_tools_go_serial(self):
        tc = {"name": "shell_exec", "args": {"cmd": "rm -rf /"}, "id": "1"}
        meta = {
            "shell_exec": DaziTool(name="shell_exec", description="", safety=ToolSafety.DESTRUCTIVE)
        }
        batch = partition_tool_calls([tc], meta)
        assert len(batch.parallel) == 0
        assert len(batch.serial) == 1

    def test_deduplicate_identical_calls(self):
        tc = {"name": "file_reader", "args": {"path": "/tmp/x"}, "id": "1"}
        meta = {"file_reader": DaziTool(name="file_reader", description="", safety=ToolSafety.SAFE)}
        batch = partition_tool_calls([tc, tc], meta)
        assert len(batch.parallel) == 1

    def test_unknown_tool_goes_serial(self):
        tc = {"name": "unknown_tool", "args": {}, "id": "1"}
        batch = partition_tool_calls([tc], {})
        assert len(batch.serial) == 1
        assert len(batch.parallel) == 0

    def test_mixed_partition(self):
        safe_tc = {"name": "file_reader", "args": {}, "id": "1"}
        write_tc = {"name": "file_writer", "args": {}, "id": "2"}
        meta = {
            "file_reader": DaziTool(name="file_reader", description="", safety=ToolSafety.SAFE),
            "file_writer": DaziTool(name="file_writer", description="", safety=ToolSafety.WRITE),
        }
        batch = partition_tool_calls([safe_tc, write_tc], meta)
        assert len(batch.parallel) == 1
        assert len(batch.serial) == 1


# ─────────────────────────────────────────────────────────
# execute_tool
# ─────────────────────────────────────────────────────────


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_success_returns_tool_message(self):
        tool = _mock_tool("test_tool", "result text")
        tc = {"name": "test_tool", "args": {}, "id": "call_1"}
        msg = await execute_tool(tc, [tool])
        assert isinstance(msg, ToolMessage)
        assert msg.content == "result text"
        assert msg.tool_call_id == "call_1"

    @pytest.mark.asyncio
    async def test_not_found_returns_error(self):
        tc = {"name": "missing_tool", "args": {}, "id": "call_1"}
        msg = await execute_tool(tc, [])
        assert isinstance(msg, ToolMessage)
        assert "not found" in msg.content

    @pytest.mark.asyncio
    async def test_exception_returns_error(self):
        tool = MagicMock()
        tool.name = "broken"
        tool.ainvoke = AsyncMock(side_effect=ValueError("crash"))
        tc = {"name": "broken", "args": {}, "id": "call_1"}
        msg = await execute_tool(tc, [tool])
        assert "Error" in msg.content
        assert "crash" in msg.content


# ─────────────────────────────────────────────────────────
# execute_tools_concurrent
# ─────────────────────────────────────────────────────────


class TestExecuteToolsConcurrent:
    @pytest.mark.asyncio
    async def test_empty_returns_empty(self):
        result = await execute_tools_concurrent([], [], {})
        assert result == []

    @pytest.mark.asyncio
    async def test_order_preserved(self):
        tool_a = _mock_tool("safe_a", "a_result")
        tool_b = _mock_tool("safe_b", "b_result")
        meta = {
            "safe_a": DaziTool(name="safe_a", description="", safety=ToolSafety.SAFE),
            "safe_b": DaziTool(name="safe_b", description="", safety=ToolSafety.SAFE),
        }
        calls = [
            {"name": "safe_a", "args": {}, "id": "1"},
            {"name": "safe_b", "args": {}, "id": "2"},
        ]
        results = await execute_tools_concurrent(calls, [tool_a, tool_b], meta)
        assert len(results) == 2
        assert results[0].content == "a_result"
        assert results[1].content == "b_result"


# ─────────────────────────────────────────────────────────
# execute_tools_sync
# ─────────────────────────────────────────────────────────


class TestExecuteToolsSync:
    def test_empty_returns_empty(self):
        result = execute_tools_sync([], [], {})
        assert result == []

    def test_executes_serially(self):
        tool = _mock_tool("test_tool", "sync_result")
        meta = {"test_tool": DaziTool(name="test_tool", description="", safety=ToolSafety.WRITE)}
        calls = [{"name": "test_tool", "args": {}, "id": "1"}]
        results = execute_tools_sync(calls, [tool], meta)
        assert len(results) == 1
        assert results[0].content == "sync_result"
