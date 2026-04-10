"""Tests for MCP connection lifecycle — connect, disconnect, reconnect, error handling.

Covers the async methods of MCPManager that the unit tests skip:
  - connect_server (success, failure, timeout, cancellation)
  - disconnect_server, disconnect_all
  - connect_all with mixed results
  - Cancel scope contamination isolation (the /reload bug)
  - _cleanup_connection edge cases
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dazi.mcp_client import (
    MCPManager,
    MCPServerConfig,
    MCPServerConnection,
    MCPServerStatus,
    MCPServerTool,
    _clear_task_cancellation,
    _force_cleanup_stale_scopes,
)

# ─────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────


def _make_config(name: str = "test-server") -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        command="echo",
        args=["hello"],
    )


def _make_connected_conn(
    name: str = "test-server",
    tool_count: int = 1,
) -> MCPServerConnection:
    """Create a connection that looks connected with mocked internals."""
    config = _make_config(name)
    conn = MCPServerConnection(config=config, status=MCPServerStatus.CONNECTED)
    # Mock SDK objects — must be AsyncMock, not None
    conn._session = MagicMock()
    conn._cm_stack = AsyncMock()
    conn._stdio_cm = AsyncMock()
    # Add fake tools
    conn.tools = [
        MCPServerTool(
            server_name=name,
            name=f"tool_{i}",
            qualified_name=f"mcp__{name}__tool_{i}",
            description=f"Test tool {i}",
            input_schema={},
        )
        for i in range(tool_count)
    ]
    return conn


def _patch_mcp_sdk(*, connect_side_effect=None, session_mock=None):
    """Patch MCP SDK imports that happen INSIDE connect_server.

    Because StdioServerParameters, stdio_client, and ClientSession are
    lazy-imported inside the function body, we must patch the source modules.
    """
    if session_mock is None:
        session_mock = AsyncMock()
        session_mock.initialize = AsyncMock()
        session_mock.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        session_mock.list_resources = AsyncMock(return_value=MagicMock(resources=[]))

    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=session_mock)

    mock_stdio_cm = AsyncMock()
    if connect_side_effect:
        mock_stdio_cm.__aenter__ = AsyncMock(side_effect=connect_side_effect)
    else:
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=("read", "write"))

    return patch.multiple(
        "mcp.client.stdio",
        StdioServerParameters=MagicMock(),
        stdio_client=MagicMock(return_value=mock_stdio_cm),
    ), patch("mcp.ClientSession", return_value=mock_session_cm)


# ─────────────────────────────────────────────────────────
# connect_server — SUCCESS
# ─────────────────────────────────────────────────────────


class TestConnectServerSuccess:
    """connect_server succeeds when the MCP subprocess connects properly."""

    @pytest.mark.asyncio
    async def test_connect_server_success(self):
        mgr = MCPManager()
        mgr.add_server(_make_config("srv"))

        sdk_patches, session_patches = _patch_mcp_sdk()
        with sdk_patches, session_patches:
            result = await mgr.connect_server("srv")

        assert result is True
        conn = mgr.get_server("srv")
        assert conn.status == MCPServerStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_server_unknown(self):
        """Connecting to an unregistered server returns False."""
        mgr = MCPManager()
        result = await mgr.connect_server("nonexistent")
        assert result is False


# ─────────────────────────────────────────────────────────
# connect_server — FAILURES
# ─────────────────────────────────────────────────────────


class TestConnectServerFailures:
    """connect_server handles failure modes gracefully."""

    @pytest.mark.asyncio
    async def test_connect_server_exception(self):
        """Exception during subprocess spawn -> error status, return False."""
        mgr = MCPManager()
        mgr.add_server(_make_config("bad"))

        sdk_patches, session_patches = _patch_mcp_sdk(
            connect_side_effect=RuntimeError("spawn failed")
        )
        with sdk_patches, session_patches:
            result = await mgr.connect_server("bad")

        assert result is False
        conn = mgr.get_server("bad")
        assert conn.status == MCPServerStatus.ERROR
        assert "spawn failed" in conn.error

    @pytest.mark.asyncio
    async def test_connect_server_session_init_fails(self):
        """Exception during session initialize -> error status, return False."""
        mgr = MCPManager()
        mgr.add_server(_make_config("bad"))

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(side_effect=RuntimeError("init fail"))
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))

        sdk_patches, session_patches = _patch_mcp_sdk(session_mock=mock_session)
        with sdk_patches, session_patches:
            result = await mgr.connect_server("bad")

        assert result is False
        conn = mgr.get_server("bad")
        assert conn.status == MCPServerStatus.ERROR
        assert "init fail" in conn.error


# ─────────────────────────────────────────────────────────
# disconnect_server / disconnect_all
# ─────────────────────────────────────────────────────────


class TestDisconnect:
    """disconnect_server and disconnect_all clean up properly."""

    @pytest.mark.asyncio
    async def test_disconnect_connected_server(self):
        mgr = MCPManager()
        conn = _make_connected_conn("srv")
        mgr._servers["srv"] = conn
        mgr._tool_map["mcp__srv__tool_0"] = conn.tools[0]

        await mgr.disconnect_server("srv")

        assert conn.status == MCPServerStatus.DISCONNECTED
        assert conn.tools == []
        assert "mcp__srv__tool_0" not in mgr._tool_map
        # Verify cleanup was called
        assert conn._cm_stack is None
        assert conn._stdio_cm is None

    @pytest.mark.asyncio
    async def test_disconnect_unknown_server(self):
        """Disconnecting unknown server is a no-op."""
        mgr = MCPManager()
        await mgr.disconnect_server("nonexistent")  # no error

    @pytest.mark.asyncio
    async def test_disconnect_already_disconnected(self):
        """Disconnecting an already-disconnected server is a no-op."""
        mgr = MCPManager()
        config = _make_config("srv")
        conn = MCPServerConnection(config=config, status=MCPServerStatus.DISCONNECTED)
        mgr._servers["srv"] = conn

        await mgr.disconnect_server("srv")
        # Should not attempt cleanup
        assert conn._cm_stack is None

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        """disconnect_all disconnects all servers."""
        mgr = MCPManager()
        conn1 = _make_connected_conn("srv1")
        conn2 = _make_connected_conn("srv2")
        mgr._servers["srv1"] = conn1
        mgr._servers["srv2"] = conn2

        await mgr.disconnect_all()

        assert conn1.status == MCPServerStatus.DISCONNECTED
        assert conn2.status == MCPServerStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_all_handles_cleanup_errors(self):
        """If one server's cleanup fails, others still disconnect."""
        mgr = MCPManager()
        conn1 = _make_connected_conn("srv1")
        conn2 = _make_connected_conn("srv2")

        # First server's session cleanup raises RuntimeError
        conn1._cm_stack.__aexit__ = AsyncMock(side_effect=RuntimeError("close fail"))
        mgr._servers["srv1"] = conn1
        mgr._servers["srv2"] = conn2

        await mgr.disconnect_all()

        # First server is cleaned up (errors caught by _cleanup_connection)
        assert conn1._session is None
        # Second server is properly disconnected
        assert conn2.status == MCPServerStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_all_empty(self):
        """disconnect_all with no servers is a no-op."""
        mgr = MCPManager()
        await mgr.disconnect_all()  # no error


# ─────────────────────────────────────────────────────────
# connect_all — MIXED RESULTS
# ─────────────────────────────────────────────────────────


class TestConnectAll:
    """connect_all handles mixed success/failure per server."""

    @pytest.mark.asyncio
    async def test_connect_all_mixed(self):
        mgr = MCPManager()
        mgr.add_server(_make_config("ok"))
        mgr.add_server(_make_config("bad"))

        async def _fake_connect(name):
            return name == "ok"

        with patch.object(mgr, "connect_server", side_effect=_fake_connect):
            results = await mgr.connect_all()

        assert results == {"ok": True, "bad": False}

    @pytest.mark.asyncio
    async def test_connect_all_empty(self):
        mgr = MCPManager()
        results = await mgr.connect_all()
        assert results == {}


# ─────────────────────────────────────────────────────────
# _cleanup_connection — EDGE CASES
# ─────────────────────────────────────────────────────────


class TestCleanupConnection:
    """_cleanup_connection is robust to internal errors."""

    @pytest.mark.asyncio
    async def test_cleanup_with_session_error(self):
        mgr = MCPManager()
        conn = _make_connected_conn()
        conn._cm_stack.__aexit__ = AsyncMock(side_effect=RuntimeError("session close fail"))

        await mgr._cleanup_connection(conn)

        assert conn._session is None
        assert conn._cm_stack is None

    @pytest.mark.asyncio
    async def test_cleanup_with_stdio_error(self):
        mgr = MCPManager()
        conn = _make_connected_conn()
        conn._stdio_cm.__aexit__ = AsyncMock(side_effect=RuntimeError("transport close fail"))

        await mgr._cleanup_connection(conn)

        assert conn._stdio_cm is None

    @pytest.mark.asyncio
    async def test_cleanup_with_both_errors(self):
        """Both session and transport errors don't prevent cleanup."""
        mgr = MCPManager()
        conn = _make_connected_conn()
        conn._cm_stack.__aexit__ = AsyncMock(side_effect=RuntimeError("session"))
        conn._stdio_cm.__aexit__ = AsyncMock(side_effect=RuntimeError("transport"))

        await mgr._cleanup_connection(conn)

        assert conn._session is None
        assert conn._cm_stack is None
        assert conn._stdio_cm is None

    @pytest.mark.asyncio
    async def test_cleanup_with_cancelled_error(self):
        """CancelledError during cleanup is caught (BaseException handler)."""
        mgr = MCPManager()
        conn = _make_connected_conn()
        conn._cm_stack.__aexit__ = AsyncMock(side_effect=asyncio.CancelledError())

        await mgr._cleanup_connection(conn)

        assert conn._session is None
        assert conn._cm_stack is None

    @pytest.mark.asyncio
    async def test_cleanup_nothing_to_clean(self):
        """Cleanup with None internals is a no-op."""
        mgr = MCPManager()
        conn = MCPServerConnection(config=_make_config())
        conn._session = None
        conn._cm_stack = None
        conn._stdio_cm = None

        await mgr._cleanup_connection(conn)  # no error


# ─────────────────────────────────────────────────────────
# RECONNECT SCENARIO (the /reload bug)
# ─────────────────────────────────────────────────────────


class TestReconnectScenario:
    """Verify that disconnect → reconnect works without cancel scope contamination.

    The /reload bug: disconnect_all exits anyio cancel scopes that contaminate
    the task's state. The fix is to run connect_mcp_servers in a separate task
    via asyncio.create_task().
    """

    @pytest.mark.asyncio
    async def test_disconnect_then_reconnect_in_fresh_task(self):
        """Simulate /reload: disconnect all, then reconnect in fresh task."""
        mgr = MCPManager()

        # Add and "connect" servers
        conn1 = _make_connected_conn("srv1")
        conn2 = _make_connected_conn("srv2")
        mgr._servers["srv1"] = conn1
        mgr._servers["srv2"] = conn2

        # Disconnect all
        await mgr.disconnect_all()
        assert conn1.status == MCPServerStatus.DISCONNECTED
        assert conn2.status == MCPServerStatus.DISCONNECTED

        # Re-register fresh configs (simulates connect_mcp_servers behavior)
        mgr.add_server(_make_config("srv1"))
        mgr.add_server(_make_config("srv2"))

        # Reconnect in a fresh task (the fix for cancel scope contamination)
        sdk_patches, session_patches = _patch_mcp_sdk()
        with sdk_patches, session_patches:
            results = await asyncio.create_task(mgr.connect_all())

        assert results["srv1"] is True
        assert results["srv2"] is True
        assert mgr.get_server("srv1").status == MCPServerStatus.CONNECTED
        assert mgr.get_server("srv2").status == MCPServerStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_reconnect_in_same_task_works_with_mocked_sdk(self):
        """With mocked SDK (no real anyio), reconnect in same task also works.

        The real bug only manifests with actual anyio cancel scopes.
        This test validates the mock setup is correct.
        """
        mgr = MCPManager()
        conn = _make_connected_conn("srv")
        mgr._servers["srv"] = conn

        await mgr.disconnect_all()
        mgr.add_server(_make_config("srv"))

        sdk_patches, session_patches = _patch_mcp_sdk()
        with sdk_patches, session_patches:
            results = await mgr.connect_all()

        assert results["srv"] is True


# ─────────────────────────────────────────────────────────
# TOOL EXECUTION — ERROR PATHS
# ─────────────────────────────────────────────────────────


class TestCallToolErrors:
    """call_tool handles missing/disconnected servers."""

    @pytest.mark.asyncio
    async def test_call_tool_unknown_server(self):
        from dazi.mcp_client import MCPConnectionError

        mgr = MCPManager()
        with pytest.raises(MCPConnectionError, match="not registered"):
            await mgr.call_tool("mcp__unknown__tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_disconnected_server(self):
        from dazi.mcp_client import MCPConnectionError

        mgr = MCPManager()
        config = _make_config("srv")
        conn = MCPServerConnection(config=config, status=MCPServerStatus.DISCONNECTED)
        mgr._servers["srv"] = conn

        with pytest.raises(MCPConnectionError, match="not connected"):
            await mgr.call_tool("mcp__srv__tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_no_session(self):
        from dazi.mcp_client import MCPConnectionError

        mgr = MCPManager()
        config = _make_config("srv")
        conn = MCPServerConnection(config=config, status=MCPServerStatus.CONNECTED)
        conn._session = None
        mgr._servers["srv"] = conn

        with pytest.raises(MCPConnectionError, match="no active session"):
            await mgr.call_tool("mcp__srv__tool", {})


# ─────────────────────────────────────────────────────────
# _clear_task_cancellation
# ─────────────────────────────────────────────────────────


class TestClearTaskCancellation:
    """Verify _clear_task_cancellation clears pending cancels.

    Each test runs the logic in a separate task so that cancelling
    doesn't propagate to the pytest-asyncio runner.
    """

    @pytest.mark.asyncio
    async def test_clears_pending_cancel(self):
        async def _body():
            task = asyncio.current_task()
            task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                _clear_task_cancellation()
            assert task.cancelling() == 0
            return "ok"

        assert await asyncio.create_task(_body()) == "ok"

    @pytest.mark.asyncio
    async def test_no_op_when_not_cancelled(self):
        async def _body():
            task = asyncio.current_task()
            assert task.cancelling() == 0
            _clear_task_cancellation()
            assert task.cancelling() == 0
            return "ok"

        assert await asyncio.create_task(_body()) == "ok"

    @pytest.mark.asyncio
    async def test_survives_multiple_cancels(self):
        async def _body():
            task = asyncio.current_task()
            task.cancel()
            task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                _clear_task_cancellation()
            assert task.cancelling() == 0
            return "ok"

        assert await asyncio.create_task(_body()) == "ok"

    @pytest.mark.asyncio
    async def test_allows_await_after_clear(self):
        """After clearing, the task can await without CancelledError."""

        async def _body():
            task = asyncio.current_task()
            task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                _clear_task_cancellation()
            # This await should NOT raise CancelledError
            await asyncio.sleep(0)
            return "ok"

        assert await asyncio.create_task(_body()) == "ok"


# ─────────────────────────────────────────────────────────
# _force_cleanup_stale_scopes
# ─────────────────────────────────────────────────────────


class TestForceCleanupStaleScopes:
    """Verify _force_cleanup_stale_scopes cleans stale anyio cancel scopes.

    The function walks UP from ts.cancel_scope via _parent_scope, so the
    stale scope must be ts.cancel_scope or an ancestor of it.
    """

    @pytest.mark.asyncio
    async def test_cleans_active_cancelled_scopes(self):
        """Scopes with active=True and cancel_called=True are force-cleaned."""
        from unittest.mock import MagicMock

        from anyio._backends._asyncio import CancelScope, TaskState

        async def _body():
            task = asyncio.current_task()
            # Stale scope IS ts.cancel_scope (the traversal start point)
            stale_scope = CancelScope()
            stale_scope._active = True
            stale_scope._cancel_called = True
            stale_scope._host_task = task
            stale_scope._tasks = {task}
            stale_scope._child_scopes = set()
            stale_scope._cancel_handle = MagicMock()

            parent_scope = CancelScope()
            parent_scope._active = True
            parent_scope._cancel_called = False
            parent_scope._host_task = task
            stale_scope._parent_scope = parent_scope
            parent_scope._child_scopes.add(stale_scope)

            ts = TaskState(None, stale_scope)
            from anyio._backends._asyncio import _task_states as ts_map

            ts_map[task] = ts

            _force_cleanup_stale_scopes()

            assert stale_scope._active is False
            assert stale_scope._host_task is None
            assert stale_scope._cancel_handle is None
            assert task not in stale_scope._tasks

            # Cleanup restored so task can await
            await asyncio.sleep(0)
            return "ok"

        assert await asyncio.create_task(_body()) == "ok"

    @pytest.mark.asyncio
    async def test_preserves_non_cancelled_active_scopes(self):
        """Scopes with active=True but cancel_called=False are left alone."""
        from anyio._backends._asyncio import CancelScope, TaskState

        async def _body():
            task = asyncio.current_task()
            scope = CancelScope()
            scope._active = True
            scope._cancel_called = False
            scope._host_task = task

            ts = TaskState(None, scope)
            from anyio._backends._asyncio import _task_states as ts_map

            ts_map[task] = ts

            _force_cleanup_stale_scopes()

            assert scope._active is True
            assert scope._host_task is task
            return "ok"

        assert await asyncio.create_task(_body()) == "ok"

    @pytest.mark.asyncio
    async def test_rewires_child_scopes(self):
        """Child scopes of cleaned scopes are rewired to the grandparent."""
        from unittest.mock import MagicMock

        from anyio._backends._asyncio import CancelScope, TaskState

        async def _body():
            task = asyncio.current_task()
            grandparent = CancelScope()
            grandparent._active = True
            grandparent._cancel_called = False
            grandparent._host_task = task

            stale = CancelScope()
            stale._parent_scope = grandparent
            stale._active = True
            stale._cancel_called = True
            stale._host_task = task
            stale._tasks = {task}
            stale._child_scopes = set()
            stale._cancel_handle = MagicMock()
            grandparent._child_scopes.add(stale)

            child = CancelScope()
            child._parent_scope = stale
            child._active = True
            child._cancel_called = False
            child._host_task = task
            stale._child_scopes.add(child)

            # stale is ts.cancel_scope, grandparent is its parent
            ts = TaskState(None, stale)
            from anyio._backends._asyncio import _task_states as ts_map

            ts_map[task] = ts

            _force_cleanup_stale_scopes()

            # Child was rewired to grandparent
            assert child._parent_scope is grandparent
            assert child in grandparent._child_scopes
            assert len(stale._child_scopes) == 0
            return "ok"

        assert await asyncio.create_task(_body()) == "ok"

    @pytest.mark.asyncio
    async def test_clears_pending_cancellation(self):
        """After cleanup, the task can await without CancelledError."""
        from unittest.mock import MagicMock

        from anyio._backends._asyncio import CancelScope, TaskState

        async def _body():
            task = asyncio.current_task()
            task.cancel()

            scope = CancelScope()
            scope._active = True
            scope._cancel_called = True
            scope._host_task = task
            scope._tasks = {task}
            scope._child_scopes = set()
            scope._cancel_handle = MagicMock()

            ts = TaskState(None, scope)
            from anyio._backends._asyncio import _task_states as ts_map

            ts_map[task] = ts

            # Flush the CancelledError from task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                pass

            _force_cleanup_stale_scopes()

            assert task.cancelling() == 0
            # This should NOT raise
            await asyncio.sleep(0)
            return "ok"

        assert await asyncio.create_task(_body()) == "ok"

    @pytest.mark.asyncio
    async def test_no_op_when_no_stale_scopes(self):
        """No stale scopes -> no-op, task still works."""
        from anyio._backends._asyncio import CancelScope, TaskState

        async def _body():
            task = asyncio.current_task()
            scope = CancelScope()
            scope._active = True
            scope._cancel_called = False
            scope._host_task = task

            ts = TaskState(None, scope)
            from anyio._backends._asyncio import _task_states as ts_map

            ts_map[task] = ts

            _force_cleanup_stale_scopes()

            assert scope._active is True
            await asyncio.sleep(0)
            return "ok"

        assert await asyncio.create_task(_body()) == "ok"
