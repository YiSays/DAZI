# DAZI — Lessons Learned

## anyio Cancel Scope Contamination on MCP Disconnect

### The Bug

The `/reload` command crashed with `CancelledError` when reconnecting MCP servers. Initial connection worked fine — only disconnect-then-reconnect failed.

### Root Cause

The MCP SDK's `BaseSession.__aexit__()` always calls `self._task_group.cancel_scope.cancel()` on exit. When our `_cleanup_connection()` exits the session/stdio context managers and catches the resulting exceptions, stale anyio cancel scopes remain on the task's scope stack with:

- `_active = True` and `_cancel_called = True`
- `_tasks` still containing the host task
- `_cancel_handle` with a pending `_deliver_cancellation` retry callback

The `_deliver_cancellation` retry chain is infinite: every event loop iteration, it calls `task.cancel()` and schedules another retry via `loop.call_soon`. This makes `task.uncancel()` alone insufficient — the retry immediately re-cancels.

### The Fix

`_force_cleanup_stale_scopes()` in `dazi/mcp_client.py` walks the task's cancel scope chain from `ts.cancel_scope` upward via `_parent_scope`, and for each scope with both `_active` and `_cancel_called` set:

1. Rewires child scopes to the grandparent (so their `__exit__` won't re-add the host task to the cleaned scope)
2. Removes the host task from `_tasks`
3. Cancels the `_cancel_handle` retry callback
4. Marks the scope as inactive

Then `_clear_task_cancellation()` drains any remaining `task.cancel()` count.

### Key Insights

- **Python 3.12 dual-cancel state**: `task.cancel()` sets BOTH `_num_cancels_requested` AND `_must_cancel`. `task.uncancel()` only decrements the former. The latter is only cleared by `Task.__step__()` when it throws `CancelledError`. This means calling `uncancel()` doesn't prevent the *next* step from raising `CancelledError` if `_must_cancel` is set.

- **anyio `_deliver_cancellation` retry**: The retry handle uses `loop.call_soon` and reschedules itself as long as `_tasks` is non-empty. Simply removing the task from `_tasks` and cancelling the handle is required — `uncancel()` alone is not enough.

- **Child scope rewiring is critical**: When a cleaned scope has children, those children's `__exit__` would call `_parent_scope._tasks.add(self._host_task)`, re-adding the task to the cleaned scope and re-triggering delivery. Rewiring children to skip over the cleaned scope prevents this.

- **Moving cleanup to a separate task doesn't help**: Cancel scopes are entered in the original task. `CancelScope.__exit__` checks `current_task() is not self._host_task` and the cleanup function operates on the wrong task's state.

- **`except BaseException` catches `CancelledError`**: In `_cleanup_connection()`, we use `except BaseException` to handle session/stdio close failures. This is correct but means anyio's scope exit exceptions are swallowed, leaving stale scopes behind.
