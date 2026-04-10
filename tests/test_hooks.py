"""Tests for dazi/hooks.py — hook events, results, and registry."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dazi.hooks import HookEvent, HookRegistry, HookResult
from dazi.permissions import PermissionBehavior

# ─────────────────────────────────────────────────────────
# HookEvent enum values
# ─────────────────────────────────────────────────────────


class TestHookEvent:
    def test_pre_tool_use_value(self):
        assert HookEvent.PRE_TOOL_USE == "pre_tool_use"

    def test_post_tool_use_value(self):
        assert HookEvent.POST_TOOL_USE == "post_tool_use"

    def test_post_tool_use_failure_value(self):
        assert HookEvent.POST_TOOL_USE_FAILURE == "post_tool_use_failure"

    def test_user_prompt_submit_value(self):
        assert HookEvent.USER_PROMPT_SUBMIT == "user_prompt_submit"

    def test_session_start_value(self):
        assert HookEvent.SESSION_START == "session_start"

    def test_all_events_count(self):
        assert len(HookEvent) == 5


# ─────────────────────────────────────────────────────────
# HookResult.merge()
# ─────────────────────────────────────────────────────────


class TestHookResultMerge:
    def test_merge_empty(self):
        result = HookResult.merge()
        assert result.modified_input is None
        assert result.modified_output is None
        assert result.permission_override is None
        assert result.should_block is False

    def test_merge_modified_input(self):
        r = HookResult(modified_input={"key": "value"})
        merged = HookResult.merge(r)
        assert merged.modified_input == {"key": "value"}

    def test_merge_modified_output(self):
        r = HookResult(modified_output="new output")
        merged = HookResult.merge(r)
        assert merged.modified_output == "new output"

    def test_merge_permission_override(self):
        r = HookResult(permission_override=PermissionBehavior.ALLOW)
        merged = HookResult.merge(r)
        assert merged.permission_override == PermissionBehavior.ALLOW

    def test_merge_should_block(self):
        r = HookResult(should_block=True, block_reason="forbidden")
        merged = HookResult.merge(r)
        assert merged.should_block is True
        assert merged.block_reason == "forbidden"

    def test_merge_later_overrides_earlier(self):
        r1 = HookResult(modified_input={"a": 1}, modified_output="first")
        r2 = HookResult(modified_input={"b": 2})
        merged = HookResult.merge(r1, r2)
        assert merged.modified_input == {"b": 2}
        assert merged.modified_output == "first"

    def test_merge_block_propagates(self):
        r1 = HookResult(should_block=False)
        r2 = HookResult(should_block=True, block_reason="stop")
        r3 = HookResult(should_block=False)
        merged = HookResult.merge(r1, r2, r3)
        assert merged.should_block is True
        assert merged.block_reason == "stop"


# ─────────────────────────────────────────────────────────
# HookRegistry
# ─────────────────────────────────────────────────────────


class TestHookRegistry:
    def test_register_handler(self):
        registry = HookRegistry()
        handler = AsyncMock(return_value=HookResult())
        registry.register(HookEvent.PRE_TOOL_USE, handler)
        assert handler in registry.get_handlers(HookEvent.PRE_TOOL_USE)

    def test_register_priority_sorted(self):
        registry = HookRegistry()
        h_low = AsyncMock(return_value=HookResult())
        h_high = AsyncMock(return_value=HookResult())
        registry.register(HookEvent.PRE_TOOL_USE, h_high, priority=10)
        registry.register(HookEvent.PRE_TOOL_USE, h_low, priority=1)
        handlers = registry.get_handlers(HookEvent.PRE_TOOL_USE)
        assert handlers[0] is h_low
        assert handlers[1] is h_high

    def test_unregister_found(self):
        registry = HookRegistry()
        handler = AsyncMock(return_value=HookResult())
        registry.register(HookEvent.PRE_TOOL_USE, handler)
        assert registry.unregister(HookEvent.PRE_TOOL_USE, handler) is True
        assert handler not in registry.get_handlers(HookEvent.PRE_TOOL_USE)

    def test_unregister_not_found(self):
        registry = HookRegistry()
        handler = AsyncMock(return_value=HookResult())
        assert registry.unregister(HookEvent.PRE_TOOL_USE, handler) is False

    def test_unregister_missing_event(self):
        registry = HookRegistry()
        handler = AsyncMock(return_value=HookResult())
        assert registry.unregister(HookEvent.SESSION_START, handler) is False

    def test_clear_specific_event(self):
        registry = HookRegistry()
        registry.register(HookEvent.PRE_TOOL_USE, AsyncMock(return_value=HookResult()))
        registry.register(HookEvent.POST_TOOL_USE, AsyncMock(return_value=HookResult()))
        registry.clear(HookEvent.PRE_TOOL_USE)
        assert registry.get_handlers(HookEvent.PRE_TOOL_USE) == []
        assert len(registry.get_handlers(HookEvent.POST_TOOL_USE)) == 1

    def test_clear_all_events(self):
        registry = HookRegistry()
        registry.register(HookEvent.PRE_TOOL_USE, AsyncMock(return_value=HookResult()))
        registry.register(HookEvent.POST_TOOL_USE, AsyncMock(return_value=HookResult()))
        registry.clear()
        assert registry.get_handlers(HookEvent.PRE_TOOL_USE) == []
        assert registry.get_handlers(HookEvent.POST_TOOL_USE) == []

    @pytest.mark.asyncio
    async def test_fire_no_handlers_returns_default(self):
        registry = HookRegistry()
        result = await registry.fire(HookEvent.SESSION_START)
        assert isinstance(result, HookResult)
        assert result.modified_input is None
        assert result.should_block is False

    @pytest.mark.asyncio
    async def test_fire_merges_multiple_results(self):
        registry = HookRegistry()
        registry.register(
            HookEvent.PRE_TOOL_USE,
            AsyncMock(return_value=HookResult(modified_input={"a": 1})),
            priority=0,
        )
        registry.register(
            HookEvent.PRE_TOOL_USE,
            AsyncMock(return_value=HookResult(modified_input={"b": 2})),
            priority=1,
        )
        result = await registry.fire(HookEvent.PRE_TOOL_USE)
        assert result.modified_input == {"b": 2}

    @pytest.mark.asyncio
    async def test_fire_exception_does_not_crash(self):
        registry = HookRegistry()

        async def bad_handler(**kwargs):
            raise RuntimeError("boom")

        registry.register(HookEvent.PRE_TOOL_USE, bad_handler)
        result = await registry.fire(HookEvent.PRE_TOOL_USE)
        assert isinstance(result, HookResult)
        assert result.should_block is False
