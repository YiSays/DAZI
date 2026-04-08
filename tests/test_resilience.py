"""Tests for dazi/resilience.py — abort signal, circuit breaker, retry."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from dazi.resilience import (
    AbortError,
    AbortSignal,
    CircuitBreaker,
    CircuitOpenError,
    MaxRetriesError,
    RetryPolicy,
    _calculate_delay,
    _is_retryable,
    with_retry,
)


# ─────────────────────────────────────────────────────────
# AbortSignal
# ─────────────────────────────────────────────────────────


class TestAbortSignal:
    def test_initial_not_aborted(self):
        sig = AbortSignal()
        assert sig.aborted is False

    def test_abort_sets_flag(self):
        sig = AbortSignal()
        sig.abort()
        assert sig.aborted is True

    def test_check_raises_when_aborted(self):
        sig = AbortSignal()
        sig.abort()
        with pytest.raises(AbortError, match="Operation aborted"):
            sig.check()

    def test_check_passes_when_not_aborted(self):
        sig = AbortSignal()
        sig.check()  # Should not raise

    @pytest.mark.asyncio
    async def test_async_check_raises_when_aborted(self):
        sig = AbortSignal()
        sig.abort()
        with pytest.raises(AbortError, match="Operation aborted"):
            await sig.async_check()

    @pytest.mark.asyncio
    async def test_async_check_passes_when_not_aborted(self):
        sig = AbortSignal()
        await sig.async_check()  # Should not raise


# ─────────────────────────────────────────────────────────
# CircuitBreaker
# ─────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_initial_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_trips_on_failures(self):
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"
        cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.0)
        cb.record_failure()
        cb.record_failure()
        # With cooldown=0.0, time.time() >= _open_until is immediately true,
        # so the breaker transitions to half_open right away
        assert cb.state == "half_open"
        assert cb.allow_request() is True

    def test_record_success_resets(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_allow_request_open(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=300.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False


# ─────────────────────────────────────────────────────────
# RetryPolicy
# ─────────────────────────────────────────────────────────


class TestRetryPolicy:
    def test_defaults(self):
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 30.0
        assert policy.jitter is True

    def test_delay_calculation_no_jitter(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=30.0, jitter=False)
        assert _calculate_delay(0, policy) == 1.0
        assert _calculate_delay(1, policy) == 2.0
        assert _calculate_delay(2, policy) == 4.0

    def test_delay_capped_at_max(self):
        policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=False)
        delay = _calculate_delay(20, policy)
        assert delay <= 10.0


# ─────────────────────────────────────────────────────────
# _is_retryable
# ─────────────────────────────────────────────────────────


class TestIsRetryable:
    def test_connection_error(self):
        policy = RetryPolicy()
        assert _is_retryable(ConnectionError("lost"), policy) is True

    def test_rate_limit_429(self):
        policy = RetryPolicy()
        assert _is_retryable(Exception("Error 429: rate limit exceeded"), policy) is True

    def test_rate_limit_text(self):
        policy = RetryPolicy()
        assert _is_retryable(Exception("rate limit hit"), policy) is True

    def test_value_error_not_retryable(self):
        policy = RetryPolicy()
        assert _is_retryable(ValueError("bad input"), policy) is False

    def test_timeout_error(self):
        policy = RetryPolicy()
        assert _is_retryable(TimeoutError("timed out"), policy) is True

    def test_os_error(self):
        policy = RetryPolicy()
        assert _is_retryable(OSError("io error"), policy) is True


# ─────────────────────────────────────────────────────────
# with_retry
# ─────────────────────────────────────────────────────────


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        func = AsyncMock(return_value="ok")
        result = await with_retry(func, policy=RetryPolicy(max_retries=2, jitter=False))
        assert result == "ok"
        assert func.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_and_succeeds(self):
        policy = RetryPolicy(max_retries=3, base_delay=0.01, jitter=False)
        call_count = 0

        async def flaky(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary")
            return "recovered"

        result = await with_retry(flaky, policy=policy)
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_exceeded_raises(self):
        policy = RetryPolicy(max_retries=2, base_delay=0.01, jitter=False)

        async def always_fail(**kwargs):
            raise ConnectionError("always down")

        with pytest.raises(MaxRetriesError, match="Failed after"):
            await with_retry(always_fail, policy=policy)

    @pytest.mark.asyncio
    async def test_abort_raises_immediately(self):
        policy = RetryPolicy(max_retries=3)
        sig = AbortSignal()
        sig.abort()

        async def func(**kwargs):
            return "never"

        with pytest.raises(AbortError):
            await with_retry(func, policy=policy, abort=sig)

    @pytest.mark.asyncio
    async def test_circuit_open_raises(self):
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=300.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        async def func(**kwargs):
            return "never"

        with pytest.raises(CircuitOpenError):
            await with_retry(func, circuit=cb)
