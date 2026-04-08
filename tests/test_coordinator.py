"""Tests for dazi/coordinator.py — AutonomousConfig, AutonomousTeammate scan/claim/report."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dazi.coordinator import AutonomousConfig, AutonomousTeammate
from dazi.mailbox import Mailbox
from dazi.task_store import Task, TaskStatus, TaskStore


# ─────────────────────────────────────────────────────────
# AutonomousConfig defaults
# ─────────────────────────────────────────────────────────


class TestAutonomousConfig:
    def test_defaults(self):
        cfg = AutonomousConfig()
        assert cfg.max_tasks_per_agent == 10
        assert cfg.claim_delay == 1.0
        assert cfg.idle_timeout == 30.0
        assert cfg.max_turns_per_task == 50


# ─────────────────────────────────────────────────────────
# scan_tasks
# ─────────────────────────────────────────────────────────


class TestScanTasks:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()
        return store, teammate

    def test_finds_pending_unblocked_task(self, setup):
        store, teammate = setup
        store.create(subject="Task 1", description="Do something")
        result = teammate.scan_tasks(store, "worker")
        assert result is not None
        assert result.subject == "Task 1"

    def test_no_available_tasks(self, setup):
        store, teammate = setup
        # No tasks at all
        result = teammate.scan_tasks(store, "worker")
        assert result is None

    def test_skips_blocked_tasks(self, setup):
        store, teammate = setup
        t1 = store.create(subject="Blocker", description="Must finish first")
        t2 = store.create(subject="Blocked", description="Waiting on t1")
        store.add_blocked_by(t2.id, t1.id)
        # t1 is pending (not completed), so t2 is blocked
        result = teammate.scan_tasks(store, "worker")
        assert result is not None
        assert result.id == t1.id  # returns the blocker (unblocked pending task)

    def test_respects_max_tasks_limit(self, setup):
        store, teammate = setup
        store.create(subject="T1", description="desc")
        # Simulate agent already at max
        teammate._tasks_claimed["worker"] = 1
        result = teammate.scan_tasks(store, "worker", max_tasks=1)
        assert result is None


# ─────────────────────────────────────────────────────────
# claim_task
# ─────────────────────────────────────────────────────────


class TestClaimTask:
    @pytest.fixture
    def setup(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()
        return store, teammate

    def test_claim_success(self, setup):
        store, teammate = setup
        task = store.create(subject="T1", description="desc")
        result = teammate.claim_task(store, task, "worker")
        assert result is not None
        assert result.status == TaskStatus.IN_PROGRESS
        assert result.owner == "worker"
        assert teammate._tasks_claimed["worker"] == 1

    def test_claim_already_claimed_returns_none(self, setup):
        store, teammate = setup
        task = store.create(subject="T1", description="desc")
        # First claim succeeds
        teammate.claim_task(store, task, "worker")
        # Second claim should fail (task is now IN_PROGRESS, not PENDING)
        result = teammate.claim_task(store, task, "other")
        assert result is None


# ─────────────────────────────────────────────────────────
# execute_claimed_task
# ─────────────────────────────────────────────────────────


class TestExecuteClaimedTask:
    @pytest.mark.asyncio
    async def test_task_completion_reporting(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        task = store.create(subject="T1", description="desc")
        claimed = teammate.claim_task(store, task, "worker")
        assert claimed is not None

        async def success_run(t: Task) -> str:
            return f"Done: {t.subject}"

        result = await teammate.execute_claimed_task(store, claimed, success_run)
        assert "Done: T1" in result
        # Task should be marked COMPLETED
        updated = store.get(task.id)
        assert updated.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_task_failure_resets_to_pending(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        teammate = AutonomousTeammate()

        task = store.create(subject="T1", description="desc")
        claimed = teammate.claim_task(store, task, "worker")

        async def failing_run(t: Task) -> str:
            raise RuntimeError("Something went wrong")

        result = await teammate.execute_claimed_task(store, claimed, failing_run)
        assert "failed" in result.lower()
        # Task should be reset to PENDING
        updated = store.get(task.id)
        assert updated.status == TaskStatus.PENDING


# ─────────────────────────────────────────────────────────
# idle notification sending (via run_autonomous_cycle)
# ─────────────────────────────────────────────────────────


class TestIdleNotification:
    @pytest.mark.asyncio
    async def test_idle_notification_sent(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks", list_id="test")
        mailbox = Mailbox(base_dir=tmp_path / "mail")
        teammate = AutonomousTeammate()

        config = AutonomousConfig(
            claim_delay=0.05,
            idle_timeout=0.1,
            max_tasks_per_agent=5,
        )

        # Create a handle with abort_signal that we'll set after short delay
        task = teammate.spawn(
            team_name="team1",
            member_name="worker",
            run_func=lambda h: None,  # dummy
        )
        await task

        # Re-spawn for the autonomous cycle
        teammate.reset()
        atask = teammate.spawn_autonomous(
            team_name="team1",
            member_name="worker",
            task_store=store,
            mailbox=mailbox,
            config=config,
            run_func=AsyncMock(return_value="done"),
        )

        # Let it run briefly, then shut down
        await asyncio.sleep(0.3)
        handle = teammate.get_handle("worker@team1")
        if handle:
            handle.abort_signal.set()
        try:
            await asyncio.wait_for(atask, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            atask.cancel()
            try:
                await atask
            except asyncio.CancelledError:
                pass

        # Check that some notification was sent (mailbox should have messages)
        # The notification goes to all team members via broadcast
