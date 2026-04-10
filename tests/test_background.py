"""Tests for dazi/background.py — BackgroundTaskStatus, BackgroundTaskManager lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dazi.background import (
    BackgroundTask,
    BackgroundTaskManager,
    BackgroundTaskStatus,
)

# ─────────────────────────────────────────────────────────
# BackgroundTaskStatus enum
# ─────────────────────────────────────────────────────────


class TestBackgroundTaskStatus:
    def test_all_values(self):
        assert BackgroundTaskStatus.PENDING == "pending"
        assert BackgroundTaskStatus.RUNNING == "running"
        assert BackgroundTaskStatus.COMPLETED == "completed"
        assert BackgroundTaskStatus.FAILED == "failed"
        assert BackgroundTaskStatus.KILLED == "killed"


# ─────────────────────────────────────────────────────────
# BackgroundTask dataclass
# ─────────────────────────────────────────────────────────


class TestBackgroundTask:
    def test_to_dict_excludes_process(self):
        task = BackgroundTask(id="t1", command="echo hi")
        d = task.to_dict()
        assert "id" in d
        assert "command" in d
        assert "process" not in d

    def test_from_dict_restores_fields(self):
        d = {
            "id": "t2",
            "command": "ls",
            "status": "completed",
            "exit_code": 0,
        }
        task = BackgroundTask.from_dict(d)
        assert task.id == "t2"
        assert task.command == "ls"
        assert task.status == BackgroundTaskStatus.COMPLETED
        assert task.process is None

    def test_is_terminal(self):
        assert BackgroundTask(id="", command="", status=BackgroundTaskStatus.COMPLETED).is_terminal
        assert BackgroundTask(id="", command="", status=BackgroundTaskStatus.FAILED).is_terminal
        assert BackgroundTask(id="", command="", status=BackgroundTaskStatus.KILLED).is_terminal
        assert not BackgroundTask(
            id="", command="", status=BackgroundTaskStatus.RUNNING
        ).is_terminal
        assert not BackgroundTask(
            id="", command="", status=BackgroundTaskStatus.PENDING
        ).is_terminal


# ─────────────────────────────────────────────────────────
# BackgroundTaskManager with real subprocesses (fast commands)
# ─────────────────────────────────────────────────────────


class TestBackgroundTaskManager:
    @pytest.fixture
    def manager(self, tmp_path: Path) -> BackgroundTaskManager:
        return BackgroundTaskManager(tmp_path / "output")

    @pytest.mark.asyncio
    async def test_submit_creates_running_task(self, manager: BackgroundTaskManager):
        task_id = await manager.submit("echo hello", description="test echo")
        assert task_id.startswith("bash_")
        task = await manager.check(task_id)
        assert task is not None
        assert task.status == BackgroundTaskStatus.RUNNING
        assert task.pid is not None
        # Wait for completion
        for _ in range(50):
            await asyncio.sleep(0.05)
            task = manager.check_sync(task_id)
            if task and task.is_terminal:
                break
        assert task.is_terminal
        manager.reset()

    @pytest.mark.asyncio
    async def test_check_returns_none_for_unknown(self, manager: BackgroundTaskManager):
        result = await manager.check("nonexistent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_active(self, manager: BackgroundTaskManager):
        task_id = await manager.submit("echo hi")
        active = manager.list_active()
        assert len(active) >= 1
        assert any(t.id == task_id for t in active)
        # Wait for it to finish before cleanup
        for _ in range(50):
            await asyncio.sleep(0.05)
            if not manager.list_active():
                break
        manager.reset()

    @pytest.mark.asyncio
    async def test_list_all(self, manager: BackgroundTaskManager):
        task_id = await manager.submit("echo test")
        all_tasks = manager.list_all()
        assert any(t.id == task_id for t in all_tasks)
        for _ in range(50):
            await asyncio.sleep(0.05)
            if not manager.list_active():
                break
        manager.reset()

    @pytest.mark.asyncio
    async def test_collect_completed_dedup(self, manager: BackgroundTaskManager):
        task_id = await manager.submit("echo done")
        # Wait for completion
        for _ in range(50):
            await asyncio.sleep(0.05)
            t = manager.check_sync(task_id)
            if t and t.is_terminal:
                break
        first_batch = manager.collect_completed()
        second_batch = manager.collect_completed()
        assert len(first_batch) >= 1
        assert len(second_batch) == 0  # dedup: notified=True set by first call
        manager.reset()

    @pytest.mark.asyncio
    async def test_output_reading(self, manager: BackgroundTaskManager):
        task_id = await manager.submit("echo output_test")
        # Wait for completion
        for _ in range(50):
            await asyncio.sleep(0.05)
            t = manager.check_sync(task_id)
            if t and t.is_terminal:
                break
        output = manager.get_output(task_id)
        assert "output_test" in output
        manager.reset()
