"""Tests for dazi/teammate.py — TeammateStatus, TeammateRunner lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from dazi.teammate import TeammateHandle, TeammateRunner, TeammateStatus

# ─────────────────────────────────────────────────────────
# TeammateStatus enum
# ─────────────────────────────────────────────────────────


class TestTeammateStatus:
    def test_all_status_values(self):
        assert TeammateStatus.SPAWNING == "spawning"
        assert TeammateStatus.ACTIVE == "active"
        assert TeammateStatus.IDLE == "idle"
        assert TeammateStatus.SHUTTING_DOWN == "shutting_down"
        assert TeammateStatus.COMPLETED == "completed"
        assert TeammateStatus.FAILED == "failed"


# ─────────────────────────────────────────────────────────
# TeammateRunner spawn
# ─────────────────────────────────────────────────────────


class TestTeammateRunnerSpawn:
    @pytest.mark.asyncio
    async def test_spawn_creates_handle_and_task(self):
        runner = TeammateRunner()
        task = runner.spawn(
            team_name="web-dev",
            member_name="frontend",
        )
        assert isinstance(task, asyncio.Task)
        handle = runner.get_handle("frontend@web-dev")
        assert handle is not None
        assert handle.name == "frontend"
        assert handle.agent_id == "frontend@web-dev"
        assert handle.team_name == "web-dev"
        assert handle.status == TeammateStatus.SPAWNING
        # Wait for the task to finish
        await task
        assert handle.status == TeammateStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_spawn_with_custom_run_func(self):
        runner = TeammateRunner()
        executed = []

        async def my_run(handle: TeammateHandle) -> None:
            executed.append(handle.agent_id)

        task = runner.spawn(
            team_name="team1",
            member_name="worker",
            run_func=my_run,
        )
        await task
        assert "worker@team1" in executed
        handle = runner.get_handle("worker@team1")
        assert handle.status == TeammateStatus.COMPLETED


# ─────────────────────────────────────────────────────────
# TeammateRunner shutdown
# ─────────────────────────────────────────────────────────


class TestTeammateRunnerShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_graceful(self):
        runner = TeammateRunner()

        async def long_run(handle: TeammateHandle) -> None:
            await asyncio.sleep(100)

        runner.spawn(team_name="team1", member_name="sleeper", run_func=long_run)
        # Give it a moment to start
        await asyncio.sleep(0.05)
        result = await runner.shutdown("team1", "sleeper")
        assert result is True
        assert runner.get_handle("sleeper@team1") is None

    @pytest.mark.asyncio
    async def test_shutdown_nonexistent_returns_false(self):
        runner = TeammateRunner()
        result = await runner.shutdown("team1", "ghost")
        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        runner = TeammateRunner()

        async def long_run(handle: TeammateHandle) -> None:
            await asyncio.sleep(100)

        runner.spawn(team_name="team1", member_name="a", run_func=long_run)
        runner.spawn(team_name="team1", member_name="b", run_func=long_run)
        await asyncio.sleep(0.05)

        count = await runner.shutdown_all("team1")
        assert count == 2
        assert runner.list_handles() == []


# ─────────────────────────────────────────────────────────
# TeammateRunner get / list / reset
# ─────────────────────────────────────────────────────────


class TestTeammateRunnerGetListReset:
    @pytest.mark.asyncio
    async def test_get_handle(self):
        runner = TeammateRunner()
        task = runner.spawn(team_name="t", member_name="x")
        handle = runner.get_handle("x@t")
        assert handle is not None
        assert handle.name == "x"
        await task

    @pytest.mark.asyncio
    async def test_get_nonexistent_handle(self):
        runner = TeammateRunner()
        assert runner.get_handle("nobody@nowhere") is None

    @pytest.mark.asyncio
    async def test_list_handles(self):
        runner = TeammateRunner()
        t1 = runner.spawn(team_name="t", member_name="a")
        t2 = runner.spawn(team_name="t", member_name="b")
        handles = runner.list_handles()
        assert len(handles) == 2
        names = {h.name for h in handles}
        assert names == {"a", "b"}
        await asyncio.gather(t1, t2)

    @pytest.mark.asyncio
    async def test_reset_clears_all(self):
        runner = TeammateRunner()
        task = runner.spawn(team_name="t", member_name="x")
        # Cancel the task before resetting so it doesn't linger
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        runner.reset()
        assert runner.list_handles() == []
