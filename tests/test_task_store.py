"""Tests for dazi.task_store — Task dataclass, TaskStatus enum, TaskStore CRUD & dependencies."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from dazi.task_store import Task, TaskStatus, TaskStore


# ─────────────────────────────────────────────────────────
# TaskStatus enum
# ─────────────────────────────────────────────────────────


class TestTaskStatus:
    def test_pending_value(self):
        assert TaskStatus.PENDING.value == "pending"

    def test_in_progress_value(self):
        assert TaskStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_value(self):
        assert TaskStatus.COMPLETED.value == "completed"

    def test_is_string_enum(self):
        assert isinstance(TaskStatus.PENDING, str)


# ─────────────────────────────────────────────────────────
# Task dataclass
# ─────────────────────────────────────────────────────────


class TestTask:
    def test_to_dict_roundtrip(self):
        task = Task(
            id=1,
            subject="Write tests",
            description="Write unit tests for task_store",
            status=TaskStatus.IN_PROGRESS,
            active_form="Writing tests",
            owner="alice",
            blocks=[2, 3],
            blocked_by=[4],
            metadata={"priority": "high"},
            created_at="2025-01-01T00:00:00",
        )
        d = task.to_dict()
        restored = Task.from_dict(d)
        assert restored.id == task.id
        assert restored.subject == task.subject
        assert restored.description == task.description
        assert restored.status == task.status
        assert restored.active_form == task.active_form
        assert restored.owner == task.owner
        assert restored.blocks == task.blocks
        assert restored.blocked_by == task.blocked_by
        assert restored.metadata == task.metadata
        assert restored.created_at == task.created_at

    def test_to_dict_status_is_string(self):
        task = Task(id=1, subject="s", description="d")
        d = task.to_dict()
        assert isinstance(d["status"], str)
        assert d["status"] == "pending"

    def test_from_dict_defaults(self):
        d = {"id": 5, "subject": "s", "description": "d", "status": "pending"}
        task = Task.from_dict(d)
        assert task.active_form == ""
        assert task.owner is None
        assert task.blocks == []
        assert task.blocked_by == []
        assert task.metadata == {}

    def test_from_dict_missing_status_raises(self):
        with pytest.raises(KeyError):
            Task.from_dict({"id": 1, "subject": "s", "description": "d"})


# ─────────────────────────────────────────────────────────
# TaskStore — CRUD
# ─────────────────────────────────────────────────────────


class TestTaskStoreCreate:
    def test_first_task_gets_id_1(self, mock_task_store):
        task = mock_task_store.create("Task A", "First task")
        assert task.id == 1
        assert task.status == TaskStatus.PENDING

    def test_sequential_ids(self, mock_task_store):
        t1 = mock_task_store.create("A", "a")
        t2 = mock_task_store.create("B", "b")
        t3 = mock_task_store.create("C", "c")
        assert t1.id == 1
        assert t2.id == 2
        assert t3.id == 3

    def test_create_with_metadata(self, mock_task_store):
        task = mock_task_store.create("T", "d", metadata={"x": 1})
        assert task.metadata == {"x": 1}

    def test_create_with_active_form(self, mock_task_store):
        task = mock_task_store.create("T", "d", active_form="Testing")
        assert task.active_form == "Testing"

    def test_create_persists_to_disk(self, mock_task_store):
        task = mock_task_store.create("T", "d")
        path = mock_task_store._list_dir / f"{task.id}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["subject"] == "T"


class TestTaskStoreGet:
    def test_get_existing(self, mock_task_store):
        created = mock_task_store.create("X", "y")
        fetched = mock_task_store.get(created.id)
        assert fetched is not None
        assert fetched.subject == "X"

    def test_get_nonexistent_returns_none(self, mock_task_store):
        assert mock_task_store.get(999) is None


class TestTaskStoreUpdate:
    def test_update_subject(self, mock_task_store):
        task = mock_task_store.create("Old", "desc")
        updated = mock_task_store.update(task.id, subject="New")
        assert updated is not None
        assert updated.subject == "New"

    def test_update_status(self, mock_task_store):
        task = mock_task_store.create("T", "d")
        updated = mock_task_store.update(task.id, status=TaskStatus.IN_PROGRESS)
        assert updated.status == TaskStatus.IN_PROGRESS

    def test_update_nonexistent_returns_none(self, mock_task_store):
        assert mock_task_store.update(999, subject="X") is None


class TestTaskStoreDelete:
    def test_delete_existing_returns_true(self, mock_task_store):
        task = mock_task_store.create("T", "d")
        assert mock_task_store.delete(task.id) is True
        assert mock_task_store.get(task.id) is None

    def test_delete_nonexistent_returns_false(self, mock_task_store):
        assert mock_task_store.delete(999) is False


class TestTaskStoreListAll:
    def test_list_all_sorted_by_id(self, mock_task_store):
        mock_task_store.create("C", "c")
        mock_task_store.create("A", "a")
        mock_task_store.create("B", "b")
        tasks = mock_task_store.list_all()
        assert [t.subject for t in tasks] == ["C", "A", "B"]
        assert [t.id for t in tasks] == [1, 2, 3]

    def test_list_all_empty(self, mock_task_store):
        assert mock_task_store.list_all() == []


# ─────────────────────────────────────────────────────────
# TaskStore — Dependencies
# ─────────────────────────────────────────────────────────


class TestTaskStoreDependencies:
    def test_add_block_bidirectional(self, mock_task_store):
        t1 = mock_task_store.create("Blocker", "d")
        t2 = mock_task_store.create("Blocked", "d")
        a, b = mock_task_store.add_block(t1.id, t2.id)
        assert t2.id in a.blocks
        assert t1.id in b.blocked_by

    def test_add_block_idempotent(self, mock_task_store):
        t1 = mock_task_store.create("A", "d")
        t2 = mock_task_store.create("B", "d")
        mock_task_store.add_block(t1.id, t2.id)
        mock_task_store.add_block(t1.id, t2.id)
        t1_check = mock_task_store.get(t1.id)
        assert t1_check.blocks.count(t2.id) == 1

    def test_add_block_nonexistent_returns_none_tuple(self, mock_task_store):
        a, b = mock_task_store.add_block(999, 998)
        assert a is None
        assert b is None

    def test_add_blocked_by_reverse(self, mock_task_store):
        t1 = mock_task_store.create("A", "d")
        t2 = mock_task_store.create("B", "d")
        a, b = mock_task_store.add_blocked_by(t2.id, t1.id)
        assert t1.id in a.blocked_by
        assert t2.id in b.blocks

    def test_get_active_blockers_excludes_completed(self, mock_task_store):
        t1 = mock_task_store.create("Blocker", "d")
        t2 = mock_task_store.create("Blocked", "d")
        t3 = mock_task_store.create("Another blocker", "d")
        mock_task_store.add_block(t1.id, t2.id)
        mock_task_store.add_block(t3.id, t2.id)
        # Complete t1
        mock_task_store.update(t1.id, status=TaskStatus.COMPLETED)
        active = mock_task_store.get_active_blockers(t2.id)
        assert t1.id not in active
        assert t3.id in active

    def test_get_active_blockers_nonexistent(self, mock_task_store):
        assert mock_task_store.get_active_blockers(999) == []


# ─────────────────────────────────────────────────────────
# TaskStore — Reset
# ─────────────────────────────────────────────────────────


class TestTaskStoreReset:
    def test_reset_removes_all_tasks(self, mock_task_store):
        mock_task_store.create("A", "a")
        mock_task_store.create("B", "b")
        mock_task_store.reset()
        assert mock_task_store.list_all() == []

    def test_reset_on_empty(self, mock_task_store):
        mock_task_store.reset()  # should not raise
        assert mock_task_store.list_all() == []
