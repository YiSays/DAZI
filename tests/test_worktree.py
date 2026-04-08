"""Tests for dazi/worktree.py — sanitize_agent_name, validate_slug, WorktreeConfig, WorktreeManager listing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dazi.worktree import WorktreeConfig, WorktreeManager, Worktree


# ─────────────────────────────────────────────────────────
# sanitize_agent_name
# ─────────────────────────────────────────────────────────


class TestSanitizeAgentName:
    def test_replaces_spaces_with_dashes(self):
        mgr = WorktreeManager()
        assert mgr.sanitize_agent_name("my agent") == "my-agent"

    def test_lowercases(self):
        mgr = WorktreeManager()
        assert mgr.sanitize_agent_name("MyAgent") == "myagent"

    def test_strips_leading_trailing_dashes(self):
        mgr = WorktreeManager()
        assert mgr.sanitize_agent_name("-hello-") == "hello"

    def test_max_length_respected_in_validate(self):
        mgr = WorktreeManager(config=WorktreeConfig(max_slug_length=10))
        slug = "a" * 11
        with pytest.raises(ValueError, match="too long"):
            mgr.validate_slug(slug)


# ─────────────────────────────────────────────────────────
# validate_slug
# ─────────────────────────────────────────────────────────


class TestValidateSlug:
    def test_valid_slug(self):
        mgr = WorktreeManager()
        mgr.validate_slug("my-agent-123")  # should not raise

    def test_empty_slug_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError, match="empty"):
            mgr.validate_slug("")

    def test_path_traversal_dotdot_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError, match="[.][.]"):
            mgr.validate_slug("..")

    def test_path_traversal_dot_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError, match='[.]"'):
            mgr.validate_slug(".")

    def test_special_chars_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError, match="Invalid"):
            mgr.validate_slug("agent!@#")

    def test_slash_segment_rejected(self):
        mgr = WorktreeManager()
        with pytest.raises(ValueError):
            mgr.validate_slug("foo/../bar")


# ─────────────────────────────────────────────────────────
# WorktreeConfig defaults
# ─────────────────────────────────────────────────────────


class TestWorktreeConfig:
    def test_defaults(self):
        cfg = WorktreeConfig()
        assert cfg.worktree_base == ".dazi/worktrees"
        assert cfg.branch_prefix == "agent-"
        assert cfg.max_slug_length == 64
        assert cfg.stale_cutoff_days == 30


# ─────────────────────────────────────────────────────────
# WorktreeManager listing
# ─────────────────────────────────────────────────────────


class TestWorktreeManagerList:
    def test_list_empty(self):
        mgr = WorktreeManager()
        assert mgr.list_all() == []

    def test_list_with_entries(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="test",
            path="/tmp/test",
            branch="agent-test",
            agent_name="test",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["test"] = wt
        result = mgr.list_all()
        assert len(result) == 1
        assert result[0].id == "test"

    def test_get_existing(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="x",
            path="/tmp/x",
            branch="agent-x",
            agent_name="x",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["x"] = wt
        assert mgr.get("x") is wt

    def test_get_nonexistent(self):
        mgr = WorktreeManager()
        assert mgr.get("nobody") is None


# ─────────────────────────────────────────────────────────
# WorktreeManager create/remove with mocked git
# ─────────────────────────────────────────────────────────


class TestWorktreeManagerCreateRemove:
    def test_create_with_mocked_git(self):
        mgr = WorktreeManager()
        mock_result = type("R", (), {"returncode": 0, "stdout": "/tmp/repo\n", "stderr": ""})()
        mock_branch = type("R", (), {"returncode": 0, "stdout": "main\n", "stderr": ""})()
        mock_create = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("dazi.worktree.subprocess.run", side_effect=[mock_result, mock_branch, mock_create]):
            wt = mgr.create("frontend")
            assert wt.id == "frontend"
            assert wt.branch == "agent-frontend"
            assert wt.agent_name == "frontend"

    def test_remove_with_mocked_git(self):
        mgr = WorktreeManager()
        wt = Worktree(
            id="test",
            path="/tmp/test",
            branch="agent-test",
            agent_name="test",
            created_at="2025-01-01T00:00:00Z",
            original_cwd="/tmp",
            original_branch="main",
        )
        mgr._worktrees["test"] = wt

        mock_status = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        mock_remove = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        mock_branch_del = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("dazi.worktree.subprocess.run", side_effect=[mock_status, mock_remove, mock_branch_del]):
            result = mgr.remove("test", force=True)
            assert result is True
            assert mgr.get("test") is None
