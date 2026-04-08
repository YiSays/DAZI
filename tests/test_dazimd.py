"""Tests for dazi/dazimd.py — DAZI.md discovery, includes, and merging."""
from __future__ import annotations

from pathlib import Path

import pytest

from dazi.dazimd import (
    DaziMdFile,
    PRIORITY_GLOBAL,
    PRIORITY_LOCAL,
    PRIORITY_PROJECT,
    discover_dazimd_files,
    merge_dazimd_content,
    resolve_includes,
)


# ─────────────────────────────────────────────────────────
# discover_dazimd_files
# ─────────────────────────────────────────────────────────


class TestDiscoverDazimdFiles:
    def test_no_files(self, tmp_path: Path):
        # tmp_path has no DAZI.md files
        files = discover_dazimd_files(project_root=tmp_path, cwd=tmp_path)
        # May still find ~/.dazi/DAZI.md, but not project files
        project_files = [f for f in files if f.priority >= PRIORITY_PROJECT]
        assert len(project_files) == 0

    def test_project_file_found(self, tmp_path: Path):
        dazi_md = tmp_path / "DAZI.md"
        dazi_md.write_text("# Project instructions\nUse Python 3.12+")
        files = discover_dazimd_files(project_root=tmp_path, cwd=tmp_path)
        names = [f.path.name for f in files]
        assert "DAZI.md" in names
        # Check priority
        project_files = [f for f in files if f.path.name == "DAZI.md"]
        assert project_files[0].priority == PRIORITY_PROJECT

    def test_priority_order(self, tmp_path: Path):
        (tmp_path / "DAZI.md").write_text("Project content")
        (tmp_path / "DAZI.local.md").write_text("Local content")
        files = discover_dazimd_files(project_root=tmp_path, cwd=tmp_path)
        # First should be local (highest priority)
        if len(files) >= 2:
            assert files[0].priority >= files[1].priority


# ─────────────────────────────────────────────────────────
# resolve_includes
# ─────────────────────────────────────────────────────────


class TestResolveIncludes:
    def test_no_includes(self):
        content = "No includes here.\nJust text."
        result = resolve_includes(content, Path("/tmp"))
        assert result == content

    def test_relative_path_include(self, tmp_path: Path):
        included = tmp_path / "extra.md"
        included.write_text("Included content")
        content = f"@include extra.md"
        result = resolve_includes(content, tmp_path)
        assert "Included content" in result

    def test_circular_include_detection(self, tmp_path: Path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("@include b.md")
        b.write_text("@include a.md")
        result = resolve_includes("@include a.md", tmp_path)
        assert "Circular include" in result

    def test_missing_file(self, tmp_path: Path):
        content = "@include nonexistent.md"
        result = resolve_includes(content, tmp_path)
        assert "not found" in result

    def test_nested_includes(self, tmp_path: Path):
        inner = tmp_path / "inner.md"
        inner.write_text("Inner content")
        outer = tmp_path / "outer.md"
        outer.write_text("@include inner.md")
        content = "@include outer.md"
        result = resolve_includes(content, tmp_path)
        assert "Inner content" in result


# ─────────────────────────────────────────────────────────
# merge_dazimd_content
# ─────────────────────────────────────────────────────────


class TestMergeDazimdContent:
    def test_empty_list(self):
        assert merge_dazimd_content([]) == ""

    def test_single_file(self):
        f = DaziMdFile(path=Path("/tmp/DAZI.md"), priority=300, content="Line 1\nLine 2")
        result = merge_dazimd_content([f])
        assert "Line 1" in result
        assert "Line 2" in result

    def test_multiple_dedup(self):
        f1 = DaziMdFile(path=Path("/tmp/DAZI.md"), priority=300, content="Common line\nUnique A")
        f2 = DaziMdFile(path=Path("/tmp/DAZI.local.md"), priority=400, content="Common line\nUnique B")
        result = merge_dazimd_content([f2, f1])
        # "Common line" should appear only once
        assert result.count("Common line") == 1
        assert "Unique A" in result
        assert "Unique B" in result

    def test_empty_content_skipped(self):
        f = DaziMdFile(path=Path("/tmp/DAZI.md"), priority=300, content="   \n  \n")
        result = merge_dazimd_content([f])
        assert result.strip() == ""
