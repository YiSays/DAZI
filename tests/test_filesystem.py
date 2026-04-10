"""Tests for dazi/filesystem.py — file_reader, file_writer, calculator, shell_exec, plan_writer
tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from dazi.filesystem import (
    calculator_tool,
    file_reader_tool,
    file_writer_tool,
    shell_exec_tool,
)

# ─────────────────────────────────────────────────────────
# file_reader
# ─────────────────────────────────────────────────────────


class TestFileReader:
    def test_read_existing_file_with_line_numbers(self, tmp_path: Path):
        f = tmp_path / "sample.txt"
        f.write_text("hello\nworld\n")
        result = file_reader_tool.invoke({"file_path": str(f)})
        assert "hello" in result
        assert "world" in result
        assert "Total lines: 2" in result
        # Line numbers should be present
        assert "1" in result
        assert "2" in result

    def test_read_nonexistent_file_returns_error(self, tmp_path: Path):
        result = file_reader_tool.invoke({"file_path": str(tmp_path / "nope.txt")})
        assert "Error" in result
        assert "not found" in result.lower() or "Error" in result

    def test_read_with_offset_and_limit(self, tmp_path: Path):
        lines = [f"line {i}" for i in range(20)]
        f = tmp_path / "many.txt"
        f.write_text("\n".join(lines))
        result = file_reader_tool.invoke({"file_path": str(f), "offset": 5, "limit": 3})
        assert "line 5" in result
        assert "line 6" in result
        assert "line 7" in result
        assert "line 8" not in result

    def test_rejects_relative_path(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            file_reader_tool.invoke({"file_path": "relative/path.txt"})


# ─────────────────────────────────────────────────────────
# file_writer
# ─────────────────────────────────────────────────────────


class TestFileWriter:
    def test_creates_file(self, tmp_path: Path):
        target = tmp_path / "new.txt"
        result = file_writer_tool.invoke(
            {
                "file_path": str(target),
                "content": "hello world",
            }
        )
        assert target.exists()
        assert target.read_text() == "hello world"
        assert "Successfully wrote" in result

    def test_creates_parent_directories(self, tmp_path: Path):
        target = tmp_path / "a" / "b" / "c" / "file.txt"
        file_writer_tool.invoke(
            {
                "file_path": str(target),
                "content": "nested",
            }
        )
        assert target.exists()
        assert target.read_text() == "nested"

    def test_overwrites_existing_file(self, tmp_path: Path):
        target = tmp_path / "overwrite.txt"
        target.write_text("old content")
        file_writer_tool.invoke(
            {
                "file_path": str(target),
                "content": "new content",
            }
        )
        assert target.read_text() == "new content"


# ─────────────────────────────────────────────────────────
# calculator
# ─────────────────────────────────────────────────────────


class TestCalculator:
    def test_basic_arithmetic(self):
        result = calculator_tool.invoke({"expression": "2 + 3 * 4"})
        assert result == "14"

    def test_builtin_functions(self):
        result = calculator_tool.invoke({"expression": "abs(-5)"})
        assert result == "5"

    def test_sum_and_max(self):
        result = calculator_tool.invoke({"expression": "sum([1, 2, 3]) + max(4, 5)"})
        assert result == "11"

    def test_invalid_expression_returns_error(self):
        result = calculator_tool.invoke({"expression": "import os"})
        assert "Error" in result


# ─────────────────────────────────────────────────────────
# shell_exec
# ─────────────────────────────────────────────────────────


class TestShellExec:
    def test_echo_command(self):
        result = shell_exec_tool.invoke({"command": "echo hello"})
        assert "hello" in result


# ─────────────────────────────────────────────────────────
# plan_writer
# ─────────────────────────────────────────────────────────


class TestPlanWriter:
    def test_writes_to_plan_file(self, tmp_path: Path, monkeypatch):
        plan_dir = tmp_path / "plans"
        plan_file = plan_dir / "plan.md"
        monkeypatch.setattr("dazi.filesystem.PLAN_DIR", plan_dir)
        monkeypatch.setattr("dazi.filesystem.PLAN_FILE", plan_file)

        from dazi.filesystem import plan_writer

        result = plan_writer("My test plan")
        assert plan_file.exists()
        assert plan_file.read_text() == "My test plan"
        assert "Plan written" in result
