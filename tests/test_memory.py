"""Tests for dazi.memory — MemoryCategory, MemoryEntry serialization, MemoryStore CRUD & search."""

from __future__ import annotations

import re

from dazi.memory import (
    MemoryCategory,
    MemoryEntry,
    _generate_id,
)

# ─────────────────────────────────────────────────────────
# MemoryCategory enum
# ─────────────────────────────────────────────────────────


class TestMemoryCategory:
    def test_user_value(self):
        assert MemoryCategory.USER.value == "user"

    def test_feedback_value(self):
        assert MemoryCategory.FEEDBACK.value == "feedback"

    def test_project_value(self):
        assert MemoryCategory.PROJECT.value == "project"

    def test_reference_value(self):
        assert MemoryCategory.REFERENCE.value == "reference"

    def test_is_string_enum(self):
        assert isinstance(MemoryCategory.USER, str)


# ─────────────────────────────────────────────────────────
# MemoryEntry serialization
# ─────────────────────────────────────────────────────────


class TestMemoryEntryFromMarkdown:
    def test_parse_with_frontmatter(self):
        md = """\
---
id: test-id-123
category: user
created_at: 2025-03-15T10:00:00
tags: [python, testing]
description: A test memory
---
This is the body content."""
        entry = MemoryEntry.from_markdown(md)
        assert entry.id == "test-id-123"
        assert entry.category == MemoryCategory.USER
        assert entry.content == "This is the body content."
        assert entry.tags == ["python", "testing"]
        assert entry.description == "A test memory"

    def test_parse_no_frontmatter(self):
        md = "Just plain content here."
        entry = MemoryEntry.from_markdown(md)
        assert entry.content == "Just plain content here."
        assert entry.category == MemoryCategory.USER  # default

    def test_parse_frontmatter_defaults(self):
        md = """\
---
id: abc
---
Body."""
        entry = MemoryEntry.from_markdown(md)
        assert entry.id == "abc"
        assert entry.content == "Body."
        assert entry.tags == []
        assert entry.description == ""


class TestMemoryEntryToMarkdown:
    def test_roundtrip(self):
        entry = MemoryEntry(
            id="round-trip",
            content="Hello world",
            category=MemoryCategory.FEEDBACK,
            tags=["a", "b"],
            description="Test desc",
            created_at="2025-01-01T00:00:00",
        )
        md = entry.to_markdown()
        assert md.startswith("---")
        assert "id: round-trip" in md
        assert "category: feedback" in md
        assert "Hello world" in md
        # Parse back
        restored = MemoryEntry.from_markdown(md)
        assert restored.id == entry.id
        assert restored.content == entry.content
        assert restored.category == entry.category
        assert restored.tags == entry.tags
        assert restored.description == entry.description

    def test_description_fallback_to_content(self):
        entry = MemoryEntry(
            id="x",
            content="A" * 200,
            category=MemoryCategory.USER,
            description="",
        )
        md = entry.to_markdown()
        assert "description: " + "A" * 100 in md


# ─────────────────────────────────────────────────────────
# _generate_id
# ─────────────────────────────────────────────────────────


class TestGenerateId:
    def test_format(self):
        mem_id = _generate_id()
        # Format: YYYYMMDD-HHMMSS-<6hex>
        assert re.match(r"\d{8}-\d{6}-[0-9a-f]{6}", mem_id)

    def test_unique(self):
        ids = {_generate_id() for _ in range(20)}
        assert len(ids) == 20


# ─────────────────────────────────────────────────────────
# MemoryStore — CRUD
# ─────────────────────────────────────────────────────────


class TestMemoryStoreWrite:
    def test_write_creates_file(self, mock_memory_store):
        entry = MemoryEntry(content="test", category=MemoryCategory.USER)
        path = mock_memory_store.write(entry)
        assert path.exists()
        assert path.name == f"{entry.id}.md"

    def test_write_rebuilds_index(self, mock_memory_store):
        entry = MemoryEntry(content="indexed", category=MemoryCategory.USER)
        mock_memory_store.write(entry)
        assert mock_memory_store.index_path.exists()


class TestMemoryStoreRead:
    def test_read_existing(self, mock_memory_store):
        entry = MemoryEntry(
            content="hello",
            category=MemoryCategory.PROJECT,
            id="read-test",
        )
        mock_memory_store.write(entry)
        result = mock_memory_store.read("read-test")
        assert result is not None
        assert result.content == "hello"
        assert result.category == MemoryCategory.PROJECT

    def test_read_nonexistent(self, mock_memory_store):
        assert mock_memory_store.read("no-such-id") is None


class TestMemoryStoreDelete:
    def test_delete_existing(self, mock_memory_store):
        entry = MemoryEntry(content="bye", id="del-me")
        mock_memory_store.write(entry)
        assert mock_memory_store.delete("del-me") is True
        assert mock_memory_store.read("del-me") is None

    def test_delete_nonexistent(self, mock_memory_store):
        assert mock_memory_store.delete("nope") is False


class TestMemoryStoreListAll:
    def test_list_all_with_entries(self, mock_memory_store):
        e1 = MemoryEntry(content="a", category=MemoryCategory.USER, id="aaa")
        e2 = MemoryEntry(content="b", category=MemoryCategory.FEEDBACK, id="bbb")
        mock_memory_store.write(e1)
        mock_memory_store.write(e2)
        entries = mock_memory_store.list_all()
        ids = {e.id for e in entries}
        assert "aaa" in ids
        assert "bbb" in ids

    def test_list_all_empty(self, mock_memory_store):
        assert mock_memory_store.list_all() == []

    def test_list_all_skips_index_file(self, mock_memory_store):
        entry = MemoryEntry(content="x", id="only-one")
        mock_memory_store.write(entry)
        entries = mock_memory_store.list_all()
        assert all(e.id != "MEMORY" for e in entries)


# ─────────────────────────────────────────────────────────
# MemoryStore — Search
# ─────────────────────────────────────────────────────────


class TestMemoryStoreFindRelevant:
    def test_match_scoring(self, mock_memory_store):
        e1 = MemoryEntry(
            content="Python is great for data science",
            category=MemoryCategory.USER,
            id="py-entry",
            tags=["python"],
        )
        e2 = MemoryEntry(
            content="JavaScript rules the web",
            category=MemoryCategory.PROJECT,
            id="js-entry",
        )
        mock_memory_store.write(e1)
        mock_memory_store.write(e2)
        results = mock_memory_store.find_relevant("python data")
        assert len(results) >= 1
        assert results[0].id == "py-entry"

    def test_limit(self, mock_memory_store):
        for i in range(10):
            mock_memory_store.write(
                MemoryEntry(
                    content=f"testing item number {i}",
                    category=MemoryCategory.USER,
                    id=f"item-{i}",
                )
            )
        results = mock_memory_store.find_relevant("testing", limit=3)
        assert len(results) <= 3

    def test_empty_query(self, mock_memory_store):
        mock_memory_store.write(MemoryEntry(content="something", id="e1"))
        results = mock_memory_store.find_relevant("")
        assert results == []

    def test_no_matching_entries(self, mock_memory_store):
        mock_memory_store.write(MemoryEntry(content="apples and oranges", id="fruit"))
        results = mock_memory_store.find_relevant("quantum physics")
        assert results == []


# ─────────────────────────────────────────────────────────
# MemoryStore — Index
# ─────────────────────────────────────────────────────────


class TestMemoryStoreRebuildIndex:
    def test_creates_memory_md(self, mock_memory_store):
        entry = MemoryEntry(
            content="test content for index",
            category=MemoryCategory.USER,
            id="idx-1",
            description="Index test",
        )
        mock_memory_store.write(entry)
        content = mock_memory_store.get_index_content()
        assert "# Memory Index" in content
        assert "idx-1" in content
        assert "User" in content

    def test_groups_by_category(self, mock_memory_store):
        e1 = MemoryEntry(
            content="u", category=MemoryCategory.USER, id="u1", description="user entry"
        )
        e2 = MemoryEntry(
            content="p", category=MemoryCategory.PROJECT, id="p1", description="project entry"
        )
        mock_memory_store.write(e1)
        mock_memory_store.write(e2)
        content = mock_memory_store.get_index_content()
        assert "## User" in content
        assert "## Project" in content
