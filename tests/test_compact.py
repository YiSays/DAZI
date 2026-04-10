"""Tests for dazi/compact.py — message grouping, micro/full compact."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from dazi.compact import (
    CLEARED_TOOL_RESULT,
    COMPACT_BOUNDARY,
    _format_for_summarization,
    auto_compact,
    full_compact,
    group_messages_by_round,
    micro_compact,
)

# ─────────────────────────────────────────────────────────
# group_messages_by_round
# ─────────────────────────────────────────────────────────


class TestGroupMessagesByRound:
    def test_empty(self):
        assert group_messages_by_round([]) == []

    def test_single_round(self):
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        rounds = group_messages_by_round(msgs)
        assert len(rounds) == 1
        assert len(rounds[0]) == 2

    def test_multiple_rounds(self):
        msgs = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
        ]
        rounds = group_messages_by_round(msgs)
        assert len(rounds) == 2

    def test_system_message_in_first_round(self):
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        rounds = group_messages_by_round(msgs)
        assert len(rounds) == 1
        assert isinstance(rounds[0][0], SystemMessage)

    def test_tool_messages_in_round(self):
        msgs = [
            HumanMessage(content="do it"),
            AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}]),
            ToolMessage(content="result", tool_call_id="1", name="test"),
        ]
        rounds = group_messages_by_round(msgs)
        assert len(rounds) == 1
        assert len(rounds[0]) == 3


# ─────────────────────────────────────────────────────────
# micro_compact
# ─────────────────────────────────────────────────────────


class TestMicroCompact:
    @patch("dazi.compact.count_messages_tokens", return_value=100)
    def test_clears_old_tool_results(self, mock_tokens):
        msgs = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            ToolMessage(content="big file content here", tool_call_id="1", name="file_reader"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
        ]
        result = micro_compact(msgs, keep_recent_rounds=1)
        assert result.method == "micro"
        assert result.tool_results_cleared == 1
        # The tool result should be replaced
        tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
        assert tool_msgs[0].content == CLEARED_TOOL_RESULT

    @patch("dazi.compact.count_messages_tokens", return_value=100)
    def test_keeps_recent_rounds(self, mock_tokens):
        msgs = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            ToolMessage(content="keep me", tool_call_id="1", name="file_reader"),
        ]
        result = micro_compact(msgs, keep_recent_rounds=3)
        assert result.method == "none"
        # Original messages unchanged
        tool_msgs = [m for m in result.messages if isinstance(m, ToolMessage)]
        assert tool_msgs[0].content == "keep me"


# ─────────────────────────────────────────────────────────
# _format_for_summarization
# ─────────────────────────────────────────────────────────


class TestFormatForSummarization:
    def test_role_labels(self):
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="usr"),
            AIMessage(content="bot"),
            ToolMessage(content="tool out", tool_call_id="1", name="reader"),
        ]
        text = _format_for_summarization(msgs)
        assert "[System]: sys" in text
        assert "[User]: usr" in text
        assert "[Assistant]: bot" in text
        assert "[Tool(reader)]: tool out" in text

    def test_empty_messages(self):
        assert _format_for_summarization([]) == ""


# ─────────────────────────────────────────────────────────
# full_compact
# ─────────────────────────────────────────────────────────


class TestFullCompact:
    @patch("dazi.compact.count_messages_tokens", return_value=100)
    async def test_with_mock_llm(self, mock_tokens):
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Summary of conversation"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        msgs = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            HumanMessage(content="q3"),
            AIMessage(content="a3"),
        ]
        result = await full_compact(msgs, mock_llm, keep_recent_rounds=1)
        assert result.method == "full"
        assert result.rounds_removed == 2
        assert "Summary of conversation" in result.summary
        assert COMPACT_BOUNDARY in result.messages[0].content

    @patch("dazi.compact.count_messages_tokens", return_value=100)
    async def test_llm_failure(self, mock_tokens):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        msgs = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
        ]
        result = await full_compact(msgs, mock_llm, keep_recent_rounds=1)
        assert result.method == "none"
        assert "Compact failed" in result.summary


# ─────────────────────────────────────────────────────────
# auto_compact
# ─────────────────────────────────────────────────────────


class TestAutoCompact:
    @patch("dazi.compact.count_messages_tokens", return_value=100)
    async def test_below_threshold(self, mock_tokens):
        mock_llm = AsyncMock()
        with patch("dazi.tokenizer.should_auto_compact", return_value=False):
            msgs = [HumanMessage(content="hi")]
            result = await auto_compact(msgs, mock_llm)
            assert result.method == "none"

    @patch("dazi.compact.count_messages_tokens", return_value=100)
    async def test_circuit_breaker(self, mock_tokens):
        mock_llm = AsyncMock()
        msgs = [HumanMessage(content="hi")]
        result = await auto_compact(msgs, mock_llm, consecutive_failures=3)
        assert result.method == "none"
        assert "Circuit breaker" in result.summary
