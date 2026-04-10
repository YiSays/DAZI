"""Tests for dazi.tokenizer — token counting, context windows, thresholds."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dazi.tokenizer import (
    AUTOCOMPACT_BUFFER_TOKENS,
    CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_WINDOW,
    MAX_CONSECUTIVE_COMPACT_FAILURES,
    MESSAGE_OVERHEAD_TOKENS,
    WARNING_THRESHOLD_BUFFER_TOKENS,
    count_message_tokens,
    count_messages_tokens,
    count_text_tokens,
    get_compact_threshold,
    get_context_window,
    get_token_warning_state,
    get_warning_threshold,
    should_auto_compact,
)


class TestCountTextTokens:
    def test_empty_string(self):
        assert count_text_tokens("") == 0

    def test_known_model(self):
        # tiktoken should give a reasonable count
        tokens = count_text_tokens("Hello world", "gpt-4o")
        assert tokens > 0
        assert tokens < 20

    def test_unknown_model_uses_tiktoken_fallback(self):
        # Unknown model falls back to cl100k_base via tiktoken
        text = "a" * 100
        tokens = count_text_tokens(text, "unknown-model-xyz")
        # cl100k_base encodes 'a' chars efficiently (~3-4 per token)
        assert tokens > 0
        assert tokens < 100

    def test_no_model_uses_char_estimation(self):
        # Empty model string skips tiktoken
        text = "a" * 100
        tokens = count_text_tokens(text, "")
        assert tokens == 25  # 100 / CHARS_PER_TOKEN

    def test_returns_positive_for_text(self):
        assert count_text_tokens("test", "gpt-4o") > 0


class TestCountMessageTokens:
    def test_human_message(self):
        msg = HumanMessage(content="Hello")
        tokens = count_message_tokens(msg, "gpt-4o")
        assert tokens >= MESSAGE_OVERHEAD_TOKENS

    def test_ai_message(self):
        msg = AIMessage(content="Hi there!")
        tokens = count_message_tokens(msg, "gpt-4o")
        assert tokens >= MESSAGE_OVERHEAD_TOKENS

    def test_ai_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "file_reader", "args": {"file_path": "/tmp/test.py"}, "id": "1"}],
        )
        tokens = count_message_tokens(msg, "gpt-4o")
        assert tokens > MESSAGE_OVERHEAD_TOKENS

    def test_tool_message(self):
        msg = ToolMessage(content="file contents here", tool_call_id="1", name="file_reader")
        tokens = count_message_tokens(msg, "gpt-4o")
        assert tokens >= MESSAGE_OVERHEAD_TOKENS

    def test_message_overhead_included(self):
        msg = HumanMessage(content="")
        tokens = count_message_tokens(msg, "gpt-4o")
        assert tokens >= MESSAGE_OVERHEAD_TOKENS


class TestCountMessagesTokens:
    def test_list(self, sample_messages):
        total = count_messages_tokens(sample_messages, "gpt-4o")
        assert total > 0

    def test_empty_list(self):
        assert count_messages_tokens([], "gpt-4o") == 0

    def test_sums_across_messages(self):
        msgs = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
        ]
        total = count_messages_tokens(msgs, "gpt-4o")
        individual = sum(count_message_tokens(m, "gpt-4o") for m in msgs)
        assert total == individual


class TestGetContextWindow:
    def test_known_model(self):
        assert get_context_window("gpt-4o") == 128_000

    def test_another_known_model(self):
        assert get_context_window("gpt-4") == 8_192

    def test_prefix_match(self):
        assert get_context_window("gpt-4o-2024-08-06") == 128_000

    def test_unknown_model(self):
        assert get_context_window("llama-3-70b") == DEFAULT_CONTEXT_WINDOW

    def test_empty_string(self):
        assert get_context_window("") == DEFAULT_CONTEXT_WINDOW

    def test_claude_models(self):
        assert get_context_window("claude-sonnet-4-20250514") == 200_000


class TestThresholds:
    def test_compact_threshold(self):
        threshold = get_compact_threshold("gpt-4o")
        assert threshold == 128_000 - AUTOCOMPACT_BUFFER_TOKENS

    def test_warning_threshold(self):
        threshold = get_warning_threshold("gpt-4o")
        assert threshold == 128_000 - WARNING_THRESHOLD_BUFFER_TOKENS

    def test_compact_greater_than_warning(self):
        # AUTOCOMPACT_BUFFER (13K) < WARNING_THRESHOLD_BUFFER (20K), so compact threshold is higher
        assert get_compact_threshold("gpt-4o") > get_warning_threshold("gpt-4o")


class TestShouldAutoCompact:
    def test_below_threshold(self, sample_messages):
        assert should_auto_compact(sample_messages, "gpt-4o") is False

    def test_above_threshold(self):
        # Create enough text to exceed 115K tokens
        # tiktoken counts ~1 token per 3-4 chars for 'x', so we need ~400K chars
        # Use empty model to get predictable char/4 token counting
        big_text = "x " * 250_000  # ~125K tokens with char/4 estimation
        msgs = [HumanMessage(content=big_text)]
        assert should_auto_compact(msgs, "") is True

    def test_circuit_breaker(self):
        big_text = "x " * 250_000
        msgs = [HumanMessage(content=big_text)]
        assert (
            should_auto_compact(msgs, "", consecutive_failures=MAX_CONSECUTIVE_COMPACT_FAILURES)
            is False
        )

    def test_circuit_breaker_below_max(self):
        big_text = "x " * 250_000
        msgs = [HumanMessage(content=big_text)]
        assert should_auto_compact(msgs, "", consecutive_failures=1) is True


class TestGetTokenWarningState:
    def test_ok(self, sample_messages):
        state = get_token_warning_state(sample_messages, "gpt-4o")
        assert state == "ok"

    def test_warning(self):
        # Create messages that exceed warning threshold (108K for gpt-4o)
        # but are below compact threshold (115K)
        # Use char/4 estimation by passing empty model
        window = DEFAULT_CONTEXT_WINDOW
        threshold = window - WARNING_THRESHOLD_BUFFER_TOKENS
        # Need text that gives > threshold tokens via char/4
        char_count = int(threshold * CHARS_PER_TOKEN) + 1000
        msgs = [HumanMessage(content="x " * (char_count // 2))]
        state = get_token_warning_state(msgs, "")
        assert state in ("warning", "compact")

    def test_compact(self):
        # Exceed compact threshold using char/4 estimation
        window = DEFAULT_CONTEXT_WINDOW
        threshold = window - AUTOCOMPACT_BUFFER_TOKENS
        char_count = int(threshold * CHARS_PER_TOKEN) + 10000
        msgs = [HumanMessage(content="x " * (char_count // 2))]
        state = get_token_warning_state(msgs, "")
        assert state == "compact"
