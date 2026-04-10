"""Tests for dazi.cost_tracker — pricing, cost calculation, persistence."""

import pytest

from dazi.cost_tracker import (
    DEFAULT_PRICING,
    CostRecord,
    _get_pricing,
    calculate_cost,
)


class TestGetPricing:
    def test_exact_match(self):
        pricing = _get_pricing("gpt-4o")
        assert pricing.model == "gpt-4o"
        assert pricing.input_cost_per_mtok == 2.50

    def test_prefix_match(self):
        pricing = _get_pricing("gpt-4o-2024-08-06")
        assert pricing.model == "gpt-4o"

    def test_unknown_model_fallback(self):
        pricing = _get_pricing("llama-3-70b")
        assert pricing.model == "default"
        assert pricing == DEFAULT_PRICING

    def test_empty_string_fallback(self):
        pricing = _get_pricing("")
        assert pricing == DEFAULT_PRICING


class TestCalculateCost:
    def test_basic(self):
        cost = calculate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.50 + 10.00)

    def test_partial_million(self):
        cost = calculate_cost("gpt-4o", 500_000, 0)
        assert cost == pytest.approx(1.25)

    def test_zero_tokens(self):
        assert calculate_cost("gpt-4o", 0, 0) == 0.0

    def test_small_values(self):
        cost = calculate_cost("gpt-4o-mini", 1000, 500)
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert cost == pytest.approx(expected)


class TestCostRecord:
    def test_to_dict(self):
        record = CostRecord(
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            estimated_cost_usd=0.001,
            request_count=2,
        )
        d = record.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["request_count"] == 2

    def test_to_dict_rounds_cost(self):
        record = CostRecord(model="test", estimated_cost_usd=0.123456789)
        d = record.to_dict()
        assert d["estimated_cost_usd"] == round(0.123456789, 6)

    def test_from_dict(self):
        d = {
            "model": "gpt-4o",
            "input_tokens": 200,
            "output_tokens": 100,
            "estimated_cost_usd": 0.002,
            "request_count": 5,
        }
        record = CostRecord.from_dict(d)
        assert record.model == "gpt-4o"
        assert record.input_tokens == 200
        assert record.request_count == 5

    def test_from_dict_defaults(self):
        record = CostRecord.from_dict({"model": "test"})
        assert record.input_tokens == 0
        assert record.output_tokens == 0
        assert record.estimated_cost_usd == 0.0
        assert record.request_count == 0

    def test_roundtrip(self):
        original = CostRecord("gpt-4o", 1000, 500, 0.001, 3)
        restored = CostRecord.from_dict(original.to_dict())
        assert restored.model == original.model
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.request_count == original.request_count


class TestCostTracker:
    def test_record_usage_new_model(self, mock_cost_tracker):
        record = mock_cost_tracker.record_usage("gpt-4o", 1000, 500)
        assert record.model == "gpt-4o"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.request_count == 1
        assert record.estimated_cost_usd > 0

    def test_record_usage_accumulates(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1000, 500)
        mock_cost_tracker.record_usage("gpt-4o", 2000, 1000)
        record = mock_cost_tracker.get_model_summary()["gpt-4o"]
        assert record.input_tokens == 3000
        assert record.output_tokens == 1500
        assert record.request_count == 2

    def test_record_usage_multi_model(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1000, 500)
        mock_cost_tracker.record_usage("gpt-4o-mini", 2000, 1000)
        assert len(mock_cost_tracker.get_model_summary()) == 2

    def test_get_total_cost_empty(self, mock_cost_tracker):
        assert mock_cost_tracker.get_total_cost() == 0.0

    def test_get_total_cost(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1_000_000, 0)
        cost = mock_cost_tracker.get_total_cost()
        assert cost == pytest.approx(2.50)

    def test_get_total_cost_multi_model(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1_000_000, 0)
        mock_cost_tracker.record_usage("gpt-4o-mini", 1_000_000, 0)
        cost = mock_cost_tracker.get_total_cost()
        assert cost == pytest.approx(2.50 + 0.15)

    def test_get_total_tokens(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1000, 500)
        mock_cost_tracker.record_usage("gpt-4o-mini", 2000, 1000)
        total_input, total_output = mock_cost_tracker.get_total_tokens()
        assert total_input == 3000
        assert total_output == 1500

    def test_get_total_request_count(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 100, 50)
        mock_cost_tracker.record_usage("gpt-4o", 200, 100)
        mock_cost_tracker.record_usage("gpt-4o-mini", 100, 50)
        assert mock_cost_tracker.get_total_request_count() == 3

    def test_format_cost(self, mock_cost_tracker):
        result = mock_cost_tracker.format_cost()
        assert result.startswith("$")

    def test_format_summary(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1000, 500)
        summary = mock_cost_tracker.format_summary()
        assert "Total cost" in summary
        assert "Total tokens" in summary
        assert "Total requests" in summary

    def test_format_summary_multi_model(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1000, 500)
        mock_cost_tracker.record_usage("gpt-4o-mini", 2000, 1000)
        summary = mock_cost_tracker.format_summary()
        assert "Usage by model" in summary

    def test_save_and_load(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1000, 500)
        mock_cost_tracker.save()

        data = mock_cost_tracker.load_last_session()
        assert data is not None
        assert "model_usage" in data
        assert "gpt-4o" in data["model_usage"]
        assert data["model_usage"]["gpt-4o"]["input_tokens"] == 1000

    def test_load_no_file(self, mock_cost_tracker):
        assert mock_cost_tracker.load_last_session() is None

    def test_load_corrupted_json(self, mock_cost_tracker, tmp_path):
        mock_cost_tracker._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        mock_cost_tracker._persistence_path.write_text("not json{{")
        assert mock_cost_tracker.load_last_session() is None

    def test_format_last_session_no_data(self, mock_cost_tracker):
        result = mock_cost_tracker.format_last_session()
        assert "No previous session" in result

    def test_format_last_session(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1000, 500)
        mock_cost_tracker.save()
        result = mock_cost_tracker.format_last_session()
        assert "Previous session" in result

    def test_reset(self, mock_cost_tracker):
        mock_cost_tracker.record_usage("gpt-4o", 1000, 500)
        mock_cost_tracker.reset()
        assert mock_cost_tracker.get_total_cost() == 0.0
        assert mock_cost_tracker.get_total_request_count() == 0
