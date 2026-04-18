"""Unit tests for the cost calculator."""

import pytest

from assistant.client import Usage
from assistant.cost import CostBreakdown, calculate_cost, format_cost


def _usage(**kwargs) -> Usage:
    return Usage(**kwargs)


# ---------------------------------------------------------------------------
# CostBreakdown
# ---------------------------------------------------------------------------

def test_total_sums_all_fields():
    b = CostBreakdown(input_cost=1.0, output_cost=2.0, cache_write_cost=0.5, cache_read_cost=0.1)
    assert b.total == pytest.approx(3.6)


def test_total_zero_by_default():
    assert CostBreakdown().total == 0.0


def test_cache_savings_positive():
    b = CostBreakdown(input_cost=0.30, cache_read_cost=0.03)
    assert b.cache_savings > 0


def test_cache_savings_never_negative():
    b = CostBreakdown(input_cost=0.01, cache_read_cost=0.05)
    assert b.cache_savings == 0.0


def test_str_shows_total():
    b = CostBreakdown(input_cost=0.001, output_cost=0.002)
    assert "$" in str(b)
    assert "cost" in str(b).lower()


# ---------------------------------------------------------------------------
# calculate_cost
# ---------------------------------------------------------------------------

def test_calculate_cost_known_model():
    usage = _usage(input_tokens=1_000_000, output_tokens=0)
    breakdown = calculate_cost(usage, "claude-sonnet-4-6")
    assert breakdown.input_cost == pytest.approx(3.00)


def test_calculate_cost_output_tokens():
    usage = _usage(input_tokens=0, output_tokens=1_000_000)
    breakdown = calculate_cost(usage, "claude-sonnet-4-6")
    assert breakdown.output_cost == pytest.approx(15.00)


def test_calculate_cost_cache_read():
    usage = _usage(cache_read_tokens=1_000_000)
    breakdown = calculate_cost(usage, "claude-sonnet-4-6")
    assert breakdown.cache_read_cost == pytest.approx(0.30)


def test_calculate_cost_cache_write():
    usage = _usage(cache_write_tokens=1_000_000)
    breakdown = calculate_cost(usage, "claude-sonnet-4-6")
    assert breakdown.cache_write_cost == pytest.approx(3.75)


def test_calculate_cost_unknown_model_uses_default():
    usage = _usage(input_tokens=1_000_000)
    # Should not raise — falls back to Sonnet pricing
    breakdown = calculate_cost(usage, "claude-unknown-model")
    assert breakdown.input_cost > 0


def test_calculate_cost_opus():
    usage = _usage(input_tokens=1_000_000)
    breakdown = calculate_cost(usage, "claude-opus-4")
    # Opus input is $15/MTok — more expensive than Sonnet
    assert breakdown.input_cost == pytest.approx(15.00)


def test_calculate_cost_zero_usage():
    usage = _usage()
    breakdown = calculate_cost(usage, "claude-sonnet-4-6")
    assert breakdown.total == 0.0


# ---------------------------------------------------------------------------
# format_cost
# ---------------------------------------------------------------------------

def test_format_cost_returns_dollar_string():
    usage = _usage(input_tokens=500, output_tokens=200)
    result = format_cost(usage, "claude-sonnet-4-6")
    assert result.startswith("~$")


def test_format_cost_small_amount():
    usage = _usage(input_tokens=10, output_tokens=5)
    result = format_cost(usage, "claude-sonnet-4-6")
    assert "~$" in result
