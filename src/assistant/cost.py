"""Cost calculator — converts token usage into estimated USD cost per model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import Usage

# Prices in USD per million tokens (as of April 2026)
# Source: https://www.anthropic.com/pricing
_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-haiku-3-5": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
}

# Fallback — use Sonnet pricing for unknown models
_DEFAULT_PRICING = _PRICING["claude-sonnet-4-6"]


@dataclass
class CostBreakdown:
    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_write_cost: float = 0.0
    cache_read_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost + self.cache_write_cost + self.cache_read_cost

    @property
    def cache_savings(self) -> float:
        """
        Estimated savings from prompt caching.

        Without caching, all cache_read tokens would be billed as input tokens.
        Savings = (input_price - cache_read_price) * cache_read_tokens / 1_000_000
        """
        return max(0.0, self.input_cost - self.cache_read_cost)

    def __str__(self) -> str:
        lines = [f"Estimated cost: ${self.total:.6f}"]
        if self.input_cost:
            lines.append(f"  Input:        ${self.input_cost:.6f}")
        if self.output_cost:
            lines.append(f"  Output:       ${self.output_cost:.6f}")
        if self.cache_write_cost:
            lines.append(f"  Cache write:  ${self.cache_write_cost:.6f}")
        if self.cache_read_cost:
            lines.append(f"  Cache read:   ${self.cache_read_cost:.6f}")
            lines.append(f"  Cache saved:  ~${self.cache_savings:.6f}")
        return "\n".join(lines)


def calculate_cost(usage: "Usage", model: str) -> CostBreakdown:
    """
    Compute the estimated USD cost for a given Usage and model.

    Args:
        usage: Token counts from a ConversationSession.
        model: The model identifier string (e.g. "claude-sonnet-4-6").

    Returns:
        A CostBreakdown with per-category and total costs.
    """
    # Match by prefix so minor version suffixes still resolve correctly
    prices = _DEFAULT_PRICING
    for key, pricing in _PRICING.items():
        if model.startswith(key) or key.startswith(model):
            prices = pricing
            break

    per_million = 1_000_000

    return CostBreakdown(
        input_cost=usage.input_tokens * prices["input"] / per_million,
        output_cost=usage.output_tokens * prices["output"] / per_million,
        cache_write_cost=usage.cache_write_tokens * prices["cache_write"] / per_million,
        cache_read_cost=usage.cache_read_tokens * prices["cache_read"] / per_million,
    )


def format_cost(usage: "Usage", model: str) -> str:
    """Return a short one-line cost summary suitable for inline display."""
    breakdown = calculate_cost(usage, model)
    total = breakdown.total
    if total < 0.001:
        return f"~${total:.6f}"
    return f"~${total:.4f}"
