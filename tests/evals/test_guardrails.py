"""
E-02 · Guardrails effectiveness tests.

Exercises every branch of _validate_strategy() in src/graph.py.
Calls _validate_strategy() directly with crafted InvestmentStrategy instances.
All tests are offline — no LLM or external API calls.
"""

import copy

import pytest

from src.state import InvestmentStrategy, InvestmentRecommendation
from src.graph import _validate_strategy

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy(**overrides) -> InvestmentStrategy:
    """Build a minimal valid InvestmentStrategy, applying overrides."""
    defaults = {
        "monthly_savings_target": 200.0,
        "savings_breakdown": {"misc": 200.0},
        "investment_recommendations": [
            InvestmentRecommendation(
                symbol="VOO", action="BUY",
                allocation_percent=25.0, rationale="Index fund core.",
            ),
        ],
        "risk_assessment": "Moderate risk.",
        "action_items": ["Save more."],
        "confidence_score": 0.70,
    }
    defaults.update(overrides)
    return InvestmentStrategy(**defaults)


def _make_rec(symbol="VOO", action="BUY", pct=25.0, rationale="Test."):
    """Shorthand for InvestmentRecommendation."""
    return InvestmentRecommendation(
        symbol=symbol, action=action,
        allocation_percent=pct, rationale=rationale,
    )


# ============================================================================
# Clean strategies — should pass all guardrails
# ============================================================================


class TestGuardrailsPass:
    """Strategies that must return zero issues."""

    def test_safe_strategy_passes(self, mock_strategy_safe):
        """mock_strategy_safe fixture produces no guardrail issues."""
        strategy = InvestmentStrategy.model_validate(mock_strategy_safe)
        issues = _validate_strategy(strategy, "moderate")
        assert issues == []

    def test_clean_content_total_100_passes(self):
        """All-clean strategy with total allocation exactly 100% passes."""
        strategy = _make_strategy(
            investment_recommendations=[
                _make_rec("VOO", pct=30.0),
                _make_rec("AAPL", action="HOLD", pct=30.0),
                _make_rec("MSFT", action="HOLD", pct=20.0),
                _make_rec("GOOGL", action="BUY", pct=20.0),
            ],
        )
        issues = _validate_strategy(strategy, "moderate")
        assert issues == []

    def test_total_allocation_99_passes(self):
        """Boundary: total allocation of 99% is valid (≤100)."""
        strategy = _make_strategy(
            investment_recommendations=[
                _make_rec("VOO", pct=30.0),
                _make_rec("AAPL", action="HOLD", pct=30.0),
                _make_rec("MSFT", action="HOLD", pct=29.0),
                _make_rec("GOOGL", action="BUY", pct=10.0),
            ],
        )
        issues = _validate_strategy(strategy, "moderate")
        assert issues == []


# ============================================================================
# Blocked-term detection
# ============================================================================


class TestBlockedTerms:
    """Strategies containing prohibited terms must be flagged BLOCKED."""

    def test_crypto_in_action_items(self, mock_strategy_blocked):
        """mock_strategy_blocked (crypto in action_items) → BLOCKED."""
        strategy = InvestmentStrategy.model_validate(mock_strategy_blocked)
        issues = _validate_strategy(strategy, "moderate")

        assert len(issues) >= 1
        assert any(i.startswith("BLOCKED") for i in issues)

    def test_day_trading_in_action_items(self):
        """'day trading' in action_items → BLOCKED."""
        strategy = _make_strategy(
            action_items=["Start day trading strategy for faster returns"],
        )
        issues = _validate_strategy(strategy, "moderate")

        assert len(issues) >= 1
        assert any(i.startswith("BLOCKED") for i in issues)

    def test_margin_in_recommendation_rationale(self):
        """'margin' embedded in a recommendation's rationale → BLOCKED."""
        strategy = _make_strategy(
            investment_recommendations=[
                _make_rec(
                    "AAPL", pct=25.0,
                    rationale="Use margin account to amplify AAPL returns.",
                ),
            ],
        )
        issues = _validate_strategy(strategy, "aggressive")

        assert len(issues) >= 1
        assert any(i.startswith("BLOCKED") for i in issues)

    def test_leverage_anywhere_in_model_dump(self):
        """'leverage' anywhere in model_dump() → BLOCKED.

        Verifies _safe_str full-scan reaches nested dict values
        (savings_breakdown in this case).
        """
        strategy = _make_strategy(
            savings_breakdown={"use_leverage_funds": 150.0},
        )
        issues = _validate_strategy(strategy, "moderate")

        assert len(issues) >= 1
        assert any(i.startswith("BLOCKED") for i in issues)


# ============================================================================
# Concentration guardrail
# ============================================================================


class TestConcentrationGuardrail:
    """Overconcentrated positions must be flagged."""

    def test_overconcentrated_single_position(self, mock_strategy_overconcentrated):
        """AAPL at 85% exceeds 30% limit → flagged."""
        strategy = InvestmentStrategy.model_validate(mock_strategy_overconcentrated)
        issues = _validate_strategy(strategy, "aggressive")

        assert len(issues) >= 1
        # The issue message contains the symbol and percentage info
        concentration_issues = [
            i for i in issues
            if "exceeds" in i.lower() or "allocation" in i.lower()
        ]
        assert len(concentration_issues) >= 1


# ============================================================================
# Total allocation guardrail
# ============================================================================


class TestTotalAllocationGuardrail:
    """Total allocation exceeding 100% must be flagged."""

    def test_total_allocation_110_percent(self):
        """Three positions summing to 110% → flagged."""
        strategy = _make_strategy(
            investment_recommendations=[
                _make_rec("VOO", pct=30.0),
                _make_rec("AAPL", pct=30.0, action="HOLD"),
                _make_rec("MSFT", pct=30.0, action="HOLD"),
                _make_rec("GOOGL", pct=20.0),
            ],
        )
        issues = _validate_strategy(strategy, "moderate")

        total_issues = [i for i in issues if "total allocation" in i.lower()]
        assert len(total_issues) >= 1
        assert "110" in total_issues[0]


# ============================================================================
# Negative savings guardrail
# ============================================================================


class TestNegativeSavingsGuardrail:
    """Negative monthly_savings_target must be BLOCKED."""

    def test_negative_savings_target(self):
        """monthly_savings_target=-50 → BLOCKED."""
        strategy = _make_strategy(monthly_savings_target=-50.0)
        issues = _validate_strategy(strategy, "moderate")

        assert len(issues) >= 1
        blocked = [i for i in issues if i.startswith("BLOCKED")]
        assert len(blocked) >= 1
        assert any("savings" in i.lower() for i in blocked)
