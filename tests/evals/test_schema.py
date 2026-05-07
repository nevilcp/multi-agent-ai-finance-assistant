"""
E-02 · Schema correctness tests.

Verifies that Pydantic domain models accept valid data, reject invalid data,
and round-trip through model_dump() → model_validate() without information loss.
All tests are offline — no LLM or external API calls.
"""

import copy

import pytest
from pydantic import ValidationError

from src.state import (
    ExpenseCategory,
    ExpenseReport,
    GraphState,
    InvestmentRecommendation,
    InvestmentStrategy,
    compute_confidence_score,
    create_initial_state,
)

pytestmark = pytest.mark.offline


# ============================================================================
# Positive validation — models accept well-formed data
# ============================================================================


class TestExpenseReportValidation:
    """ExpenseReport should accept realistic fixture data."""

    def test_accepts_mock_expense_report(self, mock_expense_report):
        """mock_expense_report fixture validates without exception."""
        report = ExpenseReport.model_validate(mock_expense_report)

        assert report.total_spending == 1885.65
        assert report.savings_potential == 285.00
        assert report.transaction_count == 20
        assert len(report.category_breakdown) > 0
        assert len(report.top_wasteful_categories) > 0


class TestInvestmentStrategyValidation:
    """InvestmentStrategy should accept clean fixture data."""

    def test_accepts_mock_strategy_safe(self, mock_strategy_safe):
        """mock_strategy_safe fixture validates without exception."""
        strategy = InvestmentStrategy.model_validate(mock_strategy_safe)

        assert strategy.monthly_savings_target == 245.00
        assert strategy.confidence_score == 0.72
        assert len(strategy.investment_recommendations) == 3
        assert len(strategy.action_items) > 0

    def test_roundtrip_preserves_all_fields(self, mock_strategy_safe):
        """model_dump() → model_validate() round-trip preserves every field."""
        original = InvestmentStrategy.model_validate(mock_strategy_safe)
        dumped = original.model_dump()
        restored = InvestmentStrategy.model_validate(dumped)

        assert restored.monthly_savings_target == original.monthly_savings_target
        assert restored.confidence_score == original.confidence_score
        assert restored.risk_assessment == original.risk_assessment
        assert restored.action_items == original.action_items
        assert restored.savings_breakdown == original.savings_breakdown

        # Verify each recommendation round-trips exactly
        for orig_rec, rest_rec in zip(
            original.investment_recommendations,
            restored.investment_recommendations,
        ):
            assert rest_rec.symbol == orig_rec.symbol
            assert rest_rec.action == orig_rec.action
            assert rest_rec.allocation_percent == orig_rec.allocation_percent
            assert rest_rec.rationale == orig_rec.rationale

        # Full equality via re-dump
        assert original.model_dump() == restored.model_dump()


# ============================================================================
# Negative validation — models reject invalid data
# ============================================================================


class TestInvestmentRecommendationRejection:
    """InvestmentRecommendation should enforce allocation_percent ∈ [0, 100]."""

    def test_rejects_allocation_above_100(self):
        """allocation_percent=150 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InvestmentRecommendation(
                symbol="AAPL",
                action="BUY",
                allocation_percent=150.0,
                rationale="Over-allocated test",
            )
        assert "allocation" in str(exc_info.value).lower()

    def test_rejects_negative_allocation(self):
        """allocation_percent=-5 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            InvestmentRecommendation(
                symbol="MSFT",
                action="HOLD",
                allocation_percent=-5.0,
                rationale="Negative allocation test",
            )
        assert "allocation" in str(exc_info.value).lower()


class TestExpenseCategoryRejection:
    """ExpenseCategory should reject negative total_amount."""

    def test_rejects_negative_total_amount(self):
        """total_amount=-100 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ExpenseCategory(
                category="Test",
                total_amount=-100.0,
                transaction_count=5,
            )
        assert "amount" in str(exc_info.value).lower()


# ============================================================================
# GraphState defaults
# ============================================================================


class TestGraphStateDefaults:
    """create_initial_state() should produce correct default field types."""

    def test_default_field_types(self):
        """Verify all default fields have expected values and types."""
        state: GraphState = create_initial_state(
            session_id="test-defaults",
            csv_path="/tmp/test.csv",
            portfolio_symbols=["AAPL", "MSFT", "VOO"],
        )

        # Errors starts as empty list
        assert state["errors"] == []
        assert isinstance(state["errors"], list)

        # Current phase starts as "input"
        assert state["current_phase"] == "input"

        # All agent outputs default to None
        assert state["expense_report"] is None
        assert state["market_intel"] is None
        assert state["investment_strategy"] is None
        assert state["approval_result"] is None

        # Feedback not applied
        assert state["feedback_applied"] is False

        # Watchlist defaults to False
        assert state["watchlist_mode"] is False

        # Inputs are set correctly
        assert state["raw_csv_path"] == "/tmp/test.csv"
        assert state["portfolio_symbols"] == ["AAPL", "MSFT", "VOO"]


# ============================================================================
# Deterministic confidence score computation
# ============================================================================


class TestConfidenceScoreComputation:
    """compute_confidence_score() produces deterministic values from pipeline signals."""

    def _make_strategy(self, num_recs: int = 3) -> InvestmentStrategy:
        """Build a minimal valid InvestmentStrategy with N recommendations."""
        recs = [
            InvestmentRecommendation(
                symbol=s, action="BUY",
                allocation_percent=round(80 / max(num_recs, 1), 1),
                rationale=f"Test recommendation for {s}",
            )
            for s in ["VOO", "AAPL", "MSFT"][:num_recs]
        ]
        return InvestmentStrategy(
            monthly_savings_target=250.0,
            savings_breakdown={"subscriptions": 100.0, "dining": 150.0},
            investment_recommendations=recs,
            risk_assessment="Moderate risk test portfolio.",
            action_items=["Action 1", "Action 2", "Action 3"],
            confidence_score=0.5,  # placeholder — will be overridden
        )

    def test_full_data_high_confidence(self, mock_expense_report, mock_market_intel):
        """Full expense data + no errors + live market → score >= 0.90."""
        strategy = self._make_strategy(num_recs=3)
        state: GraphState = {
            "session_id": "test-confidence-high",
            "user_id": None,
            "raw_csv_path": "/tmp/test.csv",
            "portfolio_symbols": ["AAPL", "MSFT", "VOO"],
            "user_preferences": None,
            "watchlist_mode": False,
            "expense_report": mock_expense_report,     # present → no -0.20
            "market_intel": mock_market_intel,          # live → no -0.10
            "investment_strategy": None,
            "approval_result": None,
            "errors": [],                               # no errors → no -0.30
            "current_phase": "advisor",
            "feedback_applied": False,
        }
        score = compute_confidence_score(strategy, state)

        # No deductions: 1.0 expected
        assert score >= 0.90, f"Expected >= 0.90 with full data, got {score}"
        assert score <= 1.0, f"Score must not exceed 1.0, got {score}"

    def test_watchlist_with_errors_low_confidence(self):
        """No expense data + 2 errors + no market data + 1 rec → score <= 0.60."""
        strategy = self._make_strategy(num_recs=1)  # under-diversified → -0.15
        state: GraphState = {
            "session_id": "test-confidence-low",
            "user_id": None,
            "raw_csv_path": "",
            "portfolio_symbols": ["VOO"],
            "user_preferences": None,
            "watchlist_mode": True,
            "expense_report": None,                     # missing → -0.20
            "market_intel": None,                       # missing → -0.10
            "investment_strategy": None,
            "approval_result": None,
            "errors": [                                 # 2 errors → -0.20
                "Market intelligence failed: connection error",
                "Data parse warning: date format",
            ],
            "current_phase": "advisor",
            "feedback_applied": False,
        }
        score = compute_confidence_score(strategy, state)

        # Deductions: -0.20 (no expense) -0.20 (2 errors) -0.10 (no market)
        #             -0.15 (< 2 recs) -0.10 (no savings) = -0.75 → 0.25
        assert score <= 0.60, f"Expected <= 0.60 with degraded data, got {score}"
        assert score >= 0.0, f"Score must not go below 0.0, got {score}"

    def test_cached_market_data_deduction(self, mock_expense_report):
        """Cached market data should reduce confidence by 0.10."""
        strategy = self._make_strategy(num_recs=3)
        state_live: GraphState = {
            "session_id": "test-fresh",
            "user_id": None,
            "raw_csv_path": "/tmp/test.csv",
            "portfolio_symbols": ["AAPL", "MSFT", "VOO"],
            "user_preferences": None,
            "watchlist_mode": False,
            "expense_report": mock_expense_report,
            "market_intel": {"data_freshness": "live", "data_points": [],
                             "portfolio_symbols": [], "market_summary": "OK"},
            "investment_strategy": None,
            "approval_result": None,
            "errors": [],
            "current_phase": "advisor",
            "feedback_applied": False,
        }
        state_cached = {**state_live, "market_intel": {
            "data_freshness": "cached", "data_points": [],
            "portfolio_symbols": [], "market_summary": "OK"
        }}

        score_live = compute_confidence_score(strategy, state_live)
        score_cached = compute_confidence_score(strategy, state_cached)

        assert score_live - score_cached == pytest.approx(0.10, abs=0.01), (
            f"Cached data should deduct 0.10: live={score_live}, cached={score_cached}"
        )

