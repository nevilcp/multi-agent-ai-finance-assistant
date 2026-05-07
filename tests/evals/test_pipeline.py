"""
E-02 · Pipeline reliability tests.

Tests graph routing functions, node execution with mocked LLM / API calls,
and graph compilation.  All tests are offline — no real Gemini or Finnhub calls.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from dataclasses import dataclass
from typing import List

import pandas as pd

from src.state import (
    ExpenseReport,
    ExpenseCategory,
    InvestmentStrategy,
    InvestmentRecommendation,
    create_initial_state,
    GraphState,
)
from src.graph import (
    route_from_entry,
    route_from_expense,
    guardrails_node,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> GraphState:
    """Build a minimal GraphState, applying overrides."""
    base = create_initial_state(
        session_id="pipeline-test",
        csv_path="/tmp/test.csv",
        portfolio_symbols=["AAPL", "MSFT", "VOO"],
    )
    base.update(overrides)
    return base


# ============================================================================
# Routing tests — route_from_entry
# ============================================================================


class TestRouteFromEntry:
    """Tests for the entry-point routing function."""

    def test_normal_mode_routes_to_expense(self):
        """watchlist_mode=False → 'expense_categorizer'."""
        state = _state(watchlist_mode=False)
        assert route_from_entry(state) == "expense_categorizer"

    def test_watchlist_mode_routes_to_market(self):
        """watchlist_mode=True → 'market_intelligence'."""
        state = _state(watchlist_mode=True)
        assert route_from_entry(state) == "market_intelligence"

    def test_excessive_errors_routes_to_error_handler(self):
        """len(errors) > 3 → 'error_handler' regardless of mode."""
        state = _state(errors=["e1", "e2", "e3", "e4"])
        assert route_from_entry(state) == "error_handler"


# ============================================================================
# Routing tests — route_from_expense
# ============================================================================


class TestRouteFromExpense:
    """Tests for the expense → market routing function."""

    def test_zero_errors_routes_to_market(self):
        """No errors → 'market_intelligence'."""
        state = _state(errors=[])
        assert route_from_expense(state) == "market_intelligence"

    def test_four_errors_routes_to_error_handler(self):
        """4 errors (> 3) → 'error_handler'."""
        state = _state(errors=["e1", "e2", "e3", "e4"])
        assert route_from_expense(state) == "error_handler"


# ============================================================================
# Node execution — expense_categorizer_node
# ============================================================================


@dataclass
class _FakeParsed:
    """Minimal stand-in for ParsedStatement returned by csv_parser.parse()."""
    df: pd.DataFrame
    total_transactions: int
    date_range_start: str
    date_range_end: str
    detected_columns: dict
    threats_detected: List[str]


class TestExpenseCategorizerNode:
    """Tests for expense_categorizer_node with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_successful_categorization(self, base_state, tmp_csv):
        """Happy path: mocked parse + generate_structured → state populated."""
        from src.agents.expense_categorizer import expense_categorizer_node

        csv_path = tmp_csv("normal")
        state = {**base_state, "raw_csv_path": csv_path}

        fake_parsed = _FakeParsed(
            df=pd.DataFrame({
                "Date": ["2025-01-02"],
                "Description": ["Whole Foods"],
                "Amount": [-87.42],
                "normalized_amount": [-87.42],
                "normalized_date": [pd.Timestamp("2025-01-02")],
            }),
            total_transactions=20,
            date_range_start="2025-01-02",
            date_range_end="2025-01-30",
            detected_columns={"date": "Date", "description": "Description", "amount": "Amount"},
            threats_detected=[],
        )

        fake_report = ExpenseReport(
            total_spending=1404.35,
            category_breakdown=[
                ExpenseCategory(
                    category="Groceries",
                    total_amount=450.0,
                    transaction_count=8,
                ),
            ],
            top_wasteful_categories=[],
            monthly_average=1404.35,
            savings_potential=0.0,
        )

        with patch(
            "src.agents.expense_categorizer.csv_parser.parse",
            return_value=fake_parsed,
        ), patch(
            "src.agents.expense_categorizer.csv_parser.to_llm_text",
            return_value="Bank Statement (20 transactions)\n  2025-01-02 | -87.42 | Whole Foods",
        ), patch(
            "src.agents.expense_categorizer.gemini_client.generate_structured",
            new=AsyncMock(return_value=fake_report),
        ):
            result = await expense_categorizer_node(state)

        # State should have expense_report set
        assert result["expense_report"] is not None
        assert result["current_phase"] == "market"
        # Validate it round-trips through the model
        validated = ExpenseReport.model_validate(result["expense_report"])
        assert validated.total_spending == 1404.35

    @pytest.mark.asyncio
    async def test_llm_failure_appends_error(self, base_state, tmp_csv):
        """generate_structured raises → error appended, node does not crash."""
        from src.agents.expense_categorizer import expense_categorizer_node

        csv_path = tmp_csv("normal")
        state = {**base_state, "raw_csv_path": csv_path}

        fake_parsed = _FakeParsed(
            df=pd.DataFrame({
                "Date": ["2025-01-02"],
                "Description": ["Test"],
                "Amount": [-10.0],
                "normalized_amount": [-10.0],
                "normalized_date": [pd.Timestamp("2025-01-02")],
            }),
            total_transactions=1,
            date_range_start="2025-01-02",
            date_range_end="2025-01-02",
            detected_columns={"date": "Date", "description": "Description", "amount": "Amount"},
            threats_detected=[],
        )

        with patch(
            "src.agents.expense_categorizer.csv_parser.parse",
            return_value=fake_parsed,
        ), patch(
            "src.agents.expense_categorizer.csv_parser.to_llm_text",
            return_value="test",
        ), patch(
            "src.agents.expense_categorizer.gemini_client.generate_structured",
            new=AsyncMock(side_effect=RuntimeError("Gemini API quota exhausted")),
        ):
            result = await expense_categorizer_node(state)

        # Node should NOT crash
        assert result["current_phase"] == "market"
        # Error should be recorded
        assert len(result["errors"]) > 0
        assert any("Expense analysis failed" in e for e in result["errors"])
        # expense_report should NOT be set
        assert result.get("expense_report") is None


# ============================================================================
# Node execution — guardrails_node
# ============================================================================


class TestGuardrailsNode:
    """Tests for guardrails_node with pre-built strategy dicts."""

    @pytest.mark.asyncio
    async def test_safe_strategy_passes_through(self, base_state, mock_strategy_safe):
        """Safe strategy → investment_strategy remains non-None."""
        state = {**base_state, "investment_strategy": mock_strategy_safe}

        result = await guardrails_node(state)

        assert result["investment_strategy"] is not None
        # Strategy should survive untouched
        validated = InvestmentStrategy.model_validate(result["investment_strategy"])
        assert validated.monthly_savings_target == 245.00

    @pytest.mark.asyncio
    async def test_blocked_strategy_nullified(self, base_state, mock_strategy_blocked):
        """Blocked strategy (crypto) → investment_strategy set to None."""
        state = {**base_state, "investment_strategy": mock_strategy_blocked}

        result = await guardrails_node(state)

        # Strategy should be nullified because "crypto" is a BLOCKED term
        assert result["investment_strategy"] is None
        # Errors should contain the BLOCKED issue
        assert len(result["errors"]) > 0
        assert any("BLOCKED" in e for e in result["errors"])


# ============================================================================
# Graph compilation
# ============================================================================


class TestGraphCompilation:
    """Import-time graph compilation must succeed."""

    def test_build_graph_returns_compiled_graph(self):
        """build_graph() should return a compiled graph without raising."""
        from src.graph import build_graph

        graph = build_graph()
        # The compiled graph should be a callable (CompiledStateGraph)
        assert graph is not None
        # It should have invoke/ainvoke methods from LangGraph
        assert hasattr(graph, "ainvoke") or hasattr(graph, "invoke")
