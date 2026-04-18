"""Tests for graph routing."""
import pytest
from unittest.mock import AsyncMock, patch

from src.graph import route_from_entry, build_graph
from src.state import (
    ExpenseCategory,
    ExpenseReport,
    InvestmentRecommendation,
    InvestmentStrategy,
    MarketDataPoint,
    MarketIntelReport,
    create_initial_state,
)


def test_route_normal_mode():
    s = create_initial_state(session_id="t", csv_path="f.csv", portfolio_symbols=["AAPL"])
    assert route_from_entry(s) == "expense_categorizer"


def test_route_watchlist_mode():
    s = create_initial_state(session_id="t", csv_path="", portfolio_symbols=["AAPL"], watchlist_mode=True)
    assert route_from_entry(s) == "market_intelligence"


def test_graph_compiles():
    assert build_graph() is not None


@pytest.mark.asyncio
async def test_graph_ainvoke_watchlist_happy_path_with_mocks():
    graph = build_graph()
    state = create_initial_state(session_id="t", csv_path="", portfolio_symbols=["AAPL"], watchlist_mode=True)

    mock_market = MarketIntelReport(
        portfolio_symbols=["AAPL"],
        data_points=[MarketDataPoint(symbol="AAPL", current_price=200.0, change_percent=1.2)],
        market_summary="Uptrend",
        data_freshness="live",
    )
    mock_strategy = InvestmentStrategy(
        monthly_savings_target=120.0,
        savings_breakdown={"Dining": 40.0},
        investment_recommendations=[
            InvestmentRecommendation(
                symbol="AAPL",
                action="HOLD",
                allocation_percent=20.0,
                rationale="Core long-term holding.",
            )
        ],
        risk_assessment="Moderate",
        action_items=["Automate monthly contribution"],
        confidence_score=0.75,
    )

    with patch(
        "src.agents.market_intelligence.finnhub_tools.get_batch_quotes",
        return_value=[],
    ), patch(
        "src.agents.market_intelligence.finnhub_tools.get_company_news",
        return_value=[],
    ), patch(
        "src.agents.market_intelligence.gemini_client.generate",
        new=AsyncMock(return_value="Market summary"),
    ), patch(
        "src.agents.wealth_advisor.gemini_client.generate_structured",
        new=AsyncMock(return_value=mock_strategy),
    ), patch(
        "src.agents.market_intelligence.MarketIntelReport",
        return_value=mock_market,
    ), patch(
        "builtins.input",
        return_value="y",
    ):
        result = await graph.ainvoke(state)

    assert result["current_phase"] == "complete"
    assert "investment_strategy" in result


@pytest.mark.asyncio
async def test_graph_routes_to_error_handler_when_errors_exceed_threshold():
    graph = build_graph()
    state = create_initial_state(session_id="t", csv_path="", portfolio_symbols=["AAPL"], watchlist_mode=True)
    state["errors"] = ["e1", "e2", "e3", "e4"]

    result = await graph.ainvoke(state)
    assert result["current_phase"] == "error"
