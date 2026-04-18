"""Tests for agent nodes."""
import pytest
from unittest.mock import AsyncMock, patch

from src.state import (
    ExpenseCategory,
    ExpenseReport,
    InvestmentRecommendation,
    InvestmentStrategy,
    MarketDataPoint,
    MarketIntelReport,
    create_initial_state,
)
from src.tools.finnhub_tools import QuoteResult


@pytest.mark.asyncio
async def test_expense_categorizer_missing_file():
    """Graceful error on missing CSV."""
    from src.agents.expense_categorizer import expense_categorizer_node

    state = create_initial_state(session_id="t", csv_path="/no/file.csv", portfolio_symbols=["AAPL"])
    result = await expense_categorizer_node(state)

    assert len(result.get("errors", [])) > 0
    assert result["current_phase"] == "market"


@pytest.mark.asyncio
async def test_market_intelligence_with_mocked_external_calls():
    """Market node uses mocked Finnhub + Gemini and returns valid output."""
    from src.agents.market_intelligence import market_intelligence_node

    state = create_initial_state(session_id="t", csv_path="", portfolio_symbols=["AAPL", "MSFT"], watchlist_mode=True)

    quotes = [
        QuoteResult(
            symbol="AAPL",
            current_price=200.0,
            change=2.0,
            change_percent=1.0,
            high=201.0,
            low=198.0,
            open_price=199.0,
            previous_close=198.0,
        )
    ]

    with patch("src.agents.market_intelligence.finnhub_tools.get_batch_quotes", return_value=quotes), patch(
        "src.agents.market_intelligence.finnhub_tools.get_company_news",
        return_value=[{"headline": "Sample headline", "summary": "Sample summary"}],
    ), patch(
        "src.agents.market_intelligence.gemini_client.generate",
        new=AsyncMock(return_value="Portfolio is stable."),
    ):
        result = await market_intelligence_node(state)

    assert result["current_phase"] == "advisor"
    assert result.get("errors") == []
    parsed = MarketIntelReport.model_validate(result["market_intel"])
    assert parsed.portfolio_symbols == ["AAPL", "MSFT"]
    assert len(parsed.data_points) == 1


@pytest.mark.asyncio
async def test_wealth_advisor_with_mocked_external_calls():
    """Wealth advisor uses mocked Gemini and returns valid strategy."""
    from src.agents.wealth_advisor import wealth_advisor_node

    state = create_initial_state(session_id="t", csv_path="", portfolio_symbols=["AAPL"], watchlist_mode=True)
    state["expense_report"] = ExpenseReport(
        total_spending=1000.0,
        category_breakdown=[
            ExpenseCategory(category="Dining", total_amount=200.0, transaction_count=10),
        ],
        top_wasteful_categories=["Dining"],
        monthly_average=1000.0,
        savings_potential=150.0,
        transaction_count=10,
        date_range_start="2024-01-01",
        date_range_end="2024-01-31",
    ).model_dump()
    state["market_intel"] = MarketIntelReport(
        portfolio_symbols=["AAPL"],
        data_points=[MarketDataPoint(symbol="AAPL", current_price=200.0, change_percent=1.0)],
        market_summary="Stable market",
        data_freshness="live",
    ).model_dump()

    strategy = InvestmentStrategy(
        monthly_savings_target=150.0,
        savings_breakdown={"Dining": 50.0},
        investment_recommendations=[
            InvestmentRecommendation(
                symbol="AAPL",
                action="BUY",
                allocation_percent=25.0,
                rationale="Quality large-cap exposure.",
            )
        ],
        risk_assessment="Moderate risk profile",
        action_items=["Move $150/month to brokerage account"],
        confidence_score=0.8,
    )

    with patch(
        "src.agents.wealth_advisor.gemini_client.generate_structured",
        new=AsyncMock(return_value=strategy),
    ):
        result = await wealth_advisor_node(state)

    assert result["current_phase"] == "complete"
    assert result.get("errors") == []
    parsed = InvestmentStrategy.model_validate(result["investment_strategy"])
    assert parsed.monthly_savings_target == 150.0
