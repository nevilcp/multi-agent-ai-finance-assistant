"""Wealth advisor — synthesizes expense + market data into strategy."""
import time
from src.state import GraphState, InvestmentStrategy, UserPreferences
from src.utils.gemini_client import gemini_client, ModelType
from src.observability.logger import get_logger

logger = get_logger(__name__)

PROMPT = """You are a personal wealth advisor. Synthesize the following data into a complete investment strategy.

## Expense Data
{expense_data}

## Market Data
{market_data}

## User Profile
- Risk tolerance: {risk_tolerance}
- Investment horizon: {horizon}
- Excluded sectors: {excluded}

Create a strategy with:
1. monthly_savings_target: realistic monthly savings from cutting waste
2. savings_breakdown: dict of category -> amount to save per month
3. investment_recommendations: list with symbol, action (BUY/HOLD/SELL/REDUCE), allocation_percent, rationale
4. risk_assessment: brief assessment of portfolio risk
5. action_items: 5-7 specific, actionable steps
6. confidence_score: 0.0-1.0 reflecting data completeness

Guidelines:
- No single allocation should exceed 30%
- Total allocations must sum to 100% or less
- Action items must be specific and immediately actionable
- Do NOT recommend day trading, margin, leverage, options, futures, or crypto
"""


async def wealth_advisor_node(state: GraphState) -> GraphState:
    """
    Generate investment strategy with one LLM call (Gemini 3 Flash).

    Passes raw expense + market data directly to LLM for synthesis.
    The LLM handles all analysis, savings calculation, and recommendation logic.
    """
    logger.agent_start("wealth_advisor")
    state = dict(state)
    t0 = time.time()

    try:
        # Format expense data
        expense_data = "No expense data available (watchlist mode)."
        if state.get("expense_report"):
            er = state["expense_report"]
            lines = [f"Total spending: ${er['total_spending']:,.2f}"]
            lines.append(f"Savings potential: ${er['savings_potential']:,.2f}")
            for cat in er.get("category_breakdown", []):
                flag = " ⚠️ WASTEFUL" if cat.get("flagged_as_wasteful") else ""
                lines.append(f"  {cat['category']}: ${cat['total_amount']:,.2f} ({cat['transaction_count']} txns){flag}")
                if cat.get("waste_reasoning"):
                    lines.append(f"    Reason: {cat['waste_reasoning']}")
            expense_data = "\n".join(lines)

        # Format market data
        market_data = "No market data available."
        if state.get("market_intel"):
            mi = state["market_intel"]
            lines = []
            for dp in mi.get("data_points", []):
                lines.append(f"  {dp['symbol']}: ${dp['current_price']:.2f} ({dp['change_percent']:+.2f}%)")
            lines.append(f"\nSummary: {mi.get('market_summary', 'N/A')}")
            market_data = "\n".join(lines)

        # User preferences
        prefs = UserPreferences.model_validate(state["user_preferences"]) if state.get("user_preferences") else None
        risk = prefs.risk_tolerance if prefs else "moderate"
        horizon = prefs.investment_horizon if prefs else "medium"
        excluded = ", ".join(prefs.exclude_sectors) if prefs and prefs.exclude_sectors else "None"

        # Single LLM call
        strategy = await gemini_client.generate_structured(
            prompt=PROMPT.format(
                expense_data=expense_data, market_data=market_data,
                risk_tolerance=risk, horizon=horizon, excluded=excluded,
            ),
            response_model=InvestmentStrategy,
            model=ModelType.FLASH,
            temperature=0.4,
        )

        state["investment_strategy"] = strategy.model_dump()
        state["current_phase"] = "complete"
        logger.agent_complete("wealth_advisor", duration_ms=(time.time() - t0) * 1000, confidence=strategy.confidence_score)

    except Exception as e:
        logger.error("advisor_error", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Wealth advisor failed: {e}"]
        state["current_phase"] = "complete"

    return state
