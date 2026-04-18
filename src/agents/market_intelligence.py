"""Market intelligence — deterministic data fetch + single LLM synthesis."""
import time
from src.state import GraphState, MarketIntelReport, MarketDataPoint
from src.utils.gemini_client import gemini_client, ModelType
from src.tools.finnhub_tools import finnhub_tools
from src.observability.logger import get_logger

logger = get_logger(__name__)

PROMPT = """You are a market intelligence analyst. Analyze this portfolio data.

Portfolio Holdings:
{quotes}

Recent News:
{news}

Provide a concise summary (under 300 words) covering:
1. Overall portfolio performance
2. Notable movers
3. Key news impacting holdings
4. General sentiment
"""


async def market_intelligence_node(state: GraphState) -> GraphState:
    """
    Gather market data and synthesize with one LLM call (Gemini 3 Flash).

    1. Fetch quotes + news from Finnhub (deterministic)
    2. Single Gemini call to summarize
    """
    logger.agent_start("market_intelligence", symbols=state["portfolio_symbols"])
    state = dict(state)
    t0 = time.time()

    try:
        symbols = state["portfolio_symbols"]

        # Step 1: Deterministic data fetch
        quotes = finnhub_tools.get_batch_quotes(symbols)

        quote_lines = []
        data_points = []
        for q in quotes:
            quote_lines.append(
                f"{q.symbol}: ${q.current_price:.2f} ({q.change_percent:+.2f}%) "
                f"H:{q.high:.2f} L:{q.low:.2f}"
            )
            data_points.append(MarketDataPoint(
                symbol=q.symbol, current_price=q.current_price,
                change_percent=q.change_percent, high=q.high, low=q.low,
            ))

        news_lines = []
        for s in symbols[:3]:
            for n in finnhub_tools.get_company_news(s, limit=3):
                news_lines.append(f"[{s}] {n['headline']}")

        # Step 2: Single LLM synthesis
        summary = await gemini_client.generate(
            prompt=PROMPT.format(
                quotes="\n".join(quote_lines) or "No data available",
                news="\n".join(news_lines) or "No news available",
            ),
            model=ModelType.FLASH,
            temperature=0.3,
            max_tokens=1024,
        )

        report = MarketIntelReport(
            portfolio_symbols=symbols, data_points=data_points,
            market_summary=summary[:500], data_freshness="live",
        )
        state["market_intel"] = report.model_dump()
        state["current_phase"] = "advisor"
        logger.agent_complete("market_intelligence", duration_ms=(time.time() - t0) * 1000)

    except Exception as e:
        logger.error("market_error", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Market intelligence failed: {e}"]
        state["current_phase"] = "advisor"

    return state
