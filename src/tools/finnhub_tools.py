"""Finnhub API wrapper with caching."""
import finnhub
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.config import config
from src.utils.cache import cache
from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QuoteResult:
    """Stock quote."""
    symbol: str
    current_price: float
    change: float
    change_percent: float
    high: float
    low: float
    open_price: float
    previous_close: float


class FinnhubTools:
    """
    Finnhub API wrapper. Free tier: 60 calls/min.

    All responses are cached (quotes: 5min, news: 1hr).
    """

    def __init__(self) -> None:
        self.client = finnhub.Client(api_key=config.finnhub_key)

    def get_quote(self, symbol: str) -> Optional[QuoteResult]:
        """Get real-time quote for a symbol."""
        cached = cache.get("quote", symbol)
        if cached:
            return QuoteResult(**cached)

        try:
            d = self.client.quote(symbol.upper())
            if d.get("c", 0) == 0:
                return None

            result = QuoteResult(
                symbol=symbol.upper(),
                current_price=d["c"], change=d["d"], change_percent=d["dp"],
                high=d["h"], low=d["l"], open_price=d["o"], previous_close=d["pc"],
            )
            cache.set("quote", symbol, result.__dict__)
            return result
        except Exception as e:
            logger.error("quote_error", symbol=symbol, error=str(e))
            return None

    def get_batch_quotes(self, symbols: List[str]) -> List[QuoteResult]:
        """Get quotes for multiple symbols (max 10)."""
        return [q for s in symbols[:10] if (q := self.get_quote(s))]

    def get_company_news(self, symbol: str, limit: int = 5) -> List[dict]:
        """Get recent news for a company (past 7 days)."""
        cached = cache.get("news", symbol)
        if cached:
            return cached

        try:
            end = datetime.now()
            start = end - timedelta(days=7)
            news = self.client.company_news(
                symbol.upper(),
                _from=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
            )
            results = [
                {"headline": n.get("headline", ""), "summary": n.get("summary", "")[:200]}
                for n in news[:limit]
            ]
            cache.set("news", symbol, results)
            return results
        except Exception as e:
            logger.error("news_error", symbol=symbol, error=str(e))
            return []


finnhub_tools = FinnhubTools()
