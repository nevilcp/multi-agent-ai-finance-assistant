"""Pydantic models and LangGraph state definition."""
from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ============================================================================
# Domain Models
# ============================================================================

class ExpenseCategory(BaseModel):
    """Single expense category."""
    category: str
    total_amount: float
    transaction_count: int
    flagged_as_wasteful: bool = False
    waste_reasoning: Optional[str] = None

    @field_validator("total_amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Amount must be non-negative")
        return round(v, 2)


class ExpenseReport(BaseModel):
    """Complete expense analysis output."""
    total_spending: float
    category_breakdown: List[ExpenseCategory]
    top_wasteful_categories: List[str]
    monthly_average: float
    savings_potential: float
    transaction_count: int = 0
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None


class UserPreferences(BaseModel):
    """User investment preferences."""
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
    investment_horizon: Literal["short", "medium", "long"]
    exclude_sectors: List[str] = Field(default_factory=list)


class MarketDataPoint(BaseModel):
    """Single stock/ETF data point."""
    symbol: str
    current_price: float
    change_percent: float
    high: Optional[float] = None
    low: Optional[float] = None
    recommendation: Optional[Literal["BUY", "HOLD", "SELL"]] = None


class MarketIntelReport(BaseModel):
    """Market intelligence output."""
    portfolio_symbols: List[str]
    data_points: List[MarketDataPoint]
    market_summary: str
    data_freshness: Literal["live", "cached"] = "live"


class InvestmentRecommendation(BaseModel):
    """Single investment recommendation."""
    symbol: str
    action: Literal["BUY", "HOLD", "SELL", "REDUCE"]
    allocation_percent: float
    rationale: str

    @field_validator("allocation_percent")
    @classmethod
    def allocation_in_range(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError("Allocation must be between 0 and 100")
        return v


class InvestmentStrategy(BaseModel):
    """Complete investment strategy output."""
    monthly_savings_target: float
    savings_breakdown: dict[str, float]
    investment_recommendations: List[InvestmentRecommendation]
    risk_assessment: str
    action_items: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)


class ApprovalResult(BaseModel):
    """Human approval decision."""
    approved: bool
    user_feedback: Optional[str] = None


# ============================================================================
# LangGraph State
# ============================================================================

class GraphState(TypedDict):
    """Workflow state passed between nodes."""

    # Session
    session_id: str
    user_id: Optional[str]

    # Inputs
    raw_csv_path: str
    portfolio_symbols: List[str]
    user_preferences: Optional[dict]
    watchlist_mode: bool

    # Agent outputs
    expense_report: Optional[dict]
    market_intel: Optional[dict]
    investment_strategy: Optional[dict]
    approval_result: Optional[dict]

    # Observability
    errors: List[str]
    current_phase: str
    feedback_applied: bool


def create_initial_state(
    session_id: str,
    csv_path: str,
    portfolio_symbols: List[str],
    user_preferences: Optional[UserPreferences] = None,
    user_id: Optional[str] = None,
    watchlist_mode: bool = False,
) -> GraphState:
    """Create initial state for a new workflow run."""
    return GraphState(
        session_id=session_id,
        user_id=user_id,
        raw_csv_path=csv_path,
        portfolio_symbols=portfolio_symbols,
        user_preferences=user_preferences.model_dump() if user_preferences else None,
        watchlist_mode=watchlist_mode,
        expense_report=None,
        market_intel=None,
        investment_strategy=None,
        approval_result=None,
        errors=[],
        current_phase="input",
        feedback_applied=False,
    )


def compute_confidence_score(
    strategy: InvestmentStrategy,
    state: GraphState,
) -> float:
    """Deterministically compute confidence_score from pipeline signals.

    Starts at 1.0 and applies deductions based on data completeness:
      - No expense report (watchlist mode):         -0.20
      - Pipeline errors: -0.10 per error, max       -0.30
      - Stale/cached market data:                    -0.10
      - Under-diversified (< 2 recommendations):    -0.15
      - No identified savings potential:             -0.10

    Returns a float in [0.0, 1.0] rounded to 2 decimal places.
    """
    score = 1.0

    # (1) No expense data — advice is market-only, less personalised
    if state.get("expense_report") is None:
        score -= 0.20

    # (2) Prior pipeline errors degrade data quality
    errors = state.get("errors", [])
    if errors:
        score -= min(len(errors) * 0.10, 0.30)

    # (3) Stale market data reduces price reliability
    market_intel = state.get("market_intel")
    if market_intel:
        freshness = market_intel.get("data_freshness", "live")
        if freshness != "live":
            score -= 0.10
    else:
        # No market data at all is worse than stale
        score -= 0.10

    # (4) Under-diversified strategy
    if len(strategy.investment_recommendations) < 2:
        score -= 0.15

    # (5) No savings potential identified
    expense_report = state.get("expense_report")
    if expense_report is None or not expense_report.get("savings_potential"):
        score -= 0.10

    return round(max(score, 0.0), 2)
