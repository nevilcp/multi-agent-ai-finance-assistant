"""
Pytest fixtures for the eval framework.

Provides reusable test fixtures for:
- Temporary CSV file creation from synthetic bank-statement constants
- Pre-populated GraphState dicts via create_initial_state()
- Realistic mock outputs for each pipeline stage (expense, market, strategy)
- Strategy fixtures that exercise guardrail pass / fail / edge-case paths
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict

import pytest

from src.state import create_initial_state, GraphState

from tests.evals.fixtures.csv_fixtures import (
    CSV_FIXTURES,
    PORTFOLIO_NORMAL,
)


# ============================================================================
# CSV file factory
# ============================================================================

@pytest.fixture
def tmp_csv(tmp_path: Path):
    """
    Factory fixture: write a named synthetic CSV to a temp file and return its path.

    Usage in tests::

        def test_something(tmp_csv):
            csv_path = tmp_csv("normal")
            assert Path(csv_path).exists()

    Valid fixture names: "normal", "wasteful", "empty", "injection", "minimal".
    """

    def _make(fixture_name: str) -> str:
        if fixture_name not in CSV_FIXTURES:
            raise ValueError(
                f"Unknown CSV fixture '{fixture_name}'. "
                f"Choose from: {sorted(CSV_FIXTURES)}"
            )
        csv_file = tmp_path / f"{fixture_name}_statement.csv"
        csv_file.write_text(CSV_FIXTURES[fixture_name], encoding="utf-8")
        return str(csv_file)

    return _make


# ============================================================================
# GraphState seed
# ============================================================================

@pytest.fixture
def base_state() -> GraphState:
    """
    Return a valid initial GraphState with blank csv_path and default portfolio.

    Callers can override individual keys after receiving the dict::

        def test_flow(base_state, tmp_csv):
            state = {**base_state, "raw_csv_path": tmp_csv("normal")}
    """
    return create_initial_state(
        session_id=f"eval-{uuid.uuid4().hex[:8]}",
        csv_path="",
        portfolio_symbols=PORTFOLIO_NORMAL,
        watchlist_mode=False,
    )


# ============================================================================
# Mock agent outputs — Expense Report (wasteful scenario)
# ============================================================================

@pytest.fixture
def mock_expense_report() -> Dict[str, Any]:
    """
    Realistic ExpenseReport.model_dump() for the WASTEFUL CSV scenario.

    Key numbers:
    - total_spending = $1,885.65
    - savings_potential = $285.00  (streaming + delivery + unused gym)
    - 7 categories; 3 flagged as wasteful
    """
    return {
        "total_spending": 1885.65,
        "category_breakdown": [
            {
                "category": "Streaming Services",
                "total_amount": 49.96,
                "transaction_count": 4,
                "flagged_as_wasteful": True,
                "waste_reasoning": (
                    "Four overlapping streaming subscriptions "
                    "(Netflix, Hulu, Disney+, Spotify). "
                    "Consolidating to 1-2 services could save ~$25/month."
                ),
            },
            {
                "category": "Food Delivery",
                "total_amount": 252.70,
                "transaction_count": 6,
                "flagged_as_wasteful": True,
                "waste_reasoning": (
                    "Six food-delivery orders totalling $252.70. "
                    "Meal-prepping could reduce this by 60-70%."
                ),
            },
            {
                "category": "Gym Membership",
                "total_amount": 49.99,
                "transaction_count": 1,
                "flagged_as_wasteful": True,
                "waste_reasoning": (
                    "Planet Fitness membership explicitly labelled "
                    "'NEVER USED'. Cancel immediately to save $49.99/month."
                ),
            },
            {
                "category": "Groceries",
                "total_amount": 462.45,
                "transaction_count": 3,
                "flagged_as_wasteful": False,
                "waste_reasoning": None,
            },
            {
                "category": "Utilities",
                "total_amount": 309.50,
                "transaction_count": 3,
                "flagged_as_wasteful": False,
                "waste_reasoning": None,
            },
            {
                "category": "Transportation",
                "total_amount": 58.40,
                "transaction_count": 1,
                "flagged_as_wasteful": False,
                "waste_reasoning": None,
            },
            {
                "category": "Shopping",
                "total_amount": 422.65,
                "transaction_count": 2,
                "flagged_as_wasteful": False,
                "waste_reasoning": None,
            },
        ],
        "top_wasteful_categories": [
            "Food Delivery",
            "Streaming Services",
            "Gym Membership",
        ],
        "monthly_average": 1885.65,
        "savings_potential": 285.00,
        "transaction_count": 20,
        "date_range_start": "2025-01-01",
        "date_range_end": "2025-01-30",
    }


# ============================================================================
# Mock agent outputs — Market Intelligence
# ============================================================================

@pytest.fixture
def mock_market_intel() -> Dict[str, Any]:
    """
    Realistic MarketIntelReport.model_dump() for PORTFOLIO_NORMAL.

    3 data points matching AAPL / MSFT / VOO with plausible prices.
    market_summary is capped at 500 characters.
    """
    return {
        "portfolio_symbols": ["AAPL", "MSFT", "VOO"],
        "data_points": [
            {
                "symbol": "AAPL",
                "current_price": 178.50,
                "change_percent": 1.2,
                "high": 180.25,
                "low": 176.80,
                "recommendation": "BUY",
            },
            {
                "symbol": "MSFT",
                "current_price": 415.30,
                "change_percent": -0.5,
                "high": 419.00,
                "low": 413.10,
                "recommendation": "HOLD",
            },
            {
                "symbol": "VOO",
                "current_price": 445.20,
                "change_percent": 0.3,
                "high": 447.00,
                "low": 443.50,
                "recommendation": "BUY",
            },
        ],
        "market_summary": (
            "US equities showed mixed performance. AAPL gained 1.2% on strong "
            "iPhone 16 demand data and services-revenue beat. MSFT dipped 0.5% "
            "amid Azure growth deceleration concerns despite solid enterprise "
            "AI adoption. VOO edged up 0.3%, reflecting broad market resilience. "
            "The S&P 500 held near all-time highs with the Fed signalling a "
            "data-dependent rate path. Sector rotation favoured tech and "
            "consumer discretionary over utilities and real estate."
        )[:500],
        "data_freshness": "live",
    }


# ============================================================================
# Mock strategy — PASSES all guardrails
# ============================================================================

@pytest.fixture
def mock_strategy_safe() -> Dict[str, Any]:
    """
    Valid InvestmentStrategy.model_dump() that passes every guardrail:
    - No blocked terms
    - No single allocation > 30%
    - Total allocation ≤ 100%
    - Positive monthly_savings_target
    """
    return {
        "monthly_savings_target": 245.00,
        "savings_breakdown": {
            "streaming_reduction": 25.00,
            "food_delivery_reduction": 170.00,
            "gym_cancellation": 50.00,
        },
        "investment_recommendations": [
            {
                "symbol": "VOO",
                "action": "BUY",
                "allocation_percent": 30.0,
                "rationale": (
                    "Broad S&P 500 exposure via VOO provides diversified "
                    "market participation at a low expense ratio of 0.03%. "
                    "Ideal core holding for long-term wealth accumulation."
                ),
            },
            {
                "symbol": "AAPL",
                "action": "HOLD",
                "allocation_percent": 25.0,
                "rationale": (
                    "AAPL shows strong momentum with 1.2% daily gain. "
                    "Maintain current position; services revenue growth "
                    "supports a positive medium-term outlook."
                ),
            },
            {
                "symbol": "MSFT",
                "action": "REDUCE",
                "allocation_percent": 20.0,
                "rationale": (
                    "MSFT underperformed today (-0.5%). Azure growth "
                    "deceleration warrants trimming to lock in gains. "
                    "Redirect freed capital toward VOO for diversification."
                ),
            },
        ],
        "risk_assessment": (
            "Moderate risk profile. Portfolio is diversified across large-cap "
            "tech (AAPL, MSFT) and a broad index ETF (VOO). No single "
            "position exceeds 30%. Recommended monthly savings of $245 is "
            "achievable based on identified spending reductions."
        ),
        "action_items": [
            "Cancel unused Planet Fitness membership ($49.99/mo savings)",
            "Consolidate streaming to Netflix + Spotify ($23/mo savings)",
            "Reduce food delivery to max 2x/week ($170/mo savings)",
            "Set up automatic $245/mo transfer to brokerage account",
            "Rebalance MSFT position from 25% to 20% over next 30 days",
        ],
        "confidence_score": 0.72,
    }


# ============================================================================
# Mock strategy — FAILS guardrails (blocked term)
# ============================================================================

@pytest.fixture
def mock_strategy_blocked() -> Dict[str, Any]:
    """
    InvestmentStrategy.model_dump() that FAILS the blocked-term guardrail.

    The action_items list contains "consider crypto allocation" which matches
    the _BLOCKED regex pattern in graph.py.
    """
    return {
        "monthly_savings_target": 200.00,
        "savings_breakdown": {
            "subscription_reduction": 50.00,
            "dining_reduction": 150.00,
        },
        "investment_recommendations": [
            {
                "symbol": "VOO",
                "action": "BUY",
                "allocation_percent": 30.0,
                "rationale": "Core index fund position for diversification.",
            },
            {
                "symbol": "AAPL",
                "action": "HOLD",
                "allocation_percent": 25.0,
                "rationale": "Maintain existing position on positive momentum.",
            },
        ],
        "risk_assessment": (
            "Moderate risk. Diversified between index and large-cap equity."
        ),
        "action_items": [
            "Cancel redundant streaming subscriptions",
            "Reduce dining-out budget by 40%",
            "Consider crypto allocation for higher growth potential",
            "Set up auto-invest for $200/mo",
        ],
        "confidence_score": 0.65,
    }


# ============================================================================
# Mock strategy — FAILS guardrails (overconcentrated)
# ============================================================================

@pytest.fixture
def mock_strategy_overconcentrated() -> Dict[str, Any]:
    """
    InvestmentStrategy.model_dump() that FAILS the concentration guardrail.

    Single recommendation: BUY AAPL at 85% — exceeds the 30% per-symbol cap
    defined by _MAX_SINGLE_ALLOC in graph.py.
    """
    return {
        "monthly_savings_target": 300.00,
        "savings_breakdown": {
            "total_identified": 300.00,
        },
        "investment_recommendations": [
            {
                "symbol": "AAPL",
                "action": "BUY",
                "allocation_percent": 85.0,
                "rationale": (
                    "All-in on AAPL due to exceptional services growth "
                    "and record iPhone revenue. Concentrated position "
                    "maximises upside potential."
                ),
            },
        ],
        "risk_assessment": (
            "High risk. Single-stock concentration of 85% in AAPL "
            "creates significant idiosyncratic exposure."
        ),
        "action_items": [
            "Allocate 85% of monthly savings to AAPL shares",
            "Monitor quarterly earnings closely",
        ],
        "confidence_score": 0.55,
    }


# ============================================================================
# Terminal summary hook — live eval summary table & JSONL persistence
# ============================================================================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print live eval summary table and persist results to JSONL.

    Uses pytest_terminal_summary instead of pytest_sessionfinish so that
    output is reliably written to the terminal via the reporter API.
    Imports the RESULTS list lazily so offline-only runs are unaffected.
    """
    try:
        from tests.evals.test_live_quality import RESULTS, _RESULTS_DIR
    except Exception:
        # Live module was skipped or not collected — nothing to report.
        return

    if not RESULTS:
        return

    import json
    from dataclasses import asdict
    from datetime import datetime

    w = terminalreporter

    # ---- Console summary ----
    w.section("LIVE EVAL RESULTS SUMMARY", sep="=")
    w.line(f"  {'Scenario':<28} {'Check':<34} {'Status':<8} {'Score':<7} Detail")
    w.line("-" * 90)

    blocking_checks = {
        "schema_valid", "no_blocked_terms", "no_injected_content_in_output",
    }
    passed_count = 0
    blocking_failures = 0

    for r in RESULTS:
        if r.passed:
            status = "PASS"
            passed_count += 1
        elif r.check == "injection_detected":
            status = "WARN"  # informational only
            passed_count += 1  # don't count as failure
        else:
            status = "FAIL"
            if r.check in blocking_checks:
                blocking_failures += 1

        detail_trunc = r.detail[:40] + "…" if len(r.detail) > 40 else r.detail
        w.line(f"  {r.scenario:<28} {r.check:<34} {status:<8} {r.score:<7.2f} {detail_trunc}")

    w.line("-" * 90)
    w.line(f"  Overall: {passed_count}/{len(RESULTS)} checks passed, "
           f"{blocking_failures} blocking failure(s)")
    w.line("=" * 90)

    # ---- JSONL persistence ----
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    jsonl_path = _RESULTS_DIR / f"live_{ts}.jsonl"

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for r in RESULTS:
            line = json.dumps({
                **asdict(r),
                "timestamp": ts,
            })
            f.write(line + "\n")

    w.line(f"\n  Results written to: {jsonl_path}")
