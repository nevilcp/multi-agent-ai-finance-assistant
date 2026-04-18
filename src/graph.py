"""LangGraph workflow assembly with inline guardrails."""
import re
from langgraph.graph import StateGraph, END

from src.state import GraphState, InvestmentStrategy
from src.agents import (
    expense_categorizer_node,
    market_intelligence_node,
    wealth_advisor_node,
)
from src.observability.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Inline Output Guardrails
# ============================================================================

_BLOCKED = re.compile(
    r"day trading|margin|leverage|options|futures|crypto|penny stock|forex|binary options",
    re.IGNORECASE,
)
_MAX_SINGLE_ALLOC = 30  # percent


def _validate_strategy(strategy: InvestmentStrategy, risk: str) -> list[str]:
    """Validate strategy. Returns list of issues (empty = valid)."""
    issues = []

    # Check blocked terms
    text = " ".join([strategy.risk_assessment, *strategy.action_items,
                     *[r.rationale for r in strategy.investment_recommendations]])
    if _BLOCKED.search(text):
        issues.append("BLOCKED: Contains prohibited recommendation types")

    # Concentration check
    for r in strategy.investment_recommendations:
        if r.allocation_percent > _MAX_SINGLE_ALLOC:
            issues.append(f"{r.symbol} allocation ({r.allocation_percent}%) exceeds {_MAX_SINGLE_ALLOC}% limit")

    # Total allocation
    total = sum(r.allocation_percent for r in strategy.investment_recommendations)
    if total > 100:
        issues.append(f"Total allocation ({total}%) exceeds 100%")

    # Negative savings
    if strategy.monthly_savings_target < 0:
        issues.append("BLOCKED: Negative savings target")

    return issues


# ============================================================================
# Routing
# ============================================================================

def route_from_entry(state: GraphState) -> str:
    """Route based on watchlist mode."""
    if len(state.get("errors", [])) > 3:
        return "error_handler"
    if state.get("watchlist_mode"):
        return "market_intelligence"
    return "expense_categorizer"


def route_from_expense(state: GraphState) -> str:
    """Route from expense to market (or error)."""
    if len(state.get("errors", [])) > 3:
        return "error_handler"
    return "market_intelligence"


def route_from_market(state: GraphState) -> str:
    """Route from market to advisor (or error)."""
    if len(state.get("errors", [])) > 3:
        return "error_handler"
    return "wealth_advisor"


# ============================================================================
# Utility Nodes
# ============================================================================

async def entry_node(state: GraphState) -> GraphState:
    """Entry point — sets initial phase."""
    state = dict(state)
    state["current_phase"] = "started"
    return state


async def guardrails_node(state: GraphState) -> GraphState:
    """Validate strategy output with automated safety checks."""
    state = dict(state)
    if state.get("investment_strategy"):
        strategy = InvestmentStrategy.model_validate(state["investment_strategy"])
        risk = (state.get("user_preferences") or {}).get("risk_tolerance", "moderate")
        issues = _validate_strategy(strategy, risk)

        if issues:
            blocked = any(i.startswith("BLOCKED") for i in issues)
            logger.warning("guardrail_issues", issues=issues, blocked=blocked)
            state["errors"] = state.get("errors", []) + issues
            if blocked:
                state["investment_strategy"] = None
    return state


async def approval_node(state: GraphState) -> GraphState:
    """Present strategy for human approval before finalizing."""
    state = dict(state)

    if not state.get("investment_strategy"):
        # Nothing to approve (strategy was blocked by guardrails)
        state["approval_result"] = {"approved": False, "user_feedback": "No strategy to approve"}
        return state

    strategy = InvestmentStrategy.model_validate(state["investment_strategy"])

    # Display strategy summary
    print("\n" + "=" * 60)
    print("INVESTMENT STRATEGY SUMMARY")
    print("=" * 60)
    print(f"\nMonthly Savings Target: ${strategy.monthly_savings_target:,.2f}")
    print(f"Confidence: {strategy.confidence_score:.0%}")
    print("\nRECOMMENDATIONS:")
    for r in strategy.investment_recommendations:
        print(f"  • {r.action} {r.symbol} ({r.allocation_percent}%): {r.rationale[:80]}")
    print("\nACTION ITEMS:")
    for a in strategy.action_items:
        print(f"  • {a}")
    print("=" * 60)

    # Prompt for approval
    try:
        answer = input("\nApprove this strategy? [y/N]: ").strip().lower()
        approved = answer in ("y", "yes")

        feedback = None
        if not approved:
            feedback = input("Feedback (optional, press Enter to skip): ").strip() or None

        state["approval_result"] = {"approved": approved, "user_feedback": feedback}
        logger.info("approval_decision", approved=approved, feedback=feedback)
    except (EOFError, KeyboardInterrupt):
        # Non-interactive mode (e.g., piped input) — auto-approve
        state["approval_result"] = {"approved": True, "user_feedback": None}
        logger.info("approval_auto", reason="non-interactive")

    return state


async def completion_node(state: GraphState) -> GraphState:
    """Final node."""
    state = dict(state)
    state["current_phase"] = "complete"
    logger.info("workflow_complete", session=state.get("session_id"),
                has_strategy=state.get("investment_strategy") is not None)
    return state


async def error_handler_node(state: GraphState) -> GraphState:
    """Error terminal node."""
    state = dict(state)
    state["current_phase"] = "error"
    logger.error("workflow_error", errors=state.get("errors", []))
    return state


# ============================================================================
# Graph Assembly
# ============================================================================

def build_graph() -> StateGraph:
    """
    Build the workflow:
    entry -> [expense -> market | market] -> advisor -> guardrails -> approval -> complete
    """
    g = StateGraph(GraphState)

    g.add_node("entry", entry_node)
    g.add_node("expense_categorizer", expense_categorizer_node)
    g.add_node("market_intelligence", market_intelligence_node)
    g.add_node("wealth_advisor", wealth_advisor_node)
    g.add_node("guardrails", guardrails_node)
    g.add_node("approval", approval_node)
    g.add_node("complete", completion_node)
    g.add_node("error_handler", error_handler_node)

    g.set_entry_point("entry")

    g.add_conditional_edges("entry", route_from_entry, {
        "expense_categorizer": "expense_categorizer",
        "market_intelligence": "market_intelligence",
        "error_handler": "error_handler",
    })
    g.add_conditional_edges("expense_categorizer", route_from_expense, {
        "market_intelligence": "market_intelligence",
        "error_handler": "error_handler",
    })
    g.add_conditional_edges("market_intelligence", route_from_market, {
        "wealth_advisor": "wealth_advisor",
        "error_handler": "error_handler",
    })

    g.add_edge("wealth_advisor", "guardrails")
    g.add_edge("guardrails", "approval")
    g.add_edge("approval", "complete")
    g.add_edge("complete", END)
    g.add_edge("error_handler", END)

    return g.compile()


graph = build_graph()
