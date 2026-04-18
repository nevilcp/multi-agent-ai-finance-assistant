"""Expense categorizer — single-shot batch analysis."""
from src.state import GraphState, ExpenseReport
from src.utils.gemini_client import gemini_client, ModelType
from src.utils.csv_parser import csv_parser
from src.observability.logger import get_logger

logger = get_logger(__name__)

PROMPT = """You are an expense analysis expert. Categorize the following bank transactions.

Categories: Dining, Groceries, Transport, Utilities, Subscriptions, Shopping, Entertainment, Health, Insurance, Housing, Other

For each category:
- total_amount: sum of expenses
- transaction_count: number of transactions
- flagged_as_wasteful: true if unnecessary (unused subscriptions, excessive spending)
- waste_reasoning: explain why it's wasteful (if flagged)

Also calculate:
- total_spending: sum of all expenses
- monthly_average: total / months in range
- savings_potential: how much could be saved by cutting waste
- top_wasteful_categories: list of category names flagged as wasteful

{transactions}
"""


async def expense_categorizer_node(state: GraphState) -> GraphState:
    """
    Analyze expenses in a single LLM call (Gemini 3.1 Flash Lite).

    1. Parse CSV deterministically
    2. Pass formatted text to LLM for categorization
    """
    logger.agent_start("expense_categorizer")
    state = dict(state)

    try:
        parsed = csv_parser.parse(state["raw_csv_path"])

        if parsed.threats_detected:
            state["errors"] = state.get("errors", []) + [
                f"Sanitized {len(parsed.threats_detected)} injection attempts"
            ]

        text = csv_parser.to_llm_text(parsed)
        if len(text) > 50000:
            text = text[:50000] + "\n... (truncated)"

        report = await gemini_client.generate_structured(
            prompt=PROMPT.format(transactions=text),
            response_model=ExpenseReport,
            model=ModelType.FLASH_LITE,
            temperature=0.2,
        )

        report.transaction_count = parsed.total_transactions
        report.date_range_start = parsed.date_range_start
        report.date_range_end = parsed.date_range_end

        state["expense_report"] = report.model_dump()
        state["current_phase"] = "market"
        logger.agent_complete("expense_categorizer", duration_ms=0, savings=report.savings_potential)

    except Exception as e:
        logger.error("expense_error", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Expense analysis failed: {e}"]
        state["current_phase"] = "market"

    return state
