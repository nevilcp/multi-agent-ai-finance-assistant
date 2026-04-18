"""CLI entry point."""
import asyncio
import argparse
import uuid
from pathlib import Path

from src.state import create_initial_state, UserPreferences, InvestmentStrategy
from src.graph import graph
from src.observability.logger import StructuredLogger, get_logger
from src.utils.rate_limiter import rate_limiter, ModelType

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Agentic Personal Portfolio & Expense Synthesizer",
        epilog="Examples:\n"
               "  %(prog)s --csv statement.csv --portfolio AAPL,MSFT,VOO\n"
               "  %(prog)s --portfolio AAPL,MSFT --watchlist --risk aggressive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=str, help="Path to bank statement CSV")
    p.add_argument("--portfolio", type=str, required=True, help="Comma-separated symbols")
    p.add_argument("--risk", choices=["conservative", "moderate", "aggressive"], default="moderate")
    p.add_argument("--horizon", choices=["short", "medium", "long"], default="medium")
    p.add_argument("--watchlist", action="store_true", help="Skip expense analysis")
    p.add_argument("--export", choices=["json", "markdown"], help="Export results")
    return p.parse_args()


def show_rate_limits() -> None:
    """Print current rate limit status."""
    f = rate_limiter.get_remaining(ModelType.FLASH)
    fl = rate_limiter.get_remaining(ModelType.FLASH_LITE)
    print(f"\n📊 Gemini 3 Flash: {f['rpd_remaining']}/20 daily, {f['rpm_remaining']}/5 per min")
    print(f"   Gemini 3.1 Flash Lite: {fl['rpd_remaining']}/500 daily, {fl['rpm_remaining']}/15 per min\n")


def show_results(state: dict) -> None:
    """Print workflow results."""
    print("\n" + "=" * 60 + "\nRESULTS\n" + "=" * 60)

    if state.get("errors"):
        print("\n⚠️  Issues:")
        for e in state["errors"]:
            print(f"  • {e}")

    if state.get("expense_report"):
        r = state["expense_report"]
        print(f"\n💰 Expenses: ${r['total_spending']:,.2f} total, ${r['savings_potential']:,.2f} savings potential")
        if r.get("top_wasteful_categories"):
            print(f"   Wasteful: {', '.join(r['top_wasteful_categories'][:3])}")

    if state.get("market_intel"):
        print("\n📈 Market:")
        for dp in state["market_intel"].get("data_points", [])[:5]:
            e = "🟢" if dp["change_percent"] > 0 else "🔴"
            print(f"  {e} {dp['symbol']}: ${dp['current_price']:.2f} ({dp['change_percent']:+.2f}%)")

    if state.get("investment_strategy"):
        s = InvestmentStrategy.model_validate(state["investment_strategy"])
        print(f"\n📋 Strategy (confidence: {s.confidence_score:.0%}):")
        print(f"   Save ${s.monthly_savings_target:,.2f}/month")
        for a in s.action_items[:5]:
            print(f"   • {a}")

    print("=" * 60)


def export_results(state: dict, fmt: str, sid: str) -> None:
    """Export to file."""
    out = Path("output")
    out.mkdir(exist_ok=True)

    if fmt == "json":
        import json
        f = out / f"results_{sid}.json"
        f.write_text(json.dumps(state, indent=2, default=str))
    elif fmt == "markdown":
        f = out / f"results_{sid}.md"
        lines = [f"# Results — {sid}\n"]
        if state.get("investment_strategy"):
            s = state["investment_strategy"]
            lines.append(f"**Savings:** ${s['monthly_savings_target']:,.2f}/mo\n")
            for a in s.get("action_items", []):
                lines.append(f"- {a}")
        f.write_text("\n".join(lines))

    print(f"📁 Exported to: {f}")


async def main() -> None:
    """Entry point."""
    args = parse_args()
    sid = str(uuid.uuid4())[:8]
    StructuredLogger.set_session(sid)

    print(f"🚀 Wealth Assistant (Session: {sid})")
    show_rate_limits()

    if not args.watchlist and not args.csv:
        print("❌ Provide --csv or use --watchlist"); return
    if args.csv and not Path(args.csv).exists():
        print(f"❌ File not found: {args.csv}"); return

    symbols = [s.strip().upper() for s in args.portfolio.split(",")]
    prefs = UserPreferences(risk_tolerance=args.risk, investment_horizon=args.horizon)

    state = create_initial_state(
        session_id=sid, csv_path=args.csv or "", portfolio_symbols=symbols,
        user_preferences=prefs, watchlist_mode=args.watchlist,
    )

    print("⏳ Running analysis...\n")
    try:
        result = await graph.ainvoke(state)
        show_results(result)
        if args.export:
            export_results(result, args.export, sid)
        show_rate_limits()
    except Exception as e:
        logger.error("failed", error=str(e))
        print(f"❌ Failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
