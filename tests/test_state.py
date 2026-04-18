"""Tests for state models."""
import pytest
from src.state import (
    ExpenseCategory, InvestmentRecommendation, create_initial_state
)


def test_expense_category_positive():
    cat = ExpenseCategory(category="Dining", total_amount=150.50, transaction_count=10)
    assert cat.total_amount == 150.50


def test_expense_category_rejects_negative():
    with pytest.raises(ValueError):
        ExpenseCategory(category="Test", total_amount=-100, transaction_count=1)


def test_allocation_in_range():
    rec = InvestmentRecommendation(symbol="AAPL", action="BUY", allocation_percent=25.0, rationale="Test")
    assert rec.allocation_percent == 25.0


def test_allocation_rejects_over_100():
    with pytest.raises(ValueError):
        InvestmentRecommendation(symbol="AAPL", action="BUY", allocation_percent=150, rationale="Test")


def test_create_initial_state():
    s = create_initial_state(session_id="t1", csv_path="f.csv", portfolio_symbols=["AAPL"])
    assert s["session_id"] == "t1"
    assert s["current_phase"] == "input"
