"""Bank statement CSV parser with format detection and input sanitization."""
import re
import pandas as pd
from pathlib import Path
from typing import List
from dataclasses import dataclass

from src.observability.logger import get_logger

logger = get_logger(__name__)

# Prompt injection patterns to neutralize
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|prior|all|above)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your)", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
]


@dataclass
class ParsedStatement:
    """Parsed bank statement."""
    df: pd.DataFrame
    total_transactions: int
    date_range_start: str
    date_range_end: str
    detected_columns: dict[str, str]
    threats_detected: List[str]


class CSVParser:
    """
    CSV parser with auto column detection and built-in sanitization.

    Handles various bank formats by mapping common column name variants
    to normalized names. Sanitizes all text fields against prompt injection.
    """

    COLUMN_MAPPINGS = {
        "date": ["date", "transaction date", "trans date", "posted date"],
        "description": ["description", "memo", "narrative", "details"],
        "amount": ["amount", "transaction amount", "value"],
        "debit": ["debit", "withdrawal", "out"],
        "credit": ["credit", "deposit", "in"],
        "balance": ["balance", "running balance"],
    }

    def _detect_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """Map CSV columns to normalized names."""
        mappings = {}
        lower_cols = {c.lower().strip(): c for c in df.columns}
        for norm, variants in self.COLUMN_MAPPINGS.items():
            for v in variants:
                if v in lower_cols:
                    mappings[norm] = lower_cols[v]
                    break
        return mappings

    def _sanitize(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """Neutralize prompt injection in all text columns."""
        threats = []
        df = df.copy()
        for col in df.select_dtypes(include=["object"]).columns:
            for idx, val in df[col].items():
                if isinstance(val, str):
                    for pattern in _INJECTION_PATTERNS:
                        if pattern.search(val):
                            threats.append(pattern.pattern[:30])
                            df.at[idx, col] = pattern.sub("[REDACTED]", val)
        return df, list(set(threats))

    def parse(self, path: str | Path) -> ParsedStatement:
        """
        Parse a bank statement CSV.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If required columns are missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        # Try common encodings
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode: {path}")

        mappings = self._detect_columns(df)
        if "date" not in mappings or "description" not in mappings:
            raise ValueError("Missing required columns: date, description")

        # Sanitize
        df, threats = self._sanitize(df)
        if threats:
            logger.warning("injection_detected", count=len(threats))

        # Normalize amount
        if "amount" in mappings:
            df["normalized_amount"] = pd.to_numeric(df[mappings["amount"]], errors="coerce")
        elif "debit" in mappings and "credit" in mappings:
            debit = pd.to_numeric(df[mappings["debit"]], errors="coerce").fillna(0)
            credit = pd.to_numeric(df[mappings["credit"]], errors="coerce").fillna(0)
            df["normalized_amount"] = credit - debit
        else:
            raise ValueError("No amount column found")

        # Parse and sort dates
        df["normalized_date"] = pd.to_datetime(df[mappings["date"]], errors="coerce", dayfirst=True)
        df = df.sort_values("normalized_date")

        valid = df["normalized_date"].dropna()
        start = valid.min().strftime("%Y-%m-%d") if len(valid) else "unknown"
        end = valid.max().strftime("%Y-%m-%d") if len(valid) else "unknown"

        logger.info("csv_parsed", rows=len(df), date_range=f"{start} to {end}")

        return ParsedStatement(
            df=df,
            total_transactions=len(df),
            date_range_start=start,
            date_range_end=end,
            detected_columns=mappings,
            threats_detected=threats,
        )

    def to_llm_text(self, parsed: ParsedStatement) -> str:
        """Format parsed data as condensed text for LLM prompts."""
        desc_col = parsed.detected_columns["description"]
        lines = [
            f"Bank Statement ({parsed.total_transactions} transactions, "
            f"{parsed.date_range_start} to {parsed.date_range_end})",
            "",
        ]
        for _, row in parsed.df.iterrows():
            date = row["normalized_date"].strftime("%Y-%m-%d") if pd.notna(row["normalized_date"]) else "N/A"
            desc = str(row[desc_col])[:50]
            amt = row["normalized_amount"]
            lines.append(f"  {date} | {amt:+.2f} | {desc}")
        return "\n".join(lines)


csv_parser = CSVParser()
