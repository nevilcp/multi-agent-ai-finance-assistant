"""Structured JSON logging."""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Optional


class StructuredLogger:
    """
    JSON logger that writes to stderr and an optional session file.

    Each entry: {"timestamp", "level", "event", "session_id", ...extras}
    """

    _session_id: Optional[str] = None
    _file_handler: Optional[logging.FileHandler] = None

    def __init__(self, name: str) -> None:
        self.name = name
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(h)
            self._logger.setLevel(logging.INFO)

    @classmethod
    def set_session(cls, session_id: str) -> None:
        """Set session ID and open a log file."""
        cls._session_id = session_id
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        cls._file_handler = logging.FileHandler(log_dir / f"session_{session_id}.jsonl")
        cls._file_handler.setFormatter(logging.Formatter("%(message)s"))

    def _emit(self, level: str, event: str, **kw: Any) -> None:
        entry = json.dumps({
            "ts": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "event": event,
            "session": self._session_id,
            **{k: v for k, v in kw.items() if v is not None},
        })
        getattr(self._logger, level.lower())(entry)
        if self._file_handler:
            self._file_handler.emit(
                logging.LogRecord(self.name, logging.INFO, "", 0, entry, (), None)
            )

    def info(self, event: str, **kw: Any) -> None:
        self._emit("INFO", event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._emit("WARNING", event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._emit("ERROR", event, **kw)

    def debug(self, event: str, **kw: Any) -> None:
        self._emit("DEBUG", event, **kw)

    def agent_start(self, agent: str, **kw: Any) -> None:
        self._emit("INFO", "agent_start", agent=agent, **kw)

    def agent_complete(self, agent: str, duration_ms: float, **kw: Any) -> None:
        self._emit("INFO", "agent_complete", agent=agent, duration_ms=duration_ms, **kw)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger."""
    return StructuredLogger(name)
