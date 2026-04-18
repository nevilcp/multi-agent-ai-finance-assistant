"""Rate limiter for Google AI Studio API calls.

Enforces per-model RPM and RPD limits based on Google AI Studio quotas:
- Gemini 3 Flash:      5 RPM, 250K TPM, 20 RPD
- Gemini 3.1 Flash Lite: 15 RPM, 250K TPM, 500 RPD
"""
import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Deque
from datetime import date

from src.config import config


class ModelType(Enum):
    """Available model types."""
    FLASH = "flash"
    FLASH_LITE = "flash_lite"


@dataclass
class _ModelState:
    """Internal rate limit tracking for one model."""
    minute_window: Deque[float] = field(default_factory=deque)
    daily_count: int = 0
    daily_tokens: int = 0
    last_reset: date = field(default_factory=date.today)


class RateLimiter:
    """
    Enforces Google AI Studio rate limits.

    Before every LLM call, the caller must `await acquire(model)`.
    - If RPM is full, the call blocks until a slot opens (auto-wait).
    - If RPD is exhausted, raises RuntimeError (cannot recover within session).
    """

    def __init__(self) -> None:
        self._limits = {
            ModelType.FLASH: {"rpm": config.google.flash_rpm, "rpd": config.google.flash_rpd},
            ModelType.FLASH_LITE: {"rpm": config.google.flash_lite_rpm, "rpd": config.google.flash_lite_rpd},
        }
        self._state: dict[ModelType, _ModelState] = {
            ModelType.FLASH: _ModelState(),
            ModelType.FLASH_LITE: _ModelState(),
        }
        self._lock = asyncio.Lock()

    def _reset_if_new_day(self, model: ModelType) -> None:
        """Reset daily counters on date change."""
        s = self._state[model]
        if s.last_reset < date.today():
            s.daily_count = 0
            s.daily_tokens = 0
            s.last_reset = date.today()

    def _prune_minute_window(self, model: ModelType) -> None:
        """Remove entries older than 60s."""
        window = self._state[model].minute_window
        now = time.time()
        while window and now - window[0] > 60:
            window.popleft()

    async def acquire(self, model: ModelType, estimated_tokens: int = 1000) -> None:
        """
        Wait until a request slot is available, then reserve it.

        Raises:
            RuntimeError: If daily limit is exhausted.
        """
        async with self._lock:
            self._reset_if_new_day(model)
            s = self._state[model]
            limits = self._limits[model]

            # Hard stop on daily limit
            if s.daily_count >= limits["rpd"]:
                raise RuntimeError(
                    f"Daily limit ({limits['rpd']}) reached for {model.value}. "
                    f"Resume tomorrow."
                )

            # Wait for RPM slot
            self._prune_minute_window(model)
            while len(s.minute_window) >= limits["rpm"]:
                wait = 60 - (time.time() - s.minute_window[0]) + 0.1
                if wait > 0:
                    await asyncio.sleep(wait)
                self._prune_minute_window(model)

            # Record request
            s.minute_window.append(time.time())
            s.daily_count += 1
            s.daily_tokens += estimated_tokens

    def get_remaining(self, model: ModelType) -> dict:
        """Get remaining quota for a model."""
        self._reset_if_new_day(model)
        self._prune_minute_window(model)
        s = self._state[model]
        limits = self._limits[model]
        return {
            "rpm_remaining": limits["rpm"] - len(s.minute_window),
            "rpd_remaining": limits["rpd"] - s.daily_count,
        }

    def get_wait_time(self, model: ModelType) -> float:
        """Seconds until next RPM slot opens (0 if available now)."""
        self._prune_minute_window(model)
        s = self._state[model]
        limits = self._limits[model]
        if len(s.minute_window) < limits["rpm"]:
            return 0.0
        return max(0.0, 60 - (time.time() - s.minute_window[0]))


# Global instance
rate_limiter = RateLimiter()
