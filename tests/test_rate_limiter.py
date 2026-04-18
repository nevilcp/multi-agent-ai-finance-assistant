"""Tests for rate limiter."""
import pytest
import asyncio
from src.utils.rate_limiter import RateLimiter, ModelType


@pytest.fixture
def limiter():
    return RateLimiter()


@pytest.mark.asyncio
async def test_acquire_within_limit(limiter):
    """RPM tracking works."""
    for _ in range(5):
        await limiter.acquire(ModelType.FLASH)
    assert limiter.get_remaining(ModelType.FLASH)["rpm_remaining"] == 0


@pytest.mark.asyncio
async def test_daily_limit_tracking(limiter):
    """RPD tracking works."""
    await limiter.acquire(ModelType.FLASH)
    assert limiter.get_remaining(ModelType.FLASH)["rpd_remaining"] == 19


def test_wait_time_at_limit(limiter):
    """Wait time is positive when at RPM limit."""
    for _ in range(5):
        asyncio.run(limiter.acquire(ModelType.FLASH))
    wait = limiter.get_wait_time(ModelType.FLASH)
    assert 0 < wait <= 60
