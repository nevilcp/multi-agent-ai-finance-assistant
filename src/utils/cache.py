"""Minimal in-memory TTL cache for API responses."""
import time
from typing import Any, Optional


class Cache:
    """
    In-memory cache with per-entry TTL.

    Default TTLs:
    - quote: 300s (5 min)
    - news: 3600s (1 hour)
    """

    DEFAULT_TTL = {"quote": 300, "news": 3600}

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)

    @staticmethod
    def _key(namespace: str, identifier: str) -> str:
        return f"{namespace}:{identifier}"

    def get(self, namespace: str, identifier: str) -> Optional[Any]:
        """Return cached value or None if missing/expired."""
        key = self._key(namespace, identifier)
        entry = self._store.get(key)
        if entry and entry[1] > time.time():
            return entry[0]
        if entry:
            del self._store[key]  # Expired — clean up
        return None

    def set(self, namespace: str, identifier: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with TTL (seconds)."""
        if ttl is None:
            ttl = self.DEFAULT_TTL.get(namespace, 300)
        self._store[self._key(namespace, identifier)] = (value, time.time() + ttl)

    def clear(self) -> int:
        """Clear all entries. Returns count cleared."""
        count = len(self._store)
        self._store.clear()
        return count


# Global instance
cache = Cache()
