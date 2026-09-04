"""
Lightweight in-process prompt/response cache (TTL + LRU eviction).

For multi-instance deployments, swap this for Redis (interface is identical:
get/set/make_key) — see README "Scaling notes".
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from app.config import get_settings

settings = get_settings()


class TTLCache:
    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            timestamp, value = item
            if time.time() - timestamp > self.ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def stats(self) -> dict:
        return {"size": len(self._store), "max_size": self.max_size, "ttl_seconds": self.ttl}


def make_key(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


response_cache = TTLCache(max_size=settings.CACHE_MAX_SIZE, ttl_seconds=settings.CACHE_TTL_SECONDS)
