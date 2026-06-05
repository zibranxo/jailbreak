"""Classification caching layer using Redis or LRU cache."""

import json
from functools import lru_cache
from typing import Any

try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False

class ClassificationCache:
    def __init__(self, redis_url: str | None = None, ttl: int = 3600):
        self.ttl = ttl
        self._redis = None
        self._local = None

        if redis_url and _HAS_REDIS:
            self._redis = redis.Redis.from_url(redis_url)
        else:
            self._local = lru_cache(maxsize=1000)(lambda key: None)
            # We'll actually manage a dict manually to handle set() properly
            self._local_dict: dict[str, Any] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        if self._redis:
            try:
                data = self._redis.get(key)
                return json.loads(data) if data else None
            except Exception:
                pass  # Fallback to None if Redis fails
        elif self._local is not None:
            return self._local_dict.get(key)
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        if self._redis:
            try:
                self._redis.setex(key, self.ttl, json.dumps(value))
            except Exception:
                pass
        elif self._local is not None:
            self._local_dict[key] = value
            if len(self._local_dict) > 1000:
                # simple eviction: pop an arbitrary item
                self._local_dict.pop(next(iter(self._local_dict)))
