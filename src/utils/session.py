"""In-memory session storage with TTL."""

import time
from collections import OrderedDict

class SessionStore:
    def __init__(self, max_turns: int = 5, ttl_seconds: int = 300):
        self.max_turns = max_turns
        self.ttl = ttl_seconds
        # Mapping of session_id -> [(timestamp, label, confidence)]
        self._store: OrderedDict[str, list[tuple[float, str, float]]] = OrderedDict()

    def get_history(self, session_id: str) -> list[tuple[float, str, float]]:
        now = time.time()
        # Clean expired entries lazily when accessed
        self._store = OrderedDict((k, v) for k, v in self._store.items() if now - v[-1][0] < self.ttl)
        return self._store.get(session_id, [])

    def add_turn(self, session_id: str, label: str, confidence: float) -> None:
        history = self._store.get(session_id, [])
        history.append((time.time(), label, confidence))
        if len(history) > self.max_turns:
            history.pop(0)
        self._store[session_id] = history
        # Move to end to maintain insertion order for LRU-like behavior
        self._store.move_to_end(session_id)
