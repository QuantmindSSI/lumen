"""D2: Event Memory Circular Buffer (RAM Tier).

Input wire: Python collections.deque
Output wire: A6 (store pipeline), D11 (consolidation)
Secret sauce: Configurable duration (default 24h), RAM-only, feeds downstream tiers
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Event:
    """A single raw interaction event."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    raw_text: str = ""
    source: str = "user"  # "user" | "agent"
    session_id: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class EventMemoryBuffer:
    """
    Lossless circular buffer of raw interactions. NOT persisted to SQLite.
    Consolidation (Task D11) drains old events into the Preference tier.
    """

    def __init__(self, max_events: int = 10_000, max_age_hours: float = 24.0) -> None:
        self._buffer: deque[Event] = deque(maxlen=max_events)
        self.max_age_hours = max_age_hours
        self._lock = threading.RLock()

    def append(self, event: Event) -> None:
        """Thread-safe append of a new event."""
        with self._lock:
            self._buffer.append(event)

    def query_since(self, since: float) -> list[Event]:
        """Return all events with timestamp >= since."""
        with self._lock:
            return [e for e in self._buffer if e.timestamp >= since]

    def drain_expired(self, cutoff_timestamp: float | None = None) -> list[Event]:
        """Return and remove events older than max_age_hours for consolidation.

        Args:
            cutoff_timestamp: If provided, drain events with timestamp < cutoff.
                              Defaults to now - max_age_hours.
        """
        if cutoff_timestamp is None:
            cutoff_timestamp = datetime.now(timezone.utc).timestamp() - (self.max_age_hours * 3600)
        with self._lock:
            expired = [e for e in self._buffer if e.timestamp < cutoff_timestamp]
            # Retain only non-expired events
            self._buffer = deque(
                [e for e in self._buffer if e.timestamp >= cutoff_timestamp],
                maxlen=self._buffer.maxlen,
            )
        return expired
