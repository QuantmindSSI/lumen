"""A12: Wear-Aware Write Batcher.

Input wire: async queue (Python asyncio), SQLite WAL
Output wire: Disk I/O
Secret sauce: SD/eMMC endurance optimisation
"""

import asyncio
import contextlib
import sqlite3
from collections import deque

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


class WearAwareBatcher:
    """
    Collects write operations and flushes as large sequential SQLite transactions.
    Target: write amplification < 1.1x vs naive per-query writes.
    """

    def __init__(
        self, conn: sqlite3.Connection, max_batch_size: int = 100, max_latency_ms: float = 500
    ):
        self.conn = conn
        self.queue: deque[tuple[str, tuple]] = deque()
        self.max_batch = max_batch_size
        self.max_latency = max_latency_ms / 1000.0
        self._lock = asyncio.Lock()
        self._flush_event = asyncio.Event()

    async def enqueue(self, sql: str, params: tuple = ()) -> None:
        async with self._lock:
            self.queue.append((sql, params))
            if len(self.queue) >= self.max_batch:
                self._flush_event.set()

    async def run(self) -> None:
        while True:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._flush_event.wait(), timeout=self.max_latency)
            await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self.queue:
                return
            batch = list(self.queue)
            self.queue.clear()
            self._flush_event.clear()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_flush, batch)

    def flush_sync(self, batch: list[tuple[str, tuple]]) -> None:
        """Synchronous batch flush exposed for non-async callers (e.g. consolidation)."""
        self._sync_flush(batch)

    def _sync_flush(self, batch: list[tuple[str, tuple]]) -> None:
        with self.conn:
            for sql, params in batch:
                self.conn.execute(sql, params)
        self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        if logger:
            logger.debug("wear_flush", count=len(batch))
