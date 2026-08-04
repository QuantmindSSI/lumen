"""C6: Sleep-Phase Consolidation Scheduler.

Idle-time palace maintenance with zero latency impact.
"""

from __future__ import annotations

from typing import Any

import psutil

from lumen.config import LumenConfig
from lumen.force.mnemonic.consolidation import run_consolidation_pass
from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


class SleepScheduler:
    """Background scheduler that runs consolidation during idle periods."""

    def __init__(self, config: LumenConfig):
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except Exception as exc:
            raise RuntimeError(
                "apscheduler is required for SleepScheduler but is not installed. "
                "Install it with: pip install 'apscheduler>=3.10'"
            ) from exc

        self.config = config
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self._maybe_consolidate,
            trigger=CronTrigger(hour=3, minute=0),
            id="sleep_consolidation",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._opportunistic_check,
            trigger="interval",
            minutes=5,
            id="opportunistic",
        )

    def start(self) -> None:
        """Start the background scheduler."""
        self.scheduler.start()

    def stop(self) -> None:
        """Shut down the scheduler gracefully."""
        self.scheduler.shutdown(wait=True)

    @property
    def is_running(self) -> bool:
        """Return whether the scheduler is currently running."""
        return self.scheduler.running

    def run_now(self) -> None:
        """Trigger a manual consolidation pass immediately."""
        self._maybe_consolidate()

    def _maybe_consolidate(self) -> None:
        """Run consolidation if conditions are met, catching all exceptions."""
        if not self._should_run():
            return
        logger.info("sleep_consolidation_start")
        try:
            from lumen.data.schema import get_connection
            from lumen.force.mnemonic.forgetting_l1_decay import ebbinghaus_decay
            from lumen.sovereign.wear import WearAwareBatcher

            conn = get_connection(self.config)
            try:
                batcher = WearAwareBatcher(conn)
                run_consolidation_pass(self.config, batcher=batcher)
                ebbinghaus_decay(conn, batcher=batcher)
                from lumen.force.mnemonic.forgetting_l3_budget import budget_curated_eviction
                budget_curated_eviction(conn, self.config, batcher=batcher)
                from lumen.curiosity import curiosity_probe
                curiosity_probe(conn, limit=10)
                if batcher.queue:
                    batcher.flush_sync(list(batcher.queue))
                    batcher.queue.clear()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.error("sleep_consolidation_error", error=str(exc))
        else:
            logger.info("sleep_consolidation_end")

    def _opportunistic_check(
        self,
        cpu_percent: float | None = None,
        battery: Any | None = None,
    ) -> None:
        """Check system load and battery; trigger consolidation if idle.

        Args:
            cpu_percent: Optional injected CPU reading for testing.
            battery: Optional injected battery object for testing.
        """
        if cpu_percent is None:
            cpu_percent = psutil.cpu_percent(interval=None)
        if battery is None:
            battery = psutil.sensors_battery()
        on_ac = battery is None or battery.power_plugged
        threshold = getattr(self.config, "consolidation_cpu_percent", 10.0)
        if cpu_percent < threshold and on_ac:
            self._maybe_consolidate()

    def _should_run(self, battery: Any | None = None) -> bool:
        """Return True if battery state permits consolidation.

        Args:
            battery: Optional injected battery object for testing.
        """
        if battery is None:
            battery = psutil.sensors_battery()
        threshold = getattr(self.config, "consolidation_battery_threshold", 50)
        return battery is None or battery.power_plugged or battery.percent >= threshold
