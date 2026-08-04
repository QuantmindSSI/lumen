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
            seconds=self.config.scheduler_granularity,
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
        conn = None
        batcher = None
        try:
            from lumen.data.schema import get_connection
            conn = get_connection(self.config)
        except Exception as exc:
            logger.error("sleep_consolidation_error", stage="connect", error=str(exc))
            return

        try:
            from lumen.sovereign.wear import WearAwareBatcher
            batcher = WearAwareBatcher(conn)
        except Exception as exc:
            logger.error("sleep_consolidation_error", stage="batcher_init", error=str(exc))
            conn.close()
            return

        try:
            run_consolidation_pass(self.config, batcher=batcher)
        except Exception as exc:
            logger.error("sleep_consolidation_error", stage="consolidation", error=str(exc))
            conn.close()
            return

        try:
            from lumen.force.mnemonic.forgetting_l1_decay import ebbinghaus_decay
            ebbinghaus_decay(conn, batcher=batcher)
        except Exception as exc:
            logger.error("sleep_consolidation_error", stage="l1_decay", error=str(exc))
            conn.close()
            return

        try:
            from lumen.force.mnemonic.forgetting_l3_budget import budget_curated_eviction
            budget_curated_eviction(conn, self.config, batcher=batcher)
        except Exception as exc:
            logger.error("sleep_consolidation_error", stage="l3_eviction", error=str(exc))
            conn.close()
            return

        try:
            from lumen.curiosity import curiosity_probe
            curiosity_probe(conn, limit=10)
        except Exception as exc:
            logger.error("sleep_consolidation_error", stage="curiosity", error=str(exc))

        try:
            if batcher and batcher.queue:
                batcher.flush_sync(list(batcher.queue))
                batcher.queue.clear()
        except Exception as exc:
            logger.error("sleep_consolidation_error", stage="flush", error=str(exc))
        finally:
            conn.close()

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
