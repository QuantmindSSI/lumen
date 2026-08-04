"""Tests for lumen.lumen.sleep."""

from unittest.mock import MagicMock, patch

import pytest

from lumen.config import LumenConfig
from lumen.sleep import SleepScheduler


@pytest.fixture
def scheduler():
    """Return a SleepScheduler with a default test config."""
    config = LumenConfig(
        vector_index="sqlite-vec",
        device="generic",
        store_path="/tmp/.lumen-test",
        consolidation_cpu_percent=10.0,
        consolidation_battery_threshold=50,
    )
    return SleepScheduler(config=config)


def test_scheduler_starts_and_stops(scheduler):
    """SleepScheduler should start and stop without error."""
    scheduler.start()
    assert scheduler.is_running
    scheduler.stop()
    assert not scheduler.is_running


def test_should_run_false_on_low_battery(scheduler):
    """_should_run should return False when battery is low and unplugged."""
    mock_battery = MagicMock()
    mock_battery.power_plugged = False
    mock_battery.percent = 30
    assert not scheduler._should_run(battery=mock_battery)


def test_should_run_true_on_ac_power(scheduler):
    """_should_run should return True when on AC power."""
    mock_battery = MagicMock()
    mock_battery.power_plugged = True
    mock_battery.percent = 30
    assert scheduler._should_run(battery=mock_battery)


def test_should_run_true_when_no_battery(scheduler):
    """_should_run should return True when there is no battery (desktop)."""
    assert scheduler._should_run(battery=None)


def test_opportunistic_check_triggers_when_idle(scheduler):
    """_opportunistic_check should trigger consolidation when CPU low and on AC."""
    mock_battery = MagicMock()
    mock_battery.power_plugged = True
    mock_battery.percent = 100

    with patch.object(scheduler, "_maybe_consolidate") as mock_maybe:
        scheduler._opportunistic_check(cpu_percent=5.0, battery=mock_battery)
        mock_maybe.assert_called_once()


def test_opportunistic_check_skips_when_cpu_high(scheduler):
    """_opportunistic_check should NOT trigger when CPU is above threshold."""
    mock_battery = MagicMock()
    mock_battery.power_plugged = True
    mock_battery.percent = 100

    with patch.object(scheduler, "_maybe_consolidate") as mock_maybe:
        scheduler._opportunistic_check(cpu_percent=15.0, battery=mock_battery)
        mock_maybe.assert_not_called()


def test_opportunistic_check_skips_when_on_battery(scheduler):
    """_opportunistic_check should NOT trigger when running on battery."""
    mock_battery = MagicMock()
    mock_battery.power_plugged = False
    mock_battery.percent = 100

    with patch.object(scheduler, "_maybe_consolidate") as mock_maybe:
        scheduler._opportunistic_check(cpu_percent=5.0, battery=mock_battery)
        mock_maybe.assert_not_called()


def test_run_now_calls_consolidation(scheduler):
    """run_now should manually trigger a consolidation pass."""
    with patch("lumen.sleep.run_consolidation_pass") as mock_run:
        scheduler.run_now()
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == scheduler.config


def test_exception_in_consolidation_caught_and_logged(scheduler):
    """An exception in run_consolidation_pass should be caught and logged."""
    with (
        patch("lumen.sleep.run_consolidation_pass") as mock_run,
        patch("lumen.sleep.logger") as mock_logger,
    ):
        mock_run.side_effect = RuntimeError("boom")
        scheduler._maybe_consolidate()
        mock_logger.error.assert_called_once()
        call_kwargs = mock_logger.error.call_args.kwargs
        assert call_kwargs.get("error") == "boom"
    # The scheduler thread must not have raised


def test_scheduler_has_expected_jobs(scheduler):
    """Scheduler should have exactly the two predefined jobs."""
    job_ids = {job.id for job in scheduler.scheduler.get_jobs()}
    assert "sleep_consolidation" in job_ids
    assert "opportunistic" in job_ids


def test_opportunistic_uses_configurable_cpu_threshold():
    """_opportunistic_check should respect config.consolidation_cpu_percent."""
    config = LumenConfig(
        vector_index="sqlite-vec",
        device="generic",
        store_path="/tmp/.lumen-test",
        consolidation_cpu_percent=20.0,
    )
    scheduler = SleepScheduler(config=config)
    mock_battery = MagicMock()
    mock_battery.power_plugged = True

    with patch.object(scheduler, "_maybe_consolidate") as mock_maybe:
        scheduler._opportunistic_check(cpu_percent=15.0, battery=mock_battery)
        mock_maybe.assert_called_once()


def test_should_run_uses_configurable_battery_threshold():
    """_should_run should respect config.consolidation_battery_threshold."""
    config = LumenConfig(
        vector_index="sqlite-vec",
        device="generic",
        store_path="/tmp/.lumen-test",
        consolidation_battery_threshold=30,
    )
    scheduler = SleepScheduler(config=config)
    mock_battery = MagicMock()
    mock_battery.power_plugged = False
    mock_battery.percent = 40
    assert scheduler._should_run(battery=mock_battery)
