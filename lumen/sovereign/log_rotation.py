"""D17: Audit Log Rotation & Disk Guard.

Input wire: Python stdlib, APScheduler
Output wire: A11, all structlog sinks

NOTE: This module is not yet imported by any code path (production or tests).
"""

from __future__ import annotations

import gzip
import shutil
from datetime import datetime
from pathlib import Path

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


def get_log_dir() -> Path:
    """Return the canonical Lumen log directory."""
    return Path.home() / ".lumen" / "logs"


def total_log_size_mb() -> float:
    """Return the combined size of all *.jsonl and *.gz files in the logs dir."""
    target = get_log_dir()
    if not target.exists():
        return 0.0
    total = 0
    for pattern in ("*.jsonl", "*.gz"):
        for f in target.glob(pattern):
            total += f.stat().st_size
    return total / (1024 * 1024)


def rotate_jsonl_logs(
    max_uncompressed_mb: float = 50.0,
    max_archives: int = 10,
) -> None:
    """Compress JSONL logs that exceed *max_uncompressed_mb* and trim old archives.

    Archives are named ``<basename>.jsonl.YYYYMMDD.gz`` (with an optional
    ``_N`` disambiguation suffix if the day already has an archive).  After
    successful compression the original file is truncated to zero bytes.  If
    more than *max_archives* exist for the same base filename, the oldest
    archives are deleted.
    """
    target = get_log_dir()
    if not target.exists():
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("log_dir_missing_and_unwritable", path=str(target))
            return

    for log_file in target.glob("*.jsonl"):
        size_mb = log_file.stat().st_size / (1024 * 1024)
        if size_mb <= max_uncompressed_mb:
            continue

        archive = _unique_archive_path(log_file)
        try:
            with open(log_file, "rb") as f_in, gzip.open(archive, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            log_file.write_text("")  # truncate
            logger.info(
                    "log_rotated",
                    source=str(log_file.name),
                    archive=str(archive.name),
                    original_mb=round(size_mb, 2),
                )
        except OSError:
            logger.exception("log_rotation_failed", source=str(log_file.name))
            continue

        _prune_archives(log_file, max_archives)


def _unique_archive_path(log_file: Path) -> Path:
    """Return a non-colliding archive path for *log_file*."""
    base = log_file.with_suffix(f".jsonl.{datetime.now():%Y%m%d}")
    candidate = base.with_suffix(base.suffix + ".gz")
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = Path(f"{base}_{counter}.gz")
        if not candidate.exists():
            return candidate
        counter += 1


def _prune_archives(log_file: Path, max_archives: int) -> None:
    """Delete oldest archives for *log_file* if the count exceeds *max_archives*."""
    stem = log_file.stem  # e.g. "compliance" for "compliance.jsonl"
    archives = sorted(
        log_file.parent.glob(f"{stem}.jsonl.*.gz"),
        key=lambda p: p.stat().st_mtime,
    )
    while len(archives) > max_archives:
        oldest = archives.pop(0)
        try:
            oldest.unlink()
            logger.info("log_archive_pruned", archive=str(oldest.name))
        except OSError:
            logger.exception("log_archive_prune_failed", archive=str(oldest.name))
