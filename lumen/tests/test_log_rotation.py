"""Tests for D17: Audit Log Rotation & Disk Guard."""

import gzip
import time
from pathlib import Path

import pytest

from lumen.sovereign import log_rotation


@pytest.fixture
def fake_log_dir(tmp_path, monkeypatch):
    """Redirect get_log_dir() to a temporary path for the duration of the test."""
    monkeypatch.setattr(log_rotation, "get_log_dir", lambda: tmp_path)
    return tmp_path


def _write_jsonl(path: Path, size_mb: float) -> None:
    """Write a JSONL file of approximately *size_mb* megabytes."""
    line = b'{"msg": "x"}\n'
    lines_needed = int((size_mb * 1024 * 1024) / len(line)) + 1
    path.write_bytes(line * lines_needed)


def test_rotate_creates_gz_and_truncates_original(fake_log_dir):
    log_file = fake_log_dir / "compliance.jsonl"
    _write_jsonl(log_file, 1.0)  # 1 MB > default 50 MB? No, 1 < 50.

    log_rotation.rotate_jsonl_logs(max_uncompressed_mb=0.5)

    archives = list(fake_log_dir.glob("compliance.jsonl.*.gz"))
    assert len(archives) == 1
    # Original should be truncated
    assert log_file.stat().st_size == 0
    # Archive should be non-empty and valid gzip
    assert archives[0].stat().st_size > 0
    with gzip.open(archives[0], "rb") as f:
        assert b'"msg": "x"' in f.read()


def test_rotate_respects_threshold(fake_log_dir):
    log_file = fake_log_dir / "small.jsonl"
    log_file.write_text('{"msg": "tiny"}\n')

    log_rotation.rotate_jsonl_logs(max_uncompressed_mb=10.0)

    assert not list(fake_log_dir.glob("small.jsonl.*.gz"))
    assert log_file.read_text() == '{"msg": "tiny"}\n'


def test_prune_oldest_archives(fake_log_dir):
    log_file = fake_log_dir / "audit.jsonl"
    _write_jsonl(log_file, 1.0)

    # Force rotation 5 times with distinct mtimes so ordering is deterministic
    for _ in range(5):
        log_rotation.rotate_jsonl_logs(max_uncompressed_mb=0.5, max_archives=3)
        # Bump mtime on the original so next rotation writes a new archive
        time.sleep(0.05)
        _write_jsonl(log_file, 1.0)

    archives = sorted(fake_log_dir.glob("audit.jsonl.*.gz"))
    # Should have been pruned down to max_archives=3
    assert len(archives) == 3


def test_total_log_size_mb(fake_log_dir):
    log_file = fake_log_dir / "compliance.jsonl"
    _write_jsonl(log_file, 1.0)

    log_rotation.rotate_jsonl_logs(max_uncompressed_mb=0.5)

    total = log_rotation.total_log_size_mb()
    # Archive + truncated original
    archives = list(fake_log_dir.glob("compliance.jsonl.*.gz"))
    expected = sum(f.stat().st_size for f in (archives + [log_file])) / (1024 * 1024)
    assert total == pytest.approx(expected, rel=1e-3)


def test_total_log_size_mb_missing_dir(fake_log_dir, monkeypatch):
    # Point get_log_dir at a non-existent path without creating it
    missing = fake_log_dir / "missing"
    monkeypatch.setattr(log_rotation, "get_log_dir", lambda: missing)
    assert log_rotation.total_log_size_mb() == 0.0


def test_rotation_creates_missing_dir(tmp_path, monkeypatch):
    missing = tmp_path / "fresh" / "logs"
    monkeypatch.setattr(log_rotation, "get_log_dir", lambda: missing)
    # No files to rotate, but it should create the dir without error
    log_rotation.rotate_jsonl_logs()
    assert missing.exists()


def test_multiple_basenames_are_isolated(fake_log_dir):
    a = fake_log_dir / "compliance.jsonl"
    b = fake_log_dir / "debug.jsonl"
    _write_jsonl(a, 1.0)
    _write_jsonl(b, 1.0)

    log_rotation.rotate_jsonl_logs(max_uncompressed_mb=0.5, max_archives=2)

    assert len(list(fake_log_dir.glob("compliance.jsonl.*.gz"))) == 1
    assert len(list(fake_log_dir.glob("debug.jsonl.*.gz"))) == 1
    assert a.stat().st_size == 0
    assert b.stat().st_size == 0
