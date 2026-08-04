"""Shared logger acquisition for Lumen.

Provides ``get_console_logger`` that tries structlog but falls back
to stdlib ``logging`` so no module ever silently loses log output.
"""

from __future__ import annotations

import logging as _stdlib_logging
from typing import Any

_logger_cache: dict[str, Any] = {}


def get_console_logger(name: str | None = None) -> Any:
    """Return a logger for *name*, preferring structlog, falling back to logging."""
    cache_key = name or "__root__"
    if cache_key in _logger_cache:
        return _logger_cache[cache_key]

    try:
        import structlog

        log = structlog.get_logger(name)
    except Exception:
        log = _stdlib_logging.getLogger(name or "lumen")
        if not log.handlers:
            handler = _stdlib_logging.StreamHandler()
            handler.setFormatter(
                _stdlib_logging.Formatter("%(asctime)s [%(levelname)s] %(name)s %(message)s")
            )
            log.addHandler(handler)
            log.setLevel(_stdlib_logging.INFO)

    _logger_cache[cache_key] = log
    return log
