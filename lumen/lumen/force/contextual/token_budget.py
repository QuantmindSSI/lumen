"""B1a: Token budget enforcement with real tokenizer.

Uses the same model tokenizer as the embedder for exact token counting.
Falls back to a character-heuristic only when the tokenizer is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import structlog

logger = structlog.get_logger()


class TokenCounter(Protocol):
    """Protocol for anything that can count tokens in a string."""

    def count(self, text: str) -> int: ...


class CharHeuristicCounter:
    """Fallback counter: ~4 chars per token (crude but zero-dep)."""

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class TransformersTokenizerCounter:
    """Exact token counter using a HuggingFace tokenizer."""

    def __init__(self, model_path: Path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def get_token_counter(model_path: Path | None = None) -> TokenCounter:
    """Return the best available token counter.

    Args:
        model_path: Path to a directory containing a HuggingFace tokenizer.
            If ``None`` or the tokenizer cannot be loaded, falls back to the
            character heuristic.

    Returns:
        A :class:`TokenCounter` instance.
    """
    if model_path is not None and model_path.exists():
        try:
            return TransformersTokenizerCounter(model_path)
        except Exception as exc:
            logger.warning("tokenizer_load_failed", path=str(model_path), error=str(exc))
    return CharHeuristicCounter()
