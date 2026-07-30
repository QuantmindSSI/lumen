"""D9: Local LLM Runtime Wrapper (llama-cpp-python).

Input wire: llama-cpp-python
Output wire: C6 (consolidation summaries), C7 (palace rebuild NLP)
Secret sauce: Demand-loaded; unloads when not in use to save SBC RAM.
"""

from __future__ import annotations

import gc
from pathlib import Path

import psutil

from lumen.config import LumenConfig
from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


def _get_n_threads() -> int:
    """Return physical CPU count or a sane default for SBCs."""
    try:
        count = psutil.cpu_count(logical=False)
        return max(1, count) if count else 4
    except Exception:
        return 4


def _get_llama_class() -> type:
    """Return the Llama class from llama_cpp, raising RuntimeError if absent."""
    try:
        from llama_cpp import Llama
    except Exception as exc:
        logger.warning("llama_cpp_not_installed")
        raise RuntimeError(
            "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
        ) from exc
    return Llama


class LocalLLM:
    """Demand-loaded wrapper around llama.cpp for local inference.

    Loads the model before inference and unloads immediately after to
    minimise RAM pressure on single-board computers.
    """

    def __init__(self, config: LumenConfig) -> None:
        self.config = config
        self._model: object | None = None
        self._path: Path = config.model_path / config.local_llm_model
        self._n_threads = _get_n_threads()

    @staticmethod
    def is_available() -> bool:
        """Return True if llama_cpp is importable *and* the configured model file exists."""
        try:
            from llama_cpp import Llama as _Llama  # noqa: F401
        except Exception:
            return False

        try:
            cfg = LumenConfig()
            return (cfg.model_path / cfg.local_llm_model).is_file()
        except Exception:
            return False

    def _load(self) -> None:
        if not self.config.enable_local_llm:
            raise RuntimeError("Local LLM is disabled in config")

        if self._model is not None:
            return

        if not self._path.is_file():
            raise RuntimeError(f"Local LLM model not found: {self._path}")

        Llama = _get_llama_class()

        logger.info("local_llm_loading", path=str(self._path), n_threads=self._n_threads)
        self._model = Llama(
            model_path=str(self._path),
            n_ctx=self.config.context_budget,
            verbose=False,
            n_threads=self._n_threads,
        )

    def _unload(self) -> None:
        self._model = None
        gc.collect()

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Run generic text generation and return the decoded output."""
        self._load()
        try:
            output = self._model(prompt, max_tokens=max_tokens, stop=["\n"])
            result: str = output["choices"][0]["text"].strip()
            return result
        finally:
            self._unload()

    def summarize(self, texts: list[str], instruction: str) -> str:
        """Generate a summary narrative used by sleep consolidation.

        Args:
            texts: List of raw text passages to summarise.
            instruction: High-level instruction for the summariser.

        Returns:
            Generated summary string.
        """
        self._load()
        try:
            prompt = (
                f"<|im_start|>user\n{instruction}\n"
                + "\n".join(texts)
                + "\n<|im_start|>assistant\n"
            )
            output = self._model(prompt, max_tokens=256, stop=["\n"])
            result: str = output["choices"][0]["text"].strip()
            return result
        finally:
            self._unload()
