"""B2: Local Embedding Pipeline (ONNX Runtime).

Input wire: ONNX Runtime + Optimum-exported BGE-small (33 MB INT8)
Output wire: A3 (vector add), A6 (store pipeline), C2 (dense query)
Secret sauce: None — pure commodity inference, but optimised for edge
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from lumen.brand.errors import ModelNotAvailableError

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:
    pass


class LocalEmbedder:
    """ONNX-based local embedder with mean pooling and L2 normalisation.

    Raises:
        ModelNotAvailableError: If the model path does not exist or ONNX
            Runtime cannot load it.
    """

    def __init__(self, model_path: Path, dims: int = 384) -> None:
        self.model_path = Path(model_path)
        self.dims = dims
        self._cache: dict[str, np.ndarray] = {}
        self.session = None
        self.tokenizer = None

        if not self.model_path.exists():
            raise ModelNotAvailableError(str(self.model_path))

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        except Exception as exc:
            raise ModelNotAvailableError(str(self.model_path)) from exc

        try:
            import onnxruntime as ort

            model_file = self.model_path / "model.onnx"
            if not model_file.exists():
                if self.model_path.is_file():
                    model_file = self.model_path
                else:
                    raise FileNotFoundError(
                        f"ONNX model not found at {self.model_path / 'model.onnx'}"
                    )
            self.session = ort.InferenceSession(
                str(model_file), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ModelNotAvailableError(str(self.model_path)) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into normalised embedding vectors."""
        if self.session is None or self.tokenizer is None:
            raise ModelNotAvailableError(str(self.model_path))
        return self._encode_onnx(texts)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text])[0]

    def _encode_onnx(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        outputs = self.session.run(None, ort_inputs)
        hidden_states = outputs[0]

        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        masked = hidden_states * mask
        summed = masked.sum(axis=1)
        counts = mask.sum(axis=1)
        embeddings = summed / np.maximum(counts, 1e-8)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)


class MockEmbedder:
    """Deterministic pseudo-embedder for **testing only**.

    Never use in production — semantic search will be random.
    """

    def __init__(self, dims: int = 384) -> None:
        self.dims = dims
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._encode_fallback(texts)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _encode_fallback(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            if t in self._cache:
                vecs.append(self._cache[t])
                continue
            h = hash(t) % (2**31)
            rng = np.random.default_rng(h)
            vec = rng.standard_normal(self.dims).astype(np.float32)
            vec = vec / np.maximum(np.linalg.norm(vec), 1e-8)
            self._cache[t] = vec
            vecs.append(vec)
        return np.stack(vecs, axis=0)


# Backward-compatible alias used by tests and benchmarks.
FallbackEmbedder = MockEmbedder


def get_embedder(config, allow_mock: bool = False) -> LocalEmbedder | MockEmbedder:
    """Return a production :class:`LocalEmbedder` for the configured model.

    Args:
        config: :class:`LumenConfig` instance.
        allow_mock: If ``True``, fall back to :class:`MockEmbedder` when the
            real model is unavailable. Defaults to ``False`` so that production
            callers fail loudly.

    Returns:
        A ready-to-use embedder.

    Raises:
        ModelNotAvailableError: When the model cannot be loaded and
            *allow_mock* is ``False``.
    """
    model_dir = Path(config.model_path) / config.embedding_model
    try:
        return LocalEmbedder(model_dir, dims=config.embedding_dims)
    except ModelNotAvailableError:
        if allow_mock:
            warnings.warn(
                "LocalEmbedder failed — falling back to MockEmbedder. "
                "Semantic search will be meaningless.",
                stacklevel=2,
            )
            return MockEmbedder(dims=config.embedding_dims)
        raise
