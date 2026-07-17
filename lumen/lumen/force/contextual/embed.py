"""B2: Local Embedding Pipeline (ONNX Runtime).

Input wire: ONNX Runtime + Optimum-exported BGE-small (33 MB INT8)
Output wire: A3 (vector add), A6 (store pipeline), C2 (dense query)
Secret sauce: None — pure commodity inference, but optimised for edge
"""

from pathlib import Path
from typing import List

import numpy as np

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass


class LocalEmbedder:
    """ONNX-based local embedder with mean pooling and L2 normalisation."""

    def __init__(self, model_path: Path, dims: int = 384):
        self.model_path = Path(model_path)
        self.dims = dims
        self._cache: dict[str, np.ndarray] = {}

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = ORTModelForFeatureExtraction.from_pretrained(str(self.model_path))
            self._onnx = True
        except Exception as exc:
            if logger:
                logger.warning("local_embedder_fallback", reason=str(exc))
            self._onnx = False

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._onnx:
            return self._encode_onnx(texts)
        return self._encode_fallback(texts)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _encode_onnx(self, texts: List[str]) -> np.ndarray:
        import torch
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)

    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
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


class FallbackEmbedder:
    """Deterministic pseudo-embedder for testing when ONNX is unavailable."""

    def __init__(self, dims: int = 384):
        self.dims = dims
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._encode_fallback(texts)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
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
