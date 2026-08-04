"""A4: FRQAD Kernel (Fisher-Rao Quantization-Aware Distance).

Input wire: NumPy arrays (embedding vectors)
Output wire: A3 (VectorBackend.search), C2 (fusion reranking)
"""

import numpy as np


def compute_frqad(query: np.ndarray, candidate: np.ndarray, resolution: str = "FP32") -> float:
    """Scalar FRQAD wrapper for single comparisons in reranking."""
    sigma_map = {"FP32": 0.0, "FP16": 1e-4, "INT8": 0.02, "BINARY": 0.3}
    sigma = sigma_map.get(resolution, 0.0)
    dot = np.dot(query, candidate)
    qn = np.linalg.norm(query)
    cn = np.linalg.norm(candidate)
    denom = qn * cn * (1.0 + sigma * sigma)
    if denom < 1e-8:
        return np.pi / 2.0
    cos_t = np.clip(dot / denom, -1.0, 1.0)
    return float(np.arccos(cos_t))


def compute_cosine_distance(query: np.ndarray, candidate: np.ndarray) -> float:
    """Fallback cosine distance for use when FRQAD is not available."""
    qn = np.linalg.norm(query)
    cn = np.linalg.norm(candidate)
    if qn == 0 or cn == 0:
        return 1.0
    dot = np.dot(query, candidate)
    sim = dot / (qn * cn)
    return 1.0 - sim  # convert similarity to distance [0, 2]
