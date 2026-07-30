"""D10: Progressive Quantization Pipeline (Actual Vector Mutation).

Called by A10 budget eviction and by A8 decay scheduler.
Re-quantizes FP32 vectors and re-inserts them into the vector index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lumen.force.mnemonic.retrieval_dense import VectorBackend

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


# ---------------------------------------------------------------------------
# Numba or plain-numpy quantizers
# ---------------------------------------------------------------------------
# NOTE: float16 is not supported by Numba's codegen, so quantize_fp16 is
# always a plain numpy function even when numba is present.
def _quantize_fp16(vec: np.ndarray) -> np.ndarray:
    return vec.astype(np.float16).astype(np.float32)  # round-trip for storage uniformity


try:
    from numba import njit

    @njit(cache=True)
    def _quantize_int8(vec: np.ndarray) -> np.ndarray:
        mn = vec.min()
        mx = vec.max()
        if mx - mn < 1e-8:
            return np.zeros_like(vec)
        scaled = (vec - mn) / (mx - mn) * 255.0 - 128.0
        return scaled.astype(np.int8).astype(np.float32)  # storage as float32, info lost

    @njit(cache=True)
    def _quantize_binary(vec: np.ndarray) -> np.ndarray:
        return np.where(vec >= 0, 1.0, -1.0).astype(np.float32)

except Exception:

    def _quantize_int8(vec: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        mn = vec.min()
        mx = vec.max()
        if mx - mn < 1e-8:
            return np.zeros_like(vec)
        scaled = (vec - mn) / (mx - mn) * 255.0 - 128.0
        return scaled.astype(np.int8).astype(np.float32)

    def _quantize_binary(vec: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        return np.where(vec >= 0, 1.0, -1.0).astype(np.float32)


# Public aliases so the spec names are satisfied without exposing the impl detail.
quantize_fp16 = _quantize_fp16
quantize_int8 = _quantize_int8
quantize_binary = _quantize_binary

QUANTIZERS = {
    "FP32": lambda v: v,
    "FP16": quantize_fp16,
    "INT8": quantize_int8,
    "BINARY": quantize_binary,
}


def quantize_vector(vector: np.ndarray, target_res: str) -> np.ndarray:
    """Standalone quantizer entrypoint.

    Args:
        vector: Input FP32 vector.
        target_res: One of ``FP32``, ``FP16``, ``INT8``, ``BINARY``.

    Returns:
        The quantized vector ( dtype float32 for storage uniformity ).

    Raises:
        KeyError: If *target_res* is not a supported resolution.
    """
    quantizer = QUANTIZERS[target_res]
    return quantizer(vector)


def degrade_chunk_vector(
    backend: VectorBackend,
    chunk_id: int,
    current_res: str,
    target_res: str,
    vector: np.ndarray,
) -> None:
    """Re-quantize *vector* to *target_res* and atomically replace it in *backend*.

    Args:
        backend: Concrete ``VectorBackend`` implementation.
        chunk_id: Primary key of the chunk being degraded.
        current_res: Existing resolution label (for logging only).
        target_res: Desired resolution label.
        vector: Original FP32 vector.
    """
    quantized = QUANTIZERS[target_res](vector)
    backend.remove(chunk_id)
    backend.add(chunk_id, quantized)
    if logger is not None:
        logger.info(
            "optical_degrade",
            chunk_id=chunk_id,
            from_res=current_res,
            to_res=target_res,
        )
