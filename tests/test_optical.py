"""Tests for D10 Progressive Quantization Pipeline."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from lumen.sovereign.optical import (
    QUANTIZERS,
    degrade_chunk_vector,
    quantize_binary,
    quantize_fp16,
    quantize_int8,
    quantize_vector,
)


@pytest.fixture
def sample_vector():
    rng = np.random.default_rng(42)
    return rng.standard_normal(384, dtype=np.float32)


class TestQuantizers:
    def test_fp32_identity(self, sample_vector):
        out = QUANTIZERS["FP32"](sample_vector)
        assert out is sample_vector
        assert out.dtype == np.float32

    def test_fp16_shape_dtype(self, sample_vector):
        out = quantize_fp16(sample_vector)
        assert out.shape == sample_vector.shape
        assert out.dtype == np.float32

    def test_fp16_loses_precision(self, sample_vector):
        out = quantize_fp16(sample_vector)
        # Some values should differ because float16 has less precision.
        assert not np.array_equal(out, sample_vector)

    def test_int8_shape_dtype(self, sample_vector):
        out = quantize_int8(sample_vector)
        assert out.shape == sample_vector.shape
        assert out.dtype == np.float32

    def test_int8_loses_information(self, sample_vector):
        out = quantize_int8(sample_vector)
        assert not np.array_equal(out, sample_vector)

    def test_int8_uniform_vector(self):
        uniform = np.ones(10, dtype=np.float32) * 0.5
        out = quantize_int8(uniform)
        expected = np.zeros_like(uniform)
        np.testing.assert_array_equal(out, expected)

    def test_binary_shape_dtype(self, sample_vector):
        out = quantize_binary(sample_vector)
        assert out.shape == sample_vector.shape
        assert out.dtype == np.float32

    def test_binary_only_produces_minus_one_or_one(self, sample_vector):
        out = quantize_binary(sample_vector)
        unique = np.unique(out)
        assert set(unique).issubset({-1.0, 1.0})

    def test_binary_all_positive(self):
        vec = np.array([0.0, 0.1, 1.0, 0.5], dtype=np.float32)
        out = quantize_binary(vec)
        np.testing.assert_array_equal(out, np.ones_like(vec))

    def test_binary_all_negative(self):
        vec = np.array([-0.1, -1.0, -0.5, -1e-6], dtype=np.float32)
        out = quantize_binary(vec)
        np.testing.assert_array_equal(out, -np.ones_like(vec))


class TestQuantizeVector:
    def test_valid_keys(self, sample_vector):
        for key in ("FP32", "FP16", "INT8", "BINARY"):
            out = quantize_vector(sample_vector, key)
            assert out.dtype == np.float32
            assert out.shape == sample_vector.shape

    def test_invalid_key_raises(self, sample_vector):
        with pytest.raises(KeyError):
            quantize_vector(sample_vector, "UNKNOWN")


class TestDegradeChunkVector:
    def test_calls_remove_and_add(self, sample_vector):
        backend = MagicMock()
        degrade_chunk_vector(backend, 7, "FP32", "INT8", sample_vector)

        backend.remove.assert_called_once_with(7)
        backend.add.assert_called_once()
        call_args = backend.add.call_args
        assert call_args[0][0] == 7
        np.testing.assert_array_equal(call_args[0][1], quantize_int8(sample_vector))

    def test_all_resolutions(self, sample_vector):
        for target in ("FP32", "FP16", "INT8", "BINARY"):
            backend = MagicMock()
            degrade_chunk_vector(backend, 1, "FP32", target, sample_vector)
            backend.remove.assert_called_once_with(1)
            backend.add.assert_called_once()
