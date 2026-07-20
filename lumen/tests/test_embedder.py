"""Tests for lumen.force.contextual.embed."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lumen.brand.errors import ModelNotAvailableError
from lumen.force.contextual.embed import (
    FallbackEmbedder,
    LocalEmbedder,
    MockEmbedder,
    get_embedder,
)


class DummyConfig:
    def __init__(self, model_path: Path, embedding_model: str, embedding_dims: int):
        self.model_path = model_path
        self.embedding_model = embedding_model
        self.embedding_dims = embedding_dims


class TestMockEmbedder:
    def test_deterministic(self):
        emb = MockEmbedder(dims=384)
        a = emb.encode(["hello", "world"])
        b = emb.encode(["hello", "world"])
        np.testing.assert_array_equal(a, b)

    def test_encode_single_shape(self):
        emb = MockEmbedder(dims=384)
        vec = emb.encode_single("test")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_backward_alias(self):
        """FallbackEmbedder is a backward-compatible alias for MockEmbedder."""
        assert FallbackEmbedder is MockEmbedder


class TestGetEmbedder:
    def test_raises_when_model_missing(self, tmp_path: Path):
        config = DummyConfig(
            model_path=tmp_path / "nonexistent",
            embedding_model="model",
            embedding_dims=128,
        )
        with pytest.raises(ModelNotAvailableError):
            get_embedder(config, allow_mock=False)

    def test_allow_mock_fallback(self, tmp_path: Path):
        config = DummyConfig(
            model_path=tmp_path / "nonexistent",
            embedding_model="model",
            embedding_dims=256,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embedder = get_embedder(config, allow_mock=True)
        assert isinstance(embedder, MockEmbedder)
        assert embedder.dims == 256
        assert len(w) == 1
        assert "MockEmbedder" in str(w[0].message)


class TestLocalEmbedder:
    def test_raises_when_model_missing(self, tmp_path: Path):
        bad_path = tmp_path / "empty_model_dir"
        bad_path.mkdir()
        with pytest.raises(ModelNotAvailableError):
            LocalEmbedder(bad_path, dims=384)

    def test_mean_pooling_math(self, tmp_path: Path):
        """Test ONNX path with mocked tokenizer and session."""
        bad_path = tmp_path / "empty_model_dir"
        bad_path.mkdir()
        # Create a dummy ONNX file so the existence check passes
        (bad_path / "model.onnx").write_text("")

        mock_tokenizer = MagicMock()
        mock_tokenizer.from_pretrained = MagicMock(return_value=mock_tokenizer)

        mock_session = MagicMock()
        hidden = np.array(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [99.0, 99.0, 99.0, 99.0],
                ]
            ],
            dtype=np.float32,
        )
        mock_session.run.return_value = [hidden]

        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ), patch(
            "onnxruntime.InferenceSession", return_value=mock_session
        ):
            embedder = LocalEmbedder(bad_path, dims=4)
            # Manually set up return values for tokenizer call inside _encode_onnx
            embedder.tokenizer = MagicMock()
            embedder.tokenizer.return_value = {
                "input_ids": np.array([[1, 2, 3, 0]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1, 0]], dtype=np.int64),
            }
            embedder.session = MagicMock()
            embedder.session.run.return_value = [hidden]

            out = embedder._encode_onnx(["dummy"])
            assert out.shape == (1, 4)

            expected_mean = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
            expected_norm = np.linalg.norm(expected_mean)
            expected = expected_mean / expected_norm
            np.testing.assert_allclose(out[0], expected, rtol=1e-5)
