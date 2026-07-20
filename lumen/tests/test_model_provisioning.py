"""Tests for D8 model provisioning."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from lumen.cli.main import app
from lumen.cli.models import KNOWN_MODELS, provision_embedding_model

runner = CliRunner()


def test_provision_returns_existing_onnx(tmp_path: Path) -> None:
    """If model.onnx already exists, return it immediately."""
    model_dir = tmp_path / "bge-small-en-v1.5"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_text("fake")
    path = provision_embedding_model("bge-small-en-v1.5", tmp_path)
    assert path == model_dir / "model.onnx"


def test_provision_calls_export_when_missing(tmp_path: Path) -> None:
    """If model.onnx is missing, export is attempted."""
    with patch(
        "lumen.cli.models._export_via_optimum",
        return_value=tmp_path / "bge-small-en-v1.5" / "model.onnx",
    ):
        path = provision_embedding_model("bge-small-en-v1.5", tmp_path)
        assert path.name == "model.onnx"


def test_model_list_command() -> None:
    result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    for alias in KNOWN_MODELS:
        assert alias in result.output


def test_model_download_existing_model(tmp_path: Path) -> None:
    """Download command should skip if model already exists."""
    model_dir = tmp_path / "bge-small-en-v1.5"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_text("fake")
    with patch("lumen.config.LumenConfig") as MockCfg:
        MockCfg.return_value.model_path = tmp_path
        result = runner.invoke(app, ["model", "download", "bge-small-en-v1.5"])
        assert result.exit_code == 0
        assert "exported" in result.output.lower() or "Model exported" in result.output
