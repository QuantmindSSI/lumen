"""D8: Model Download & ONNX Export Pipeline."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # type: ignore[assignment,misc]

console = Console()

KNOWN_MODELS = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


class HuggingfaceUnavailableError(RuntimeError):
    """Raised when huggingface_hub is required but not installed."""

    def __init__(self, message: str = "huggingface_hub is not installed.") -> None:
        super().__init__(message)


def _export_via_optimum(repo_id: str, dest: Path) -> Path:
    """Use ``optimum-cli`` or Python API to export a model to ONNX.

    Args:
        repo_id: HuggingFace model repository ID.
        dest: Output directory for the ONNX export.

    Returns:
        Path to the exported ``model.onnx`` file.

    Raises:
        RuntimeError: If neither ``optimum-cli`` nor the Python API is available.
    """
    dest.mkdir(parents=True, exist_ok=True)

    # Attempt 1: optimum-cli subprocess (fastest, handles caching)
    try:
        import subprocess

        subprocess.run(
            [
                "optimum-cli",
                "export",
                "onnx",
                "--model",
                repo_id,
                str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        onnx_file = dest / "model.onnx"
        if onnx_file.exists():
            return onnx_file
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Attempt 2: Python API fallback
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = ORTModelForFeatureExtraction.from_pretrained(repo_id)
        model.save_pretrained(str(dest))
        tokenizer.save_pretrained(str(dest))

        onnx_file = dest / "model.onnx"
        if onnx_file.exists():
            return onnx_file
        # some exports name it model.onnx, others onnx/model.onnx
        alt = dest / "onnx" / "model.onnx"
        if alt.exists():
            return alt
    except Exception as exc:
        raise RuntimeError(
            f"ONNX export failed for {repo_id}. Install optimum[onnxruntime] and try again."
        ) from exc

    raise RuntimeError(f"ONNX export succeeded but model.onnx not found in {dest}")


def provision_embedding_model(model_name: str, dest: Path) -> Path:
    """Ensure an ONNX embedding model exists at *dest*.

    Strategy:
    1. If ``model.onnx`` already exists at ``dest / model_name``, return it.
    2. Download the HF model and export to ONNX via ``optimum-cli``.
    """
    model_dir = dest / model_name
    onnx_path = model_dir / "model.onnx"
    if onnx_path.exists():
        return onnx_path

    repo_id = KNOWN_MODELS.get(model_name, model_name)
    return _export_via_optimum(repo_id, model_dir)


model_app = typer.Typer(name="model", help="Model operations")


@model_app.command(name="list")
def model_list() -> None:
    """Print a table of known embedding models."""
    table = Table(title="Known Embedding Models")
    table.add_column("Alias", style="cyan")
    table.add_column("Repo ID", style="magenta")
    for alias, repo_id in KNOWN_MODELS.items():
        table.add_row(alias, repo_id)
    console.print(table)


@model_app.command(name="download")
def model_download(
    model_name: str = typer.Argument(..., help="Model alias or repo ID"),
) -> None:
    """Download and export the embedding model ONNX file to the default Lumen model path."""
    from lumen.config import LumenConfig

    config = LumenConfig()
    dest = config.model_path
    dest.mkdir(parents=True, exist_ok=True)

    try:
        path = provision_embedding_model(model_name, dest)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[bold green]Model exported to {path}[/bold green]")


@model_app.command(name="export")
def model_export(
    repo_id: str = typer.Argument(..., help="HuggingFace repo ID (e.g. BAAI/bge-small-en-v1.5)"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory"),  # noqa: B008
) -> None:
    """Export a HuggingFace model to ONNX using optimum-cli."""
    from lumen.config import LumenConfig

    config = LumenConfig()
    if output is None:
        output = config.model_path / repo_id.split("/")[-1]

    try:
        path = _export_via_optimum(repo_id, output)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[bold green]Model exported to {path}[/bold green]")
