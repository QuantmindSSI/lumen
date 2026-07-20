"""Tests for lumen.sovereign.local_llm."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lumen.config import LumenConfig
from lumen.sovereign.local_llm import LocalLLM, _get_llama_class


@pytest.fixture
def enabled_config(tmp_path: Path):
    """Return a config with local LLM enabled and a temporary model path."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    return LumenConfig(
        enable_local_llm=True,
        model_path=model_dir,
        local_llm_model="dummy.gguf",
        context_budget=2048,
    )


@pytest.fixture
def disabled_config(tmp_path: Path):
    """Return a config with local LLM disabled."""
    return LumenConfig(
        enable_local_llm=False,
        model_path=tmp_path / "models",
        local_llm_model="dummy.gguf",
    )


def test_is_available_returns_false_when_llama_cpp_not_installed():
    """is_available() should be False if the llama_cpp module cannot be imported."""

    def _broken_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("No module named llama_cpp")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_broken_import):
        assert LocalLLM.is_available() is False


def test_is_available_returns_false_when_model_missing():
    """is_available() should be False when the model file does not exist."""
    # Ensure llama_cpp is importable by patching the inline import in is_available
    with patch(
        "lumen.sovereign.local_llm.LumenConfig",
        return_value=LumenConfig(
            model_path=Path("/nonexistent"),
            local_llm_model="missing.gguf",
        ),
    ):
        assert LocalLLM.is_available() is False


def test_is_available_returns_true_when_model_present():
    """is_available() should be True when llama_cpp is installed and the model exists."""
    model_path_obj = MagicMock()
    model_path_obj.__truediv__.return_value.is_file.return_value = True
    fake_cfg = MagicMock()
    fake_cfg.model_path = model_path_obj
    fake_cfg.local_llm_model = "dummy.gguf"
    fake_mod = MagicMock()
    with (
        patch("lumen.sovereign.local_llm.LumenConfig", return_value=fake_cfg),
        patch.dict("sys.modules", {"llama_cpp": fake_mod}),
    ):
        assert LocalLLM.is_available() is True


def test_load_raises_when_disabled(disabled_config):
    """_load() must raise RuntimeError when enable_local_llm is False."""
    llm = LocalLLM(disabled_config)
    with pytest.raises(RuntimeError, match="Local LLM is disabled in config"):
        llm._load()


def test_load_raises_when_model_file_missing(enabled_config):
    """_load() must raise RuntimeError when the GGUF file is absent."""
    llm = LocalLLM(enabled_config)
    with pytest.raises(RuntimeError, match="Local LLM model not found"):
        llm._load()


def test_load_unload_lifecycle(enabled_config):
    """_load() should instantiate Llama and _unload() should clear it."""
    model_file = enabled_config.model_path / enabled_config.local_llm_model
    model_file.write_text("dummy")

    llm = LocalLLM(enabled_config)
    assert llm._model is None

    mock_instance = MagicMock()
    mock_llama_class = MagicMock(return_value=mock_instance)

    with patch("lumen.sovereign.local_llm._get_llama_class", return_value=mock_llama_class):
        llm._load()
        assert llm._model is mock_instance
        mock_llama_class.assert_called_once()
        call_kwargs = mock_llama_class.call_args.kwargs
        assert call_kwargs["model_path"] == str(model_file)
        assert call_kwargs["n_ctx"] == enabled_config.context_budget
        assert call_kwargs["verbose"] is False
        assert call_kwargs["n_threads"] == llm._n_threads

        llm._unload()
        assert llm._model is None


def test_generate_calls_load_and_unload(enabled_config):
    """generate() must load before inference and unload after."""
    model_file = enabled_config.model_path / enabled_config.local_llm_model
    model_file.write_text("dummy")

    llm = LocalLLM(enabled_config)
    mock_instance = MagicMock()
    mock_instance.return_value = {"choices": [{"text": "  generated text  "}]}
    mock_llama_class = MagicMock(return_value=mock_instance)

    with patch("lumen.sovereign.local_llm._get_llama_class", return_value=mock_llama_class):
        result = llm.generate("test prompt", max_tokens=128)
        assert result == "generated text"
        mock_instance.assert_called_once_with("test prompt", max_tokens=128, stop=["\n"])
        assert llm._model is None  # unloaded after inference


def test_summarize_calls_load_and_unload(enabled_config):
    """summarize() must load before inference and unload after."""
    model_file = enabled_config.model_path / enabled_config.local_llm_model
    model_file.write_text("dummy")

    llm = LocalLLM(enabled_config)
    mock_instance = MagicMock()
    mock_instance.return_value = {"choices": [{"text": "  summary result  "}]}
    mock_llama_class = MagicMock(return_value=mock_instance)

    with patch("lumen.sovereign.local_llm._get_llama_class", return_value=mock_llama_class):
        result = llm.summarize(["text one", "text two"], "Summarise these")
        assert result == "summary result"
        assert llm._model is None


def test_get_llama_class_raises_informative_error():
    """_get_llama_class() must raise RuntimeError with helpful message when import fails."""
    import builtins

    _orig_import = builtins.__import__

    def _broken_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("No module named llama_cpp")
        return _orig_import(name, *args, **kwargs)

    with (
        patch.object(builtins, "__import__", side_effect=_broken_import),
        pytest.raises(RuntimeError, match="llama-cpp-python is not installed"),
    ):
        _get_llama_class()
