"""D1: Configuration Schema & Runtime Loader.

Input wire: pydantic-settings, tomli, env vars
Output wire: EVERY other task
Secret sauce: Device-specific defaults (RPi5 vs Jetson vs generic)
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class LumenConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LUMEN_",
        toml_file=[".lumen/config.toml", "~/.lumen/config.toml"],
    )

    device: Literal["rpi5", "jetson-orin", "orange-pi", "generic"] = "generic"
    context_budget: int = 2048
    memory_limit_mb: int = 300
    embedding_model: str = "bge-small-en-v1.5"
    embedding_dims: int = 384
    vector_index: Literal["sqlite-vec", "usearch"] = "sqlite-vec"
    enable_kuzu: bool = False
    enable_frqad: bool = False
    enable_local_llm: bool = False
    local_llm_model: str = "Qwen3-1.7B-Q4_K_M.gguf"
    consolidation_cpu_percent: float = 5.0
    scheduler_granularity: int = 300
    store_path: Path = Path.home() / ".lumen" / "store"
    model_path: Path = Path.home() / ".lumen" / "models"
    cache_path: Path = Path.home() / ".lumen" / "cache"
    sovereign: bool = True
    log_level: str = "info"

    @property
    def db_uri(self) -> str:
        return f"{self.store_path}/lumen.db"
