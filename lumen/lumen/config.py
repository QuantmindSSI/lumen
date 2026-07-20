"""D1: Configuration Schema & Runtime Loader.

Input wire: pydantic-settings, tomli, env vars
Output wire: EVERY other task
Secret sauce: Device-specific defaults (RPi5 vs Jetson vs generic)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_config_path() -> Path:
    return Path.home() / ".lumen" / "config.toml"


def _default_store_path() -> Path:
    return Path.home() / ".lumen" / "store"


def _default_model_path() -> Path:
    return Path.home() / ".lumen" / "models"


def _default_cache_path() -> Path:
    return Path.home() / ".lumen" / "cache"


DEVICE_PROFILES: dict[str, dict[str, object]] = {
    "rpi5": {"context_budget": 1024, "memory_limit_mb": 256, "vector_index": "sqlite-vec"},
    "jetson-orin": {"context_budget": 2048, "memory_limit_mb": 512, "vector_index": "usearch"},
    "orange-pi": {"context_budget": 1024, "memory_limit_mb": 256, "vector_index": "sqlite-vec"},
    "generic": {"context_budget": 2048, "memory_limit_mb": 512, "vector_index": "sqlite-vec"},
}


class LumenConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LUMEN_",
        toml_file=_default_config_path(),
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
    consolidation_battery_threshold: int = 50
    device_name: str = "lumen-device"
    local_ip: str = "0.0.0.0"
    beam_port: int = 8847
    store_path: Path = Path.home() / ".lumen" / "store"
    model_path: Path = Path.home() / ".lumen" / "models"
    cache_path: Path = Path.home() / ".lumen" / "cache"
    sovereign: bool = True
    log_level: str = "info"
    api_host: str = "0.0.0.0"
    api_port: int = 8848
    api_key: str | None = None
    api_rate_limit: str = "60/minute"
    allowed_origins: str = "*"
    request_max_size_bytes: int = 1_048_576
    release_threshold: float = 0.05

    @property
    def db_uri(self) -> str:
        return f"{self.store_path}/lumen.db"

    @property
    def db_path(self) -> Path:
        return self.store_path / "lumen.db"

    def resolve_device_defaults(self) -> None:
        profile = DEVICE_PROFILES.get(self.device, DEVICE_PROFILES["generic"])
        self.context_budget = int(profile.get("context_budget", self.context_budget))
        self.memory_limit_mb = int(profile.get("memory_limit_mb", self.memory_limit_mb))
        self.vector_index = str(profile.get("vector_index", self.vector_index))