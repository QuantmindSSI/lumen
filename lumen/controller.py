"""C7: Twin-Force Controller (TFC).

Manages the dynamic equilibrium between Mnemonic and Contextual forces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lumen.config import LumenConfig

from pydantic import BaseModel, Field


class TFCState(BaseModel):
    """Pydantic model for TFC state variables."""

    e: float = Field(default=0.5, ge=0.0, le=1.0, description="mnemonic/conservation bias")
    a: float = Field(default=0.5, ge=0.0, le=1.0, description="attentional temperature")
    tau: float = Field(default=7.0, ge=0.0, description="temporal horizon (days)")
    r: int = Field(default=3, ge=0, le=5, description="resolution level")

    def to_env(self) -> dict[str, float]:
        return {"e": self.e, "a": self.a, "tau": self.tau, "r": self.r}


class TwinForceController:
    """TFC state machine with deterministic update rules."""

    def __init__(self, state: TFCState | None = None, config: LumenConfig | None = None):
        self.state = state or TFCState()
        self.config = config

    @property
    def _cfg(self) -> dict:
        if self.config:
            return {
                "novelty_threshold": self.config.tfc_novelty_threshold,
                "repetition_threshold": self.config.tfc_repetition_threshold,
                "context_pressure_threshold": self.config.tfc_context_pressure_threshold,
                "satisfaction_threshold": self.config.tfc_satisfaction_threshold,
                "step_conservation": self.config.tfc_step_conservation,
                "step_attention": self.config.tfc_step_attention,
                "step_attention_repair": self.config.tfc_step_attention_repair,
                "default_tau": self.config.tfc_default_tau,
                "default_resolution": self.config.tfc_default_resolution,
            }
        return {
            "novelty_threshold": 0.7,
            "repetition_threshold": 0.7,
            "context_pressure_threshold": 0.9,
            "satisfaction_threshold": -0.3,
            "step_conservation": 0.1,
            "step_attention": 0.1,
            "step_attention_repair": 0.2,
            "default_tau": 7.0,
            "default_resolution": 3,
        }

    def update(self, interaction_signal: dict) -> None:
        """
        Update TFC state based on interaction signal.
        Expected keys: novelty, repetition, context_pressure, satisfaction, goal_changed
        """
        cfg = self._cfg
        novelty = interaction_signal.get("novelty", 0.0)
        repetition = interaction_signal.get("repetition", 0.0)
        context_pressure = interaction_signal.get("context_pressure", 0.0)
        satisfaction = interaction_signal.get("satisfaction", 0.0)
        goal_changed = interaction_signal.get("goal_changed", False)

        if novelty > cfg["novelty_threshold"]:
            self.state.e = max(0.0, self.state.e - cfg["step_conservation"])
            self.state.a = min(1.0, self.state.a + cfg["step_attention"])
        if repetition > cfg["repetition_threshold"]:
            self.state.e = min(1.0, self.state.e + cfg["step_conservation"])
            self.state.a = max(0.0, self.state.a - cfg["step_attention"])
        if context_pressure > cfg["context_pressure_threshold"]:
            self.state.r = max(0, self.state.r - 1)
        if satisfaction < cfg["satisfaction_threshold"]:
            self.state.a = min(1.0, self.state.a + cfg["step_attention_repair"])
        if goal_changed:
            self.state.tau = cfg["default_tau"]
            self.state.r = int(cfg["default_resolution"])

    def to_env(self) -> dict[str, float]:
        return self.state.to_env()
