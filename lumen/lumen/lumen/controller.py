"""C7: Twin-Force Controller (TFC).

Manages the dynamic equilibrium between Mnemonic and Contextual forces.
"""

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

    def __init__(self, state: TFCState | None = None):
        self.state = state or TFCState()

    def update(self, interaction_signal: dict) -> None:
        """
        Update TFC state based on interaction signal.
        Expected keys: novelty, repetition, context_pressure, satisfaction, goal_changed
        """
        novelty = interaction_signal.get("novelty", 0.0)
        repetition = interaction_signal.get("repetition", 0.0)
        context_pressure = interaction_signal.get("context_pressure", 0.0)
        satisfaction = interaction_signal.get("satisfaction", 0.0)
        goal_changed = interaction_signal.get("goal_changed", False)

        if novelty > 0.7:
            self.state.e = max(0.0, self.state.e - 0.1)
            self.state.a = min(1.0, self.state.a + 0.1)
        if repetition > 0.7:
            self.state.e = min(1.0, self.state.e + 0.1)
            self.state.a = max(0.0, self.state.a - 0.1)
        if context_pressure > 0.9:
            self.state.r = max(0, self.state.r - 1)
        if satisfaction < -0.3:
            self.state.a = min(1.0, self.state.a + 0.2)
        if goal_changed:
            self.state.tau = 7.0
            self.state.r = 3

    def to_env(self) -> dict[str, float]:
        return self.state.to_env()
