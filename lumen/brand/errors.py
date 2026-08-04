"""D15: Centralized Error/Exception Hierarchy.

Input wire: Brand bible error code spec
Output wire: All modules
Secret sauce: Unified error identity across CLI, logs, and API
"""


class LumenError(Exception):
    """Base for all Lumen exceptions."""

    code = "LME-0000"


class PalaceError(LumenError):
    """Errors within the memory palace (Force A)."""

    pass


class RoomNotFoundError(PalaceError):
    """Requested room does not exist in the palace."""

    code = "LME-1001"

    def __init__(self, room: str) -> None:
        super().__init__(f"The room '{room}' has faded. Let me search nearby loci.")


class LocusConflictError(PalaceError):
    """Two loci collide or violate uniqueness."""

    code = "LME-1002"

    def __init__(self, room: str, name: str) -> None:
        super().__init__(
            f"The locus '{name}' already exists within '{room}'. "
            "Try a more specific name to avoid overlap."
        )


class ConsolidationFailedError(PalaceError):
    """Sleep-phase consolidation could not complete."""

    code = "LME-1003"

    def __init__(self, detail: str = "") -> None:
        msg = "Consolidation failed. The palace will retry on the next sleep cycle."
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class ContextError(LumenError):
    """Errors within the context window (Force B)."""

    pass


class BudgetExceededError(ContextError):
    """Context window budget exceeded."""

    code = "LCX-2001"


class RetrievalEmptyError(ContextError):
    """No results returned for a retrieval query."""

    code = "LCX-2002"

    def __init__(self, query: str = "") -> None:
        msg = "No memories matched your query."
        if query:
            msg += f" Try rephrasing '{query[:60]}'."
        super().__init__(msg)


class AssemblyTimeoutError(ContextError):
    """Context assembly exceeded its time budget."""

    code = "LCX-2003"

    def __init__(self, budget_ms: float = 0) -> None:
        msg = "Context assembly timed out."
        if budget_ms > 0:
            msg += f" Budget was {budget_ms:.0f} ms."
        super().__init__(msg)


class TFCError(LumenError):
    """Twin-Force Controller state error."""

    code = "LLM-3001"


class TFCStuckError(TFCError):
    """TFC has entered an unrecoverable attractor state."""

    code = "LLM-3002"

    def __init__(self) -> None:
        super().__init__(
            "The Twin-Force Controller is stuck. Resetting attention to neutral. "
            "Run 'lumen tfc set --a 0.5 --e 0.5' to recover."
        )


class ValueModelUncalibratedError(TFCError):
    """V(m) weights are not yet learned for this user."""

    code = "LLM-3004"

    def __init__(self, user_id: str = "default") -> None:
        super().__init__(
            f"V(m) weights for '{user_id}' have not been calibrated yet. "
            "Provide explicit feedback on at least 5 results to train the value model."
        )


class SovereignViolationError(LumenError):
    """Attempted external API call when sovereign mode is enabled."""

    code = "LLM-3003"

    def __init__(self, attempted_call: str) -> None:
        super().__init__(f"Sovereign boundary: external API call blocked ({attempted_call}).")


class ModelNotAvailableError(LumenError):
    """Required ONNX embedding model is missing or cannot be loaded."""

    code = "LME-2001"

    def __init__(self, model_path: str) -> None:
        super().__init__(
            f"Embedding model not available at '{model_path}'. "
            "Run 'lumen model download <alias>' or 'lumen init --download-model' to provision it."
        )


class FeedbackLogError(LumenError):
    """Could not write or read feedback log."""

    code = "LME-2002"

    def __init__(self, detail: str = "") -> None:
        msg = "Feedback could not be recorded. The palace will attempt recovery."
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)
