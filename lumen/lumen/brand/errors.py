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


class ContextError(LumenError):
    """Errors within the context window (Force B)."""

    pass


class BudgetExceededError(ContextError):
    """Context window budget exceeded."""

    code = "LCX-2001"


class RetrievalEmptyError(ContextError):
    """No results returned for a retrieval query."""

    code = "LCX-2002"


class TFCError(LumenError):
    """Twin-Force Controller state error."""

    code = "LLM-3001"


class SovereignViolationError(LumenError):
    """Attempted external API call when sovereign mode is enabled."""

    code = "LLM-3003"

    def __init__(self, attempted_call: str) -> None:
        super().__init__(f"Sovereign boundary: external API call blocked ({attempted_call}).")
