"""D12: Search Repair Loop (Stage 5 Feedback).

Input wire: C8 (failed search signal), TFC state (C5)
Output wire: C8 (re-execute with wider net), A9 (feedback_log via D3)
Secret sauce: Self-healing retrieval; teaches TFC to widen `a`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lumen.lumen.controller import TwinForceController
    from lumen.lumen.search import SearchPipeline

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


class SearchRepair:
    """Self-healing retrieval that adjusts TFC state and re-executes search."""

    MAX_ATTEMPTS = 2

    def __init__(self, tfc: TwinForceController, pipeline: SearchPipeline) -> None:
        self.tfc = tfc
        self.pipeline = pipeline
        self._attempts = 0

    def attempt_repair(self, query: str, reason: str) -> list:
        """Attempt to repair a failed search by mutating TFC and re-executing.

        Args:
            query: Original query string.
            reason: One of "empty_results", "budget_exceeded", "low_confidence".

        Returns:
            List of RetrievedChunk from the repaired query, or empty list.
        """
        if self._attempts >= self.MAX_ATTEMPTS:
            if logger:
                logger.debug("search_repair_maxed", attempts=self._attempts)
            return []

        self._attempts += 1

        if reason == "empty_results":
            old_a = self.tfc.state.a
            self.tfc.state.a = min(1.0, self.tfc.state.a + 0.3)
            if logger:
                logger.info("search_repair", reason=reason, old_a=old_a, new_a=self.tfc.state.a)
            return self.pipeline.execute(query, max_repair_attempts=0, k=50)

        if reason == "budget_exceeded":
            old_r = self.tfc.state.r
            self.tfc.state.r = max(0, self.tfc.state.r - 1)
            if logger:
                logger.info("search_repair", reason=reason, old_r=old_r, new_r=self.tfc.state.r)
            return self.pipeline.execute(query, max_repair_attempts=0)

        if reason == "low_confidence":
            old_a = self.tfc.state.a
            self.tfc.state.a = min(1.0, self.tfc.state.a + 0.15)
            if logger:
                logger.info("search_repair", reason=reason, old_a=old_a, new_a=self.tfc.state.a)
            return self.pipeline.execute(query, max_repair_attempts=0)

        if logger:
            logger.warning("search_repair_unknown_reason", reason=reason)
        return []
