"""C3: Real EpistemicTracker.

Known facts, assumed gaps, established truths tracking.
Persists to the ``epistemic_state`` SQLite table.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field


@dataclass
class EpistemicState:
    known_facts: set[str] = field(default_factory=set)
    assumed_gaps: set[str] = field(default_factory=set)
    established_truths: set[str] = field(default_factory=set)


class EpistemicTracker:
    """Session-resident + SQLite-persisted epistemic state."""

    def __init__(self, conn: sqlite3.Connection | None = None, user_id: str = "default"):
        self.state = EpistemicState()
        self.conn = conn
        self.user_id = user_id
        if conn is not None:
            self._load()

    def mark_known(self, chunk_ids: list[int]) -> None:
        for cid in chunk_ids:
            self.state.known_facts.add(str(cid))
        self._save()

    def mark_gap(self, descriptions: list[str]) -> None:
        for d in descriptions:
            self.state.assumed_gaps.add(d)
        self._save()

    def confirm_truth(self, description: str) -> None:
        self.state.established_truths.add(description)
        self.state.assumed_gaps.discard(description)
        self._save()

    def search_bias(self, query_type: str) -> dict[str, float]:
        """Return strategy hints based on epistemic state."""
        hints = {
            "boost_reliability": 0.0,
            "widen_retrieval": 0.0,
            "skip_retrieval": False,
        }
        if query_type in self.state.assumed_gaps:
            hints["widen_retrieval"] = 0.3
        if query_type in self.state.established_truths:
            hints["boost_reliability"] = 0.2
        if query_type in self.state.known_facts:
            hints["skip_retrieval"] = True
        return hints

    def _save(self) -> None:
        if self.conn is None:
            return
        self.conn.execute(
            """INSERT INTO epistemic_state (user_id, known_facts_json, assumed_gaps_json, established_truths_json, updated_at)
               VALUES (?, ?, ?, ?, unixepoch())
               ON CONFLICT(user_id) DO UPDATE SET
                   known_facts_json = excluded.known_facts_json,
                   assumed_gaps_json = excluded.assumed_gaps_json,
                   established_truths_json = excluded.established_truths_json,
                   updated_at = unixepoch()""",
            (
                self.user_id,
                json.dumps(sorted(self.state.known_facts)),
                json.dumps(sorted(self.state.assumed_gaps)),
                json.dumps(sorted(self.state.established_truths)),
            ),
        )
        self.conn.commit()

    def _load(self) -> None:
        if self.conn is None:
            return
        row = self.conn.execute(
            "SELECT known_facts_json, assumed_gaps_json, established_truths_json FROM epistemic_state WHERE user_id = ?",
            (self.user_id,),
        ).fetchone()
        if row:
            self.state.known_facts = set(json.loads(row["known_facts_json"] or "[]"))
            self.state.assumed_gaps = set(json.loads(row["assumed_gaps_json"] or "[]"))
            self.state.established_truths = set(json.loads(row["established_truths_json"] or "[]"))
