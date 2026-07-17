"""C3: Real EpistemicTracker.

Known facts, assumed gaps, established truths tracking.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class EpistemicState:
    known_facts: Set[str] = field(default_factory=set)
    assumed_gaps: Set[str] = field(default_factory=set)
    established_truths: Set[str] = field(default_factory=set)


class EpistemicTracker:
    """Session-resident; optionally flushed to SQLite for cross-session continuity."""

    def __init__(self, conn: sqlite3.Connection | None = None):
        self.state = EpistemicState()
        self.conn = conn

    def mark_known(self, chunk_ids: List[int]) -> None:
        for cid in chunk_ids:
            self.state.known_facts.add(str(cid))

    def mark_gap(self, descriptions: List[str]) -> None:
        for d in descriptions:
            self.state.assumed_gaps.add(d)

    def confirm_truth(self, description: str) -> None:
        self.state.established_truths.add(description)
        self.state.assumed_gaps.discard(description)

    def search_bias(self, query_type: str) -> Dict[str, float]:
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

    def save(self) -> None:
        if self.conn is None:
            return
        blob = json.dumps({
            "known_facts": list(self.state.known_facts),
            "assumed_gaps": list(self.state.assumed_gaps),
            "established_truths": list(self.state.established_truths),
        })
        self.conn.execute(
            """INSERT INTO event_buffer_meta(meta_id, last_consolidation_at)
               VALUES (1, ?)
               ON CONFLICT(meta_id) DO UPDATE SET last_consolidation_at=excluded.last_consolidation_at""",
            (blob,)
        )
        self.conn.commit()

    def load(self) -> None:
        if self.conn is None:
            return
        row = self.conn.execute(
            "SELECT last_consolidation_at FROM event_buffer_meta WHERE meta_id = 1"
        ).fetchone()
        if row and row[0]:
            data = json.loads(row[0])
            self.state.known_facts = set(data.get("known_facts", []))
            self.state.assumed_gaps = set(data.get("assumed_gaps", []))
            self.state.established_truths = set(data.get("established_truths", []))
