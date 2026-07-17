"""A3b: GraphChannel using NetworkX or Kùzu fallback.

Input wire: SQLite adjacency + optional Kùzu
Output wire: C2 (fusion engine)
"""

import sqlite3
from dataclasses import dataclass
from typing import List

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass


@dataclass(frozen=True)
class GraphHit:
    chunk_id: int
    depth: int
    path: List[int]


class GraphChannel:
    """Graph traversal backed by SQLite adjacency, with optional NetworkX/Kùzu."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._nx = None
        try:
            import networkx as nx
            self._nx = nx
            self._build_nx_graph()
        except Exception:
            pass

    def _build_nx_graph(self):
        if self._nx is None:
            return
        self._graph = self._nx.DiGraph()
        # Build edges from provenance parent links and room/locus relationships
        rows = self.conn.execute(
            "SELECT chunk_id, locus_id, room_id, provenance_root FROM chunk WHERE valid_to IS NULL"
        ).fetchall()
        chunk_to_locus = {}
        chunk_to_room = {}
        for cid, lid, rid, prov in rows:
            self._graph.add_node(cid)
            chunk_to_locus[cid] = lid
            chunk_to_room[cid] = rid
        # Locus-level edges
        locus_groups: dict = {}
        for cid, lid, _, _ in rows:
            if lid is not None:
                locus_groups.setdefault(lid, []).append(cid)
        for lid, cids in locus_groups.items():
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    self._graph.add_edge(cids[i], cids[j])
                    self._graph.add_edge(cids[j], cids[i])
        # Provenance parent edges
        prov_rows = self.conn.execute(
            "SELECT chunk_id, parent_provenance FROM provenance WHERE parent_provenance IS NOT NULL"
        ).fetchall()
        for cid, parent in prov_rows:
            parent_chunk = self.conn.execute(
                "SELECT chunk_id FROM provenance WHERE provenance_id = ?", (parent,)
            ).fetchone()
            if parent_chunk:
                self._graph.add_edge(parent_chunk[0], cid)

    def traverse_from_seed(self, seed_chunk_id: int, hops: int = 2) -> List[GraphHit]:
        """BFS traversal from seed with depth limit."""
        if self._nx is not None and hasattr(self, '_graph'):
            return self._traverse_nx(seed_chunk_id, hops)
        return self._traverse_sqlite(seed_chunk_id, hops)

    def _traverse_nx(self, seed_chunk_id: int, hops: int) -> List[GraphHit]:
        hits = []
        visited = {seed_chunk_id: [seed_chunk_id]}
        queue = [(seed_chunk_id, 0)]
        while queue:
            node, depth = queue.pop(0)
            if depth >= hops:
                continue
            for neighbor in self._graph.neighbors(node):
                if neighbor not in visited:
                    path = visited[node] + [neighbor]
                    visited[neighbor] = path
                    queue.append((neighbor, depth + 1))
                    hits.append(GraphHit(neighbor, depth + 1, path))
        return hits

    def _traverse_sqlite(self, seed_chunk_id: int, hops: int) -> List[GraphHit]:
        """Pure SQLite BFS fallback."""
        hits = []
        visited = {seed_chunk_id}
        queue = [(seed_chunk_id, 0, [seed_chunk_id])]
        while queue:
            node, depth, path = queue.pop(0)
            if depth >= hops:
                continue
            # Neighbors: same locus
            locus_row = self.conn.execute(
                "SELECT locus_id, room_id FROM chunk WHERE chunk_id = ?", (node,)
            ).fetchone()
            if locus_row is None:
                continue
            lid, rid = locus_row
            neighbors = set()
            if lid is not None:
                rows = self.conn.execute(
                    "SELECT chunk_id FROM chunk WHERE locus_id = ? AND valid_to IS NULL", (lid,)
                ).fetchall()
                neighbors.update(r[0] for r in rows)
            if rid is not None:
                rows = self.conn.execute(
                    "SELECT chunk_id FROM chunk WHERE room_id = ? AND valid_to IS NULL", (rid,)
                ).fetchall()
                neighbors.update(r[0] for r in rows)
            # Parent provenance neighbors
            prov_rows = self.conn.execute(
                """SELECT p2.chunk_id
                    FROM provenance p1
                    JOIN provenance p2 ON p1.parent_provenance = p2.provenance_id
                    WHERE p1.chunk_id = ?""", (node,)
            ).fetchall()
            neighbors.update(r[0] for r in prov_rows if r[0] is not None)
            for n in neighbors:
                if n not in visited and n != node:
                    visited.add(n)
                    new_path = path + [n]
                    queue.append((n, depth + 1, new_path))
                    hits.append(GraphHit(n, depth + 1, new_path))
        return hits
