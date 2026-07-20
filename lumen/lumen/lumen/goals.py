"""C6: Real GoalTree.

GoalNode with anytree-compatible structure.
Persists to SQLite for cross-session continuity.
"""

from __future__ import annotations

import sqlite3


class GoalNode:
    def __init__(self, name: str, parent: GoalNode | None = None):
        self.name = name
        self.parent = parent
        self.children: list[GoalNode] = []
        self.is_active = False

    def add_child(self, node: GoalNode) -> None:
        self.children.append(node)
        node.parent = self

    def find(self, name: str) -> GoalNode | None:
        if self.name == name:
            return self
        for c in self.children:
            found = c.find(name)
            if found:
                return found
        return None

    def path_names(self) -> list[str]:
        path = []
        current: GoalNode | None = self
        while current:
            path.append(current.name)
            current = current.parent
        return list(reversed(path))


class GoalTree:
    """Goal tree with SQLite persistence."""

    def __init__(self, conn: sqlite3.Connection | None = None, user_id: str = "default"):
        self.root = GoalNode("root")
        self._active: GoalNode | None = None
        self.conn = conn
        self.user_id = user_id
        if conn is not None:
            self._load()

    def add_goal(self, name: str, parent_name: str | None = None) -> GoalNode:
        if parent_name:
            parent = self.root.find(parent_name)
            if parent is None:
                parent = self.root
        else:
            parent = self.root
        existing = parent.find(name)
        if existing and existing != self.root:
            return existing
        node = GoalNode(name, parent=parent)
        parent.add_child(node)
        if self.conn is not None:
            self._save_node(node)
        return node

    def set_active(self, name: str) -> None:
        node = self.root.find(name)
        if node:
            self._deactivate_all(self.root)
            node.is_active = True
            self._active = node
            if self.conn is not None:
                self.conn.execute(
                    """UPDATE goals SET is_active = CASE WHEN name = ? THEN 1 ELSE 0 END
                       WHERE user_id = ?""",
                    (name, self.user_id),
                )
                self.conn.commit()

    def _deactivate_all(self, node: GoalNode) -> None:
        node.is_active = False
        for c in node.children:
            self._deactivate_all(c)

    def active_path_keywords(self) -> list[str]:
        if self._active is None:
            return []
        return self._active.path_names()

    def all_goals(self) -> list[str]:
        result: list[str] = []
        self._collect(self.root, result)
        return result

    def _collect(self, node: GoalNode, result: list[str]) -> None:
        if node != self.root:
            result.append(node.name)
        for c in node.children:
            self._collect(c, result)

    def _save_node(self, node: GoalNode) -> None:
        """Insert or update a single goal row."""
        parent_name = node.parent.name if node.parent and node.parent != self.root else None
        # Find parent_id
        parent_id = None
        if parent_name:
            row = self.conn.execute(
                "SELECT goal_id FROM goals WHERE user_id = ? AND name = ?",
                (self.user_id, parent_name),
            ).fetchone()
            if row:
                parent_id = row["goal_id"]
        self.conn.execute(
            """INSERT INTO goals (user_id, parent_id, name, is_active)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, name) DO UPDATE SET
                   parent_id = excluded.parent_id,
                   is_active = excluded.is_active,
                   updated_at = unixepoch()""",
            (self.user_id, parent_id, node.name, 1 if node.is_active else 0),
        )
        self.conn.commit()

    def _load(self) -> None:
        """Rebuild tree from SQLite."""
        rows = self.conn.execute(
            "SELECT goal_id, parent_id, name, is_active FROM goals WHERE user_id = ?",
            (self.user_id,),
        ).fetchall()
        if not rows:
            return

        # Build nodes
        id_to_node: dict[int, GoalNode] = {}
        id_to_parent_id: dict[int, int | None] = {}
        for row in rows:
            gid, pid, name, active = row
            node = GoalNode(name)
            node.is_active = bool(active)
            id_to_node[gid] = node
            id_to_parent_id[gid] = pid
            if node.is_active:
                self._active = node

        # Wire parents
        for gid, node in id_to_node.items():
            pid = id_to_parent_id[gid]
            if pid is not None and pid in id_to_node:
                parent = id_to_node[pid]
                parent.add_child(node)
            else:
                self.root.add_child(node)
