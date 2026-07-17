"""C6: Real GoalTree.

GoalNode with anytree-compatible structure.
Store in-memory for now.
"""

from typing import List, Optional, Set


class GoalNode:
    def __init__(self, name: str, parent: Optional["GoalNode"] = None):
        self.name = name
        self.parent = parent
        self.children: List["GoalNode"] = []
        self.is_active = False

    def add_child(self, node: "GoalNode") -> None:
        self.children.append(node)
        node.parent = self

    def find(self, name: str) -> Optional["GoalNode"]:
        if self.name == name:
            return self
        for c in self.children:
            found = c.find(name)
            if found:
                return found
        return None

    def path_names(self) -> List[str]:
        path = []
        current: Optional[GoalNode] = self
        while current:
            path.append(current.name)
            current = current.parent
        return list(reversed(path))


class GoalTree:
    """In-memory goal tree with active path tracking."""

    def __init__(self):
        self.root = GoalNode("root")
        self._active: Optional[GoalNode] = None

    def add_goal(self, name: str, parent_name: Optional[str] = None) -> GoalNode:
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
        return node

    def set_active(self, name: str) -> None:
        node = self.root.find(name)
        if node:
            # Deactivate all
            self._deactivate_all(self.root)
            node.is_active = True
            self._active = node

    def _deactivate_all(self, node: GoalNode) -> None:
        node.is_active = False
        for c in node.children:
            self._deactivate_all(c)

    def active_path_keywords(self) -> List[str]:
        if self._active is None:
            return []
        return self._active.path_names()

    def all_goals(self) -> List[str]:
        result: List[str] = []
        self._collect(self.root, result)
        return result

    def _collect(self, node: GoalNode, result: List[str]) -> None:
        if node != self.root:
            result.append(node.name)
        for c in node.children:
            self._collect(c, result)
