"""Tests for lumen.lumen.goals persistence."""

import pytest

from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.goals import GoalTree


@pytest.fixture
def fresh_config(tmp_path):
    cfg = LumenConfig(
        store_path=tmp_path / "store",
        model_path=tmp_path / "models",
        vector_index="sqlite-vec",
        device="generic",
    )
    cfg.store_path.mkdir(parents=True, exist_ok=True)
    cfg.model_path.mkdir(parents=True, exist_ok=True)
    return cfg


class TestGoalTreePersistence:
    def test_add_and_load(self, fresh_config):
        conn = get_connection(fresh_config)
        tree = GoalTree(conn, user_id="alice")
        tree.add_goal("ship v1", parent_name=None)
        tree.add_goal("backend", parent_name="ship v1")
        tree.add_goal("tests", parent_name="ship v1")
        tree.set_active("backend")

        # Re-instantiate — should load from SQLite
        tree2 = GoalTree(conn, user_id="alice")
        assert "ship v1" in tree2.all_goals()
        assert "backend" in tree2.all_goals()
        assert "tests" in tree2.all_goals()
        assert tree2._active is not None
        assert tree2._active.name == "backend"
        conn.close()

    def test_user_isolation(self, fresh_config):
        conn = get_connection(fresh_config)
        tree_alice = GoalTree(conn, user_id="alice")
        tree_alice.add_goal("alice_goal")
        tree_bob = GoalTree(conn, user_id="bob")
        tree_bob.add_goal("bob_goal")

        alice_goals = GoalTree(conn, user_id="alice").all_goals()
        bob_goals = GoalTree(conn, user_id="bob").all_goals()
        assert "alice_goal" in alice_goals
        assert "alice_goal" not in bob_goals
        assert "bob_goal" in bob_goals
        assert "bob_goal" not in alice_goals
        conn.close()
