"""Tests for lumen.lumen.epistemic persistence."""

import pytest

from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.lumen.epistemic import EpistemicTracker


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


class TestEpistemicPersistence:
    def test_mark_known_and_load(self, fresh_config):
        conn = get_connection(fresh_config)
        ep = EpistemicTracker(conn, user_id="alice")
        ep.mark_known([1, 2, 3])
        ep.mark_gap(["pricing_info", "api_docs"])
        ep.confirm_truth("pricing_info")

        ep2 = EpistemicTracker(conn, user_id="alice")
        assert "1" in ep2.state.known_facts
        assert "pricing_info" in ep2.state.established_truths
        assert "pricing_info" not in ep2.state.assumed_gaps
        conn.close()

    def test_user_isolation(self, fresh_config):
        conn = get_connection(fresh_config)
        ep_alice = EpistemicTracker(conn, user_id="alice")
        ep_alice.mark_known([42])
        ep_bob = EpistemicTracker(conn, user_id="bob")
        ep_bob.mark_known([99])

        alice_loaded = EpistemicTracker(conn, user_id="alice")
        bob_loaded = EpistemicTracker(conn, user_id="bob")
        assert "42" in alice_loaded.state.known_facts
        assert "42" not in bob_loaded.state.known_facts
        assert "99" in bob_loaded.state.known_facts
        conn.close()
