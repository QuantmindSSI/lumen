"""Trivial scaffold verification tests."""

import sqlite3

from lumen.config import LumenConfig
from lumen.data.schema import init_db
from lumen.force.mnemonic.event_buffer import Event, EventMemoryBuffer


def test_imports():
    """Smoke-test that all scaffold modules are importable."""
    import lumen
    import lumen.brand.errors
    import lumen.config
    import lumen.data.migrate
    import lumen.data.schema
    import lumen.force.mnemonic.event_buffer

    assert lumen.__version__ == "0.2.0"


def test_schema_init():
    """A1: schema.sql must execute cleanly on an in-memory DB."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cursor.fetchall()}
    expected = {"room", "locus", "chunk", "provenance", "feedback_log", "user_profile"}
    assert expected <= tables
    conn.close()


def test_config_defaults():
    """D1: Config must instantiate with sensible defaults."""
    cfg = LumenConfig()
    assert cfg.device == "generic"
    assert cfg.embedding_dims == 384
    assert cfg.vector_index == "sqlite-vec"


def test_event_buffer():
    """D2: EventMemoryBuffer append and drain."""
    buf = EventMemoryBuffer(max_events=100, max_age_hours=1.0)
    evt = Event(raw_text="hello", session_id="s1")
    buf.append(evt)
    assert len(buf.query_since(0)) == 1
    expired = buf.drain_expired()
    # Event is not yet expired (age < 1h)
    assert len(expired) == 0
    assert len(buf.query_since(0)) == 1
