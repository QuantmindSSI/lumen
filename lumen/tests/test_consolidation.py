"""Tests for lumen.force.mnemonic.consolidation."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lumen.config import LumenConfig
from lumen.data.schema import get_connection, init_db
from lumen.force.mnemonic.consolidation import (
    _get_recent_fact_bullets,
    _get_rooms_with_new_activity,
    _group_similar_chunks,
    run_consolidation_pass,
)
from lumen.force.mnemonic.event_buffer import Event, EventMemoryBuffer


@pytest.fixture
def memory_db():
    """Yield an in-memory SQLite connection with the Lumen schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def tmp_config(tmp_path: Path):
    """Return a LumenConfig backed by a temporary store path."""
    store_path = tmp_path / "store"
    store_path.mkdir(parents=True)
    return LumenConfig(
        vector_index="sqlite-vec",
        device="generic",
        store_path=str(store_path),
    )


def test_group_similar_chunks_groups_by_hash(memory_db):
    """Chunks with identical content_hash should be grouped together."""
    memory_db.execute("INSERT INTO room(name) VALUES ('test_room')")
    room_id = memory_db.execute(
        "SELECT room_id FROM room WHERE name = 'test_room'"
    ).fetchone()[0]

    memory_db.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, valid_from) "
        "VALUES (?, ?, ?, ?, ?, unixepoch())",
        (None, room_id, "content A", "samehash123", 0.8),
    )
    memory_db.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, valid_from) "
        "VALUES (?, ?, ?, ?, ?, unixepoch())",
        (None, room_id, "content B", "samehash123", 0.9),
    )
    memory_db.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, valid_from) "
        "VALUES (?, ?, ?, ?, ?, unixepoch())",
        (None, room_id, "content C", "otherhash456", 0.7),
    )
    memory_db.commit()

    groups = _group_similar_chunks(memory_db)
    multi_groups = [g for g in groups if len(g) > 1]
    assert len(multi_groups) == 1
    assert len(multi_groups[0]) == 2
    ids = {c.chunk_id for c in multi_groups[0]}
    assert ids == {1, 2}


def test_dedup_merge_supersedes_losers(tmp_config):
    """Two valid chunks with identical hash should collapse to one winner."""
    conn = get_connection(tmp_config)
    conn.execute("INSERT INTO room(name) VALUES ('dedup_room')")
    room_id = conn.execute(
        "SELECT room_id FROM room WHERE name = 'dedup_room'"
    ).fetchone()[0]

    conn.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, valid_from) "
        "VALUES (?, ?, ?, ?, ?, unixepoch())",
        (None, room_id, "dup content", "duphash789", 0.5),
    )
    conn.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, valid_from) "
        "VALUES (?, ?, ?, ?, ?, unixepoch())",
        (None, room_id, "dup content 2", "duphash789", 0.9),
    )
    conn.commit()
    conn.close()

    with patch("lumen.force.mnemonic.consolidation.LocalLLM") as MockLLM:
        MockLLM.is_available.return_value = False
        ops = run_consolidation_pass(tmp_config, event_buffer=None, embedder=None)

    assert ops == 1  # exactly one merge

    conn = get_connection(tmp_config)
    rows = conn.execute(
        "SELECT chunk_id, valid_to, superseded_by FROM chunk WHERE content_hash = 'duphash789'"
    ).fetchall()
    valid = [r for r in rows if r["valid_to"] is None]
    invalid = [r for r in rows if r["valid_to"] is not None]
    assert len(valid) == 1
    assert len(invalid) == 1
    assert invalid[0]["superseded_by"] == valid[0]["chunk_id"]
    conn.close()


def test_narrative_generation_skipped_when_llm_unavailable(tmp_config):
    """If LocalLLM.is_available() is False, no narrative chunks should be created."""
    conn = get_connection(tmp_config)
    conn.execute("INSERT INTO room(name) VALUES ('narrative_room')")
    room_id = conn.execute(
        "SELECT room_id FROM room WHERE name = 'narrative_room'"
    ).fetchone()[0]

    for i in range(6):
        conn.execute(
            "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, created_at) "
            "VALUES (?, ?, ?, ?, ?, unixepoch())",
            (None, room_id, f"fact {i}", f"hash{i}", 0.8 - i * 0.01),
        )
    conn.commit()
    conn.close()

    with patch("lumen.force.mnemonic.consolidation.LocalLLM") as MockLLM:
        MockLLM.is_available.return_value = False
        ops = run_consolidation_pass(tmp_config, event_buffer=None, embedder=None)

    assert ops == 0

    conn = get_connection(tmp_config)
    count = conn.execute(
        """
        SELECT COUNT(*) FROM chunk c
        JOIN provenance p ON p.chunk_id = c.chunk_id
        WHERE p.source_type = 'consolidation'
        """
    ).fetchone()[0]
    assert count == 0
    conn.close()


def test_narrative_generation_when_llm_available(tmp_config):
    """If LocalLLM is available and >5 facts exist, a narrative should be stored."""
    conn = get_connection(tmp_config)
    conn.execute("INSERT INTO room(name) VALUES ('narrative_room')")
    room_id = conn.execute(
        "SELECT room_id FROM room WHERE name = 'narrative_room'"
    ).fetchone()[0]

    for i in range(6):
        conn.execute(
            "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, created_at) "
            "VALUES (?, ?, ?, ?, ?, unixepoch())",
            (None, room_id, f"fact {i}", f"hash{i}", 0.8 - i * 0.01),
        )
    conn.commit()
    conn.close()

    mock_llm_instance = MagicMock()
    mock_llm_instance.summarize.return_value = "User enjoys facts."

    with patch(
        "lumen.force.mnemonic.consolidation.LocalLLM",
        return_value=mock_llm_instance,
    ) as MockLLMClass:
        MockLLMClass.is_available.return_value = True
        ops = run_consolidation_pass(tmp_config, event_buffer=None, embedder=None)

    assert ops == 1  # one narrative generated

    conn = get_connection(tmp_config)
    count = conn.execute(
        """
        SELECT COUNT(*) FROM chunk c
        JOIN provenance p ON p.chunk_id = c.chunk_id
        WHERE p.source_type = 'consolidation'
        """
    ).fetchone()[0]
    assert count == 1
    conn.close()


def test_get_rooms_with_new_activity(memory_db):
    """Only rooms with valid chunks from the last 7 days should be returned."""
    memory_db.execute("INSERT INTO room(name) VALUES ('active_room')")
    memory_db.execute("INSERT INTO room(name) VALUES ('inactive_room')")

    room_active = memory_db.execute(
        "SELECT room_id FROM room WHERE name = 'active_room'"
    ).fetchone()[0]
    room_inactive = memory_db.execute(
        "SELECT room_id FROM room WHERE name = 'inactive_room'"
    ).fetchone()[0]

    memory_db.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, created_at) "
        "VALUES (?, ?, ?, ?, ?, unixepoch())",
        (None, room_active, "recent", "hash_recent", 0.5),
    )
    memory_db.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, created_at) "
        "VALUES (?, ?, ?, ?, ?, unixepoch() - 10 * 86400)",
        (None, room_inactive, "old", "hash_old", 0.5),
    )
    memory_db.commit()

    rooms = _get_rooms_with_new_activity(memory_db)
    assert len(rooms) == 1
    assert rooms[0] == (room_active, "active_room")


def test_get_recent_fact_bullets(memory_db):
    """Fact bullets should be recent, valid, ordered by vm_score desc."""
    memory_db.execute("INSERT INTO room(name) VALUES ('bullet_room')")
    room_id = memory_db.execute(
        "SELECT room_id FROM room WHERE name = 'bullet_room'"
    ).fetchone()[0]

    facts = [
        ("high vm", "hash1", 0.9),
        ("low vm", "hash2", 0.3),
        ("medium vm", "hash3", 0.6),
    ]
    for content, chash, vm in facts:
        memory_db.execute(
            "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, created_at) "
            "VALUES (?, ?, ?, ?, ?, unixepoch())",
            (None, room_id, content, chash, vm),
        )
    memory_db.commit()

    bullets = _get_recent_fact_bullets(memory_db, room_id, days=7)
    assert len(bullets) == 3
    assert bullets[0] == "high vm"
    assert bullets[1] == "medium vm"
    assert bullets[2] == "low vm"


def test_event_buffer_drained_into_db(tmp_config):
    """Events from the buffer should be stored as chunks during consolidation."""
    buf = EventMemoryBuffer(max_events=10, max_age_hours=0.0)
    buf.append(Event(raw_text="hello world", session_id="sess_1"))
    buf.append(Event(raw_text="goodbye world", session_id="sess_2"))

    with patch("lumen.force.mnemonic.consolidation.LocalLLM") as MockLLM:
        MockLLM.is_available.return_value = False
        ops = run_consolidation_pass(tmp_config, event_buffer=buf, embedder=None)

    # ops should be 0 because neither dedup nor narratives occurred
    assert ops == 0
    assert len(buf.query_since(0)) == 0  # buffer drained

    conn = get_connection(tmp_config)
    rows = conn.execute(
        "SELECT content FROM chunk ORDER BY chunk_id"
    ).fetchall()
    contents = [r[0] for r in rows]
    assert "hello world" in contents
    assert "goodbye world" in contents
    conn.close()


def test_no_circular_supersession(tmp_config):
    """A winner should never supersede itself."""
    conn = get_connection(tmp_config)
    conn.execute("INSERT INTO room(name) VALUES ('circle_room')")
    room_id = conn.execute(
        "SELECT room_id FROM room WHERE name = 'circle_room'"
    ).fetchone()[0]

    conn.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, valid_from) "
        "VALUES (?, ?, ?, ?, ?, unixepoch())",
        (None, room_id, "solo", "solohash", 0.5),
    )
    conn.commit()
    conn.close()

    with patch("lumen.force.mnemonic.consolidation.LocalLLM") as MockLLM:
        MockLLM.is_available.return_value = False
        ops = run_consolidation_pass(tmp_config, event_buffer=None, embedder=None)

    assert ops == 0

    conn = get_connection(tmp_config)
    row = conn.execute(
        "SELECT valid_to, superseded_by FROM chunk WHERE content_hash = 'solohash'"
    ).fetchone()
    assert row["valid_to"] is None
    assert row["superseded_by"] is None
    conn.close()
