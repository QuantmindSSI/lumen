import sqlite3

import numpy as np
import pytest

from lumen.config import LumenConfig
from lumen.data.schema import init_db
from lumen.force.mnemonic.store import store_memory


@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def config():
    return LumenConfig(
        device="generic",
        vector_index="sqlite-vec",
        store_path="/tmp/.lumen-test-store",
    )


def test_store_memory_returns_chunk_id(db_conn, config):
    chunk_id = store_memory(
        db_conn,
        "hello world",
        room_name="store_room",
        config=config,
    )
    assert isinstance(chunk_id, int)
    assert chunk_id > 0


def test_store_memory_persists_chunk(db_conn, config):
    chunk_id = store_memory(
        db_conn,
        "remember this message",
        room_name="persist_room",
        config=config,
    )
    row = db_conn.execute(
        "SELECT content, room_id FROM chunk WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    assert row is not None
    assert row["content"] == "remember this message"


def test_store_memory_with_embedding(db_conn, config):
    emb = np.random.randn(384).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    chunk_id = store_memory(
        db_conn,
        "vectorized memory",
        room_name="vec_room",
        embedding=emb,
        config=config,
    )
    row = db_conn.execute(
        "SELECT embedding FROM vec_fallback WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    assert row is not None


def test_store_memory_dedup_same_content(db_conn, config):
    content = "unique content for dedup check"
    cid1 = store_memory(db_conn, content, room_name="dedup_room", config=config)
    cid2 = store_memory(db_conn, content, room_name="dedup_room", config=config)
    assert cid1 == cid2


def test_store_memory_creates_room(db_conn, config):
    store_memory(db_conn, "new room test", room_name="new_room_xyz", config=config)
    row = db_conn.execute(
        "SELECT room_id FROM room WHERE name = ?", ("new_room_xyz",)
    ).fetchone()
    assert row is not None


def test_store_memory_with_locus(db_conn, config):
    cid = store_memory(
        db_conn,
        "placed at locus",
        room_name="locus_room",
        locus_name="my_locus",
        config=config,
    )
    row = db_conn.execute(
        "SELECT c.locus_id, l.name FROM chunk c JOIN locus l ON c.locus_id = l.locus_id WHERE c.chunk_id = ?",
        (cid,),
    ).fetchone()
    assert row["name"] == "my_locus"
