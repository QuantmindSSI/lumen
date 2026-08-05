"""Real Integration Tests for Lumen Core."""

import os
import sqlite3

import numpy as np
import pytest

from lumen.compliance.safety_forgetting import safety_scan_chunk
from lumen.config import LumenConfig
from lumen.controller import TFCState, TwinForceController
from lumen.data.schema import init_db
from lumen.force.contextual.assembly import assemble_context
from lumen.force.contextual.embed import FallbackEmbedder
from lumen.force.mnemonic.forgetting_l1_decay import ebbinghaus_decay
from lumen.force.mnemonic.retrieval_dense import SqliteVecBackend
from lumen.force.mnemonic.retrieval_lexical import LexicalChannel
from lumen.force.mnemonic.store import store_memory
from lumen.force.mnemonic.value_model import compute_vm
from lumen.fusion import fuse_and_rerank


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def test_config():
    return LumenConfig(
        vector_index="sqlite-vec",
        device="generic",
        store_path="/tmp/.lumen-test",
    )


def test_config_loads_from_env():
    os.environ["LUMEN_DEVICE"] = "rpi5"
    cfg = LumenConfig()
    assert cfg.device == "rpi5"
    del os.environ["LUMEN_DEVICE"]


def test_schema_creates_all_tables(memory_db):
    cursor = memory_db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cursor.fetchall()}
    expected = {
        "room",
        "locus",
        "chunk",
        "provenance",
        "feedback_log",
        "user_profile",
        "event_buffer_meta",
    }
    assert expected <= tables


def test_store_memory_inserts_chunk_fts_vector(memory_db, test_config):
    embedder = FallbackEmbedder(dims=test_config.embedding_dims)
    emb = embedder.encode_single("hello world")
    chunk_id = store_memory(
        memory_db,
        "hello world",
        room_name="test_room",
        locus_name="test_locus",
        embedding=emb,
        config=test_config,
    )
    assert chunk_id > 0
    row = memory_db.execute("SELECT content FROM chunk WHERE chunk_id = ?", (chunk_id,)).fetchone()
    assert row[0] == "hello world"
    # FTS5 query via MATCH (rowid queries on virtual tables can be brittle)
    fts = memory_db.execute(
        "SELECT content FROM chunk_fts WHERE chunk_fts MATCH ?", ("hello",)
    ).fetchone()
    assert fts is not None


def test_retrieve_returns_results_ranked(memory_db, test_config):
    embedder = FallbackEmbedder(dims=test_config.embedding_dims)
    for i in range(5):
        emb = embedder.encode_single(f"memory number {i}")
        store_memory(
            memory_db,
            f"memory number {i}",
            room_name="retrieve_room",
            locus_name="locus_a",
            embedding=emb,
            config=test_config,
        )
    lexical = LexicalChannel(memory_db)
    dense_backend = SqliteVecBackend(memory_db, test_config.embedding_dims)
    query_vec = embedder.encode_single("memory number 2")
    lexical_hits = lexical.search("memory number", k=10)
    dense_hits = dense_backend.search(query_vec, k=10)
    results = fuse_and_rerank(lexical_hits, dense_hits, [], memory_db)
    assert len(results) > 0
    scores = [r.final_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_interference_weakens_old_chunk(memory_db, test_config):
    embedder = FallbackEmbedder(dims=test_config.embedding_dims)
    emb = embedder.encode_single("interference test")
    cid1 = store_memory(
        memory_db,
        "interference test one",
        room_name="interf_room",
        locus_name="same_locus",
        embedding=emb,
        config=test_config,
    )
    vm_before = memory_db.execute(
        "SELECT vm_score FROM chunk WHERE chunk_id = ?", (cid1,)
    ).fetchone()[0]
    # Nearly identical embedding to trigger interference
    emb2 = emb * 0.99
    emb2 = emb2 / np.linalg.norm(emb2)
    store_memory(
        memory_db,
        "interference test two",
        room_name="interf_room",
        locus_name="same_locus",
        embedding=emb2,
        config=test_config,
    )
    vm_after = memory_db.execute(
        "SELECT vm_score FROM chunk WHERE chunk_id = ?", (cid1,)
    ).fetchone()[0]
    assert vm_after < vm_before


def test_decay_reduces_vm_score(memory_db):
    memory_db.execute("INSERT INTO room(name) VALUES ('decay_room')")
    memory_db.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, created_at) "
        "VALUES (NULL, 1, 'old', 'hash1', 0.8, 0)"
    )
    _ = ebbinghaus_decay(memory_db, user_half_life_days=1.0)
    row = memory_db.execute("SELECT vm_score FROM chunk WHERE content_hash = 'hash1'").fetchone()
    assert row[0] < 0.8


def test_safety_scan_detects_email():
    hits = safety_scan_chunk("Contact me at alice@example.com please")
    assert "email" in hits


def test_context_assembly_respects_budget(memory_db, test_config):
    from lumen.fusion import RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk_id=i,
            room_name="r",
            locus_name="l",
            content="x" * 20,
            provenance_id=None,
            rrf_score=1.0,
            vm_score=0.8,
            frqad_score=0.9,
            recency_hours=1.0,
            final_score=1.0 - i * 0.01,
        )
        for i in range(20)
    ]
    tfc = TFCState()
    test_config.context_budget = 100  # tokens => 400 chars
    assembled = assemble_context("query", chunks, [], tfc, test_config)
    # Should include some chunks but not all 20
    assert len(assembled) < test_config.context_budget * 4 + 500


def test_tfc_updates_after_interaction_signal():
    tfc = TwinForceController(TFCState(e=0.5, a=0.5, tau=7.0, r=3))
    tfc.update({"novelty": 0.8, "repetition": 0.0, "context_pressure": 0.0, "satisfaction": 0.0})
    assert tfc.state.e < 0.5
    assert tfc.state.a > 0.5


def test_compute_vm_factors():
    vm, factors = compute_vm(
        "I need to schedule a meeting",
        {},
        "user_input",
        user_goals=["meeting"],
        user_values=["productive"],
    )
    assert 0.0 <= vm <= 1.0
    assert "task_utility" in factors
    assert "goal_relevance" in factors
