"""User-centric scenario integration tests.

These tests simulate realistic user journeys through the Lumen memory
framework, exercising the CLI-adjacent programmatic API end-to-end.
"""

from pathlib import Path

import numpy as np
import pytest

from lumen.compliance.safety_forgetting import safety_forget_chunk, safety_scan_chunk
from lumen.config import LumenConfig
from lumen.controller import TFCState, TwinForceController
from lumen.curiosity import curiosity_probe
from lumen.data.schema import get_connection
from lumen.force.contextual.assembly import assemble_context
from lumen.force.contextual.embed import FallbackEmbedder
from lumen.force.mnemonic.event_buffer import Event, EventMemoryBuffer
from lumen.force.mnemonic.forgetting_l1_decay import ebbinghaus_decay
from lumen.force.mnemonic.retrieval_temporal import temporal_point_query
from lumen.force.mnemonic.store import store_memory
from lumen.force.mnemonic.value_model import compute_vm
from lumen.fusion import RetrievedChunk
from lumen.search import SearchPipeline
from lumen.sovereign.optical import quantize_vector


@pytest.fixture
def fresh_config(tmp_path: Path):
    """A config pointing to a temporary store for scenario tests."""
    return LumenConfig(
        device="generic",
        vector_index="sqlite-vec",
        store_path=str(tmp_path / "store"),
        model_path=str(tmp_path / "models"),
        context_budget=512,
    )


@pytest.fixture
def fresh_conn(fresh_config: LumenConfig):
    """Yield a freshly initialised DB connection."""
    conn = get_connection(fresh_config)
    yield conn
    conn.close()


@pytest.fixture
def embedder(fresh_config: LumenConfig):
    return FallbackEmbedder(dims=fresh_config.embedding_dims)


class TestUserStoryDailyLogging:
    """A user stores memories throughout the day and retrieves them."""

    def test_store_and_retrieve_daily_memories(self, fresh_conn, fresh_config, embedder):
        memories = [
            ("Buy oat milk and sourdough", " errands ", "grocery"),
            ("Call dentist to reschedule", "health", "appointments"),
            ("Lumen FRQAD kernel needs ARM NEON SIMD", "work", "projects"),
        ]
        for content, room, locus in memories:
            emb = embedder.encode_single(content)
            store_memory(
                fresh_conn,
                content,
                room_name=room.strip(),
                locus_name=locus,
                embedding=emb,
                config=fresh_config,
            )

        pipeline = SearchPipeline(fresh_conn, fresh_config, embedder=embedder)
        results = pipeline.execute("dentist reschedule")
        assert len(results) >= 1
        assert any("dentist" in r.content.lower() for r in results)

    def test_palace_map_reflects_rooms_and_loci(self, fresh_conn, fresh_config, embedder):
        store_memory(
            fresh_conn,
            "Book flight to Tokyo",
            room_name="travel",
            locus_name="planning",
            embedding=embedder.encode_single("tokyo flight"),
            config=fresh_config,
        )
        rooms = [
            tuple(r) for r in fresh_conn.execute("SELECT name FROM room ORDER BY name").fetchall()
        ]
        assert ("travel",) in rooms
        loci = [
            tuple(r)
            for r in fresh_conn.execute(
                "SELECT name FROM locus WHERE room_id = (SELECT room_id FROM room WHERE name = 'travel')"
            ).fetchall()
        ]
        assert ("planning",) in loci

    def test_duplicate_content_is_deduplicated(self, fresh_conn, fresh_config, embedder):
        content = "Remember to water the plants"
        emb = embedder.encode_single(content)
        cid1 = store_memory(
            fresh_conn,
            content,
            room_name="home",
            locus_name="chores",
            embedding=emb,
            config=fresh_config,
        )
        cid2 = store_memory(
            fresh_conn,
            content,
            room_name="home",
            locus_name="chores",
            embedding=emb,
            config=fresh_config,
        )
        assert cid1 == cid2


class TestUserStoryMemoryInterference:
    """Similar memories in the same locus interfere with each other."""

    def test_similar_memories_weaken_older_ones(self, fresh_conn, fresh_config, embedder):
        content_a = "Alpha protocol version 2.1"
        content_b = "Alpha protocol version 2.2"
        emb_a = embedder.encode_single(content_a)
        # Perturb slightly so similarity is very high (> 0.85 threshold)
        emb_b = emb_a * 0.99 + np.random.randn(*emb_a.shape).astype(np.float32) * 0.01
        emb_b = emb_b / np.linalg.norm(emb_b)

        cid_a = store_memory(
            fresh_conn,
            content_a,
            room_name="tech",
            locus_name="protocols",
            embedding=emb_a,
            config=fresh_config,
        )
        vm_before = fresh_conn.execute(
            "SELECT vm_score FROM chunk WHERE chunk_id = ?", (cid_a,)
        ).fetchone()[0]

        store_memory(
            fresh_conn,
            content_b,
            room_name="tech",
            locus_name="protocols",
            embedding=emb_b,
            config=fresh_config,
        )
        vm_after = fresh_conn.execute(
            "SELECT vm_score FROM chunk WHERE chunk_id = ?", (cid_a,)
        ).fetchone()[0]

        assert vm_after < vm_before

    def test_dissimilar_memories_do_not_interfere(self, fresh_conn, fresh_config, embedder):
        content_a = "Quantum computing lecture notes"
        content_b = "Chocolate cake recipe"
        emb_a = embedder.encode_single(content_a)
        emb_b = embedder.encode_single(content_b)

        cid_a = store_memory(
            fresh_conn,
            content_a,
            room_name="notes",
            locus_name="bucket",
            embedding=emb_a,
            config=fresh_config,
        )
        vm_before = fresh_conn.execute(
            "SELECT vm_score FROM chunk WHERE chunk_id = ?", (cid_a,)
        ).fetchone()[0]

        store_memory(
            fresh_conn,
            content_b,
            room_name="notes",
            locus_name="bucket",
            embedding=emb_b,
            config=fresh_config,
        )
        vm_after = fresh_conn.execute(
            "SELECT vm_score FROM chunk WHERE chunk_id = ?", (cid_a,)
        ).fetchone()[0]

        assert vm_after == pytest.approx(vm_before, abs=1e-6)


class TestUserStorySafetyAndCompliance:
    """A user accidentally stores PII and the safety system catches it."""

    def test_safety_scan_detects_pii(self):
        assert "email" in safety_scan_chunk("Reach me at alice@example.com")
        assert "ssn" in safety_scan_chunk("SSN: 123-45-6789")
        assert "phone" in safety_scan_chunk("Call 555-123-4567")
        assert safety_scan_chunk("Just a normal thought") == []

    def test_safety_forget_redacts_chunk(self, fresh_conn, fresh_config, embedder):
        content = "My secret is alice@example.com"
        cid = store_memory(
            fresh_conn,
            content,
            room_name="secrets",
            locus_name="tmp",
            embedding=embedder.encode_single(content),
            config=fresh_config,
        )
        hits = safety_scan_chunk(content)
        assert "email" in hits

        safety_forget_chunk(fresh_conn, cid, reason="pii_detected")
        row = fresh_conn.execute(
            "SELECT content, valid_to, optical_level FROM chunk WHERE chunk_id = ?", (cid,)
        ).fetchone()
        assert row["content"] == "[REDACTED]"
        assert row["valid_to"] is not None
        assert row["optical_level"] == 2


class TestUserStoryTemporalQueries:
    """A user asks 'What did I know last Tuesday?'"""

    def test_temporal_point_query_finds_valid_facts(self, fresh_conn):
        now = 1_700_000_000
        # Fact valid from now, no end
        fresh_conn.execute("INSERT INTO room(name) VALUES ('history')")
        fresh_conn.execute(
            """INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, created_at, valid_from, valid_to)
                VALUES (NULL, 1, 'Fact A', 'hash_a', 0.8, ?, ?, NULL)""",
            (now, now),
        )
        # Fact valid only in the past
        fresh_conn.execute(
            """INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, created_at, valid_from, valid_to)
                VALUES (NULL, 1, 'Fact B old', 'hash_b', 0.8, ?, ?, ?)""",
            (now - 100_000, now - 100_000, now - 50_000),
        )
        hits = temporal_point_query(fresh_conn, ["Fact"], as_of_unix=now - 60_000)
        contents = [h.content for h in hits]
        assert "Fact B old" in contents
        assert "Fact A" not in contents

    def test_temporal_query_with_empty_keywords_returns_empty(self, fresh_conn):
        hits = temporal_point_query(fresh_conn, [])
        assert hits == []


class TestUserStoryForgettingAndDecay:
    """Old memories fade unless reinforced."""

    def test_ebbinghaus_decay_reduces_vm(self, fresh_conn):
        from datetime import datetime, timezone

        fresh_conn.execute("INSERT INTO room(name) VALUES ('aging')")
        now = int(datetime.now(timezone.utc).timestamp())
        one_hour_ago = now - 3600
        fresh_conn.execute(
            """INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, created_at, valid_from)
                VALUES (NULL, 1, 'Old memory', 'hash_old', 0.8, ?, ?)""",
            (one_hour_ago, one_hour_ago),
        )
        ebbinghaus_decay(fresh_conn, user_half_life_days=1.0, now=datetime.now(timezone.utc))
        vm = fresh_conn.execute(
            "SELECT vm_score FROM chunk WHERE content_hash = 'hash_old'"
        ).fetchone()[0]
        assert vm < 0.8
        assert vm > 0  # not fully released yet


class TestUserStoryCuriosity:
    """The system proactively surfaces stale or neglected memories."""

    def test_curiosity_surfaces_oldest_memories(self, fresh_conn, fresh_config, embedder):
        for text in [
            "First memory from weeks ago",
            "Second memory from yesterday",
            "Third memory just now",
        ]:
            store_memory(
                fresh_conn,
                text,
                room_name="journal",
                locus_name="diary",
                embedding=embedder.encode_single(text),
                config=fresh_config,
            )
        # Manually age the first memory
        fresh_conn.execute(
            "UPDATE chunk SET last_access_at = 0 WHERE content LIKE ?", ("First memory%",)
        )
        ids = curiosity_probe(fresh_conn, limit=2)
        assert len(ids) >= 1
        row = fresh_conn.execute(
            "SELECT content FROM chunk WHERE chunk_id = ?", (ids[0],)
        ).fetchone()
        assert "First memory" in row["content"]


class TestUserStoryContextAssembly:
    """Retrieved memories are packed into a prompt respecting the budget."""

    def test_assembled_context_includes_memories_and_budget(self, fresh_conn, fresh_config):
        chunks = [
            RetrievedChunk(
                chunk_id=i,
                room_name="r",
                locus_name="l",
                content=f"Memory number {i} with some text.",
                provenance_id=None,
                rrf_score=1.0,
                vm_score=0.9 - i * 0.05,
                frqad_score=0.9,
                recency_hours=float(i),
                final_score=1.0 - i * 0.01,
            )
            for i in range(10)
        ]
        tfc = TFCState()
        assembled = assemble_context("What do I know?", chunks, [], tfc, fresh_config)
        assert "Memory number 0" in assembled or "Memory number 1" in assembled
        # With budget 512 tokens (~2048 chars) it should not fit all 10 memories + boilerplate
        assert len(assembled) < fresh_config.context_budget * 4 + 300


class TestUserStoryTFCAdaptation:
    """The twin-force controller shifts personality based on signals."""

    def test_high_novelty_makes_explorer(self):
        tfc = TwinForceController(TFCState(e=0.5, a=0.5, tau=7.0, r=3))
        tfc.update(
            {"novelty": 0.8, "repetition": 0.0, "context_pressure": 0.0, "satisfaction": 0.0}
        )
        assert tfc.state.e < 0.5
        assert tfc.state.a > 0.5

    def test_high_repetition_makes_builder(self):
        tfc = TwinForceController(TFCState(e=0.5, a=0.5, tau=7.0, r=3))
        tfc.update(
            {"novelty": 0.0, "repetition": 0.8, "context_pressure": 0.0, "satisfaction": 0.0}
        )
        assert tfc.state.e > 0.5
        assert tfc.state.a < 0.5


class TestUserStoryOpticalDegradation:
    """Memories lose resolution under memory pressure."""

    def test_quantization_changes_values(self, embedder):
        vec = embedder.encode_single("test sentence for quantization")
        assert np.array_equal(quantize_vector(vec, "FP32"), vec)

        fp16 = quantize_vector(vec, "FP16")
        assert fp16.dtype == np.float32
        assert not np.array_equal(fp16, vec)

        binary = quantize_vector(vec, "BINARY")
        assert set(np.unique(binary)).issubset({-1.0, 1.0})

    def test_quantization_preserves_shape(self, embedder):
        vec = embedder.encode_single("another test")
        for res in ("FP32", "FP16", "INT8", "BINARY"):
            q = quantize_vector(vec, res)
            assert q.shape == vec.shape


class TestUserStoryValueModel:
    """Memories are scored by importance at store time."""

    def test_actionable_memory_gets_high_task_utility(self):
        vm, factors = compute_vm(
            "I need to schedule a meeting with the team tomorrow",
            user_weights=None,
            source_type="user_input",
            user_goals=["meeting"],
            user_values=["productive"],
        )
        assert 0.0 <= vm <= 1.0
        assert factors["task_utility"] > 0.5

    def test_self_referential_memory_gets_ego_score(self):
        vm, factors = compute_vm(
            "I feel happy about my progress today",
            user_weights=None,
            source_type="user_input",
        )
        assert factors["self_relevance"] > 0.0


class TestUserStoryEventBuffer:
    """Raw interactions live briefly in RAM before consolidation."""

    def test_buffer_holds_events_and_drains(self):
        buf = EventMemoryBuffer(max_events=100, max_age_hours=1.0)
        buf.append(Event(raw_text="User asked about weather", session_id="s1"))
        buf.append(Event(raw_text="Agent replied sunny", session_id="s1"))
        assert len(buf.query_since(0)) == 2
        expired = buf.drain_expired()
        assert len(expired) == 0  # not old enough
        assert len(buf.query_since(0)) == 2
