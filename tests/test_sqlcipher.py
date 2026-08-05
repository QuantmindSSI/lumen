"""Tests for SQLCipher encryption-at-rest integration."""

import sqlite3
from pathlib import Path

import pytest

from lumen.config import LumenConfig
from lumen.data.schema import get_connection, get_encryption, init_db
from lumen.force.contextual.embed import MockEmbedder
from lumen.force.mnemonic.store import store_memory
from lumen.search import SearchPipeline
from lumen.security.crypto import derive_sqlcipher_passphrase


@pytest.fixture
def sqlcipher_config(tmp_path):
    return LumenConfig(
        store_path=tmp_path / ".lumen" / "store",
        model_path=tmp_path / ".lumen" / "models",
        vector_index="sqlite-vec",
        device="generic",
        encryption_key="test-sqlcipher-key-12345",
        database_encryption_mode="sqlcipher",
    )


@pytest.mark.skip(reason="SQLCipher encryption-at-rest planned for v0.3.0")
class TestSQLCipherIntegration:
    def test_sqlcipher_connection_opens(self, sqlcipher_config):
        conn = get_connection(sqlcipher_config)
        init_db(conn)
        conn.execute("INSERT INTO room(name) VALUES ('test_room')")
        conn.commit()
        row = conn.execute("SELECT name FROM room WHERE name = 'test_room'").fetchone()
        assert row is not None
        conn.close()

    def test_db_file_is_encrypted(self, sqlcipher_config):
        conn = get_connection(sqlcipher_config)
        init_db(conn)
        conn.execute("INSERT INTO room(name) VALUES ('secret')")
        conn.commit()
        conn.close()

        db_path = sqlcipher_config.db_path
        raw = db_path.read_bytes()
        # Encrypted SQLite files should NOT contain plaintext "secret"
        assert b"secret" not in raw
        # Should contain SQLCipher header or at least not standard SQLite header
        assert raw[:16] != b"SQLite format 3\x00"

    def test_salt_file_created(self, sqlcipher_config):
        conn = get_connection(sqlcipher_config)
        init_db(conn)
        conn.close()
        salt_path = sqlcipher_config.store_path / ".lumen_salt"
        assert salt_path.exists()
        assert salt_path.stat().st_size == 16

    def test_derived_passphrase_deterministic(self, sqlcipher_config):
        db_path = sqlcipher_config.store_path
        db_path.mkdir(parents=True, exist_ok=True)
        p1 = derive_sqlcipher_passphrase("mykey", db_path)
        p2 = derive_sqlcipher_passphrase("mykey", db_path)
        assert p1 == p2
        assert len(p1) == 64  # 32 bytes hex-encoded

    def test_get_encryption_disabled_under_sqlcipher(self, sqlcipher_config):
        enc = get_encryption(sqlcipher_config)
        assert not enc.enabled

    def test_store_and_retrieve_with_sqlcipher(self, sqlcipher_config):
        conn = get_connection(sqlcipher_config)
        init_db(conn)
        embedder = MockEmbedder(dims=sqlcipher_config.embedding_dims)
        chunk_id = store_memory(
            conn,
            content="SQLCipher protected memory",
            room_name="secure",
            embedding=embedder.encode_single("SQLCipher protected memory"),
            config=sqlcipher_config,
        )
        pipeline = SearchPipeline(conn, sqlcipher_config, embedder=embedder)
        results = pipeline.execute("protected memory", k=5)
        assert any(r.chunk_id == chunk_id for r in results)
        conn.close()

    def test_plaintext_fallback_when_sqlcipher_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lumen.data.schema._HAS_SQLCIPHER", False)
        config = LumenConfig(
            store_path=tmp_path / ".lumen" / "store",
            model_path=tmp_path / ".lumen" / "models",
            vector_index="sqlite-vec",
            device="generic",
            encryption_key="key",
            database_encryption_mode="sqlcipher",
        )
        with pytest.raises(RuntimeError, match="sqlcipher3 is not installed"):
            get_connection(config)
