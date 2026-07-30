"""Tests for the P2P Beam protocol."""

import asyncio
import socket
import sqlite3
import struct
import sys

import pytest

from lumen.config import LumenConfig
from lumen.data.schema import init_db
from lumen.p2p.beam import BeamNode, availability, decode_frame, encode_frame


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def beam_config(tmp_path):
    return LumenConfig(
        device="generic",
        vector_index="sqlite-vec",
        store_path=str(tmp_path / ".lumen"),
        beam_port=9999,
    )


class MockStreamWriter:
    def __init__(self):
        self.written = []

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


def test_frame_roundtrip():
    packet = {
        "room": "test",
        "ttl": 5,
        "chunks": [{"content": "c1", "vm": 0.5, "hash": "abc"}],
    }
    framed = encode_frame(packet)
    assert isinstance(framed, bytes)
    assert framed[:4] == struct.pack(">I", len(framed) - 4)

    async def _run():
        reader = asyncio.StreamReader()
        reader.feed_data(framed)
        reader.feed_eof()
        return await decode_frame(reader)

    decoded = asyncio.run(_run())
    assert decoded == packet


def test_discovery_mock(beam_config):
    node = BeamNode(beam_config)

    class MockInfo:
        addresses = [socket.inet_aton("192.168.1.5")]
        port = 8847

    class MockZc:
        def get_service_info(self, type_, name):
            return MockInfo()

    node.add_service(MockZc(), "_lumen-beam._tcp.local.", "device1._lumen-beam._tcp.local.")
    assert "device1._lumen-beam._tcp.local." in node.peers
    assert node.peers["device1._lumen-beam._tcp.local."] == ("192.168.1.5", 8847)

    node.update_service(MockZc(), "_lumen-beam._tcp.local.", "device1._lumen-beam._tcp.local.")
    assert node.peers["device1._lumen-beam._tcp.local."] == ("192.168.1.5", 8847)

    node.remove_service(MockZc(), "_lumen-beam._tcp.local.", "device1._lumen-beam._tcp.local.")
    assert "device1._lumen-beam._tcp.local." not in node.peers


def test_share_room_packet(memory_db, beam_config, monkeypatch):
    memory_db.execute("INSERT INTO room(name) VALUES ('share_room')")
    memory_db.execute(
        "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score) "
        "VALUES (NULL, 1, 'shared memory', 'hash123', 0.7)"
    )
    monkeypatch.setattr("lumen.p2p.beam.get_connection", lambda cfg: memory_db)

    node = BeamNode(beam_config)
    node.peers = {"peer1": ("192.168.1.2", 8847)}

    sent_packets = []

    async def mock_send(addr, packet):
        sent_packets.append((addr, packet))

    monkeypatch.setattr(node, "_send", mock_send)

    asyncio.run(node.share_room("share_room", ttl_hours=12))

    assert len(sent_packets) == 1
    addr, packet = sent_packets[0]
    assert addr == ("192.168.1.2", 8847)
    assert packet["room"] == "share_room"
    assert packet["ttl"] == 12
    assert len(packet["chunks"]) == 1
    assert packet["chunks"][0]["content"] == "shared memory"
    assert packet["chunks"][0]["vm"] == 0.7
    assert packet["chunks"][0]["hash"] == "hash123"


class _NoCloseConnection:
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return self._conn.__exit__(*args, **kwargs)


def test_receive_stores_memory(memory_db, beam_config, monkeypatch):
    monkeypatch.setattr("lumen.p2p.beam.get_connection", lambda cfg: _NoCloseConnection(memory_db))

    node = BeamNode(beam_config)

    packet = {
        "room": "recv_room",
        "ttl": 24,
        "chunks": [{"content": "hello beam", "vm": 0.6, "hash": "h1"}],
    }
    framed = encode_frame(packet)

    async def _run():
        reader = asyncio.StreamReader()
        reader.feed_data(framed)
        reader.feed_eof()
        writer = MockStreamWriter()
        await node._handle_peer(reader, writer)

    asyncio.run(_run())

    row = memory_db.execute(
        "SELECT content FROM chunk WHERE content = ?", ("hello beam",)
    ).fetchone()
    assert row is not None
    assert row[0] == "hello beam"

    prov = memory_db.execute(
        "SELECT confidence FROM provenance p "
        "JOIN chunk c ON c.chunk_id = p.chunk_id WHERE c.content = ?",
        ("hello beam",),
    ).fetchone()
    assert prov is not None
    assert prov[0] == pytest.approx(0.5)


def test_receive_empty_packet_graceful(beam_config):
    node = BeamNode(beam_config)

    packet = {"room": "", "ttl": 24, "chunks": []}
    framed = encode_frame(packet)

    async def _run():
        reader = asyncio.StreamReader()
        reader.feed_data(framed)
        reader.feed_eof()
        writer = MockStreamWriter()
        await node._handle_peer(reader, writer)

    asyncio.run(_run())


def test_nacl_missing(beam_config, monkeypatch):
    monkeypatch.setitem(availability, "nacl", False)
    node = BeamNode(beam_config)
    assert node._private_key is None
    assert node._public_key_bytes is None


def test_zeroconf_missing():
    real_modules = {}
    for key in list(sys.modules.keys()):
        if key.startswith("zeroconf"):
            real_modules[key] = sys.modules.pop(key)

    if "lumen.p2p.beam" in sys.modules:
        del sys.modules["lumen.p2p.beam"]

    broken = type(sys)("zeroconf")
    sys.modules["zeroconf"] = broken

    try:
        import lumen.p2p.beam as beam_module
        from lumen.config import LumenConfig

        _ = beam_module.BeamNode(LumenConfig())
        pytest.fail("Expected RuntimeError on missing zeroconf")
    except RuntimeError as exc:
        assert "zeroconf is required" in str(exc)
    finally:
        for key in list(sys.modules.keys()):
            if key.startswith("zeroconf"):
                del sys.modules[key]
        for key, mod in real_modules.items():
            sys.modules[key] = mod
