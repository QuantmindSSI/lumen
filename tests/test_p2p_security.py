"""Tests for hardened Beam P2P protocol (signing, replay, TTL)."""

import asyncio
import struct
import time
from datetime import datetime, timedelta, timezone

import pytest

from lumen.config import LumenConfig
from lumen.p2p.beam import BeamNode, decode_frame, encode_frame
from lumen.security.crypto import P2PCrypto, P2PSign


@pytest.fixture
def secure_config(tmp_path):
    return LumenConfig(
        device="generic",
        vector_index="sqlite-vec",
        store_path=str(tmp_path / ".lumen"),
        beam_port=9998,
        sovereign=False,
        p2p_encryption_key="test-p2p-key-9876",
    )


class _FakeReader:
    def __init__(self, data: bytes):
        self._data = data

    async def readexactly(self, n: int) -> bytes:
        out, self._data = self._data[:n], self._data[n:]
        if len(out) < n:
            raise asyncio.IncompleteReadError(out, n)
        return out


class _FakeWriter:
    def __init__(self):
        self.written: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


class TestP2PSigning:
    def test_sign_verify_roundtrip(self):
        signer = P2PSign("mykey")
        payload = b"hello world"
        signed = signer.sign(payload)
        assert len(signed) == 8 + 32 + len(payload)
        recovered = signer.verify(signed)
        assert recovered == payload

    def test_tampered_frame_rejected(self):
        signer = P2PSign("mykey")
        signed = signer.sign(b"original")
        # Flip a byte in the payload region
        tampered = signed[:-1] + bytes([signed[-1] ^ 0xFF])
        with pytest.raises(ValueError, match="signature verification failed"):
            signer.verify(tampered)

    def test_replay_rejected(self):
        signer = P2PSign("mykey")
        signed = signer.sign(b"replay me")
        signer.verify(signed)  # first time OK
        with pytest.raises(ValueError, match="Replay detected"):
            signer.verify(signed)

    def test_expired_timestamp_rejected(self):
        signer = P2PSign("mykey", replay_window_seconds=1)
        signed = signer.sign(b"old")
        time.sleep(2.1)
        with pytest.raises(ValueError, match="outside replay window"):
            signer.verify(signed)


class TestHardenedBeamProtocol:
    def test_plaintext_fallback_rejected(self, tmp_path):
        config = LumenConfig(
            device="generic",
            vector_index="sqlite-vec",
            store_path=str(tmp_path / ".lumen"),
            beam_port=9997,
            sovereign=False,
            p2p_encryption_key=None,
        )
        with pytest.raises(RuntimeError, match="requires a p2p_encryption_key"):
            BeamNode(config)

    def test_insecure_override_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LUMEN_P2P_INSECURE", "1")
        config = LumenConfig(
            device="generic",
            vector_index="sqlite-vec",
            store_path=str(tmp_path / ".lumen"),
            beam_port=9996,
            sovereign=False,
            p2p_encryption_key=None,
        )
        # Should not raise; the node is created with crypto disabled
        node = BeamNode(config)
        assert not node._crypto.enabled

    def test_encode_decode_with_crypto_and_sign(self, secure_config):
        node = BeamNode(secure_config)
        packet = {"room": "test", "ttl": 5, "chunks": []}
        framed = encode_frame(packet, crypto=node._crypto, signer=node._signer)

        reader = _FakeReader(framed)
        decoded = asyncio.run(decode_frame(reader, crypto=node._crypto, signer=node._signer))
        assert decoded == packet

    def test_ttl_enforcement(self, secure_config, monkeypatch):
        node = BeamNode(secure_config)

        old_packet = {
            "room": "test",
            "ttl": 24,
            "_sent_at": (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(),
            "chunks": [{"content": "old news", "vm": 0.5, "hash": "h1"}],
        }
        framed = encode_frame(old_packet, crypto=node._crypto, signer=node._signer)

        calls = []
        monkeypatch.setattr("lumen.p2p.beam.store_memory", lambda *a, **k: calls.append(k))
        monkeypatch.setattr(
            "lumen.p2p.beam.get_connection",
            lambda cfg: type("C", (), {
                "execute": lambda *a, **k: None,
                "close": lambda: None,
                "__enter__": lambda s: s,
                "__exit__": lambda *a: None,
            })(),
        )

        reader = _FakeReader(framed)
        writer = _FakeWriter()
        asyncio.run(node._handle_peer(reader, writer))
        assert len(calls) == 0  # TTL expired → dropped