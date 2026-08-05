"""S1: Encryption primitives for Lumen.

Provides:
  - Fernet field-level encryption (fallback when SQLCipher is unavailable).
  - SQLCipher passphrase derivation with persisted salt.
  - P2P transport encryption (AES-256-GCM) + HMAC-SHA256 frame signing.
"""

from __future__ import annotations

import base64
import hashlib
import os
import struct
import time
from pathlib import Path
from typing import Literal

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def derive_sqlcipher_passphrase(key: str, store_path: Path, iterations: int = 256_000) -> str:
    """Derive a high-entropy hex passphrase for SQLCipher PRAGMA key.

    Persists a random 16-byte salt in *store_path/.lumen_salt* so the
    same user key always yields the same derived passphrase across restarts.
    """
    salt_path = store_path / ".lumen_salt"
    if salt_path.exists():
        salt = salt_path.read_bytes()
    else:
        salt = os.urandom(16)
        store_path.mkdir(parents=True, exist_ok=True)
        salt_path.write_bytes(salt)
        os.chmod(salt_path, 0o600)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    derived = kdf.derive(key.encode("utf-8"))
    return derived.hex()


class FernetEncryption:
    """Field-level encryption using Fernet (AES-128-CBC + HMAC-SHA256)."""

    def __init__(self, key: str | None = None):
        """Initialise with a user-supplied key.

        Args:
            key: Any string; it is hashed to a 32-byte Fernet key.
        """
        if key:
            # Derive a URL-safe base64-encoded 32-byte key from arbitrary input
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            self._fernet = Fernet(base64.urlsafe_b64encode(digest))
        else:
            self._fernet = None

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt *plaintext* and return URL-safe base64 bytes."""
        if self._fernet is None:
            raise RuntimeError("Encryption not initialised with a key")
        return self._fernet.encrypt(plaintext.encode("utf-8"))

    def decrypt(self, token: bytes | str) -> str:
        """Decrypt *token* and return the plaintext string."""
        if self._fernet is None:
            raise RuntimeError("Encryption not initialised with a key")
        if isinstance(token, str):
            token = token.encode("utf-8")
        return self._fernet.decrypt(token).decode("utf-8")


class P2PCrypto:
    """AES-256-GCM transport encryption for Beam P2P packets.

    Each encrypted payload is: nonce (12 bytes) || ciphertext || tag (16 bytes).
    The framing layer handles length-prefixing of this combined blob.
    """

    def __init__(self, key: str | None = None):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if key:
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            self._aesgcm = AESGCM(digest)
        else:
            self._aesgcm = None

    @property
    def enabled(self) -> bool:
        return self._aesgcm is not None

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt *plaintext* and return nonce+ciphertext+tag."""
        if self._aesgcm is None:
            raise RuntimeError("P2P encryption not initialised with a key")
        nonce = os.urandom(12)
        return nonce + self._aesgcm.encrypt(nonce, plaintext, None)

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt *ciphertext* (nonce+ciphertext+tag) and plaintext."""
        if self._aesgcm is None:
            raise RuntimeError("P2P encryption not initialised with a key")
        nonce, ct = ciphertext[:12], ciphertext[12:]
        return self._aesgcm.decrypt(nonce, ct, None)


class P2PSign:
    """HMAC-SHA256 frame signing with embedded timestamp and replay-window protection.

    Wire format (prepended to encrypted or plaintext payload):
        timestamp_be64 (8 bytes) || signature (32 bytes) || original_payload
    """

    def __init__(self, key: str | None = None, replay_window_seconds: int = 30):
        self._key = hashlib.sha256(key.encode("utf-8")).digest() if key else None
        self._replay_window = replay_window_seconds
        self._seen_nonces: set[bytes] = set()

    @property
    def enabled(self) -> bool:
        return self._key is not None

    def sign(self, payload: bytes) -> bytes:
        """Return timestamp || HMAC || payload."""
        if self._key is None:
            raise RuntimeError("P2P signing not initialised with a key")
        ts = struct.pack(">Q", int(time.time()))
        mac = hmac.HMAC(self._key, hashes.SHA256())
        mac.update(ts + payload)
        return ts + mac.finalize() + payload

    def verify(self, framed: bytes) -> bytes:
        """Verify HMAC and timestamp; return original payload.

        Raises ValueError on tampering, expiry, or replay.
        """
        if self._key is None:
            raise RuntimeError("P2P signing not initialised with a key")
        if len(framed) < 40:
            raise ValueError("Frame too short for signed payload")
        ts_bytes, sig, payload = framed[:8], framed[8:40], framed[40:]
        ts = struct.unpack(">Q", ts_bytes)[0]
        now = int(time.time())
        if abs(now - ts) > self._replay_window:
            raise ValueError("Frame outside replay window")

        mac = hmac.HMAC(self._key, hashes.SHA256())
        mac.update(ts_bytes + payload)
        try:
            mac.verify(sig)
        except Exception as exc:
            raise ValueError("Frame signature verification failed") from exc

        nonce = hashlib.sha256(ts_bytes + sig).digest()
        if nonce in self._seen_nonces:
            raise ValueError("Replay detected")
        self._seen_nonces.add(nonce)
        # Cap memory usage of replay set
        if len(self._seen_nonces) > 10_000:
            self._seen_nonces = set(list(self._seen_nonces)[-5_000:])
        return payload
