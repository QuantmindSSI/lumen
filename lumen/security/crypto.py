"""S1: Fernet field-level encryption for sensitive memory content.

Provides transparent encryption/decryption for chunk content when an
encryption key is configured. This is a pragmatic fallback when SQLCipher
is not available.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet


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
        import os

        nonce = os.urandom(12)
        return nonce + self._aesgcm.encrypt(nonce, plaintext, None)

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt *ciphertext* (nonce+ciphertext+tag) and return plaintext."""
        if self._aesgcm is None:
            raise RuntimeError("P2P encryption not initialised with a key")
        nonce, ct = ciphertext[:12], ciphertext[12:]
        return self._aesgcm.decrypt(nonce, ct, None)
