"""Tests for P2P Beam transport encryption."""

import pytest

from lumen.security.crypto import P2PCrypto


class TestP2PCrypto:
    def test_encrypt_decrypt_roundtrip(self):
        crypto = P2PCrypto(key="shared-secret-key")
        plaintext = b'{"room": "test", "chunks": [{"content": "hello"}]}'
        ciphertext = crypto.encrypt(plaintext)
        assert ciphertext != plaintext
        decrypted = crypto.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_disabled_by_default(self):
        crypto = P2PCrypto(key=None)
        assert not crypto.enabled
        with pytest.raises(RuntimeError):
            crypto.encrypt(b"data")

    def test_different_keys_fail(self):
        from cryptography.exceptions import InvalidTag

        crypto1 = P2PCrypto(key="key-one")
        crypto2 = P2PCrypto(key="key-two")
        ciphertext = crypto1.encrypt(b"secret")
        with pytest.raises(InvalidTag):
            crypto2.decrypt(ciphertext)

    def test_ciphertext_is_unique(self):
        crypto = P2PCrypto(key="shared-secret-key")
        plaintext = b'{"room": "test"}'
        ct1 = crypto.encrypt(plaintext)
        ct2 = crypto.encrypt(plaintext)
        assert ct1 != ct2  # nonce makes each encryption unique
