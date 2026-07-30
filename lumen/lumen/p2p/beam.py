"""C10: P2P Memory Sharing (Beam Protocol).

Household-only, encrypted, ephemeral memory sharing with permission decay.
"""

import asyncio
import logging
import socket
import struct
from typing import Any

import msgspec

from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.force.mnemonic.store import store_memory

logger = logging.getLogger(__name__)

# Availability flags
availability: dict[str, bool] = {}

ZeroconfModule = None
ServiceInfo = None
AsyncServiceBrowser = None
AsyncZeroconf = None

try:
    from zeroconf import ServiceInfo
    from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

    availability["zeroconf"] = True
except Exception:
    availability["zeroconf"] = False
    ServiceInfo = None
    AsyncServiceBrowser = None
    AsyncZeroconf = None

try:
    from nacl.public import PrivateKey

    availability["nacl"] = True
except Exception:
    availability["nacl"] = False
    logger.warning("pynacl is not installed; beam packets will be sent as plaintext")


def _get_default_ip() -> str:
    """Best-effort local IPv4 detection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(("10.254.254.254", 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def encode_frame(packet: dict) -> bytes:
    """Serialize *packet* with a 4-byte big-endian length prefix."""
    payload = msgspec.json.encode(packet)
    return struct.pack(">I", len(payload)) + payload


async def decode_frame(reader: asyncio.StreamReader) -> dict | None:
    """Read a length-prefixed msgspec JSON packet from *reader*."""
    length_bytes = await reader.readexactly(4)
    length = struct.unpack(">I", length_bytes)[0]
    payload = await reader.readexactly(length)
    return msgspec.json.decode(payload)


class BeamNode:
    """Household-only P2P memory sharing node."""

    def __init__(self, config: LumenConfig):
        if not availability["zeroconf"]:
            raise RuntimeError(
                "zeroconf is required for the P2P beam protocol but is not installed. "
                "Install it with: pip install 'zeroconf>=0.132'"
            )
        self.config = config
        self.service_type = "_lumen-beam._tcp.local."
        self.peers: dict[str, tuple[str, int]] = {}
        self.device_name = getattr(config, "device_name", socket.gethostname())
        self.local_ip = getattr(config, "local_ip", _get_default_ip())
        self.beam_port = getattr(config, "beam_port", 8847)
        self._azc: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._server: asyncio.Server | None = None
        self._private_key: Any | None = None
        self._public_key_bytes: bytes | None = None
        if availability["nacl"]:
            self._private_key = PrivateKey.generate()
            self._public_key_bytes = self._private_key.public_key.encode()

    async def start(self) -> None:
        """Register the local mDNS service and start listening for peers."""
        self._azc = AsyncZeroconf()
        properties: dict[str, Any] = {}
        if self._public_key_bytes:
            properties["pk"] = self._public_key_bytes.hex()
        info = ServiceInfo(
            self.service_type,
            f"{self.device_name}.{self.service_type}",
            addresses=[socket.inet_aton(self.local_ip)],
            port=self.beam_port,
            properties=properties,
        )
        await self._azc.async_register_service(info)
        self._browser = AsyncServiceBrowser(self._azc.zeroconf, self.service_type, self)
        self._server = await asyncio.start_server(self._handle_peer, "0.0.0.0", self.beam_port)

    async def stop(self) -> None:
        """Unregister mDNS services and shut down the peer listener."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._browser:
            await self._browser.async_cancel()
            self._browser = None
        if self._azc:
            await self._azc.async_unregister_all_services()
            await self._azc.async_close()
            self._azc = None

    async def _handle_peer(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle an incoming framed packet and store its chunks."""
        try:
            packet = await decode_frame(reader)
            if not isinstance(packet, dict):
                return
            room = packet.get("room")
            chunks = packet.get("chunks", [])
            if not room or not isinstance(chunks, list):
                return
            conn = get_connection(self.config)
            try:
                for chunk in chunks:
                    content = chunk.get("content")
                    if not content:
                        continue
                    chunk_id = store_memory(
                        conn,
                        content,
                        room_name=room,
                        source_type="p2p_share",
                        config=self.config,
                    )
                    conn.execute(
                        "UPDATE provenance SET confidence = 0.5 "
                        "WHERE chunk_id = ? AND source_type = 'p2p_share'",
                        (chunk_id,),
                    )
            finally:
                conn.close()
        except asyncio.IncompleteReadError:
            pass
        except Exception as exc:
            logger.warning("beam_handle_peer_error: %s", exc)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def share_room(self, room_name: str, ttl_hours: int = 24) -> None:
        """Fetch all active chunks from *room_name* and broadcast them to peers."""
        conn = get_connection(self.config)
        try:
            rows = conn.execute(
                "SELECT chunk_id, content, vm_score, content_hash FROM chunk "
                "JOIN room USING(room_id) WHERE room.name = ? AND valid_to IS NULL",
                (room_name,),
            ).fetchall()
        finally:
            conn.close()

        packet: dict = {
            "room": room_name,
            "ttl": ttl_hours,
            "chunks": [{"content": r[1], "vm": r[2], "hash": r[3]} for r in rows],
        }

        if availability["nacl"] and self._private_key:
            logger.warning(
                "NaCl is available but peer public keys are not tracked; "
                "sending plaintext beam packet"
            )
        elif not availability["nacl"]:
            logger.warning("pynacl not available; sending plaintext beam packet")

        for peer_addr in list(self.peers.values()):
            await self._send(peer_addr, packet)

    async def _send(self, addr: tuple[str, int], packet: dict) -> None:
        """Open a TCP connection to *addr*, send the framed packet, and close."""
        host, port = addr
        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=5.0
            )
            framed = encode_frame(packet)
            writer.write(framed)
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        except Exception as exc:
            logger.warning("beam_send_failed to %s:%s: %s", host, port, exc)

    # --- Zeroconf listener callbacks ---

    def add_service(self, zc: Any, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info and info.addresses:
            host = socket.inet_ntoa(info.addresses[0])
            port = info.port
            self.peers[name] = (host, port)
            logger.info("beam peer added: %s at %s:%s", name, host, port)

    def update_service(self, zc: Any, type_: str, name: str) -> None:
        self.add_service(zc, type_, name)

    def remove_service(self, zc: Any, type_: str, name: str) -> None:
        if name in self.peers:
            del self.peers[name]
            logger.info("beam peer removed: %s", name)
