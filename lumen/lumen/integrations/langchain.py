"""LangChain integration for Lumen memory.

Provides ``LumenChatMemory`` — a convenience wrapper around
:class:`lumen.lumen.conversation.ConversationMemory` designed for
pre/post-processing in LangChain agents.

Usage::

    from langchain.agents import create_agent
    from lumen.integrations.langchain import LumenChatMemory

    memory = LumenChatMemory(room="support")

    # Before the LLM call
    enriched_prompt = memory.retrieve_context("How do I reset my password?")

    # After the LLM call
    memory.save_turn(
        "How do I reset my password?",
        "You can reset it via Settings > Account > Reset Password."
    )

If you want to pass Lumen as a native ``store=`` to
``langchain.agents.create_agent``, use :class:`LumenStore`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from lumen.config import LumenConfig
from lumen.lumen.conversation import ConversationMemory

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:
    pass


class LumenChatMemory:
    """High-level memory adapter for LangChain-style agents.

    Manages the retrieve → assemble → store → feedback loop for a single
    user/room.  The connection and embedder are lazy-initialised on first use.

    Args:
        config: :class:`LumenConfig`.  Created automatically if omitted.
        user_id: Multi-tenancy identifier.
        room: Room name for turn storage.
        system_prompt_override: Optional static system prompt injected into
            every assembled context.
    """

    def __init__(
        self,
        config: LumenConfig | None = None,
        user_id: str = "default",
        room: str = "conversations",
        system_prompt_override: str | None = None,
        embedder=None,
    ) -> None:
        self.config = config or LumenConfig()
        self.user_id = user_id
        self.room = room
        self.system_prompt_override = system_prompt_override
        self._embedder = embedder
        self._memory: ConversationMemory | None = None

    @property
    def memory(self) -> ConversationMemory:
        """Lazy-initialised ``ConversationMemory`` instance."""
        if self._memory is None:
            self._memory = ConversationMemory(
                config=self.config, embedder=self._embedder
            )
        return self._memory

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve memories for *query* and return an assembled prompt.

        The returned string contains a system prompt, palace minimap,
        active goals, and the ranked memory snippets.
        """
        turn = self.memory.retrieve_and_assemble(
            query=query,
            active_goals=self._active_goals(),
            system_prompt_override=self.system_prompt_override,
            top_k=top_k,
        )
        return turn.assembled_context

    def save_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Store a conversation turn and log implicit feedback.

        Internally re-runs retrieval so we know which chunks were shown to the
        agent for this turn.
        """
        turn = self.memory.retrieve_and_assemble(
            query=user_msg,
            active_goals=self._active_goals(),
            system_prompt_override=self.system_prompt_override,
            top_k=5,
        )
        self.memory.store_turn(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            retrieved_chunks=turn.retrieved_chunks,
            room_name=self.room,
        )
        self._maybe_learn_weights()

    def log_feedback(self, chunk_id: int, was_useful: bool) -> None:
        """Log explicit user feedback for a single chunk."""
        self.memory.log_explicit_feedback(
            chunk_id=chunk_id,
            was_useful=was_useful,
            user_id=self.user_id,
            feedback_type="explicit",
        )
        self._maybe_learn_weights()

    def clear(self) -> None:
        """Reset the in-memory conversation state."""
        if self._memory is not None:
            self._memory.close()
            self._memory = None

    def _active_goals(self) -> list[str]:
        """Placeholder for active-goal retrieval (M3)."""
        return []

    def _maybe_learn_weights(self) -> None:
        """Schedule V(m) weight update when enough feedback exists."""
        from lumen.force.mnemonic.value_model import learn_weights_from_feedback

        try:
            new_weights = learn_weights_from_feedback(
                self.memory.conn, user_id=self.user_id
            )
            if logger:
                logger.info("vm_weights_learned", user_id=self.user_id, weights=new_weights)
            # Persist back to user_profile
            import json
            self.memory.conn.execute(
                """UPDATE user_profile
                   SET vm_weights_json = ?
                   WHERE user_id = ?""",
                (json.dumps(new_weights), self.user_id),
            )
            self.memory.conn.commit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LumenStore — langchain_core.stores.BaseStore implementation
# ---------------------------------------------------------------------------

try:
    from langchain_core.stores import BaseStore
except Exception:  # pragma: no cover
    BaseStore = None  # type: ignore[assignment,misc]

if BaseStore is not None:

    class LumenStore(BaseStore[str, bytes]):
        """Lumen-backed key-value store for LangChain ``store=`` compatibility.

        Keys are expected in the form ``"user_id|namespace|key"``.
        Values are raw bytes stored as Lumen memories in the room named by
        *namespace*.
        """

        def __init__(
            self,
            config: LumenConfig | None = None,
            default_namespace: str = "store",
            embedder=None,
        ) -> None:
            self.config = config or LumenConfig()
            self.default_namespace = default_namespace
            self._embedder = embedder
            self._memory: ConversationMemory | None = None

        @property
        def memory(self) -> ConversationMemory:
            if self._memory is None:
                self._memory = ConversationMemory(
                    config=self.config, embedder=self._embedder
                )
            return self._memory

        def _parse_key(self, key: str) -> tuple[str, str, str]:
            parts = key.split("|", 2)
            if len(parts) == 3:
                return parts[0], parts[1], parts[2]
            return "default", self.default_namespace, key

        def mget(self, keys: Sequence[str]) -> list[bytes | None]:
            results: list[bytes | None] = []
            for key in keys:
                user_id, room, subkey = self._parse_key(key)
                row = self.memory.conn.execute(
                    "SELECT content FROM chunk WHERE room_id = (SELECT room_id FROM room WHERE name = ?) AND content LIKE ? AND valid_to IS NULL ORDER BY created_at DESC LIMIT 1",
                    (room, f"{subkey}:%"),
                ).fetchone()
                if row:
                    results.append(row["content"].split(":", 1)[1].encode())
                else:
                    results.append(None)
            return results

        def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
            from lumen.force.mnemonic.store import store_memory

            for key, value in key_value_pairs:
                _user_id, room, subkey = self._parse_key(key)
                # Ensure room exists
                self.memory.conn.execute(
                    "INSERT OR IGNORE INTO room(name, room_type) VALUES (?, 'ephemeral')",
                    (room,),
                )
                self.memory.conn.commit()
                content = f"{subkey}:{value.decode('utf-8', errors='replace')}"
                embedder = self.memory.embedder
                emb = embedder.encode_single(content)
                store_memory(
                    self.memory.conn,
                    content=content,
                    room_name=room,
                    source_type="import",
                    embedding=emb,
                    config=self.config,
                )

        def mdelete(self, keys: Sequence[str]) -> None:
            for key in keys:
                user_id, room, subkey = self._parse_key(key)
                self.memory.conn.execute(
                    """UPDATE chunk
                       SET valid_to = unixepoch()
                       WHERE room_id = (SELECT room_id FROM room WHERE name = ?)
                         AND content LIKE ?
                         AND valid_to IS NULL""",
                    (room, f"{subkey}:%"),
                )
            self.memory.conn.commit()

        def yield_keys(self, *, prefix: str | None = None) -> Iterator[str]:
            sql = """SELECT c.content, r.name
                     FROM chunk c JOIN room r ON c.room_id = r.room_id
                     WHERE c.valid_to IS NULL"""
            params: tuple[Any, ...] = ()
            if prefix:
                sql += " AND c.content LIKE ?"
                params = (f"{prefix}%",)
            for row in self.memory.conn.execute(sql, params):
                content = row["content"]
                room = row["name"]
                if ":" in content:
                    key = content.split(":", 1)[0]
                    yield f"default|{room}|{key}"
