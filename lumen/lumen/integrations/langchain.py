"""LangChain integration for Lumen memory.

Provides ``LumenChatMemory`` — implements ``langchain.memory.BaseChatMemory``
for native LangChain agent compatibility, and ``LumenStore`` for
``langchain_core.stores.BaseStore`` compatibility.

Usage::

    from langchain.agents import create_react_agent
    from langchain.memory import ConversationBufferMemory
    from lumen.integrations.langchain import LumenChatMemory

    memory = LumenChatMemory(room="support")
    agent = create_react_agent(llm, tools, prompt, memory=memory)
"""

from __future__ import annotations

import json
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

# ---------------------------------------------------------------------------
# LumenChatMemory — native LangChain memory
# ---------------------------------------------------------------------------

try:
    from langchain.memory.chat_memory import BaseChatMemory
except Exception:
    BaseChatMemory = None  # type: ignore[assignment,misc]

_LumenChatMemoryBase = BaseChatMemory if BaseChatMemory is not None else object


class LumenChatMemory(_LumenChatMemoryBase):
    """Lumen-backed memory adapter for LangChain agents.

    When ``langchain`` is installed (``pip install lumen[langchain]``),
    this class extends ``langchain.memory.BaseChatMemory`` for native
    agent integration.  When ``langchain`` is absent it falls back to a
    standalone mode that still exposes the same API surface.

    Args:
        config: :class:`LumenConfig`.  Created automatically if omitted.
        user_id: Multi-tenancy identifier.
        room: Room name for turn storage.
        system_prompt_override: Optional static system prompt injected into
            every assembled context.
        return_messages: When True (default), returns LangChain ``HumanMessage``
            / ``AIMessage`` objects.
        input_key: Dict key to read from the input (default: ``"input"``).
        output_key: Dict key to read from the output (default: ``"output"``).
    """

    def __init__(
        self,
        config: LumenConfig | None = None,
        user_id: str = "default",
        room: str = "conversations",
        system_prompt_override: str | None = None,
        embedder: Any = None,
        return_messages: bool = True,
        input_key: str = "input",
        output_key: str = "output",
    ) -> None:
        if BaseChatMemory is not None:
            BaseChatMemory.__init__(
                self,
                return_messages=return_messages,
                input_key=input_key,
                output_key=output_key,
            )

        self.config = config or LumenConfig()
        self.user_id = user_id
        self.room = room
        self.system_prompt_override = system_prompt_override
        self._embedder = embedder
        self._memory: ConversationMemory | None = None
        self._last_context: str = ""

    @property
    def memory(self) -> ConversationMemory:
        if self._memory is None:
            embedder = self._embedder
            if embedder is None:
                try:
                    from lumen.force.contextual.embed import get_embedder
                    embedder = get_embedder(self.config, allow_mock=False)
                except Exception:
                    from lumen.force.contextual.embed import MockEmbedder
                    if logger:
                        logger.warning("langchain_memory_using_mock_embedder")
                    embedder = MockEmbedder(dims=self.config.embedding_dims)
            self._memory = ConversationMemory(
                config=self.config, embedder=embedder,
            )
        return self._memory

    @property
    def memory_variables(self) -> list[str]:
        """Return the keys this memory expects to load."""
        return ["lumen_context"]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Load memory context for the current input.

        Called by LangChain agents before the LLM call.  Returns the
        assembled Lumen context under the ``"lumen_context"`` key.
        """
        query = inputs.get(self.input_key, "")
        if not query:
            return {"lumen_context": ""}
        self._last_context = self.retrieve_context(query)
        return {"lumen_context": self._last_context}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save conversational context to Lumen.

        Called by LangChain agents after the LLM call.
        """
        user_msg = inputs.get(self.input_key, "")
        assistant_msg = outputs.get(self.output_key, "")
        if user_msg and assistant_msg:
            self.save_turn(user_msg, assistant_msg)

    def clear(self) -> None:
        """Reset the in-memory conversation state."""
        self._last_context = ""
        if self._memory is not None:
            self._memory.close()
            self._memory = None

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve memories for *query* and return an assembled prompt."""
        turn = self.memory.retrieve_and_assemble(
            query=query,
            active_goals=self._active_goals(),
            system_prompt_override=self.system_prompt_override,
            top_k=top_k,
        )
        return turn.assembled_context

    def save_turn(self, user_msg: str, assistant_msg: str) -> None:
        """Store a conversation turn and log implicit feedback."""
        try:
            turn = self.memory.retrieve_and_assemble(
                query=user_msg,
                active_goals=self._active_goals(),
                system_prompt_override=self.system_prompt_override,
                top_k=5,
            )
        except Exception:
            if logger:
                logger.warning("save_turn_retrieval_failed", user_id=self.user_id)
            return
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

    def _active_goals(self) -> list[str]:
        """Return currently active goal keywords from the persistent GoalTree."""
        if self._memory is not None and hasattr(self._memory, "goals"):
            return self._memory.goals.active_path_keywords()
        try:
            mem = self.memory
            if hasattr(mem, "goals"):
                return mem.goals.active_path_keywords()
        except Exception:
            if logger:
                logger.debug("active_goals_fallback", user_id=self.user_id)
        return []

    def _maybe_learn_weights(self) -> None:
        from lumen.force.mnemonic.value_model import learn_weights_from_feedback

        try:
            new_weights = learn_weights_from_feedback(
                self.memory.conn, user_id=self.user_id
            )
            if logger:
                logger.info("vm_weights_learned", user_id=self.user_id)
            self.memory.conn.execute(
                "UPDATE user_profile SET vm_weights_json = ? WHERE user_id = ?",
                (json.dumps(new_weights), self.user_id),
            )
            self.memory.conn.commit()
        except Exception:
            if logger:
                logger.warning("vm_weights_learn_failed", user_id=self.user_id)


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
            embedder: Any = None,
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
                _user_id, room, subkey = self._parse_key(key)
                row = self.memory.conn.execute(
                    """SELECT content
                       FROM chunk
                       WHERE room_id = (
                           SELECT room_id FROM room WHERE name = ?
                       )
                       AND content LIKE ?
                       AND valid_to IS NULL
                       ORDER BY created_at DESC
                       LIMIT 1""",
                    (room, f"{subkey}:%"),
                ).fetchone()
                if row:
                    parts = row["content"].split(":", 1)
                    results.append(parts[1].encode("utf-8") if len(parts) > 1 else b"")
                else:
                    results.append(None)
            return results

        def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
            from lumen.force.mnemonic.store import store_memory

            for key, value in key_value_pairs:
                _user_id, room, subkey = self._parse_key(key)
                self.memory.conn.execute(
                    "INSERT OR IGNORE INTO room(name, room_type) VALUES (?, 'ephemeral')",
                    (room,),
                )
                self.memory.conn.commit()
                content = f"{subkey}:{value.decode('utf-8', errors='replace')}"
                emb = self.memory.embedder.encode_single(content) if self.memory.embedder else None
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
                _user_id, room, subkey = self._parse_key(key)
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
