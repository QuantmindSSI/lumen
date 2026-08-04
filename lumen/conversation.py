"""ConversationMemory: production memory loop for agentic businesses.

Orchestrates retrieve → assemble → (caller runs LLM) → store turn → log feedback.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from lumen.brand.errors import ModelNotAvailableError
from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.force.contextual.assembly import assemble_context
from lumen.force.contextual.embed import get_embedder
from lumen.force.mnemonic.retrieval_graph import GraphChannel
from lumen.force.mnemonic.store import _tenant_id_supported, store_memory
from lumen.force.mnemonic.value_model import DEFAULT_WEIGHTS, learn_weights_from_feedback
from lumen.logging import get_console_logger
from lumen.controller import TwinForceController
from lumen.epistemic import EpistemicTracker
from lumen.fusion import RetrievedChunk
from lumen.goals import GoalTree
from lumen.search import SearchPipeline

logger = get_console_logger(__name__)


@dataclass
class TurnResult:
    """Result of a single conversation turn."""

    assembled_context: str
    retrieved_chunks: list[RetrievedChunk]
    stored_user_chunk_id: int | None
    stored_assistant_chunk_id: int | None


class ConversationMemory:
    """Drop-in memory manager for agent conversations.

    Typical usage::

        mem = ConversationMemory(config, user_id="alice")

        # 1. Retrieve & assemble context before sending to LLM
        turn = mem.retrieve_and_assemble("What did we decide about pricing?")
        response = my_llm_call(turn.assembled_context)

        # 2. Store the turn and log implicit feedback
        mem.store_turn(
            user_msg="What did we decide about pricing?",
            assistant_msg=response,
            retrieved_chunks=turn.retrieved_chunks,
            room_name="sales",
        )

    Attributes:
        config: :class:`LumenConfig` instance.
        conn: Shared SQLite connection (caller may manage lifecycle).
        tfc: :class:`TwinForceController` instance.
        embedder: Production embedder (raises if model missing).
        pipeline: :class:`SearchPipeline` wired with embedder.
        user_id: Multi-tenancy identifier.
        goals: Persistent :class:`GoalTree`.
        epistemic: Persistent :class:`EpistemicTracker`.
    """

    def __init__(
        self,
        config: LumenConfig | None = None,
        conn: sqlite3.Connection | None = None,
        tfc: TwinForceController | None = None,
        embedder=None,
        user_id: str = "default",
        tenant_id: str = "default",
    ) -> None:
        self.config = config or LumenConfig()
        self.conn = conn or get_connection(self.config)
        self.tfc = tfc or TwinForceController()
        self._own_conn = conn is None
        self.user_id = user_id
        self.tenant_id = tenant_id

        if embedder is not None:
            self.embedder = embedder
        else:
            try:
                self.embedder = get_embedder(self.config, allow_mock=False)
            except ModelNotAvailableError:
                from lumen.force.contextual.embed import MockEmbedder

                logger.warning("conversation_using_mock_embedder")
                self.embedder = MockEmbedder(dims=self.config.embedding_dims)

        self.pipeline = SearchPipeline(
            self.conn,
            self.config,
            tfc=self.tfc,
            embedder=self.embedder,
            graph=GraphChannel(self.conn) if GraphChannel.is_available(self.conn) else None,
        )
        self.goals = GoalTree(self.conn, user_id=user_id)
        self.epistemic = EpistemicTracker(self.conn, user_id=user_id)

        # Ensure user_profile row exists
        self._ensure_user_profile()

    def _ensure_user_profile(self) -> None:
        self.conn.execute(
            """INSERT INTO user_profile (user_id, vm_weights_json)
               VALUES (?, ?)
               ON CONFLICT(user_id) DO NOTHING""",
            (self.user_id, json.dumps(DEFAULT_WEIGHTS)),
        )
        self.conn.commit()

    def retrieve_and_assemble(
        self,
        query: str,
        active_goals: list[str] | None = None,
        system_prompt_override: str | None = None,
        top_k: int = 5,
    ) -> TurnResult:
        """Retrieve memories for *query* and assemble a context window.

        Args:
            query: User message or agent query string.
            active_goals: Optional list of active goal strings for boosting.
                If omitted, the persistent :attr:`goals` tree is queried.
            system_prompt_override: Replace the default palace system prompt.
            top_k: Max memories to retrieve.

        Returns:
            :class:`TurnResult` containing assembled prompt and raw chunks.
        """
        goals = active_goals if active_goals is not None else self.goals.active_path_keywords()
        results = self.pipeline.execute(query, goal_tree_keywords=goals, tenant_id=self.tenant_id)
        top = results[:top_k]

        if not top:
            ctx = assemble_context(
                query,
                [],
                goals,
                self.tfc.state,
                self.config,
                token_counter=None,
                system_prompt_override=system_prompt_override,
            )
            return TurnResult(ctx, [], None, None)

        ctx = assemble_context(
            query,
            top,
            goals,
            self.tfc.state,
            self.config,
            token_counter=None,
            system_prompt_override=system_prompt_override,
        )
        return TurnResult(ctx, top, None, None)

    def store_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        retrieved_chunks: list[RetrievedChunk] | None = None,
        room_name: str = "conversations",
    ) -> tuple[int | None, int | None]:
        """Store a conversation turn and log implicit feedback.

        Args:
            user_msg: Raw user message.
            assistant_msg: Raw assistant response.
            retrieved_chunks: Chunks that were injected into the prompt for this
                turn. Implicit positive feedback is logged for each.
            room_name: Room to store the turn in. Defaults to ``conversations``.

        Returns:
            Tuple of ``(user_chunk_id, assistant_chunk_id)``.
        """
        user_emb = self.embedder.encode_single(user_msg)
        assistant_emb = self.embedder.encode_single(assistant_msg)

        user_chunk_id = store_memory(
            self.conn,
            content=user_msg,
            room_name=room_name,
            source_type="user_input",
            embedding=user_emb,
            config=self.config,
            tenant_id=self.tenant_id,
        )
        assistant_chunk_id = store_memory(
            self.conn,
            content=assistant_msg,
            room_name=room_name,
            source_type="agent_reasoning",
            embedding=assistant_emb,
            config=self.config,
            tenant_id=self.tenant_id,
        )

        if retrieved_chunks:
            for rc in retrieved_chunks:
                self._log_implicit_feedback(rc.chunk_id, positive=True)

        # Mark retrieved chunk IDs as known facts
        if retrieved_chunks:
            self.epistemic.mark_known([rc.chunk_id for rc in retrieved_chunks])

        logger.info(
                "turn_stored",
                user_chunk_id=user_chunk_id,
                assistant_chunk_id=assistant_chunk_id,
                room=room_name,
                feedback_chunks=len(retrieved_chunks or []),
            )

        return user_chunk_id, assistant_chunk_id

    def log_explicit_feedback(
        self,
        chunk_id: int,
        was_useful: bool,
        user_id: str | None = None,
        feedback_type: str = "explicit",
    ) -> None:
        """Log explicit user feedback for a retrieved chunk.

        Args:
            chunk_id: Target chunk primary key.
            was_useful: ``True`` for positive, ``False`` for negative.
            user_id: Multi-tenancy identifier. Defaults to :attr:`user_id`.
            feedback_type: ``explicit``, ``implicit``, or ``repair``.
        """
        uid = user_id or self.user_id
        self._write_feedback(chunk_id, 1 if was_useful else 0, uid, feedback_type)
        self._maybe_learn_weights(uid)

    def learn_weights(self, user_id: str | None = None) -> dict[str, float]:
        """Force an immediate V(m) weight update from feedback history.

        Returns:
            New learned weights dictionary.
        """
        uid = user_id or self.user_id
        return self._maybe_learn_weights(uid, force=True)

    def _log_implicit_feedback(self, chunk_id: int, positive: bool = True) -> None:
        """Write an implicit feedback row."""
        self._write_feedback(chunk_id, 1 if positive else 0, self.user_id, "implicit")

    def _write_feedback(
        self,
        chunk_id: int,
        positive: int,
        user_id: str,
        feedback_type: str,
        tenant_id: str | None = None,
    ) -> None:
        """Insert a row into ``feedback_log``."""
        tid = tenant_id if tenant_id is not None else self.tenant_id
        if _tenant_id_supported(self.conn):
            self.conn.execute(
                """INSERT INTO feedback_log (chunk_id, user_id, positive, feedback_type, created_at, tenant_id)
                   VALUES (?, ?, ?, ?, unixepoch(), ?)""",
                (chunk_id, user_id, positive, feedback_type, tid),
            )
        else:
            self.conn.execute(
                """INSERT INTO feedback_log (chunk_id, user_id, positive, feedback_type, created_at)
                   VALUES (?, ?, ?, ?, unixepoch())""",
                (chunk_id, user_id, positive, feedback_type),
            )
        self.conn.commit()
        logger.info("feedback_logged", chunk_id=chunk_id, positive=positive, type=feedback_type)

    def _maybe_learn_weights(self, user_id: str, force: bool = False) -> dict[str, float]:
        """Update V(m) weights when enough feedback exists."""
        try:
            # Count feedback signals for this user
            count_row = self.conn.execute(
                "SELECT COUNT(*) FROM feedback_log WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not force and (count_row is None or count_row[0] < 10):
                return DEFAULT_WEIGHTS.copy()

            new_weights = learn_weights_from_feedback(self.conn, user_id=user_id)
            self.conn.execute(
                """UPDATE user_profile
                   SET vm_weights_json = ?
                   WHERE user_id = ?""",
                (json.dumps(new_weights), user_id),
            )
            self.conn.commit()
            logger.info("vm_weights_learned", user_id=user_id, weights=new_weights)
            return new_weights
        except Exception as exc:
            logger.warning("vm_weights_learn_failed", user_id=user_id, error=str(exc))
            return DEFAULT_WEIGHTS.copy()

    def close(self) -> None:
        """Close the connection if this instance created it."""
        if self._own_conn:
            self.conn.close()
