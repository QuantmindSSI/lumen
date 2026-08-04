"""LangGraph integration for Lumen.

Provides ``LumenCheckpointSaver`` — implements LangGraph's
``BaseCheckpointSaver`` protocol so Lumen can persist graph state
sovereignly, and ``LumenGraphStore`` for cross-thread long-term memory.

Usage::

    from langgraph.graph import StateGraph
    from lumen.integrations.langgraph import LumenCheckpointSaver

    graph = StateGraph(...)
    ...
    checkpointer = LumenCheckpointSaver()
    app = graph.compile(checkpointer=checkpointer)
    app.invoke({"input": "hello"}, config={"configurable": {"thread_id": "t1"}})
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.logging import get_console_logger

logger = get_console_logger(__name__)

# ---------------------------------------------------------------------------
# Conditional LangGraph imports
# ---------------------------------------------------------------------------
try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )
except Exception:
    from collections import namedtuple

    BaseCheckpointSaver = None  # type: ignore[assignment,misc]
    Checkpoint = Any  # type: ignore[misc]
    CheckpointMetadata = Any  # type: ignore[misc]
    CheckpointTuple = namedtuple(  # type: ignore[misc]
        "CheckpointTuple",
        ["config", "checkpoint", "metadata", "parent_config", "pending_writes"],
    )

_LumenSaverBase = BaseCheckpointSaver if BaseCheckpointSaver is not None else object


# ---------------------------------------------------------------------------
# LumenCheckpointSaver
# ---------------------------------------------------------------------------
class LumenCheckpointSaver(_LumenSaverBase):
    """Lumen-backed checkpoint saver for LangGraph.

    Stores LangGraph checkpoints as structured memories in the SQLite palace.
    Each ``thread_id`` gets its own locus inside a dedicated ``langgraph`` room,
    providing natural isolation and thread-scoped retrieval.

    When ``langgraph`` is installed (``pip install lumen[langgraph]``), this
    class extends ``BaseCheckpointSaver`` for native compatibility.  When
    ``langgraph`` is absent it falls back to a standalone mode that still
    exposes the same API surface.
    """

    ROOM_NAME = "langgraph"
    SOURCE_TYPE = "agent_reasoning"

    def __init__(
        self,
        config: LumenConfig | None = None,
        room: str = ROOM_NAME,
    ) -> None:
        self.config = config or LumenConfig()
        self.room = room
        self._conn: Any = None
        self._room_id: int | None = None

    # -----------------------------------------------------------------------
    # Connection helpers
    # -----------------------------------------------------------------------
    @property
    def conn(self) -> Any:
        if self._conn is None:
            self._conn = get_connection(self.config)
            self._ensure_room()
        return self._conn

    def _ensure_room(self) -> int:
        if self._room_id is not None:
            return self._room_id
        conn = self.conn
        row = conn.execute(
            "SELECT room_id FROM room WHERE name = ?", (self.room,)
        ).fetchone()
        if row:
            self._room_id = row[0]
        else:
            cur = conn.execute(
                "INSERT INTO room(name, room_type) VALUES (?, 'ephemeral')",
                (self.room,),
            )
            self._room_id = cur.lastrowid
            conn.commit()
        return self._room_id

    def _resolve_locus(self, thread_id: str) -> int:
        room_id = self._ensure_room()
        row = self._conn.execute(
            "SELECT locus_id FROM locus WHERE room_id = ? AND name = ?",
            (room_id, thread_id),
        ).fetchone()
        if row:
            return row[0]
        cur = self._conn.execute(
            "INSERT INTO locus(room_id, name) VALUES (?, ?)",
            (room_id, thread_id),
        )
        self._conn.commit()
        return cur.lastrowid

    def _thread_id(self, config: dict[str, Any]) -> str:
        return str(
            config.get("configurable", {}).get("thread_id", "default")
        )

    def _checkpoint_id(self, config: dict[str, Any]) -> str | None:
        return config.get("configurable", {}).get("checkpoint_id")

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------
    @staticmethod
    def _serialize(
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        parent_config: dict[str, Any] | None,
        pending_writes: list[tuple[str, str, Any]] | None,
    ) -> str:
        payload = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "parent_config": parent_config,
            "pending_writes": pending_writes or [],
        }
        return json.dumps(payload, default=str)

    @staticmethod
    def _deserialize(content: str) -> dict[str, Any]:
        return json.loads(content)

    # -----------------------------------------------------------------------
    # Core protocol (sync)
    # -----------------------------------------------------------------------
    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        """Retrieve the latest checkpoint tuple for *config*."""
        thread_id = self._thread_id(config)
        locus_id = self._resolve_locus(thread_id)
        checkpoint_id = self._checkpoint_id(config)

        if checkpoint_id:
            row = self.conn.execute(
                """SELECT content, created_at
                   FROM chunk
                   WHERE locus_id = ?
                     AND valid_to IS NULL
                     AND content_hash = ?
                   ORDER BY created_at DESC, chunk_id DESC
                   LIMIT 1""",
                (locus_id, checkpoint_id),
            ).fetchone()
        else:
            row = self.conn.execute(
                """SELECT content, created_at
                   FROM chunk
                   WHERE locus_id = ? AND valid_to IS NULL
                   ORDER BY created_at DESC, chunk_id DESC
                   LIMIT 1""",
                (locus_id,),
            ).fetchone()

        if not row:
            return None

        data = self._deserialize(row["content"])
        return CheckpointTuple(
            config=config,
            checkpoint=data["checkpoint"],
            metadata=data["metadata"],
            parent_config=data.get("parent_config"),
            pending_writes=data.get("pending_writes", []),
        )

    def put(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a checkpoint and return an updated config."""
        thread_id = self._thread_id(config)
        locus_id = self._resolve_locus(thread_id)
        room_id = self._ensure_room()
        checkpoint_id = checkpoint.get("id", "")

        parent_config = None
        if config.get("configurable", {}).get("checkpoint_id"):
            parent_config = {
                "configurable": {
                    **config.get("configurable", {}),
                    "checkpoint_ns": "",
                }
            }

        payload = self._serialize(checkpoint, metadata, parent_config, [])
        content_hash = hashlib.sha256(
            (checkpoint_id + thread_id).encode()
        ).hexdigest()[:32]

        self.conn.execute(
            """INSERT INTO chunk
               (locus_id, room_id, content, content_hash,
                vm_score, resolution, provenance_root)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                locus_id,
                room_id,
                payload,
                checkpoint_id or content_hash,
                0.5,
                "FP32",
                None,
            ),
        )
        self.conn.commit()

        logger.info(
                "langgraph_checkpoint_saved",
                thread_id=thread_id,
                checkpoint_id=checkpoint_id,
                step=metadata.get("step") if isinstance(metadata, dict) else None,
            )

        return {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "checkpoint_ns": "",
                "checkpoint_id": checkpoint_id,
            },
        }

    def put_writes(
        self,
        writes: list[tuple[str, str, Any]],
        config: dict[str, Any],
    ) -> None:
        """Store pending writes for a thread.

        Writes are stored as ephemeral chunks in a ``writes`` locus under the
        same room.  They are garbage-collected automatically by L3 eviction.
        """
        if not writes:
            return
        thread_id = self._thread_id(config)
        room_id = self._ensure_room()
        write_locus = self._resolve_locus(f"{thread_id}__writes")

        for channel, value in writes:
            payload = json.dumps(
                {"thread_id": thread_id, "channel": channel, "value": value},
                default=str,
            )
            self.conn.execute(
                """INSERT INTO chunk
                   (locus_id, room_id, content, content_hash,
                    vm_score, resolution)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (write_locus, room_id, payload, "lg_write", 0.1, "FP32"),
            )
        self.conn.commit()

    def list(
        self,
        config: dict[str, Any],
        *,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield checkpoint tuples for *thread_id*, newest first."""
        thread_id = self._thread_id(config)
        locus_id = self._resolve_locus(thread_id)

        # Optional filter on checkpoint metadata source/step
        where_extra = ""
        params: list[Any] = [locus_id]

        if filter:
            # Very basic filter support: exact match on metadata fields
            for key, val in filter.items():
                where_extra += f" AND json_extract(content, '$.metadata.{key}') = ?"
                params.append(val)

        sql = f"""SELECT content
                   FROM chunk
                   WHERE locus_id = ? AND valid_to IS NULL
                   {where_extra}
                   ORDER BY created_at DESC, chunk_id DESC
                   LIMIT ?"""
        params.append(limit or 100)

        for row in self.conn.execute(sql, params):
            data = self._deserialize(row["content"])
            yield CheckpointTuple(
                config=config,
                checkpoint=data["checkpoint"],
                metadata=data["metadata"],
                parent_config=data.get("parent_config"),
                pending_writes=data.get("pending_writes", []),
            )

    # -----------------------------------------------------------------------
    # Async methods
    # -----------------------------------------------------------------------
    async def aget_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def aput(
        self,
        config: dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, Any],
    ) -> dict[str, Any]:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        writes: list[tuple[str, str, Any]],
        config: dict[str, Any],
    ) -> None:
        self.put_writes(writes, config)

    async def alist(
        self,
        config: dict[str, Any],
        *,
        before: dict[str, Any] | None = None,
        limit: int | None = None,
        filter: dict[str, Any] | None = None,
    ):
        for item in self.list(config, before=before, limit=limit, filter=filter):
            yield item


# ---------------------------------------------------------------------------
# LumenGraphStore — cross-thread long-term memory for LangGraph
# ---------------------------------------------------------------------------
try:
    from langgraph.store.base import BaseStore as LangGraphBaseStore
except Exception:
    LangGraphBaseStore = None  # type: ignore[assignment,misc]

_LumenGraphStoreBase = LangGraphBaseStore if LangGraphBaseStore is not None else object


class LumenGraphStore(_LumenGraphStoreBase):
    """Lumen-backed store for LangGraph cross-thread memory.

    LangGraph's ``Store`` API lets graphs read and write long-term facts
    that persist across conversation threads.  LumenGraphStore maps each
    ``namespace`` to a room and each ``key`` to a chunked memory.
    """

    def __init__(
        self,
        config: LumenConfig | None = None,
        default_namespace: str = "memories",
        embedder: Any = None,
    ) -> None:
        self.config = config or LumenConfig()
        self.default_namespace = default_namespace
        self._embedder = embedder
        self._conn: Any = None

    @property
    def conn(self) -> Any:
        if self._conn is None:
            self._conn = get_connection(self.config)
        return self._conn

    def _embedding(self, text: str) -> Any:
        if self._embedder is not None:
            return self._embedder.encode_single(text)
        try:
            from lumen.force.contextual.embed import get_embedder

            e = get_embedder(self.config, allow_mock=False)
            return e.encode_single(text)
        except Exception:
            from lumen.force.contextual.embed import MockEmbedder

            return MockEmbedder(dims=self.config.embedding_dims).encode_single(text)

    def _namespace_to_room(self, namespace: tuple[str, ...] | str) -> str:
        if isinstance(namespace, tuple):
            return "_".join(namespace) or self.default_namespace
        return namespace or self.default_namespace

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> dict[str, Any] | None:
        room = self._namespace_to_room(namespace)
        row = self.conn.execute(
            """SELECT c.content, c.created_at, c.vm_score
               FROM chunk c
               JOIN locus l ON l.locus_id = c.locus_id
               JOIN room r ON r.room_id = c.room_id
               WHERE r.name = ? AND l.name = ? AND c.valid_to IS NULL
               ORDER BY c.created_at DESC
               LIMIT 1""",
            (room, key),
        ).fetchone()
        if not row:
            return None
        return {
            "key": key,
            "namespace": namespace,
            "value": row["content"],
            "created_at": row["created_at"],
            "score": row["vm_score"],
        }

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Any,
    ) -> None:
        from lumen.force.mnemonic.store import store_memory

        room = self._namespace_to_room(namespace)
        content = json.dumps({"key": key, "value": value}, default=str)
        emb = self._embedding(content)
        store_memory(
            self.conn,
            content=content,
            room_name=room,
            locus_name=key,
            embedding=emb,
            config=self.config,
        )

    def delete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        room = self._namespace_to_room(namespace)
        self.conn.execute(
            """UPDATE chunk
               SET valid_to = unixepoch()
               WHERE room_id = (SELECT room_id FROM room WHERE name = ?)
                 AND locus_id = (SELECT locus_id FROM locus WHERE room_id = (SELECT room_id FROM room WHERE name = ?) AND name = ?)
                 AND valid_to IS NULL""",
            (room, room, key),
        )
        self.conn.commit()

    def search(
        self,
        namespace: tuple[str, ...],
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Semantic search within a namespace using Lumen's pipeline."""
        from lumen.search import SearchPipeline

        room = self._namespace_to_room(namespace)
        # Ensure room exists so SearchPipeline can find chunks
        self.conn.execute(
            "INSERT OR IGNORE INTO room(name, room_type) VALUES (?, 'ephemeral')",
            (room,),
        )
        self.conn.commit()

        embedder_obj = self._embedder
        if embedder_obj is None:
            try:
                from lumen.force.contextual.embed import get_embedder

                embedder_obj = get_embedder(self.config, allow_mock=False)
            except Exception:
                from lumen.force.contextual.embed import MockEmbedder

                embedder_obj = MockEmbedder(dims=self.config.embedding_dims)

        pipeline = SearchPipeline(self.conn, self.config, embedder=embedder_obj)
        results = pipeline.execute(query, k=limit)

        out = []
        for r in results:
            if r.room_name != room:
                continue
            out.append({
                "key": r.locus_name,
                "namespace": namespace,
                "value": r.content,
                "score": r.final_score,
            })
        return out

    def list_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
    ) -> list[tuple[str, ...]]:
        """Return namespaces (rooms) that match prefix/suffix filters."""
        rows = self.conn.execute(
            "SELECT name FROM room WHERE name LIKE ?",
            (f"{self.default_namespace}%",),
        ).fetchall()
        namespaces = []
        for row in rows:
            name = row["name"]
            ns = tuple(name.split("_"))
            if prefix and ns[: len(prefix)] != prefix:
                continue
            if suffix and ns[-len(suffix) :] != suffix:
                continue
            if max_depth is not None and len(ns) > max_depth:
                continue
            namespaces.append(ns)
        return namespaces
