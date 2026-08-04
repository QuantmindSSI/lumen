"""B1: Context Assembly Jinja Schema.

Input wire: Jinja2, retrieval results (C2), goal-tree (C6), TFC state (C7)
Output wire: LLM prompt string
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import BaseLoader, Environment

from lumen.brand.errors import BudgetExceededError
from lumen.config import LumenConfig
from lumen.force.contextual.token_budget import get_token_counter
from lumen.logging import get_console_logger
from lumen.controller import TFCState
from lumen.fusion import RetrievedChunk

ASSEMBLY_TEMPLATE_STRING = """{{ system }}

{{ minimap }}

{{ goals }}

--- Retrieved Memories ---
{{ memories }}
--- End Memories ---

User: {{ query }}
Agent:
"""

env = Environment(loader=BaseLoader())
ASSEMBLY_TEMPLATE = env.from_string(ASSEMBLY_TEMPLATE_STRING)


def assemble_context(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    active_goals: list[str],
    tfc_state: TFCState,
    config: LumenConfig,
    token_counter=None,
    system_prompt_override: str | None = None,
) -> str:
    """Build the working window and enforce the token budget.

    Recovery strategy when budget is exceeded:
    1. Truncate individual chunks that are >30 % of budget.
    2. Drop lowest-scoring chunks until everything fits.
    3. If *still* nothing fits, return a ``BudgetExceededError`` (pilot
       mode will catch this and allow caller to handle).

    Args:
        query: User query string.
        retrieved_chunks: Ranked chunks from the fusion engine.
        active_goals: List of active goal strings.
        tfc_state: Current Twin-Force Controller state.
        config: Runtime configuration (provides context_budget).
        token_counter: Optional callable that returns token count for a string.
            If ``None``, one is constructed from the configured embedding model.
        system_prompt_override: If provided, replaces the default system prompt.

    Returns:
        Assembled prompt string.

    Raises:
        BudgetExceededError: If even the highest-scoring chunk exceeds budget.
    """
    if token_counter is None:
        model_dir = Path(config.model_path) / config.embedding_model
        token_counter = get_token_counter(model_dir)

    budget_tokens = config.context_budget
    system_prompt = (
        system_prompt_override
        if system_prompt_override is not None
        else _build_system_prompt(tfc_state)
    )
    minimap = _build_minimap(retrieved_chunks)
    goal_block = _render_goal_tree(active_goals)

    preamble = system_prompt + minimap + goal_block + query
    used = token_counter.count(preamble)

    # Stage 1: pack chunks into budget
    packed = []
    truncated_any = False
    for rc in sorted(retrieved_chunks, key=lambda x: x.final_score, reverse=True):
        chunk_text = (
            f"[Room:{rc.room_name} Locus:{rc.locus_name} Prov:{rc.provenance_id}]\n{rc.content}\n\n"
        )
        chunk_tokens = token_counter.count(chunk_text)

        # If a single chunk is absurdly large, truncate it aggressively
        if chunk_tokens > int(budget_tokens * 0.3):
            chunk_text = _truncated_chunk_text(rc, budget_tokens, token_counter)
            truncated_any = True
            chunk_tokens = token_counter.count(chunk_text)

        if used + chunk_tokens > budget_tokens:
            if packed:
                # We have at least something; log truncation and stop packing
                _log_truncation(
                    query,
                    packed=len(packed),
                    dropped=len(retrieved_chunks) - len(packed),
                    truncated=truncated_any,
                )
                break
            # Stage 2: emergency — drop lowest-score items to make room for top chunk
            if len(retrieved_chunks) > 1:
                # Just include the top chunk alone
                _log_truncation(
                    query, packed=1, dropped=len(retrieved_chunks) - 1, truncated=truncated_any
                )
                packed = [chunk_text]
                used += chunk_tokens
                break
            # Nothing can fit — truly broken
            raise BudgetExceededError(
                "LCX-2001 BudgetExceeded: even top chunk exceeds context budget"
            )

        packed.append(chunk_text)
        used += chunk_tokens

    if not packed and retrieved_chunks:
        raise BudgetExceededError("LCX-2001 BudgetExceeded: even top chunk exceeds context budget")

    return ASSEMBLY_TEMPLATE.render(
        system=system_prompt,
        minimap=minimap,
        goals=goal_block,
        memories="".join(packed),
        query=query,
        tfc=tfc_state,
    )


def _truncated_chunk_text(
    rc: RetrievedChunk,
    budget_tokens: int,
    token_counter,
) -> str:
    """Return a truncated version of *rc* content that fits ~30 % of budget."""
    hard_limit_chars = int(budget_tokens * 0.25 * 4)  # rough heuristic
    truncated = rc.content[:hard_limit_chars]
    if len(rc.content) > hard_limit_chars:
        truncated += "\n[...truncated]"
    return f"[Room:{rc.room_name} Locus:{rc.locus_name} Prov:{rc.provenance_id}]\n{truncated}\n\n"


def _log_truncation(query: str, packed: int, dropped: int, truncated: bool) -> None:
    """Log context-window truncation events."""
    if logger is not None:
        logger.info(
            "context_truncated",
            query=query[:50],
            packed=packed,
            dropped=dropped,
            truncated=truncated,
        )


def _build_minimap(chunks: list[RetrievedChunk]) -> str:
    rooms = sorted({c.room_name for c in chunks})
    return "Palace minimap: " + " -> ".join(rooms) + "\n"


def _build_system_prompt(tfc: TFCState) -> str:
    personality = "builder" if tfc.e > 0.6 else "explorer"
    return (
        f"You are a sovereign agent with a memory palace. "
        f"Your mnemonic bias is {tfc.e:.2f} ({personality} mode). "
        f"Attend to retrieved memories with care."
    )


def _render_goal_tree(goals: list[str]) -> str:
    if not goals:
        return "Goals: (none)\n"
    return "Goals:\n" + "\n".join(f"  - {g}" for g in goals) + "\n"


logger = get_console_logger(__name__)
