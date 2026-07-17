"""B1: Context Assembly Jinja Schema.

Input wire: Jinja2, retrieval results (C2), goal-tree (C6), TFC state (C7)
Output wire: LLM prompt string
"""

from typing import List

from jinja2 import BaseLoader, Environment

from lumen.brand.errors import BudgetExceededError
from lumen.config import LumenConfig
from lumen.lumen.controller import TFCState
from lumen.lumen.fusion import RetrievedChunk

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
    retrieved_chunks: List[RetrievedChunk],
    active_goals: List[str],
    tfc_state: TFCState,
    config: LumenConfig,
) -> str:
    """
    Build the working window. Enforce token budget via character proxy
    (1 token ~ 4 chars for English text on BPE tokenisers).
    """
    budget_chars = config.context_budget * 4
    system_prompt = _build_system_prompt(tfc_state)
    minimap = _build_minimap(retrieved_chunks)
    goal_block = _render_goal_tree(active_goals)

    packed = []
    used = len(system_prompt) + len(minimap) + len(goal_block) + len(query)
    for rc in sorted(retrieved_chunks, key=lambda x: x.final_score, reverse=True):
        chunk_text = f"[Room:{rc.room_name} Locus:{rc.locus_name} Prov:{rc.provenance_id}]\n{rc.content}\n\n"
        if used + len(chunk_text) > budget_chars:
            break
        packed.append(chunk_text)
        used += len(chunk_text)

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


def _build_minimap(chunks: List[RetrievedChunk]) -> str:
    rooms = sorted({c.room_name for c in chunks})
    return "Palace minimap: " + " -> ".join(rooms) + "\n"


def _build_system_prompt(tfc: TFCState) -> str:
    personality = "builder" if tfc.e > 0.6 else "explorer"
    return (
        f"You are a sovereign agent with a memory palace. "
        f"Your mnemonic bias is {tfc.e:.2f} ({personality} mode). "
        f"Attend to retrieved memories with care."
    )


def _render_goal_tree(goals: List[str]) -> str:
    if not goals:
        return "Goals: (none)\n"
    return "Goals:\n" + "\n".join(f"  - {g}" for g in goals) + "\n"
