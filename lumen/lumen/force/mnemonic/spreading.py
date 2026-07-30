"""A92: Spreading Activation Module.

Input wire: Kùzu / NetworkX entity graph, TFC attentional temperature `a`
Output wire: C8 (prefetch buffer)
Secret sauce: Controlled echo-location; γ decay prevents explosion
"""

from typing import Any

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)

SPREAD_GAMMA = 0.4


def spread_activation(
    graph: Any | None,
    seed_ids: list[int],
    a: float,
) -> dict[int, float]:
    """
    Spread activation across a graph from seed nodes.

    `a` ∈ [0,1] maps directly to TFC attentional temperature.
    Higher `a` → wider spread. Formula from brainstorm 9.2.6.

    Args:
        graph: Graph-like object supporting membership testing (``node in graph``),
            ``.neighbors(node)``, and edge data accessible via
            ``graph[node][neighbor]``. May be ``None``.
        seed_ids: List of integer node IDs to activate.
        a: Attentional temperature in [0, 1].

    Returns:
        Mapping of node ID -> activation score.
    """
    if not seed_ids:
        return {}

    activations: dict[int, float] = dict.fromkeys(seed_ids, 1.0)

    if graph is None:
        if logger:
            logger.debug("spread_activation_no_graph", seed_count=len(seed_ids))
        return activations

    threshold = max(0.1, 1.0 - a)  # a=1 → threshold=0.1 (max spread)
    frontier = set(seed_ids)

    for hop in range(1, 3):  # max 2 hops on SBC
        next_frontier: set[int] = set()
        for node in frontier:
            if node not in graph:
                continue
            for neighbor in graph.neighbors(node):
                if neighbor not in graph:
                    continue
                try:
                    weight = float(graph[node][neighbor].get("weight", 1.0))
                except (AttributeError, KeyError, TypeError):
                    weight = 1.0

                activation = activations.get(node, 0.0) * (SPREAD_GAMMA**hop) * weight
                if activation > threshold:
                    activations[neighbor] = max(activations.get(neighbor, 0.0), activation)
                    next_frontier.add(neighbor)

        frontier = next_frontier
        if not frontier:
            break

    if logger:
        logger.debug(
            "spread_activation_complete",
            seeds=len(seed_ids),
            a=a,
            activated=len(activations),
        )

    return activations
