"""A9: 7-Factor Value Model V(m).

Input wire: spaCy / sklearn (B4), user interaction history (C6, D3)
Output wire: A6 (store pipeline), C3 (fusion reranking), C7 (TFC)
Secret sauce: Per-user learned weights, no API calls, CPU-only
"""

import json
import math

import numpy as np

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass

# Default weights for cold-start user (untrained)
DEFAULT_WEIGHTS = {
    "goal_relevance": 0.20,
    "value_alignment": 0.15,
    "self_relevance": 0.15,
    "task_utility": 0.15,
    "emotional_intensity": 0.15,
    "reliability": 0.10,
    "usage_history": 0.10,
}

FACTOR_KEYS = list(DEFAULT_WEIGHTS.keys())


def _max_similarity_to_phrases(content: str, phrases: list[str]) -> float:
    """Compute max cosine-like overlap via simple word-set Jaccard."""
    if not phrases:
        return 0.0
    content_words = set(content.lower().split())
    best = 0.0
    for ph in phrases:
        ph_words = set(ph.lower().split())
        inter = len(content_words & ph_words)
        union = len(content_words | ph_words)
        sim = inter / union if union else 0.0
        if sim > best:
            best = sim
    return best


def _jaccard_overlap(content: str, values: list[str]) -> float:
    """Jaccard overlap between content words and value words."""
    if not values:
        return 0.0
    content_words = set(content.lower().split())
    value_words = set()
    for v in values:
        value_words.update(v.lower().split())
    inter = len(content_words & value_words)
    union = len(content_words | value_words)
    return inter / union if union else 0.0


def _simple_sentiment_polarity(text: str) -> float:
    """Crude polarity proxy without TextBlob."""
    positive = {"good", "great", "excellent", "happy", "love", "best", "awesome", "fantastic"}
    negative = {"bad", "terrible", "awful", "hate", "worst", "sad", "angry", "poor"}
    tokens = text.lower().split()
    pos = sum(1 for t in tokens if t in positive)
    neg = sum(1 for t in tokens if t in negative)
    total = len(tokens) or 1
    return (pos - neg) / total


def extract_factors(
    content: str,
    source_type: str,
    user_goals: list[str],
    user_values: list[str],
    sentiment_pipeline=None,
) -> dict[str, float]:
    """Compute raw factor scores from content and user profile."""
    # 1. Goal relevance
    g_rel = _max_similarity_to_phrases(content, user_goals) if user_goals else 0.5

    # 2. Value alignment
    v_align = _jaccard_overlap(content, user_values) if user_values else 0.5

    # 3. Self relevance: pronoun density
    self_words = {"i", "me", "my", "myself"}
    tokens = content.lower().split()
    self_rel = min(1.0, sum(1 for t in tokens if t in self_words) / max(len(tokens), 10))

    # 4. Task utility
    action_verbs = {"schedule", "book", "buy", "call", "email", "remind", "need", "must", "should"}
    task_u = 1.0 if any(v in tokens for v in action_verbs) else 0.3

    # 5. Emotional intensity
    if sentiment_pipeline is not None:
        try:
            pol = abs(sentiment_pipeline(content).sentiment.polarity)
        except Exception:
            pol = abs(_simple_sentiment_polarity(content))
    else:
        pol = abs(_simple_sentiment_polarity(content))
    emo = max(0.3, pol)

    # 6. Reliability
    rel_map = {
        "user_input": 0.9,
        "agent_reasoning": 0.7,
        "consolidation": 0.75,
        "import": 0.6,
        "p2p_share": 0.5,
    }
    rel = rel_map.get(source_type, 0.5)

    # 7. Usage history
    usage = 0.5

    return {
        "goal_relevance": round(g_rel, 3),
        "value_alignment": round(v_align, 3),
        "self_relevance": round(self_rel, 3),
        "task_utility": round(task_u, 3),
        "emotional_intensity": round(emo, 3),
        "reliability": round(rel, 3),
        "usage_history": round(usage, 3),
    }


def compute_vm(
    content: str,
    user_weights: dict[str, float] | None,
    source_type: str,
    user_goals: list[str] | None = None,
    user_values: list[str] | None = None,
    sentiment_pipeline=None,
) -> tuple[float, dict[str, float]]:
    """Scalar V(m) = sigmoid( dot(weights, factors) )."""
    weights = {**DEFAULT_WEIGHTS, **(user_weights or {})}
    factors = extract_factors(
        content, source_type, user_goals or [], user_values or [], sentiment_pipeline
    )
    vec = np.array([factors[k] for k in FACTOR_KEYS])
    w = np.array([weights[k] for k in FACTOR_KEYS])
    z = float(np.dot(w, vec))
    vm = 1.0 / (1.0 + math.exp(-z))
    return vm, factors


def learn_weights_from_feedback(
    conn,
    user_id: str = "default",
    method: str = "nelder-mead",
) -> dict[str, float]:
    """Learn per-user weights from click/retrieval-success feedback."""
    import sqlite3
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be sqlite3.Connection")

    rows = conn.execute(
        """SELECT c.vm_factors, f.positive
           FROM feedback_log f JOIN chunk c ON f.chunk_id = c.chunk_id
           WHERE f.user_id = ?""",
        (user_id,),
    ).fetchall()
    if len(rows) < 10:
        return DEFAULT_WEIGHTS.copy()

    def loss(w_array):
        pos_scores = []
        neg_scores = []
        for vm_factors_json, positive in rows:
            factors = json.loads(vm_factors_json)
            vec = np.array([factors[k] for k in FACTOR_KEYS])
            score = 1.0 / (1.0 + math.exp(-np.dot(w_array, vec)))
            if positive:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
        mean_pos = np.mean(pos_scores) if pos_scores else 0.5
        mean_neg = np.mean(neg_scores) if neg_scores else 0.5
        return -(mean_pos - mean_neg)

    try:
        from scipy.optimize import minimize
    except Exception:
        return DEFAULT_WEIGHTS.copy()

    x0 = np.array([DEFAULT_WEIGHTS[k] for k in FACTOR_KEYS])
    result = minimize(loss, x0, method="Nelder-Mead",
                      bounds=[(0.01, 0.99)] * len(FACTOR_KEYS))
    learned = dict(zip(FACTOR_KEYS, result.x.tolist(), strict=True))
    s = sum(learned.values())
    if s == 0:
        s = 1.0
    return {k: round(v / s, 4) for k, v in learned.items()}
