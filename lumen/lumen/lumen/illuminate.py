"""C7: Palace Construction Pipeline (Onboarding -> Blueprint).

Input wire: spaCy, Rich CLI wizard, SQLite schema
Output wire: A1 (room/locus creation)
Secret sauce: Cognitive mapping from user research to palace topology
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import TYPE_CHECKING, Any

try:
    from rich.prompt import Confirm, Prompt
except Exception:
    Confirm = None  # type: ignore[misc,assignment]
    Prompt = None  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    Prompt = Any  # type: ignore[misc,assignment]

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)

_NLP_CACHE = None


def _load_nlp():
    """Lazy-load spaCy model with caching."""
    global _NLP_CACHE
    if _NLP_CACHE is not None:
        return _NLP_CACHE
    try:
        import spacy

        _NLP_CACHE = spacy.load("en_core_web_sm")
        return _NLP_CACHE
    except Exception as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required for onboarding. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exc


def run_onboarding_wizard(
    conn: sqlite3.Connection,
    nlp=None,
    console=None,
) -> int:
    """Run the interactive palace onboarding wizard.

    Args:
        conn: SQLite connection.
        nlp: Optional spaCy pipeline. If None, loads en_core_web_sm.
        console: Optional Rich Console. If None, instantiates one.

    Returns:
        Number of rooms created.
    """
    if nlp is None:
        nlp = _load_nlp()

    if console is None:
        from rich.console import Console

        console = Console()

    if Prompt is None or Confirm is None:
        raise RuntimeError("Rich is required for the CLI wizard")

    console.print("[bold #3D5A80]Welcome to Lumen. Let's build your memory palace.[/]")

    desc = Prompt.ask("Describe what you do in 2-3 sentences")
    doc = nlp(desc)

    noun_chunks = [nc.text.lower() for nc in doc.noun_chunks if len(nc.text) > 3]
    top_domains = [item for item, _ in Counter(noun_chunks).most_common(7)]

    console.print(f"\n[#E8A838]I detected these domains in your work:[/] {', '.join(top_domains)}")
    confirmed = []
    for domain in top_domains:
        if Confirm.ask(f"Is '{domain}' a major area of your work?", default=True):
            confirmed.append(domain)

    rankings = _pairwise_rank(confirmed)

    for rank, domain in enumerate(rankings, 1):
        cur = conn.execute(
            "INSERT INTO room(name, room_type, topological_order) VALUES (?,?,?)",
            (domain, "domain", float(rank)),
        )
        room_id = cur.lastrowid
        seed_loci = _extract_sub_entities(doc, domain)
        for locus in seed_loci[:5]:
            conn.execute("INSERT INTO locus(room_id, name) VALUES (?,?)", (room_id, locus))

    conn.commit()
    console.print(f"\n[bold #2D8A5E]Palace blueprint created.[/] Rooms: {len(rankings)}")
    if logger:
        logger.info("onboarding_complete", rooms=len(rankings))
    return len(rankings)


def _pairwise_rank(items: list[str], ask_fn=None) -> list[str]:
    """Simple bubble sort by user preference — 5 min max for <=7 items.

    Args:
        items: List of domain strings to rank.
        ask_fn: Optional callable(prompt) -> str for testing. If None,
            uses Rich Prompt.ask.

    Returns:
        Ranked list, capped at 7 items.
    """
    if ask_fn is None:
        ask_fn = Prompt.ask

    items = items[:7]
    if len(items) <= 1:
        return items

    arr = items[:]
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            prompt_text = f"Which is more important? 1) {arr[i]}  2) {arr[j]}"
            if ask_fn is Prompt.ask:
                pref = ask_fn(prompt_text, choices=["1", "2"], default="1")
            else:
                pref = ask_fn(prompt_text)
                if pref not in ("1", "2"):
                    pref = "1"
            if pref == "2":
                arr[i], arr[j] = arr[j], arr[i]
    return arr


def _extract_sub_entities(doc, domain: str) -> list[str]:
    """Heuristic: tokens that appear near the domain noun in the description."""
    loci: set[str] = set()
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if domain in sent_text:
            # Handle both spaCy Span (has .ents) and test mocks (may have _ents)
            sent_ents = getattr(sent, "ents", None) or getattr(sent, "_ents", [])
            if not sent_ents:
                # Fallback: pull doc-level entities that appear in this sentence
                doc_ents = getattr(doc, "ents", None) or getattr(doc, "_ents", [])
                sent_ents = [e for e in doc_ents if e.text in sent.text]
            for ent in sent_ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"}:
                    loci.add(ent.text)
            for token in sent:
                if token.pos_ == "NOUN" and token.text != domain:
                    loci.add(token.lemma_)
    return list(loci)[:5]
