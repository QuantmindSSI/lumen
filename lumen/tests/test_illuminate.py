"""Tests for C7: Palace Construction Pipeline."""

import sqlite3

import pytest
from rich.console import Console

from lumen.data.schema import init_db


class MockChunk:
    def __init__(self, text: str):
        self.text = text


class MockEnt:
    def __init__(self, text: str, label_: str):
        self.text = text
        self.label_ = label_


class MockToken:
    def __init__(self, text: str, pos_: str, lemma_: str | None = None):
        self.text = text
        self.pos_ = pos_
        self.lemma_ = lemma_ if lemma_ is not None else text


class MockSent:
    def __init__(
        self,
        text: str,
        tokens: list[MockToken],
        ents: list[MockEnt] | None = None,
    ):
        self.text = text
        self._tokens = tokens
        self._ents = ents or []

    def __iter__(self):
        return iter(self._tokens)


class MockDoc:
    def __init__(
        self,
        noun_chunks: list[MockChunk] | None = None,
        sents: list[MockSent] | None = None,
        ents: list[MockEnt] | None = None,
    ):
        self._noun_chunks = noun_chunks or []
        self._sents = sents or []
        self._ents = ents or []

    @property
    def noun_chunks(self):
        return self._noun_chunks

    @property
    def sents(self):
        return self._sents

    @property
    def ents(self):
        return self._ents


def test_extract_sub_entities():
    from lumen.lumen.illuminate import _extract_sub_entities

    t1 = MockToken("software", "NOUN", "software")
    t2 = MockToken("engineering", "NOUN", "engineering")
    t3 = MockToken("Google", "PROPN", "Google")
    sent = MockSent("I do software engineering at Google.", [t1, t2, t3])
    ent = MockEnt("Google", "ORG")
    doc = MockDoc(
        noun_chunks=[MockChunk("software engineering")],
        sents=[sent],
        ents=[ent],
    )

    result = _extract_sub_entities(doc, "software engineering")
    assert "Google" in result
    assert any(token in result for token in ("software", "engineering"))


def test_pairwise_rank_prefers_first():
    from lumen.lumen.illuminate import _pairwise_rank

    result = _pairwise_rank(["a", "b", "c"], ask_fn=lambda prompt: "1")
    assert result == ["a", "b", "c"]


def test_pairwise_rank_prefers_second():
    from lumen.lumen.illuminate import _pairwise_rank

    result = _pairwise_rank(["a", "b", "c"], ask_fn=lambda prompt: "2")
    assert result == ["c", "b", "a"]


def test_pairwise_rank_caps_at_seven():
    from lumen.lumen.illuminate import _pairwise_rank

    result = _pairwise_rank(list("abcdefgh"), ask_fn=lambda prompt: "1")
    assert len(result) == 7


def test_run_onboarding_wizard(monkeypatch):
    import lumen.lumen.illuminate as illum
    from lumen.lumen.illuminate import run_onboarding_wizard

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)

    def fake_prompt(prompt, choices=None, default=None):
        if choices is not None:
            return "1"
        return "I do software engineering and machine learning."

    monkeypatch.setattr(illum.Prompt, "ask", fake_prompt)
    monkeypatch.setattr(illum.Confirm, "ask", lambda prompt, default=True: True)

    chunk1 = MockChunk("software engineering")
    chunk2 = MockChunk("machine learning")
    chunk3 = MockChunk("natural language processing")
    t1 = MockToken("software", "NOUN", "software")
    t2 = MockToken("engineering", "NOUN", "engineering")
    t3 = MockToken("machine", "NOUN", "machine")
    t4 = MockToken("learning", "NOUN", "learning")
    t5 = MockToken("Alice", "PROPN", "Alice")
    sent = MockSent(
        "I do software engineering and machine learning with Alice.",
        [t1, t2, t3, t4, t5],
    )
    ent = MockEnt("Alice", "PERSON")
    doc = MockDoc(
        noun_chunks=[chunk1, chunk2, chunk3, chunk1, chunk2],
        sents=[sent],
        ents=[ent],
    )

    def mock_nlp(text):
        return doc

    console = Console()
    run_onboarding_wizard(conn, nlp=mock_nlp, console=console)

    rooms = conn.execute("SELECT name, room_type, topological_order FROM room").fetchall()
    assert len(rooms) == 3
    assert all(r["room_type"] == "domain" for r in rooms)

    loci = conn.execute("SELECT name FROM locus").fetchall()
    assert len(loci) >= 1
    names = {row["name"] for row in loci}
    assert "Alice" in names

    conn.close()


def test_load_nlp_graceful_failure():
    import lumen.lumen.illuminate as illum

    illum._NLP_CACHE = None
    with pytest.raises(RuntimeError, match="en_core_web_sm"):
        illum._load_nlp()
