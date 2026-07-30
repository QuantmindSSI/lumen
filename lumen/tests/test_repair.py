"""D12: Tests for Search Repair Loop."""

from lumen.lumen.controller import TFCState, TwinForceController
from lumen.lumen.fusion import RetrievedChunk
from lumen.lumen.repair import SearchRepair
from lumen.lumen.search import SearchPipeline


class FakePipeline:
    def __init__(self, results_sequence):
        self.results_sequence = results_sequence
        self.call_count = 0
        self.kwargs_log = []

    def execute(self, query, goal_tree_keywords=None, k=20, max_repair_attempts=1):
        self.call_count += 1
        self.kwargs_log.append(
            {
                "query": query,
                "goal_tree_keywords": goal_tree_keywords,
                "k": k,
                "max_repair_attempts": max_repair_attempts,
            }
        )
        if self.call_count <= len(self.results_sequence):
            return self.results_sequence[self.call_count - 1]
        return []


def _make_chunk(final_score=0.5):
    return RetrievedChunk(
        chunk_id=1,
        room_name="r",
        locus_name="l",
        content="hello",
        provenance_id=None,
        rrf_score=0.5,
        vm_score=0.5,
        frqad_score=0.5,
        recency_hours=1.0,
        final_score=final_score,
    )


def test_empty_results_widens_a():
    tfc = TwinForceController(TFCState(a=0.5, r=3))
    fake = FakePipeline([[_make_chunk()]])
    repair = SearchRepair(tfc, fake)
    results = repair.attempt_repair("q", "empty_results")
    assert len(results) == 1
    assert tfc.state.a == 0.8
    assert fake.kwargs_log[-1]["k"] == 50
    assert fake.kwargs_log[-1]["max_repair_attempts"] == 0


def test_empty_results_caps_a():
    tfc = TwinForceController(TFCState(a=0.9))
    fake = FakePipeline([[_make_chunk()]])
    repair = SearchRepair(tfc, fake)
    repair.attempt_repair("q", "empty_results")
    assert tfc.state.a == 1.0


def test_budget_exceeded_degrades_r():
    tfc = TwinForceController(TFCState(r=3))
    fake = FakePipeline([[_make_chunk()]])
    repair = SearchRepair(tfc, fake)
    repair.attempt_repair("q", "budget_exceeded")
    assert tfc.state.r == 2
    assert fake.kwargs_log[-1]["max_repair_attempts"] == 0


def test_budget_exceeded_floors_r():
    tfc = TwinForceController(TFCState(r=0))
    fake = FakePipeline([])
    repair = SearchRepair(tfc, fake)
    repair.attempt_repair("q", "budget_exceeded")
    assert tfc.state.r == 0


def test_low_confidence_increases_a():
    tfc = TwinForceController(TFCState(a=0.5))
    fake = FakePipeline([[_make_chunk()]])
    repair = SearchRepair(tfc, fake)
    repair.attempt_repair("q", "low_confidence")
    assert tfc.state.a == 0.65
    assert fake.kwargs_log[-1]["max_repair_attempts"] == 0


def test_max_attempts_enforced():
    tfc = TwinForceController(TFCState(a=0.5))
    fake = FakePipeline([[_make_chunk()], [_make_chunk()], [_make_chunk()]])
    repair = SearchRepair(tfc, fake)
    # First repair attempt
    results1 = repair.attempt_repair("q", "empty_results")
    assert len(results1) == 1
    assert fake.call_count == 1
    # Second repair attempt on same instance — allowed (MAX_ATTEMPTS == 2)
    results2 = repair.attempt_repair("q", "empty_results")
    assert len(results2) == 1
    assert fake.call_count == 2
    # Third repair attempt on same instance — should be blocked
    results3 = repair.attempt_repair("q", "empty_results")
    assert results3 == []
    assert fake.call_count == 2


def test_unknown_reason_returns_empty():
    tfc = TwinForceController(TFCState())
    fake = FakePipeline([])
    repair = SearchRepair(tfc, fake)
    assert repair.attempt_repair("q", "unknown") == []


def test_no_repair_when_results_good(monkeypatch, memory_db, test_config, mock_embedder):
    tfc = TwinForceController(TFCState(a=0.5))
    pipe = SearchPipeline(memory_db, test_config, tfc=tfc, embedder=mock_embedder)

    call_count = [0]

    def fake_fuse(*args, **kwargs):
        call_count[0] += 1
        return [_make_chunk(final_score=0.5)]

    monkeypatch.setattr("lumen.lumen.search.fuse_and_rerank", fake_fuse)

    results = pipe.execute("good query")
    assert len(results) == 1
    assert call_count[0] == 1  # only once, no repair triggered
    assert tfc.state.a == 0.5  # unchanged by repair


def test_repair_triggered_on_empty_results(monkeypatch, memory_db, test_config, mock_embedder):
    tfc = TwinForceController(TFCState(a=0.5))
    pipe = SearchPipeline(memory_db, test_config, tfc=tfc, embedder=mock_embedder)

    call_count = [0]

    def fake_fuse(*args, **kwargs):
        call_count[0] += 1
        return []

    monkeypatch.setattr("lumen.lumen.search.fuse_and_rerank", fake_fuse)

    results = pipe.execute("empty query", max_repair_attempts=1)
    assert results == []
    assert call_count[0] == 2  # original + repair
    assert tfc.state.a == 0.8  # repaired


def test_repair_triggered_on_low_confidence(monkeypatch, memory_db, test_config, mock_embedder):
    tfc = TwinForceController(TFCState(a=0.5))
    pipe = SearchPipeline(memory_db, test_config, tfc=tfc, embedder=mock_embedder)

    call_count = [0]

    def fake_fuse(*args, **kwargs):
        call_count[0] += 1
        return [_make_chunk(final_score=0.005)]

    monkeypatch.setattr("lumen.lumen.search.fuse_and_rerank", fake_fuse)

    pipe.execute("low confidence query", max_repair_attempts=1)
    assert call_count[0] == 2
    assert tfc.state.a == 0.65


def test_max_repair_attempts_zero(monkeypatch, memory_db, test_config, mock_embedder):
    tfc = TwinForceController(TFCState(a=0.5))
    pipe = SearchPipeline(memory_db, test_config, tfc=tfc, embedder=mock_embedder)

    call_count = [0]

    def fake_fuse(*args, **kwargs):
        call_count[0] += 1
        return []

    monkeypatch.setattr("lumen.lumen.search.fuse_and_rerank", fake_fuse)

    results = pipe.execute("empty query", max_repair_attempts=0)
    assert results == []
    assert call_count[0] == 1  # no repair attempt
    assert tfc.state.a == 0.5


def test_repair_returns_original_if_repair_also_empty(
    monkeypatch, memory_db, test_config, mock_embedder
):
    tfc = TwinForceController(TFCState(a=0.5))
    pipe = SearchPipeline(memory_db, test_config, tfc=tfc, embedder=mock_embedder)

    call_count = [0]

    def fake_fuse(*args, **kwargs):
        call_count[0] += 1
        return []

    monkeypatch.setattr("lumen.lumen.search.fuse_and_rerank", fake_fuse)

    results = pipe.execute("empty query", max_repair_attempts=1)
    assert results == []
    assert call_count[0] == 2
    # TFC was still modified by the repair attempt
    assert tfc.state.a == 0.8
