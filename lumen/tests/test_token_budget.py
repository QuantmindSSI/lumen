"""Tests for lumen.force.contextual.token_budget."""


from lumen.force.contextual.token_budget import (
    CharHeuristicCounter,
    get_token_counter,
)


class TestCharHeuristicCounter:
    def test_basic_count(self):
        counter = CharHeuristicCounter()
        # "hello world" = 11 chars -> ~2-3 tokens
        assert counter.count("hello world") == 2

    def test_empty_string(self):
        counter = CharHeuristicCounter()
        assert counter.count("") == 1

    def test_long_text(self):
        counter = CharHeuristicCounter()
        text = "a" * 400
        assert counter.count(text) == 100


class TestTransformersTokenizerCounter:
    def test_real_tokenizer(self, tmp_path):
        # We can't easily ship a real tokenizer in tests, so we test the
        # fallback path when the model_path does not exist.
        counter = get_token_counter(tmp_path / "nonexistent")
        assert isinstance(counter, CharHeuristicCounter)

    def test_get_token_counter_with_none(self):
        counter = get_token_counter(None)
        assert isinstance(counter, CharHeuristicCounter)


class TestTokenBudgetIntegration:
    def test_budget_enforcement(self):
        counter = CharHeuristicCounter()
        text = "x" * 400  # 100 tokens
        assert counter.count(text) == 100
        assert counter.count(text * 2) == 200
