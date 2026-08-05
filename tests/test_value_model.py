from lumen.force.mnemonic.value_model import (
    DEFAULT_WEIGHTS,
    FACTOR_KEYS,
    compute_vm,
    extract_factors,
    learn_weights_from_feedback,
)


def test_compute_vm_returns_value_in_range():
    vm, factors = compute_vm(
        "I need to schedule a meeting tomorrow",
        user_weights={},
        source_type="user_input",
        user_goals=["meeting", "schedule"],
        user_values=["productive"],
    )
    assert 0.0 <= vm <= 1.0


def test_compute_vm_returns_all_factor_keys():
    vm, factors = compute_vm("hello world", {}, "user_input")
    for key in FACTOR_KEYS:
        assert key in factors
        assert isinstance(factors[key], float)
        assert 0.0 <= factors[key] <= 1.0


def test_extract_factors_user_input_reliability():
    factors = extract_factors("test content", "user_input", [], [])
    assert factors["reliability"] == 0.9


def test_extract_factors_agent_reasoning_reliability():
    factors = extract_factors("test content", "agent_reasoning", [], [])
    assert factors["reliability"] == 0.7


def test_compute_vm_default_weights_used():
    vm, factors = compute_vm(
        "schedule a meeting about budget review requirements",
        user_weights=None,
        source_type="user_input",
        user_goals=[],
        user_values=[],
    )
    assert 0.0 <= vm <= 1.0
    assert "goal_relevance" in factors


def test_compute_vm_with_custom_weights():
    custom_weights = dict.fromkeys(FACTOR_KEYS, 0.0)
    custom_weights["task_utility"] = 1.0
    vm, factors = compute_vm(
        "I must schedule a call",
        user_weights=custom_weights,
        source_type="user_input",
    )
    assert 0.0 <= vm <= 1.0


def test_extract_factors_goal_relevance_with_matching_goal():
    factors = extract_factors(
        "schedule meeting calendar agenda",
        "user_input",
        user_goals=["schedule meetings"],
        user_values=[],
    )
    assert factors["goal_relevance"] > 0.0


def test_extract_factors_no_goals_defaults():
    factors = extract_factors("content here", "user_input", [], [])
    assert factors["goal_relevance"] == 0.5
    assert factors["value_alignment"] == 0.5


def test_learn_weights_from_feedback_insufficient_data():
    import sqlite3

    from lumen.data.schema import init_db
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    result = learn_weights_from_feedback(conn)
    conn.close()
    assert result == DEFAULT_WEIGHTS


def test_learn_weights_from_feedback_wrong_type_raises():
    with __import__("pytest").raises(TypeError):
        learn_weights_from_feedback("not_a_connection")
