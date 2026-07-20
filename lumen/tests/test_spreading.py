"""Tests for spreading activation module (D6)."""

import pytest

from lumen.force.mnemonic.spreading import SPREAD_GAMMA, spread_activation


@pytest.fixture
def chain_graph():
    """Return a simple 3-node chain 0–1–2."""
    pytest.importorskip("networkx")
    import networkx as nx

    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(1, 2, weight=1.0)
    return g


@pytest.fixture
def star_graph():
    """Return a star graph with node 0 connected to 1, 2, 3."""
    pytest.importorskip("networkx")
    import networkx as nx

    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)
    g.add_edge(0, 2, weight=1.0)
    g.add_edge(0, 3, weight=1.0)
    return g


def test_spread_activation_basic(chain_graph):
    """Test basic spread on a toy NetworkX graph."""
    result = spread_activation(chain_graph, [0], a=1.0)

    assert result[0] == 1.0
    # Hop 1: 1.0 * 0.4^1 * 1.0 = 0.4 > threshold(0.1)
    assert result[1] == pytest.approx(1.0 * SPREAD_GAMMA)
    # Hop 2: 0.4 * 0.4^2 * 1.0 = 0.064 < threshold(0.1), so node 2 not activated
    assert 2 not in result


def test_higher_a_wider_spread(star_graph):
    """Test that higher `a` produces wider spread than lower `a`."""
    low = spread_activation(star_graph, [0], a=0.0)
    high = spread_activation(star_graph, [0], a=1.0)

    # a=0.0 → threshold=1.0, nothing spreads
    assert set(low.keys()) == {0}
    # a=1.0 → threshold=0.1, neighbors spread
    assert set(high.keys()) == {0, 1, 2, 3}


def test_empty_seeds():
    """Empty seeds returns empty dict."""
    pytest.importorskip("networkx")
    import networkx as nx

    g = nx.Graph()
    g.add_edge(0, 1)

    result = spread_activation(g, [], a=0.5)
    assert result == {}


def test_seed_not_in_graph():
    """Seed not in graph returns just the seed with activation 1.0."""
    pytest.importorskip("networkx")
    import networkx as nx

    g = nx.Graph()
    g.add_edge(0, 1)

    result = spread_activation(g, [42], a=0.5)
    assert result == {42: 1.0}


def test_graph_none():
    """graph=None returns seed activations without error."""
    result = spread_activation(None, [1, 2], a=0.5)
    assert result == {1: 1.0, 2: 1.0}


def test_multiple_seeds():
    """Multiple seeds both activate and can spread."""
    pytest.importorskip("networkx")
    import networkx as nx

    g = nx.Graph()
    g.add_edge(10, 20, weight=1.0)
    g.add_edge(20, 30, weight=1.0)

    result = spread_activation(g, [10, 30], a=1.0)

    assert result[10] == 1.0
    assert result[30] == 1.0
    # Node 20 reached from both seeds at hop 1
    assert result[20] == pytest.approx(SPREAD_GAMMA)
