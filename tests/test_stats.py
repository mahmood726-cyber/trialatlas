"""Stats engine test suite -- 15 tests for TrialAtlas network statistics."""

import json
import math
import os
import sys
import pytest
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stats_engine import (
    betweenness_centrality,
    eigenvector_centrality,
    degree_distribution,
    modularity_significance,
    morans_i,
    stochastic_block_model,
    ergm_pseudolikelihood,
    degree_assortativity,
    attribute_assortativity,
    lorenz_curve,
    atkinson_index,
    spectral_gap,
    motif_census,
    link_prediction,
    core_periphery,
    network_robustness,
)


# ──────────────────────────────────────────────────────────────────────
# Fixture: sample_network (loads sample_network.json → list of trials)
# ──────────────────────────────────────────────────────────────────────

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fixtures")


@pytest.fixture
def sample_network():
    """Load sample trial data from fixture."""
    path = os.path.join(FIXTURES_DIR, "sample_network.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────
# Betweenness Centrality (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_betweenness_star():
    """Star graph center has highest betweenness."""
    graph = {
        "nodes": {f"n{i}": {"type": "site"} for i in range(5)},
        "edges": [
            {"source": "n0", "target": f"n{i}", "weight": 1}
            for i in range(1, 5)
        ],
    }
    graph["nodes"]["n0"]["type"] = "sponsor"
    bc = betweenness_centrality(graph)
    assert bc["n0"] == max(bc.values())


def test_betweenness_values():
    """All betweenness values in [0, 1] after normalization."""
    graph = {
        "nodes": {f"n{i}": {"type": "site"} for i in range(4)},
        "edges": [
            {"source": "n0", "target": "n1", "weight": 1},
            {"source": "n1", "target": "n2", "weight": 1},
            {"source": "n2", "target": "n3", "weight": 1},
        ],
    }
    bc = betweenness_centrality(graph)
    assert all(0 <= v <= 1 for v in bc.values())


# ──────────────────────────────────────────────────────────────────────
# Eigenvector Centrality (1 test)
# ──────────────────────────────────────────────────────────────────────

def test_eigenvector_connected():
    """Eigenvector centrality returns values for all nodes in a cycle."""
    graph = {
        "nodes": {"a": {}, "b": {}, "c": {}, "d": {}},
        "edges": [
            {"source": "a", "target": "b", "weight": 1},
            {"source": "b", "target": "c", "weight": 1},
            {"source": "c", "target": "d", "weight": 1},
            {"source": "a", "target": "d", "weight": 1},
        ],
    }
    ec = eigenvector_centrality(graph)
    assert len(ec) == 4
    assert all(v >= 0 for v in ec.values())


# ──────────────────────────────────────────────────────────────────────
# Degree Distribution (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_degree_dist_counts():
    """Degree distribution returns non-empty degree list."""
    graph = {
        "nodes": {f"n{i}": {} for i in range(6)},
        "edges": [
            {"source": "n0", "target": "n1", "weight": 1},
            {"source": "n0", "target": "n2", "weight": 1},
            {"source": "n0", "target": "n3", "weight": 1},
            {"source": "n1", "target": "n4", "weight": 1},
            {"source": "n2", "target": "n5", "weight": 1},
        ],
    }
    result = degree_distribution(graph)
    assert len(result["degrees"]) > 0


def test_degree_alpha_positive():
    """Power-law alpha is > 1 for a star graph."""
    graph = {
        "nodes": {f"n{i}": {} for i in range(20)},
        "edges": [
            {"source": "n0", "target": f"n{i}", "weight": 1}
            for i in range(1, 20)
        ],
    }
    result = degree_distribution(graph)
    assert result["power_law_alpha"] > 1


# ──────────────────────────────────────────────────────────────────────
# Modularity Significance (1 test)
# ──────────────────────────────────────────────────────────────────────

def test_modularity_sig(sample_network):
    """Modularity significance returns valid z_score and p_value."""
    from src.network_builder import build_sponsor_site_network
    graph = build_sponsor_site_network(sample_network)
    result = modularity_significance(graph, n_permutations=100, seed=42)
    assert "z_score" in result
    assert "p_value" in result
    assert 0 <= result["p_value"] <= 1


# ──────────────────────────────────────────────────────────────────────
# Moran's I (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_morans_i_clustered():
    """Positive Moran's I for spatially clustered values."""
    values = [10, 10, 10, 1, 1, 1]
    W = [
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    result = morans_i(values, W)
    assert result["I"] > 0  # positive spatial autocorrelation


def test_morans_i_random():
    """Random values should not show strong spatial autocorrelation."""
    np.random.seed(42)
    values = list(np.random.randn(10))
    W = [[1 if abs(i - j) == 1 else 0 for j in range(10)] for i in range(10)]
    result = morans_i(values, W)
    assert abs(result["z_score"]) < 3  # not significant


# ──────────────────────────────────────────────────────────────────────
# Stochastic Block Model (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_sbm_finds_blocks():
    """SBM identifies at least 2 blocks in a graph with 2 clear cliques."""
    graph = {
        "nodes": {**{f"a{i}": {} for i in range(5)},
                  **{f"b{i}": {} for i in range(5)}},
        "edges": [],
    }
    for i in range(5):
        for j in range(i + 1, 5):
            graph["edges"].append(
                {"source": f"a{i}", "target": f"a{j}", "weight": 1}
            )
            graph["edges"].append(
                {"source": f"b{i}", "target": f"b{j}", "weight": 1}
            )
    result = stochastic_block_model(graph, max_k=3, seed=42)
    assert result["k_best"] >= 2


def test_sbm_soft_assignments():
    """Soft assignments sum to 1 for each node."""
    graph = {
        "nodes": {"a": {}, "b": {}, "c": {}},
        "edges": [{"source": "a", "target": "b", "weight": 1}],
    }
    result = stochastic_block_model(graph, max_k=2, seed=42)
    for node, probs in result["assignments"].items():
        assert abs(sum(probs.values()) - 1.0) < 0.01


# ──────────────────────────────────────────────────────────────────────
# Assortativity (1 test)
# ──────────────────────────────────────────────────────────────────────

def test_degree_assortativity_range():
    """Degree assortativity is in [-1, 1]."""
    graph = {
        "nodes": {f"n{i}": {} for i in range(6)},
        "edges": [
            {"source": "n0", "target": "n1", "weight": 1},
            {"source": "n0", "target": "n2", "weight": 1},
            {"source": "n3", "target": "n4", "weight": 1},
            {"source": "n3", "target": "n5", "weight": 1},
        ],
    }
    r = degree_assortativity(graph)
    assert -1 <= r <= 1


# ──────────────────────────────────────────────────────────────────────
# ERGM (1 test)
# ──────────────────────────────────────────────────────────────────────

def test_ergm_coefficients(sample_network):
    """ERGM pseudo-likelihood returns coefficients with valid structure."""
    from src.network_builder import build_sponsor_site_network
    graph = build_sponsor_site_network(sample_network)
    result = ergm_pseudolikelihood(graph)
    assert len(result["coefficients"]) > 0
    for coef in result["coefficients"]:
        assert "name" in coef
        assert "coef" in coef
        assert "se" in coef
        assert "p_value" in coef
        assert 0 <= coef["p_value"] <= 1
    assert 0 <= result["pseudo_r2"] <= 1


# ──────────────────────────────────────────────────────────────────────
# Attribute Assortativity (1 test)
# ──────────────────────────────────────────────────────────────────────

def test_attribute_assortativity_type(sample_network):
    """Attribute assortativity by node type is in [-1, 1]."""
    from src.network_builder import build_sponsor_site_network
    graph = build_sponsor_site_network(sample_network)
    r = attribute_assortativity(graph, "type")
    assert -1 <= r <= 1
    # Bipartite graph: all edges cross types, so assortativity should be negative
    assert r < 0


# ──────────────────────────────────────────────────────────────────────
# Lorenz + Atkinson (1 test)
# ──────────────────────────────────────────────────────────────────────

def test_lorenz_atkinson():
    """Lorenz curve and Atkinson index for highly unequal distribution."""
    values = [1, 1, 1, 100]
    lc = lorenz_curve(values)
    assert len(lc["percentiles"]) == len(lc["cumulative_shares"])
    assert lc["gini"] > 0.5  # very unequal
    ai = atkinson_index(values, epsilon=0.5)
    assert 0 < ai < 1


# ──────────────────────────────────────────────────────────────────────
# Integration (1 test)
# ──────────────────────────────────────────────────────────────────────

def test_full_stats_pipeline(sample_network):
    """Full stats pipeline runs on real fixture data."""
    from src.network_builder import build_sponsor_site_network
    graph = build_sponsor_site_network(sample_network)
    bc = betweenness_centrality(graph)
    ec = eigenvector_centrality(graph)
    dd = degree_distribution(graph)
    lc = lorenz_curve(list(bc.values()))
    assert len(bc) > 0 and len(ec) > 0
    assert "power_law_alpha" in dd
    assert len(lc["percentiles"]) > 0


# ──────────────────────────────────────────────────────────────────────
# Spectral Gap (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_spectral_gap_simple():
    """Two-clique graph should have gap at k=2."""
    graph = {"nodes": {f"a{i}": {} for i in range(4)} | {f"b{i}": {} for i in range(4)},
             "edges": []}
    for i in range(4):
        for j in range(i + 1, 4):
            graph["edges"].append({"source": f"a{i}", "target": f"a{j}", "weight": 1})
            graph["edges"].append({"source": f"b{i}", "target": f"b{j}", "weight": 1})
    # One bridge edge
    graph["edges"].append({"source": "a0", "target": "b0", "weight": 1})
    result = spectral_gap(graph)
    assert result["suggested_k"] == 2
    assert result["fiedler_value"] > 0  # connected


def test_spectral_eigenvalues():
    """Cycle graph eigenvalues: first eigenvalue is 0."""
    graph = {"nodes": {f"n{i}": {} for i in range(5)},
             "edges": [{"source": f"n{i}", "target": f"n{(i+1)%5}", "weight": 1} for i in range(5)]}
    result = spectral_gap(graph)
    assert abs(result["eigenvalues"][0]) < 0.01  # first eigenvalue is 0


# ──────────────────────────────────────────────────────────────────────
# Motif Census (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_motif_census_bipartite(sample_network):
    """Motif census runs on real bipartite network."""
    from src.network_builder import build_sponsor_site_network
    graph = build_sponsor_site_network(sample_network)
    result = motif_census(graph)
    assert "triads" in result
    assert result["total_triads"] >= 0


def test_motif_significance():
    """Motif census computes significance for small graph."""
    graph = {"nodes": {f"n{i}": {} for i in range(6)},
             "edges": [{"source": "n0", "target": "n1", "weight": 1},
                       {"source": "n0", "target": "n2", "weight": 1},
                       {"source": "n1", "target": "n2", "weight": 1},
                       {"source": "n3", "target": "n4", "weight": 1},
                       {"source": "n4", "target": "n5", "weight": 1}]}
    result = motif_census(graph)
    assert "triad_significance" in result


# ──────────────────────────────────────────────────────────────────────
# Link Prediction (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_link_prediction_auc():
    """Link prediction AUC in valid range for ring+skip graph."""
    graph = {"nodes": {f"n{i}": {} for i in range(10)},
             "edges": [{"source": f"n{i}", "target": f"n{(i+1)%10}", "weight": 1} for i in range(10)] +
                      [{"source": f"n{i}", "target": f"n{(i+2)%10}", "weight": 1} for i in range(10)]}
    result = link_prediction(graph, test_fraction=0.2, seed=42)
    assert 0 <= result["auc_common_neighbors"] <= 1
    assert len(result["top_predicted"]) > 0


def test_link_prediction_scores():
    """Link prediction generates predictions for small graph."""
    graph = {"nodes": {"a": {}, "b": {}, "c": {}, "d": {}, "e": {}},
             "edges": [{"source": "a", "target": "b", "weight": 1},
                       {"source": "b", "target": "c", "weight": 1},
                       {"source": "a", "target": "c", "weight": 1},
                       {"source": "c", "target": "d", "weight": 1},
                       {"source": "d", "target": "e", "weight": 1}]}
    result = link_prediction(graph, test_fraction=0.0, seed=42)
    # Non-edges between distance-2 nodes should be predicted
    assert len(result["predictions"]) > 0


# ──────────────────────────────────────────────────────────────────────
# Core-Periphery (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_core_periphery_detection():
    """Dense core + sparse periphery correctly identified."""
    graph = {"nodes": {f"c{i}": {} for i in range(4)} | {f"p{i}": {} for i in range(4)},
             "edges": []}
    # Core is fully connected
    for i in range(4):
        for j in range(i + 1, 4):
            graph["edges"].append({"source": f"c{i}", "target": f"c{j}", "weight": 1})
    # Periphery connects to one core node each
    for i in range(4):
        graph["edges"].append({"source": f"p{i}", "target": f"c{i}", "weight": 1})
    result = core_periphery(graph, seed=42)
    assert len(result["core_nodes"]) > 0
    assert len(result["periphery_nodes"]) > 0
    assert result["correlation"] > 0


def test_core_periphery_quality():
    """Core-periphery correlation in valid range for small graph."""
    graph = {"nodes": {f"n{i}": {} for i in range(6)},
             "edges": [{"source": "n0", "target": "n1", "weight": 1},
                       {"source": "n0", "target": "n2", "weight": 1},
                       {"source": "n1", "target": "n2", "weight": 1}]}
    result = core_periphery(graph, seed=42)
    assert -1 <= result["correlation"] <= 1


# ──────────────────────────────────────────────────────────────────────
# Network Robustness (2 tests)
# ──────────────────────────────────────────────────────────────────────

def test_robustness_targeted():
    """Complete graph should be robust under targeted attack."""
    graph = {"nodes": {f"n{i}": {} for i in range(8)},
             "edges": []}
    for i in range(8):
        for j in range(i + 1, 8):
            graph["edges"].append({"source": f"n{i}", "target": f"n{j}", "weight": 1})
    result = network_robustness(graph, strategy='targeted')
    assert result["robustness_index"] > 0.3  # robust graph
    assert len(result["removal_fractions"]) == len(result["giant_component_sizes"])


def test_robustness_fragile():
    """Star graph is fragile — center removal kills connectivity."""
    graph = {"nodes": {f"n{i}": {} for i in range(6)},
             "edges": [{"source": "n0", "target": f"n{i}", "weight": 1} for i in range(1, 6)]}
    result = network_robustness(graph, strategy='targeted')
    assert result["critical_threshold"] <= 0.3  # fragile — center removal kills it
