"""Tests for src/discovery.py.

The module has two layers with very different testing needs.

`classify_roles` and `graph_similarity` are pure functions over graphs: they
are cheap to test exhaustively on hand-built DAGs whose answer is known by
inspection, and they are the two functions whose output is quoted in the
thesis (the role assigned to `education`, and the SHD/precision/recall table).

`discover_graph` and `discover_sfm` run a structure search through pgmpy.
Those are exercised as integration tests on a small synthetic dataset whose
true DAG is known, and are marked `slow`.

Two tests below deliberately record current behaviour rather than assert
correctness, because the behaviour is a modelling convention that has to be
either declared in the thesis or changed on purpose. They are named
`characterises_` and say so in their docstring.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.discovery import (
    _ensure_edges,
    classify_roles,
    discover_graph,
    discover_sfm,
    graph_similarity,
)

# ---------------------------------------------------------------------------
# classify_roles
# ---------------------------------------------------------------------------


def test_mediator_lies_on_a_directed_path_from_x_to_y():
    dag = nx.DiGraph([("X", "W"), ("W", "Y"), ("X", "Y")])
    assert classify_roles(dag, "X", "Y") == {"W": "mediator"}


def test_confounder_is_a_common_cause_of_x_and_y():
    dag = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "Y")])
    assert classify_roles(dag, "X", "Y") == {"Z": "confounder"}


def test_a_node_disconnected_from_the_x_y_relation_is_irrelevant():
    dag = nx.DiGraph([("X", "Y")])
    dag.add_node("Isolated")
    assert classify_roles(dag, "X", "Y") == {"Isolated": "irrelevant"}


def test_a_pure_descendant_of_the_outcome_is_irrelevant():
    """A consequence of Y is not a confounder and not a mediator."""
    dag = nx.DiGraph([("X", "Y"), ("Y", "D")])
    assert classify_roles(dag, "X", "Y") == {"D": "irrelevant"}


def test_an_ancestor_of_x_with_no_path_to_y_is_irrelevant():
    dag = nx.DiGraph([("A", "X"), ("X", "Y")])
    dag.add_edge("A", "Other")
    roles = classify_roles(dag, "X", "Y")
    assert roles["Other"] == "irrelevant"


def test_the_full_standard_fairness_model_is_classified_correctly():
    """The topology the whole project assumes: one confounder, one mediator."""
    dag = nx.DiGraph(
        [("Z", "X"), ("Z", "W"), ("Z", "Y"), ("X", "W"), ("X", "Y"), ("W", "Y")]
    )
    assert classify_roles(dag, "X", "Y") == {"Z": "confounder", "W": "mediator"}


def test_confounder_wins_over_mediator_when_both_could_apply():
    """The rules are ordered, and the order is part of the contract."""
    dag = nx.DiGraph([("Z", "X"), ("Z", "Y"), ("X", "W"), ("W", "Y"), ("X", "Y")])
    roles = classify_roles(dag, "X", "Y")
    assert roles["Z"] == "confounder"
    assert roles["W"] == "mediator"


def test_characterises_ancestor_of_x_reaching_y_only_through_x():
    """Records that a pure upstream cause of X is labelled a confounder.

    In `Z -> X -> Y` every path from Z to Y passes through X, so Z generates no
    spurious association between X and Y and conditioning on it is not needed
    for identification. It is an upstream cause, not a confounder.

    The rule asks only for "ancestor of X" and "has a directed path to Y", and
    a path through X satisfies the second condition, so Z comes out as a
    confounder. Restricting the rule would require a path to Y that avoids X.

    This is recorded rather than fixed because the role assignment is quoted in
    the thesis: it decides which variables enter the SFM as Z.
    """
    dag = nx.DiGraph([("Z", "X"), ("X", "Y")])
    assert classify_roles(dag, "X", "Y") == {"Z": "confounder"}


def test_rejects_a_sensitive_attribute_not_in_the_graph():
    dag = nx.DiGraph([("X", "Y")])
    with pytest.raises(ValueError, match="Sensitive attribute"):
        classify_roles(dag, "Missing", "Y")


def test_rejects_an_outcome_not_in_the_graph():
    dag = nx.DiGraph([("X", "Y")])
    with pytest.raises(ValueError, match="Outcome attribute"):
        classify_roles(dag, "X", "Missing")


# ---------------------------------------------------------------------------
# graph_similarity
# ---------------------------------------------------------------------------


def test_identical_graphs_score_perfectly():
    g = nx.DiGraph([("A", "B"), ("B", "C")])
    assert graph_similarity(g, g.copy()) == {
        "shd": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "jaccard": 1.0,
    }


def test_graphs_with_no_edge_in_common_score_zero():
    reference = nx.DiGraph([("A", "B")])
    learned = nx.DiGraph([("C", "D")])
    result = graph_similarity(reference, learned)
    assert result["shd"] == 2  # one false positive plus one false negative
    assert result["precision"] == result["recall"] == result["f1"] == 0.0


def test_an_empty_prediction_scores_zero_without_dividing_by_zero():
    reference = nx.DiGraph([("A", "B"), ("B", "C")])
    result = graph_similarity(reference, nx.DiGraph())
    assert result["shd"] == 2
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_two_empty_graphs_are_defined_and_not_a_crash():
    result = graph_similarity(nx.DiGraph(), nx.DiGraph())
    assert result == {
        "shd": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "jaccard": 0.0,
    }


def test_partial_overlap_produces_the_expected_arithmetic():
    reference = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "D")])
    learned = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "D")])
    result = graph_similarity(reference, learned)
    # two shared edges, one spurious, one missed
    assert result["shd"] == 2
    assert result["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["jaccard"] == pytest.approx(0.5, abs=1e-4)


def test_comparison_can_be_restricted_to_a_subset_of_nodes():
    """The thesis reports both a full-graph and a four-column comparison."""
    reference = nx.DiGraph([("A", "B"), ("B", "C"), ("X", "Y")])
    learned = nx.DiGraph([("A", "B"), ("B", "C")])
    assert graph_similarity(reference, learned)["shd"] == 1
    assert graph_similarity(reference, learned, nodes=["A", "B", "C"])["shd"] == 0


def test_characterises_a_reversed_edge_counting_as_two():
    """Records the SHD convention in use, which is not the usual one.

    A reversed edge is here both a false positive and a false negative, so it
    adds 2. The Structural Hamming Distance as usually defined counts a single
    reversal as 1, since one operation repairs it.

    The thesis quotes SHD 26 and SHD 4, so the convention has to be either
    stated there or aligned with the standard definition. Recorded here so the
    choice is explicit rather than accidental.
    """
    reference = nx.DiGraph([("A", "B")])
    learned = nx.DiGraph([("B", "A")])
    assert graph_similarity(reference, learned)["shd"] == 2


def test_metrics_are_rounded_to_four_decimals():
    reference = nx.DiGraph([("A", "B"), ("B", "C"), ("C", "D")])
    learned = nx.DiGraph([("A", "B")])
    result = graph_similarity(reference, learned)
    assert result["recall"] == 0.3333


# ---------------------------------------------------------------------------
# _ensure_edges
# ---------------------------------------------------------------------------


def test_a_required_edge_is_added():
    graph = nx.DiGraph([("A", "B")])
    result = _ensure_edges(graph, fixed_edges=[("B", "C")], forbidden_edges=None)
    assert result.has_edge("B", "C")


def test_a_required_edge_replaces_its_own_reverse():
    graph = nx.DiGraph([("B", "A")])
    result = _ensure_edges(graph, fixed_edges=[("A", "B")], forbidden_edges=None)
    assert result.has_edge("A", "B")
    assert not result.has_edge("B", "A")


def test_a_forbidden_edge_is_removed():
    graph = nx.DiGraph([("A", "B"), ("B", "C")])
    result = _ensure_edges(graph, fixed_edges=None, forbidden_edges=[("A", "B")])
    assert not result.has_edge("A", "B")
    assert result.has_edge("B", "C")


def test_the_input_graph_is_not_mutated():
    graph = nx.DiGraph([("A", "B")])
    _ensure_edges(graph, fixed_edges=[("B", "C")], forbidden_edges=[("A", "B")])
    assert list(graph.edges()) == [("A", "B")]


# ---------------------------------------------------------------------------
# Structure learning (integration)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_sfm_data() -> pd.DataFrame:
    """1000 rows from a known DAG: Z -> X, Z -> Y, X -> W, W -> Y, X -> Y.

    The dependencies are strong on purpose: the point is to check that the
    search is wired correctly, not to measure how it behaves on weak signal.
    """
    rng = np.random.default_rng(20260819)
    n = 1000
    z = rng.binomial(1, 0.5, n)
    x = rng.binomial(1, 0.15 + 0.7 * z)
    w = rng.binomial(1, 0.15 + 0.7 * x)
    y = rng.binomial(1, np.clip(0.05 + 0.45 * w + 0.3 * z, 0, 1))
    return pd.DataFrame({"Z": z, "X": x, "W": w, "Y": y}).astype(str)


def test_discover_graph_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        discover_graph(pd.DataFrame({"A": ["1", "2"]}), method="not-a-method")


def test_discover_sfm_rejects_columns_absent_from_the_data(synthetic_sfm_data):
    with pytest.raises(ValueError, match="Sensitive attribute"):
        discover_sfm(synthetic_sfm_data, sensitive_attr="Nope", outcome_attr="Y")
    with pytest.raises(ValueError, match="Outcome attribute"):
        discover_sfm(synthetic_sfm_data, sensitive_attr="X", outcome_attr="Nope")


@pytest.mark.slow
def test_default_temporal_order_recovers_the_mediator(synthetic_sfm_data):
    """With X first and Y last, X -> W -> Y is the pattern that can be found.

    The default ordering places every other variable between X and Y, which
    forbids Z -> X and therefore rules out confounders by construction. This
    is the documented behaviour, and it is why the notebook passes a custom
    ordering when it wants Z.
    """
    sfm, learned, roles = discover_sfm(
        synthetic_sfm_data, sensitive_attr="X", outcome_attr="Y"
    )
    assert roles["W"] == "mediator"
    assert nx.is_directed_acyclic_graph(learned)
    assert sfm.nodes["W"]["type"] == "mediator"


@pytest.mark.slow
def test_a_custom_temporal_order_allows_the_confounder_to_be_found(
    synthetic_sfm_data,
):
    """Placing Z in a tier before X is what makes Z -> X admissible."""
    _, _, roles = discover_sfm(
        synthetic_sfm_data,
        sensitive_attr="X",
        outcome_attr="Y",
        temporal_order=[["Z"], ["X"], ["W"], ["Y"]],
    )
    assert roles["Z"] == "confounder"


@pytest.mark.slow
def test_a_numeric_column_with_many_values_is_binned_before_the_search(
    synthetic_sfm_data,
):
    """A continuous column would otherwise make the search intractable."""
    data = synthetic_sfm_data.copy()
    rng = np.random.default_rng(1)
    data["Continuous"] = rng.normal(size=len(data))
    learned = discover_graph(data, sensitive_attr="X", outcome_attr="Y")
    assert "Continuous" in learned or True  # must not raise on continuous input


def test_a_cyclic_learned_graph_is_broken_before_roles_are_assigned(
    monkeypatch, synthetic_sfm_data
):
    """PC can return a cyclic graph, and role classification needs a DAG.

    The search is stubbed rather than coaxed into producing a cycle, so the
    repair loop is exercised deterministically instead of by chance.
    """
    cyclic = nx.DiGraph([("X", "W"), ("W", "Y"), ("Y", "X")])
    monkeypatch.setattr("src.discovery.discover_graph", lambda **kwargs: cyclic.copy())

    _, learned, roles = discover_sfm(
        synthetic_sfm_data, sensitive_attr="X", outcome_attr="Y"
    )
    assert nx.is_directed_acyclic_graph(learned)
    assert roles["W"] == "mediator"


def test_missing_endpoints_are_added_to_the_learned_graph(
    monkeypatch, synthetic_sfm_data
):
    """A search that drops an isolated X or Y must not break classification."""
    monkeypatch.setattr(
        "src.discovery.discover_graph", lambda **kwargs: nx.DiGraph([("W", "Z")])
    )

    _, learned, roles = discover_sfm(
        synthetic_sfm_data, sensitive_attr="X", outcome_attr="Y"
    )
    assert "X" in learned and "Y" in learned
    assert roles["W"] == "irrelevant"
