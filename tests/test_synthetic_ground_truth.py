"""Ground-truth stress tests for effects.py using exact synthetic CPTs.

Each scenario builds a Bayesian network from hand-specified CPDs whose true
TV/TE/SE/NDE/NIE decomposition is knowable by construction -- either exactly
zero (an edge is simply absent from the graph) or via closed-form probability
arithmetic on the same constants used to build the CPTs. There is no
sampling/estimation noise: these tests validate the identification formulas
in effects.py directly, independent of any single worked dataset example.
"""

import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from src.effects import (
    decompose_indirect_effect,
    decompose_spurious_effect,
    natural_direct_effect,
    natural_indirect_effect,
    spurious_effect,
    total_effect,
    total_variation,
)


def _mark_types(bn: DiscreteBayesianNetwork, **types: str) -> None:
    for node, node_type in types.items():
        bn.nodes[node]["type"] = node_type


def test_pure_direct_effect_no_mediator_no_confounder():
    """X -> Y only: TV = TE = NDE exactly; SE and NIE are exactly zero."""
    p_y1_x0, p_y1_x1 = 0.3, 0.7

    bn = DiscreteBayesianNetwork([("X", "Y")])
    _mark_types(bn, X="sensitive", Y="outcome")
    bn.add_cpds(
        TabularCPD("X", 2, [[0.5], [0.5]], state_names={"X": ["x0", "x1"]}),
        TabularCPD(
            "Y",
            2,
            [[1 - p_y1_x0, 1 - p_y1_x1], [p_y1_x0, p_y1_x1]],
            evidence=["X"],
            evidence_card=[2],
            state_names={"Y": ["y0", "y1"], "X": ["x0", "x1"]},
        ),
    )
    bn.check_model()

    target = ("Y", "y1")
    expected = p_y1_x1 - p_y1_x0

    assert total_variation(bn, target, "X", "x0", "x1") == pytest.approx(expected)
    assert total_effect(bn, target, "X", "x0", "x1") == pytest.approx(expected)
    assert natural_direct_effect(bn, target, "X", "x0", "x1") == pytest.approx(expected)
    assert natural_indirect_effect(bn, target, "X", "x0", "x1") == pytest.approx(
        0.0, abs=1e-9
    )
    assert natural_indirect_effect(bn, target, "X", "x1", "x0") == pytest.approx(
        0.0, abs=1e-9
    )
    assert spurious_effect(bn, target, "X", "x0") == pytest.approx(0.0, abs=1e-9)
    assert spurious_effect(bn, target, "X", "x1") == pytest.approx(0.0, abs=1e-9)


def test_pure_indirect_effect_via_single_mediator():
    """X -> W -> Y with no X->Y edge: NDE is exactly zero, TE=TV is fully mediated."""
    p_w1_x0, p_w1_x1 = 0.2, 0.8
    p_y1_w0, p_y1_w1 = 0.1, 0.9

    bn = DiscreteBayesianNetwork([("X", "W"), ("W", "Y")])
    _mark_types(bn, X="sensitive", Y="outcome", W="mediator")
    bn.add_cpds(
        TabularCPD("X", 2, [[0.5], [0.5]], state_names={"X": ["x0", "x1"]}),
        TabularCPD(
            "W",
            2,
            [[1 - p_w1_x0, 1 - p_w1_x1], [p_w1_x0, p_w1_x1]],
            evidence=["X"],
            evidence_card=[2],
            state_names={"W": ["w0", "w1"], "X": ["x0", "x1"]},
        ),
        TabularCPD(
            "Y",
            2,
            [[1 - p_y1_w0, 1 - p_y1_w1], [p_y1_w0, p_y1_w1]],
            evidence=["W"],
            evidence_card=[2],
            state_names={"Y": ["y0", "y1"], "W": ["w0", "w1"]},
        ),
    )
    bn.check_model()

    target = ("Y", "y1")
    p_y_x0 = p_y1_w0 * (1 - p_w1_x0) + p_y1_w1 * p_w1_x0
    p_y_x1 = p_y1_w0 * (1 - p_w1_x1) + p_y1_w1 * p_w1_x1
    expected_te = p_y_x1 - p_y_x0

    assert natural_direct_effect(bn, target, "X", "x0", "x1") == pytest.approx(
        0.0, abs=1e-9
    )
    assert total_effect(bn, target, "X", "x0", "x1") == pytest.approx(expected_te)
    assert total_variation(bn, target, "X", "x0", "x1") == pytest.approx(expected_te)
    assert natural_indirect_effect(bn, target, "X", "x0", "x1") == pytest.approx(
        expected_te
    )

    contributions = decompose_indirect_effect(bn, target, "X", "x0", "x1")
    assert contributions == pytest.approx({"W": expected_te})


def test_confounding_creates_apparent_disparity_with_zero_causal_effect():
    """Z -> X, Z -> Y, no X->Y edge: TE is exactly zero but TV is not (Simpson's paradox).

    This is the core case the framework exists to catch: a purely spurious
    disparity, with zero causal effect, that a naive observational comparison
    (TV alone) would mistake for discrimination.
    """
    p_z0 = 0.5
    p_x1_z0, p_x1_z1 = 0.2, 0.8
    p_y1_z0, p_y1_z1 = 0.1, 0.9

    bn = DiscreteBayesianNetwork([("Z", "X"), ("Z", "Y")])
    _mark_types(bn, X="sensitive", Y="outcome", Z="confounder")
    bn.add_cpds(
        TabularCPD("Z", 2, [[p_z0], [1 - p_z0]], state_names={"Z": ["z0", "z1"]}),
        TabularCPD(
            "X",
            2,
            [[1 - p_x1_z0, 1 - p_x1_z1], [p_x1_z0, p_x1_z1]],
            evidence=["Z"],
            evidence_card=[2],
            state_names={"X": ["x0", "x1"], "Z": ["z0", "z1"]},
        ),
        TabularCPD(
            "Y",
            2,
            [[1 - p_y1_z0, 1 - p_y1_z1], [p_y1_z0, p_y1_z1]],
            evidence=["Z"],
            evidence_card=[2],
            state_names={"Y": ["y0", "y1"], "Z": ["z0", "z1"]},
        ),
    )
    bn.check_model()

    target = ("Y", "y1")

    # Closed-form Bayes posterior P(Z|X), derived from the same constants
    # used to build the CPDs above.
    p_x1 = p_x1_z0 * p_z0 + p_x1_z1 * (1 - p_z0)
    p_z0_given_x1 = p_x1_z0 * p_z0 / p_x1
    p_y_x1 = p_y1_z0 * p_z0_given_x1 + p_y1_z1 * (1 - p_z0_given_x1)

    p_x0 = 1 - p_x1
    p_z0_given_x0 = (1 - p_x1_z0) * p_z0 / p_x0
    p_y_x0 = p_y1_z0 * p_z0_given_x0 + p_y1_z1 * (1 - p_z0_given_x0)

    expected_tv = p_y_x1 - p_y_x0

    assert total_effect(bn, target, "X", "x0", "x1") == pytest.approx(0.0, abs=1e-9)
    assert total_variation(bn, target, "X", "x0", "x1") == pytest.approx(expected_tv)
    assert abs(expected_tv) > 0.1  # a substantial apparent disparity despite zero causal effect

    for x in ["x0", "x1"]:
        se = spurious_effect(bn, target, "X", x)
        contributions = decompose_spurious_effect(bn, target, "X", x)
        assert contributions == pytest.approx({"Z": se})


def test_direct_and_indirect_effects_can_cancel_in_total_effect():
    """Opposing direct/indirect pathways cancel: TE ~= 0 while NDE stays large.

    A fairness audit that only looks at the aggregate total effect would
    conclude there is no disparity here. The point of the decomposition is to
    reveal the large, real direct effect that TE alone masks.
    """
    base, c, d = 0.5, 0.3, -1 / 3
    p_w1_x0, p_w1_x1 = 0.05, 0.95

    def y1_prob(x_is_x1: bool, w_is_w1: bool) -> float:
        return base + (c if x_is_x1 else 0.0) + (d if w_is_w1 else 0.0)

    bn = DiscreteBayesianNetwork([("X", "W"), ("X", "Y"), ("W", "Y")])
    _mark_types(bn, X="sensitive", Y="outcome", W="mediator")
    y1_row = [
        y1_prob(False, False),
        y1_prob(False, True),
        y1_prob(True, False),
        y1_prob(True, True),
    ]
    bn.add_cpds(
        TabularCPD("X", 2, [[0.5], [0.5]], state_names={"X": ["x0", "x1"]}),
        TabularCPD(
            "W",
            2,
            [[1 - p_w1_x0, 1 - p_w1_x1], [p_w1_x0, p_w1_x1]],
            evidence=["X"],
            evidence_card=[2],
            state_names={"W": ["w0", "w1"], "X": ["x0", "x1"]},
        ),
        TabularCPD(
            "Y",
            2,
            [[1 - p for p in y1_row], y1_row],
            evidence=["X", "W"],
            evidence_card=[2, 2],
            state_names={"Y": ["y0", "y1"], "X": ["x0", "x1"], "W": ["w0", "w1"]},
        ),
    )
    bn.check_model()

    target = ("Y", "y1")
    expected_nde = c
    expected_te = c + d * (p_w1_x1 - p_w1_x0)

    nde = natural_direct_effect(bn, target, "X", "x0", "x1")
    te = total_effect(bn, target, "X", "x0", "x1")
    nie_swapped = natural_indirect_effect(bn, target, "X", "x1", "x0")

    assert nde == pytest.approx(expected_nde)
    assert te == pytest.approx(expected_te, abs=1e-9)
    assert te == pytest.approx(0.0, abs=1e-9)
    assert abs(nde) > 0.2  # the direct effect is large in isolation
    # Identity compute_fairness_report relies on (NIE called with x0/x1 swapped).
    assert te == pytest.approx(nde - nie_swapped)


def test_indirect_decomposition_sums_to_total_with_two_mediators():
    """Two independent mediators: per-mediator contributions must sum to the total NIE (Thm 6.6)."""
    bn = DiscreteBayesianNetwork(
        [("X", "W1"), ("X", "W2"), ("X", "Y"), ("W1", "Y"), ("W2", "Y")]
    )
    _mark_types(bn, X="sensitive", Y="outcome", W1="mediator", W2="mediator")
    bn.add_cpds(
        TabularCPD("X", 2, [[0.5], [0.5]], state_names={"X": ["x0", "x1"]}),
        TabularCPD(
            "W1",
            2,
            [[0.8, 0.2], [0.2, 0.8]],
            evidence=["X"],
            evidence_card=[2],
            state_names={"W1": ["w0", "w1"], "X": ["x0", "x1"]},
        ),
        TabularCPD(
            "W2",
            2,
            [[0.6, 0.3], [0.4, 0.7]],
            evidence=["X"],
            evidence_card=[2],
            state_names={"W2": ["w0", "w1"], "X": ["x0", "x1"]},
        ),
        TabularCPD(
            "Y",
            2,
            [
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9],
            ],
            evidence=["X", "W1", "W2"],
            evidence_card=[2, 2, 2],
            state_names={
                "Y": ["y0", "y1"],
                "X": ["x0", "x1"],
                "W1": ["w0", "w1"],
                "W2": ["w0", "w1"],
            },
        ),
    )
    bn.check_model()

    target = ("Y", "y1")
    nie = natural_indirect_effect(bn, target, "X", "x0", "x1")
    contributions = decompose_indirect_effect(bn, target, "X", "x0", "x1")

    assert set(contributions) == {"W1", "W2"}
    assert sum(contributions.values()) == pytest.approx(nie)
    assert all(abs(v) > 1e-6 for v in contributions.values())


def test_spurious_decomposition_sums_to_total_with_two_confounders():
    """Two independent confounders: per-confounder contributions must sum to the total SE."""
    bn = DiscreteBayesianNetwork(
        [("Z1", "X"), ("Z2", "X"), ("Z1", "Y"), ("Z2", "Y"), ("X", "Y")]
    )
    _mark_types(bn, X="sensitive", Y="outcome", Z1="confounder", Z2="confounder")
    bn.add_cpds(
        TabularCPD("Z1", 2, [[0.5], [0.5]], state_names={"Z1": ["z0", "z1"]}),
        TabularCPD("Z2", 2, [[0.7], [0.3]], state_names={"Z2": ["z0", "z1"]}),
        TabularCPD(
            "X",
            2,
            [[0.8, 0.6, 0.4, 0.1], [0.2, 0.4, 0.6, 0.9]],
            evidence=["Z1", "Z2"],
            evidence_card=[2, 2],
            state_names={"X": ["x0", "x1"], "Z1": ["z0", "z1"], "Z2": ["z0", "z1"]},
        ),
        TabularCPD(
            "Y",
            2,
            [
                [0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.3, 0.1],
                [0.1, 0.3, 0.5, 0.6, 0.4, 0.5, 0.7, 0.9],
            ],
            evidence=["X", "Z1", "Z2"],
            evidence_card=[2, 2, 2],
            state_names={
                "Y": ["y0", "y1"],
                "X": ["x0", "x1"],
                "Z1": ["z0", "z1"],
                "Z2": ["z0", "z1"],
            },
        ),
    )
    bn.check_model()

    target = ("Y", "y1")
    for x in ["x0", "x1"]:
        se = spurious_effect(bn, target, "X", x)
        contributions = decompose_spurious_effect(bn, target, "X", x)

        assert set(contributions) == {"Z1", "Z2"}
        assert sum(contributions.values()) == pytest.approx(se)
        assert all(abs(v) > 1e-6 for v in contributions.values())
