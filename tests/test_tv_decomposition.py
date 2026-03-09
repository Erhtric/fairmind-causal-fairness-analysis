"""Tests for decomposition identities in effects.py."""

import numpy as np
import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
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


@pytest.fixture
def sfm_bn() -> DiscreteBayesianNetwork:
    """Build a deterministic SFM-like BN with one confounder and one mediator."""
    bn = DiscreteBayesianNetwork(
        [
            ("Z", "X"),
            ("Z", "W"),
            ("Z", "Y"),
            ("X", "W"),
            ("X", "Y"),
            ("W", "Y"),
        ]
    )

    # Node typing is required by effect helpers that filter by role.
    bn.nodes["X"]["type"] = "sensitive"
    bn.nodes["Y"]["type"] = "outcome"
    bn.nodes["Z"]["type"] = "confounder"
    bn.nodes["W"]["type"] = "mediator"

    state_names = {
        "X": ["x0", "x1"],
        "Y": ["y0", "y1"],
        "Z": ["z0", "z1"],
        "W": ["w0", "w1"],
    }

    cpd_z = TabularCPD(
        variable="Z",
        variable_card=2,
        values=[[0.6], [0.4]],
        state_names={"Z": state_names["Z"]},
    )

    cpd_x = TabularCPD(
        variable="X",
        variable_card=2,
        values=[
            [0.8, 0.3],
            [0.2, 0.7],
        ],
        evidence=["Z"],
        evidence_card=[2],
        state_names={"X": state_names["X"], "Z": state_names["Z"]},
    )

    cpd_w = TabularCPD(
        variable="W",
        variable_card=2,
        values=[
            [0.9, 0.8, 0.3, 0.2],
            [0.1, 0.2, 0.7, 0.8],
        ],
        evidence=["X", "Z"],
        evidence_card=[2, 2],
        state_names={
            "W": state_names["W"],
            "X": state_names["X"],
            "Z": state_names["Z"],
        },
    )

    cpd_y = TabularCPD(
        variable="Y",
        variable_card=2,
        values=[
            [0.95, 0.85, 0.55, 0.45, 0.75, 0.65, 0.30, 0.20],
            [0.05, 0.15, 0.45, 0.55, 0.25, 0.35, 0.70, 0.80],
        ],
        evidence=["X", "W", "Z"],
        evidence_card=[2, 2, 2],
        state_names={
            "Y": state_names["Y"],
            "X": state_names["X"],
            "W": state_names["W"],
            "Z": state_names["Z"],
        },
    )

    bn.add_cpds(cpd_z, cpd_x, cpd_w, cpd_y)
    bn.check_model()
    return bn


def test_total_variation_matches_observational_definition(sfm_bn):
    """TV should match P(Y|X=x1) - P(Y|X=x0) for the selected target value."""
    tv = total_variation(sfm_bn, ("Y", "y1"), "X", "x0", "x1")

    ve = VariableElimination(sfm_bn)
    p_y_given_x1 = ve.query(
        variables=["Y"], evidence={"X": "x1"}, show_progress=False
    ).values[1]
    p_y_given_x0 = ve.query(
        variables=["Y"], evidence={"X": "x0"}, show_progress=False
    ).values[1]

    assert tv == pytest.approx(p_y_given_x1 - p_y_given_x0)


def test_tv_te_and_spurious_decomposition_identity(sfm_bn):
    """TV identity: TV = TE + (SE(x1) - SE(x0))."""
    tv = total_variation(sfm_bn, ("Y", "y1"), "X", "x0", "x1")
    te = total_effect(sfm_bn, ("Y", "y1"), "X", "x0", "x1")
    se_x0 = spurious_effect(sfm_bn, ("Y", "y1"), "X", "x0")
    se_x1 = spurious_effect(sfm_bn, ("Y", "y1"), "X", "x1")

    assert tv == pytest.approx(te + (se_x1 - se_x0))


def test_natural_effects_are_computable_and_bounded(sfm_bn):
    """Natural effects should return finite probabilities in a valid range."""
    de = natural_direct_effect(sfm_bn, ("Y", "y1"), "X", "x0", "x1")
    ie = natural_indirect_effect(sfm_bn, ("Y", "y1"), "X", "x0", "x1")

    assert np.isfinite(de)
    assert np.isfinite(ie)
    assert -1.0 <= de <= 1.0
    assert -1.0 <= ie <= 1.0


def test_indirect_decomposition_matches_total_indirect_effect_single_mediator(sfm_bn):
    """With one mediator, its contribution should equal the full NIE."""
    nie = natural_indirect_effect(sfm_bn, ("Y", "y1"), "X", "x0", "x1")
    contributions = decompose_indirect_effect(sfm_bn, ("Y", "y1"), "X", "x0", "x1")

    assert set(contributions.keys()) == {"W"}
    assert contributions["W"] == pytest.approx(nie)


def test_spurious_decomposition_matches_total_spurious_effect_single_confounder(sfm_bn):
    """With one confounder, its contribution should equal the full SE for each x."""
    for x in ["x0", "x1"]:
        se = spurious_effect(sfm_bn, ("Y", "y1"), "X", x)
        contributions = decompose_spurious_effect(sfm_bn, ("Y", "y1"), "X", x)

        assert set(contributions.keys()) == {"Z"}
        assert contributions["Z"] == pytest.approx(se)


def test_total_effect_rejects_non_sensitive_private_attribute(sfm_bn):
    """If graph has sensitive node annotations, private_attr must be sensitive."""
    with pytest.raises(ValueError, match="is not marked as sensitive"):
        total_effect(sfm_bn, ("Y", "y1"), "Z", "z0", "z1")
