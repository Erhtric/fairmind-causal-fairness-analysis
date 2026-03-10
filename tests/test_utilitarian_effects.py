import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from src.effects import effect_distribution, total_effect, utility_weighted_effect


@pytest.fixture
def numeric_target_bn() -> DiscreteBayesianNetwork:
    bn = DiscreteBayesianNetwork([("X", "Y")])

    bn.nodes["X"]["type"] = "sensitive"
    bn.nodes["Y"]["type"] = "outcome"

    cpd_x = TabularCPD(
        variable="X",
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={"X": ["x0", "x1"]},
    )

    cpd_y = TabularCPD(
        variable="Y",
        variable_card=3,
        values=[
            [0.7, 0.1],
            [0.2, 0.3],
            [0.1, 0.6],
        ],
        evidence=["X"],
        evidence_card=[2],
        state_names={"Y": [0, 1, 2], "X": ["x0", "x1"]},
    )

    bn.add_cpds(cpd_x, cpd_y)
    bn.check_model()
    return bn


@pytest.fixture
def categorical_target_bn() -> DiscreteBayesianNetwork:
    bn = DiscreteBayesianNetwork([("X", "Y")])

    bn.nodes["X"]["type"] = "sensitive"
    bn.nodes["Y"]["type"] = "outcome"

    cpd_x = TabularCPD(
        variable="X",
        variable_card=2,
        values=[[0.5], [0.5]],
        state_names={"X": ["x0", "x1"]},
    )

    cpd_y = TabularCPD(
        variable="Y",
        variable_card=2,
        values=[[0.8, 0.3], [0.2, 0.7]],
        evidence=["X"],
        evidence_card=[2],
        state_names={"Y": ["low", "high"], "X": ["x0", "x1"]},
    )

    bn.add_cpds(cpd_x, cpd_y)
    bn.check_model()
    return bn


def test_effect_distribution_returns_all_statewise_total_effects(numeric_target_bn):
    dist = effect_distribution(total_effect, numeric_target_bn, "Y", "X", "x0", "x1")

    assert dist == pytest.approx([-0.6, 0.1, 0.5])


def test_utilitarian_effect_matches_difference_of_expectations(numeric_target_bn):
    effect = utility_weighted_effect(
        total_effect, numeric_target_bn, "Y", "X", "x0", "x1"
    )

    assert effect == pytest.approx(1.1)


def test_utilitarian_effect_accepts_callable_utility(numeric_target_bn):
    effect = utility_weighted_effect(
        total_effect,
        numeric_target_bn,
        "Y",
        "X",
        "x0",
        "x1",
        lambda state: state**2,
    )

    assert effect == pytest.approx(2.1)


def test_utilitarian_effect_accepts_dictionary_utility(numeric_target_bn):
    effect = utility_weighted_effect(
        total_effect,
        numeric_target_bn,
        "Y",
        "X",
        "x0",
        "x1",
        {0: -1.0, 1: 0.5, 2: 3.0},
    )

    assert effect == pytest.approx(2.15)


def test_utilitarian_effect_accepts_tuple_target_and_ignores_state_value(
    numeric_target_bn,
):
    effect = utility_weighted_effect(
        total_effect,
        numeric_target_bn,
        ("Y", 1),
        "X",
        "x0",
        "x1",
    )

    assert effect == pytest.approx(1.1)


def test_utilitarian_effect_rejects_non_numeric_target_states(categorical_target_bn):
    with pytest.raises(ValueError, match="requires inherently numeric target states"):
        utility_weighted_effect(
            total_effect, categorical_target_bn, "Y", "X", "x0", "x1"
        )


def test_utilitarian_effect_rejects_invalid_callable_utility(categorical_target_bn):
    with pytest.raises(ValueError, match="valid callable function"):
        utility_weighted_effect(
            total_effect,
            categorical_target_bn,
            "Y",
            "X",
            "x0",
            "x1",
            lambda state: float(state),
        )


def test_utilitarian_effect_rejects_incomplete_dictionary_utility(numeric_target_bn):
    with pytest.raises(ValueError, match="valid utility dictionary"):
        utility_weighted_effect(
            total_effect,
            numeric_target_bn,
            "Y",
            "X",
            "x0",
            "x1",
            {0: 1.0, 1: 2.0},
        )
