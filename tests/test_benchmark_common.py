"""Tests for the functions shared by the benchmark notebooks.

These functions were extracted from the notebooks, where run_fairmind existed
in five copies that had drifted apart. The point of the tests is therefore not
only to check that the code works, but to pin the reference values so that a
future change to the shared copy is visible immediately, instead of surfacing
as a number in the thesis that quietly stopped matching.

The effects are checked by rounding to six decimals, which is the precision
the thesis quotes. Bit exact comparison would be wrong here: the values
computed on this machine differ from the ones produced on the cluster by about
1e-16, because floating point summation order depends on the hardware. That
noise is ten orders of magnitude below the sixth decimal, so the rounded figure
is stable, while any real change of behaviour would move it.
"""

import pandas as pd
import pytest

from src.benchmark_common import (
    build_llm_prompt,
    compute_discrepancies,
    run_fairmind,
)

ADULT_CONFIG = {
    "dataset_name": "adult",
    "csv_path": "data/processed/adult.csv",
    "target_col": "T_income",
    "target_val": ">50K",
    "protected": "S2_gender",
    "x0": "Female",
    "x1": "Male",
    "mediators": ["hours-per-week"],
    "confounders": ["education"],
    # Declares what run_fairmind used to hardcode. The hour bands keep the
    # mediator at five states; the education tiers bring the confounder from
    # sixteen levels down to five, so the prompt enumerates twenty five pairs
    # instead of eighty.
    "binning": {
        "hours-per-week": {
            "bins": [0, 20, 35, 45, 60, 100],
            "labels": ["<=20", "21-35", "36-45", "46-60", ">60"],
        },
        "education": {
            "mapping": {
                "Preschool": "<HS", "1st-4th": "<HS", "5th-6th": "<HS",
                "7th-8th": "<HS", "9th": "<HS", "10th": "<HS",
                "11th": "<HS", "12th": "<HS",
                "HS-grad": "HS-grad",
                "Some-college": "Some-college", "Assoc-acdm": "Some-college",
                "Assoc-voc": "Some-college",
                "Bachelors": "Bachelors",
                "Masters": "Grad", "Prof-school": "Grad", "Doctorate": "Grad",
            }
        },
    },
}

# The values reported in the experimental chapter.
ADULT_EFFECTS = {
    "TV": 0.193714,
    "TE": 0.184736,
    "SE": 0.008977,
    "DE": 0.138404,
    "IE": 0.016869,
    "IE_reverse": -0.046333,
}

TOL = 1e-9


@pytest.fixture(scope="module")
def adult_run():
    """Fit the network once and share it across the tests in this module."""
    effects, bn, n_rows, elapsed = run_fairmind(ADULT_CONFIG)
    return effects, bn, n_rows, elapsed


# ---------------------------------------------------------------------------
# run_fairmind
# ---------------------------------------------------------------------------


def test_adult_effects_match_the_reported_values(adult_run):
    """Pins every number the experimental chapter quotes for Adult."""
    effects = adult_run[0]
    for name, expected in ADULT_EFFECTS.items():
        assert round(float(effects[name]), 6) == expected, name


def test_the_whole_dataset_is_used(adult_run):
    """48842 rows, with no row silently dropped by the role selection."""
    assert adult_run[2] == 48842


def test_spurious_effect_is_derived_and_not_computed_separately(adult_run):
    """SE must equal TV - TE exactly, being defined as their difference.

    Regression test for the first of the two misalignments: SE used to be
    obtained from a one-argument formula, which gave the opposite sign.
    """
    effects = adult_run[0]
    assert float(effects["SE"]) == pytest.approx(
        float(effects["TV"]) - float(effects["TE"]), abs=1e-12
    )


def test_the_decomposition_closes_on_the_reverse_indirect_effect(adult_run):
    """TE = DE - IE_{x1,x0}, the identity of Prop. 2.

    Regression test for the second misalignment. Only the reverse form closes
    the decomposition; the direct form does not, and the assertion below would
    fail by a wide margin if the two were swapped.
    """
    effects = adult_run[0]
    assert float(effects["DE"]) - float(effects["IE_reverse"]) == pytest.approx(
        float(effects["TE"]), abs=TOL
    )


def test_the_two_indirect_forms_are_distinct_quantities(adult_run):
    """They are not one another's opposite, and confusing them changes results."""
    effects = adult_run[0]
    assert float(effects["IE"]) != pytest.approx(-float(effects["IE_reverse"]), abs=1e-3)


def test_both_effects_and_a_fitted_network_are_returned(adult_run):
    effects, bn, n_rows, elapsed = adult_run
    assert set(effects) == {"TV", "TE", "SE", "DE", "IE", "IE_reverse"}
    assert bn.get_cpds(ADULT_CONFIG["target_col"]) is not None
    assert elapsed >= 0


def test_the_mediator_is_discretised_into_five_bands(adult_run):
    """The binning is what keeps the enumeration in the prompt tractable."""
    bn = adult_run[1]
    states = bn.get_cpds("hours-per-week").state_names["hours-per-week"]
    assert len(states) == 5


def test_the_confounder_is_reduced_to_five_tiers(adult_run):
    """Sixteen education levels would give eighty (z,w) combinations."""
    bn = adult_run[1]
    states = bn.get_cpds("education").state_names["education"]
    assert len(states) == 5


# ---------------------------------------------------------------------------
# compute_discrepancies
# ---------------------------------------------------------------------------


def test_a_perfect_answer_has_no_error():
    table = compute_discrepancies(ADULT_EFFECTS, dict(ADULT_EFFECTS))
    assert (table["abs_error"] == 0).all()
    assert (table["rel_error_%"] == 0).all()


def test_the_relative_error_is_computed_against_the_reference():
    ground_truth = {"TV": 0.2, "TE": 0.2, "SE": 0.2, "DE": 0.2, "IE": 0.2}
    llm = {"TV": 0.1, "TE": 0.2, "SE": 0.2, "DE": 0.2, "IE": 0.2}
    row = compute_discrepancies(ground_truth, llm).set_index("effect").loc["TV"]
    assert row["abs_error"] == pytest.approx(0.1, abs=1e-9)
    assert row["rel_error_%"] == pytest.approx(50.0, abs=1e-6)


def test_only_the_five_requested_effects_are_compared():
    """IE_reverse stays out: the model is never asked for it.

    Comparing an answer against an estimand nobody requested measures nothing,
    which is exactly how the second misalignment produced an apparent error of
    138 per cent where the real one was under 7.
    """
    table = compute_discrepancies(ADULT_EFFECTS, dict(ADULT_EFFECTS))
    assert list(table["effect"]) == ["TV", "TE", "SE", "DE", "IE"]


def test_a_missing_answer_becomes_not_a_number_instead_of_raising():
    table = compute_discrepancies(ADULT_EFFECTS, {"TV": 0.19})
    assert pd.isna(table.set_index("effect").loc["TE", "llm"])


def test_a_zero_reference_does_not_divide_by_zero():
    ground_truth = dict.fromkeys(["TV", "TE", "SE", "DE", "IE"], 0.0)
    table = compute_discrepancies(ground_truth, dict.fromkeys(ground_truth, 0.1))
    assert table["rel_error_%"].isna().all()


# ---------------------------------------------------------------------------
# build_llm_prompt
# ---------------------------------------------------------------------------


def test_the_prompt_carries_the_tables_and_not_the_answers(adult_run):
    """The model receives aggregated probabilities, never the effects."""
    _, bn, n_rows, _ = adult_run
    prompt = build_llm_prompt(ADULT_CONFIG, bn, n_rows)

    assert "S2_gender" in prompt
    assert "T_income" in prompt
    # none of the reference values may leak into the prompt
    for value in ("0.193714", "0.184736", "0.138404"):
        assert value not in prompt


def test_the_prompt_enumerates_every_confounder_mediator_pair(adult_run):
    """Five education tiers by five hour bands gives twenty five rows."""
    _, bn, n_rows, _ = adult_run
    prompt = build_llm_prompt(ADULT_CONFIG, bn, n_rows)
    for tier in bn.get_cpds("education").state_names["education"]:
        assert tier in prompt
