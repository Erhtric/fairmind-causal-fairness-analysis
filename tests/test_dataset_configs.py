"""Tests for the dataset configurations of the multi-dataset campaign.

The configurations are data, not code, and that is exactly why they need
checking: a wrong state name or a missing binning specification fails deep
inside pgmpy, with a message that says nothing about which dataset is at fault.

The cheap structural checks run always. The one that actually fits every
network is marked slow, since it reads two hundred thousand rows for the
largest dataset.
"""

from pathlib import Path

import pytest

from src.dataset_configs import DATASET_CONFIGS, config_by_name

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = {
    "dataset_name", "csv_path", "target_col", "target_val",
    "protected", "x0", "x1", "mediators", "confounders",
    "binning", "combinations",
}


def _resolve(config: dict) -> Path:
    """The csv_path is written relative to the notebooks directory."""
    return REPO_ROOT / config["csv_path"].replace("../../", "")


@pytest.mark.parametrize("config", DATASET_CONFIGS, ids=lambda c: c["dataset_name"])
def test_every_configuration_declares_the_required_keys(config):
    assert REQUIRED_KEYS <= set(config), REQUIRED_KEYS - set(config)


@pytest.mark.parametrize("config", DATASET_CONFIGS, ids=lambda c: c["dataset_name"])
def test_the_csv_exists(config):
    assert _resolve(config).is_file()


@pytest.mark.parametrize("config", DATASET_CONFIGS, ids=lambda c: c["dataset_name"])
def test_one_mediator_and_one_confounder(config):
    """More than one of either multiplies the pairs the prompt has to list."""
    assert len(config["mediators"]) == 1
    assert len(config["confounders"]) == 1


@pytest.mark.parametrize("config", DATASET_CONFIGS, ids=lambda c: c["dataset_name"])
def test_the_declared_combinations_stay_within_the_prompt_budget(config):
    """Past roughly forty pairs the model stops finishing the computation.

    Recorded as a number on each entry so that adding a dataset makes the
    author state the figure rather than discover it on the cluster.
    """
    assert config["combinations"] <= 40


@pytest.mark.parametrize("config", DATASET_CONFIGS, ids=lambda c: c["dataset_name"])
def test_binning_specifications_are_well_formed(config):
    """Each entry is either numeric intervals or a categorical mapping."""
    for column, spec in config["binning"].items():
        if "mapping" in spec:
            assert isinstance(spec["mapping"], dict) and spec["mapping"]
        else:
            assert len(spec["bins"]) == len(spec["labels"]) + 1, column


def test_dataset_names_are_unique():
    names = [c["dataset_name"] for c in DATASET_CONFIGS]
    assert len(names) == len(set(names))


def test_config_by_name_finds_a_dataset_and_returns_a_copy():
    config = config_by_name("adult")
    assert config["protected"] == "S2_gender"
    config["protected"] = "changed"
    assert config_by_name("adult")["protected"] == "S2_gender"


def test_config_by_name_names_the_alternatives_when_it_fails():
    with pytest.raises(KeyError, match="adult"):
        config_by_name("not-a-dataset")


@pytest.mark.slow
@pytest.mark.parametrize("config", DATASET_CONFIGS, ids=lambda c: c["dataset_name"])
def test_every_configuration_fits_and_matches_its_declared_size(config):
    """Fit the network and check the pair count against the declared figure.

    This is the test that catches a state name that does not exist in the data,
    a target value spelled differently, or a binning that leaves the confounder
    with more levels than expected.
    """
    from src.benchmark_common import run_fairmind

    resolved = dict(config)
    resolved["csv_path"] = str(_resolve(config))
    effects, bn, n_rows, _ = run_fairmind(resolved)

    assert n_rows > 0
    assert set(effects) == {"TV", "TE", "SE", "DE", "IE", "IE_reverse"}

    confounder = config["confounders"][0]
    mediator = config["mediators"][0]
    n_z = len(bn.get_cpds(confounder).state_names[confounder])
    n_w = len(bn.get_cpds(mediator).state_names[mediator])
    assert n_z * n_w == config["combinations"]

    # Eq. 9 has to close on every dataset, not only on Adult.
    assert abs(effects["DE"] - effects["IE_reverse"] - effects["TE"]) < 1e-9


@pytest.mark.slow
def test_adult_still_reproduces_the_reference_values():
    """The shared configuration must not have changed the reference numbers.

    Adult is quoted throughout the experimental chapter, and its configuration
    now lives here rather than inside a notebook. If moving it changed a digit,
    every figure in that chapter would be stale.
    """
    from src.benchmark_common import run_fairmind

    config = config_by_name("adult")
    config["csv_path"] = str(_resolve(config))
    effects, _, _, _ = run_fairmind(config)

    for key, expected in [
        ("TV", 0.193714), ("TE", 0.184736), ("SE", 0.008977),
        ("DE", 0.138404), ("IE", 0.016869), ("IE_reverse", -0.046333),
    ]:
        assert effects[key] == pytest.approx(expected, abs=1e-6), key
