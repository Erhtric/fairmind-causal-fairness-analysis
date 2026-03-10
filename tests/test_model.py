"""Tests for model.py module."""

import networkx as nx
import pandas as pd
import pgmpy.estimators
import pytest
from pgmpy.models import DiscreteBayesianNetwork

from src.graph import build_sfm
from src.model import fit_discrete_bayesian_model


@pytest.fixture
def sample_data():
    """Create sample discrete data for testing."""
    return pd.DataFrame(
        {
            "X": [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            "Z": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            "W": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
            "Y": [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def simple_sfm():
    """Create a simple SFM graph for testing."""
    return build_sfm(
        sensitive_attr="X",
        outcome_attr="Y",
        confounder_attrs=["Z"],
        mediator_attrs=["W"],
    )


class TestFitDiscreteBayesianModel:
    """Test suite for fit_discrete_bayesian_model function."""

    def test_basic_mle_fitting(self, simple_sfm, sample_data):
        """Test basic model fitting with Maximum Likelihood Estimator."""
        model = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.MaximumLikelihoodEstimator,
                {},
            ),
        )

        assert isinstance(model, DiscreteBayesianNetwork)
        assert set(model.nodes()) == {"X", "Y", "Z", "W"}

    def test_bayesian_estimator_with_bdeu(self, simple_sfm, sample_data):
        """Test model fitting with BayesianEstimator using BDeu prior."""
        model = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.BayesianEstimator,
                {"prior_type": "BDeu", "equivalent_sample_size": 5},
            ),
        )

        assert isinstance(model, DiscreteBayesianNetwork)
        assert set(model.nodes()) == {"X", "Y", "Z", "W"}

    def test_bayesian_estimator_with_k2(self, simple_sfm, sample_data):
        """Test model fitting with BayesianEstimator using K2 prior."""
        model = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.BayesianEstimator,
                {"prior_type": "K2"},
            ),
        )

        assert isinstance(model, DiscreteBayesianNetwork)
        assert set(model.nodes()) == {"X", "Y", "Z", "W"}

    def test_bayesian_estimator_with_dirichlet(self, simple_sfm, sample_data):
        """Test model fitting with BayesianEstimator using Dirichlet prior."""
        model = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.BayesianEstimator,
                {"prior_type": "dirichlet", "pseudo_counts": 1.0},
            ),
        )

        assert isinstance(model, DiscreteBayesianNetwork)
        assert set(model.nodes()) == {"X", "Y", "Z", "W"}

    def test_model_has_cpds(self, simple_sfm, sample_data):
        """Test that fitted model has Conditional Probability Distributions."""
        model = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.MaximumLikelihoodEstimator,
                {},
            ),
        )

        cpds = model.get_cpds()
        assert len(cpds) == 4  # One CPD for each node

        # Check that each node has a CPD
        cpd_variables = {cpd.variable for cpd in cpds}
        assert cpd_variables == {"X", "Y", "Z", "W"}

    def test_model_structure_matches_sfm(self, simple_sfm, sample_data):
        """Test that fitted model structure matches the input SFM."""
        model = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.MaximumLikelihoodEstimator,
                {},
            ),
        )

        # Check edges match
        assert set(model.edges()) == set(simple_sfm.edges())

    def test_simple_graph_no_mediators_or_confounders(self, sample_data):
        """Test fitting with minimal SFM (only X -> Y)."""
        minimal_sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=[],
            mediator_attrs=[],
        )

        model = fit_discrete_bayesian_model(
            sfm=minimal_sfm,
            data=sample_data[["X", "Y"]],
            estimator_instance=(
                pgmpy.estimators.MaximumLikelihoodEstimator,
                {},
            ),
        )

        assert isinstance(model, DiscreteBayesianNetwork)
        assert set(model.nodes()) == {"X", "Y"}
        assert model.has_edge("X", "Y")

    def test_different_pseudo_counts(self, simple_sfm, sample_data):
        """Test that different pseudo_counts produce different models."""
        model1 = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.BayesianEstimator,
                {"prior_type": "dirichlet", "pseudo_counts": 0.5},
            ),
        )

        model2 = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.BayesianEstimator,
                {"prior_type": "dirichlet", "pseudo_counts": 2.0},
            ),
        )

        # Both should be valid models
        assert isinstance(model1, DiscreteBayesianNetwork)
        assert isinstance(model2, DiscreteBayesianNetwork)

        # CPDs should differ due to different smoothing
        cpd1 = model1.get_cpds("Y")
        cpd2 = model2.get_cpds("Y")

        # At least some values should be different
        assert not (cpd1.values == cpd2.values).all()

    def test_invalid_sfm_type(self, sample_data):
        """Test that non-DiGraph SFM raises ValueError."""
        with pytest.raises(ValueError, match="The SFM must be a directed graph"):
            fit_discrete_bayesian_model(
                sfm="not a graph",
                data=sample_data,
                estimator_instance=(
                    pgmpy.estimators.MaximumLikelihoodEstimator,
                    {},
                ),
            )

    def test_invalid_data_type(self, simple_sfm):
        """Test that non-DataFrame data raises ValueError."""
        with pytest.raises(ValueError, match="Data must be a pandas DataFrame"):
            fit_discrete_bayesian_model(
                sfm=simple_sfm,
                data=[[0, 1], [1, 0]],
                estimator_instance=(
                    pgmpy.estimators.MaximumLikelihoodEstimator,
                    {},
                ),
            )

    def test_model_check_passes(self, simple_sfm, sample_data):
        """Test that the model passes internal consistency checks."""
        model = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.MaximumLikelihoodEstimator,
                {},
            ),
        )

        # check_model() is called internally
        # If it passes without raising, the model is valid
        model.check_model()  # Should not raise

    def test_empty_estimator_params(self, simple_sfm, sample_data):
        """Test that empty estimator_params dict works correctly."""
        model = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.MaximumLikelihoodEstimator,
                {},
            ),
        )

        assert isinstance(model, DiscreteBayesianNetwork)

    def test_larger_dataset(self, simple_sfm):
        """Test model fitting with a larger dataset."""
        large_data = pd.DataFrame(
            {
                "X": [i % 2 for i in range(1000)],
                "Z": [i % 3 for i in range(1000)],
                "W": [(i + 1) % 2 for i in range(1000)],
                "Y": [(i + 2) % 2 for i in range(1000)],
            }
        )

        model = fit_discrete_bayesian_model(
            sfm=simple_sfm,
            data=large_data,
            estimator_instance=(
                pgmpy.estimators.BayesianEstimator,
                {"prior_type": "BDeu", "equivalent_sample_size": 10},
            ),
        )

        assert isinstance(model, DiscreteBayesianNetwork)
        assert len(model.get_cpds()) == 4

    def test_model_with_multiple_mediators(self, sample_data):
        """Test fitting model with multiple mediators."""
        # Add another mediator column
        sample_data["W2"] = [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]

        multi_med_sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=["Z"],
            mediator_attrs=["W", "W2"],
        )

        model = fit_discrete_bayesian_model(
            sfm=multi_med_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.MaximumLikelihoodEstimator,
                {},
            ),
        )

        assert isinstance(model, DiscreteBayesianNetwork)
        assert "W2" in model.nodes()
        assert len(model.get_cpds()) == 5  # X, Z, W, W2, Y

    def test_model_with_multiple_confounders(self, sample_data):
        """Test fitting model with multiple confounders."""
        # Add another confounder column
        sample_data["Z2"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1]

        multi_conf_sfm = build_sfm(
            sensitive_attr="X",
            outcome_attr="Y",
            confounder_attrs=["Z", "Z2"],
            mediator_attrs=["W"],
        )

        model = fit_discrete_bayesian_model(
            sfm=multi_conf_sfm,
            data=sample_data,
            estimator_instance=(
                pgmpy.estimators.MaximumLikelihoodEstimator,
                {},
            ),
        )

        assert isinstance(model, DiscreteBayesianNetwork)
        assert "Z2" in model.nodes()
        assert len(model.get_cpds()) == 5  # X, Z, Z2, W, Y
