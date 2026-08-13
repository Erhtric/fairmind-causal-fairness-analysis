from typing import Any

import networkx as nx
import pandas as pd
from loguru import logger
from pgmpy.models import DiscreteBayesianNetwork

from .graph import filter_nodes_by_type


from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteMLE


def fit_discrete_bayesian_model(
    sfm: nx.DiGraph,
    data: pd.DataFrame,
    estimator_instance: tuple[Any, dict],
) -> DiscreteBayesianNetwork:
    """
    Fits a Discrete Bayesian Model to the given Standard Fairness Model (SFM) graph.

    Args:
        sfm (nx.DiGraph): A directed graph representing the SFM template.
        data (pd.DataFrame): The dataset to fit the model on.
        estimator_instance (Tuple[pgmpy.estimators.BaseEstimator, dict]): A tuple containing an instance of a pgmpy estimator and its parameters.

    Returns:
        DiscreteBayesianNetwork: A fitted Discrete Bayesian Network based on the SFM graph.
    """
    if not isinstance(sfm, nx.DiGraph):
        raise ValueError("The SFM must be a directed graph (nx.DiGraph).")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    estimator_class, estimator_params = estimator_instance
    logger.debug(
        f"Using estimator: {estimator_class} with parameters: {estimator_params}"
    )

    latents = filter_nodes_by_type(sfm.nodes(data=True), category="latent")
    model = DiscreteBayesianNetwork(sfm, latents=set(latents))

    if hasattr(estimator_class, "__name__") and estimator_class.__name__ in (
        "MaximumLikelihoodEstimator",
        "DiscreteMLE",
    ):
        estimator = DiscreteMLE(**estimator_params)
    elif hasattr(estimator_class, "__name__") and estimator_class.__name__ in (
        "DiscreteBayesianEstimator",
        "BayesianEstimator",
    ):
        estimator = DiscreteBayesianEstimator(**estimator_params)
    elif isinstance(estimator_class, type):
        estimator = estimator_class(**estimator_params)
    else:
        estimator = estimator_class

    model.fit(data, estimator=estimator)

    return model


# def fit_discrete_bayesian_model_with_mle(
#     sfm: nx.DiGraph,
#     data: pd.DataFrame,
# ) -> DiscreteBayesianNetwork:
#     """
#     Fits a Discrete Bayesian Model to the given Standard Fairness Model (SFM) graph using Maximum Likelihood Estimation (MLE).

#     Args:
#         sfm (nx.DiGraph): A directed graph representing the SFM template.
#         data (pd.DataFrame): The dataset to fit the model on.

#     Returns:
#         DiscreteBayesianNetwork: A fitted Discrete Bayesian Network based on the SFM graph using MLE.
#     """
#     if not isinstance(sfm, nx.DiGraph):
#         raise ValueError("The SFM must be a directed graph (nx.DiGraph).")
#     if not isinstance(data, pd.DataFrame):
#         raise ValueError("Data must be a pandas DataFrame.")

#     latents = filter_nodes_by_type(sfm.nodes(data=True), category="latent")
#     model = DiscreteBayesianNetwork(sfm, latents=set(latents))
#     model.fit(data, estimator=DiscreteMLE())
#     return model
