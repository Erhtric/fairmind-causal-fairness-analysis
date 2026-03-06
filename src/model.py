import networkx as nx
import pandas as pd
import pgmpy
import pgmpy.estimators
from loguru import logger
from pgmpy.models import DiscreteBayesianNetwork


def fit_discrete_bayesian_model(
    sfm: nx.DiGraph,
    data: pd.DataFrame,
    estimator_instance: tuple[pgmpy.estimators.BaseEstimator, dict],
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

    model = DiscreteBayesianNetwork(sfm)
    model.fit(data, estimator=estimator_class, **estimator_params)

    try:
        model.check_model()
    except ValueError as e:
        raise ValueError("Creation and fitting of the model failed.") from e
    return model
