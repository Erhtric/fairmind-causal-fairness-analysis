import pandas as pd
import numpy as np
import networkx as nx
from itertools import product

from src.graph import build_sfm
from src.model import fit_discrete_bayesian_model
from src.effects import (
    total_variation,
    total_effect,
    spurious_effect,
    natural_direct_effect,
    natural_indirect_effect,
)
from pgmpy.estimators import BayesianEstimator


def combine_sensitive_attrs(
    df: pd.DataFrame,
    attrs: list[str],
    sep: str = "\u00d7",
) -> pd.DataFrame:
    """Create a single composite sensitive attribute from multiple columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    attrs : list of str
        Column names to combine (e.g. ["S2_gender", "S3_race"]).
    sep : str
        Separator between values (default: multiplication sign).

    Returns
    -------
    pd.DataFrame
        A copy of df with a new column ``"_sensitive_composite"`` added.
    """
    df = df.copy()
    composite = df[attrs[0]].astype(str)
    for attr in attrs[1:]:
        composite = composite + sep + df[attr].astype(str)
    df["_sensitive_composite"] = composite
    return df


def compute_intersectional_effects(
    df: pd.DataFrame,
    sensitive_attrs: list[str],
    target_col: str,
    target_val: str,
    mediators: list[str],
    confounders: list[str],
    x0: str | None = None,
    x1: str | None = None,
) -> dict[str, dict[str, float]]:
    """Compute causal fairness effects for an intersectional protected attribute.

    The function creates a composite attribute from ``sensitive_attrs``,
    then computes TV, TE, SE, DE, IE for ALL pairs of (x0, x1) where
    x0 iterates over all groups and x1 is either a fixed reference or
    also iterates over all groups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset.
    sensitive_attrs : list of str
        Columns forming the intersectional attribute.
    target_col : str
        Outcome column name.
    target_val : str
        Target outcome value (e.g. ">50K").
    mediators : list of str
        Mediator column names.
    confounders : list of str
        Confounder column names.
    x0 : str or None
        Baseline composite value. If None, uses all groups as baselines.
    x1 : str or None
        Comparison composite value. If None, uses all groups as comparisons.

    Returns
    -------
    dict of dict:
        {
            "(Female×White, Male×White)": {"TV": ..., "TE": ..., "SE": ..., "DE": ..., "IE": ...},
            ...
        }
    """
    df = combine_sensitive_attrs(df, sensitive_attrs)
    protected = "_sensitive_composite"

    cols = [protected] + mediators + confounders + [target_col]
    df_model = df[cols].dropna()

    groups = sorted(df_model[protected].unique())

    if x0 is None:
        x0_list = groups
    else:
        x0_list = [x0]

    if x1 is None:
        x1_list = groups
    else:
        x1_list = [x1]

    target = (target_col, target_val)

    all_effects = {}
    for x0_val, x1_val in product(x0_list, x1_list):
        if x0_val == x1_val:
            continue

        sfm = build_sfm(
            sensitive_attr=protected,
            outcome_attr=target_col,
            confounder_attrs=confounders,
            mediator_attrs=mediators,
        )
        bn = fit_discrete_bayesian_model(
            sfm=sfm,
            data=df_model,
            estimator_instance=(BayesianEstimator, {"prior_type": "BDeu"}),
        )

        effects = {
            "TV": total_variation(bn, target, protected, x0_val, x1_val),
            "TE": total_effect(bn, target, protected, x0_val, x1_val),
            "SE": spurious_effect(bn, target, protected, x0_val),
            "DE": natural_direct_effect(bn, target, protected, x0_val, x1_val),
            "IE": natural_indirect_effect(bn, target, protected, x1_val, x0_val),
        }

        key = f"({x0_val}, {x1_val})"
        all_effects[key] = effects

    return all_effects
