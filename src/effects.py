import openai.types.webhooks.eval_run_succeeded_webhook_event
from collections.abc import Callable
from itertools import product
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import CausalInference, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.graph import filter_nodes_by_type

############################################################
######### Effect computation via ID Expr on SFM ############
############################################################


def utility_weighted_effect(
    effect_func: Callable[
        [DiscreteBayesianNetwork, tuple[str, Any], str, Any, Any], float
    ],
    bn: DiscreteBayesianNetwork,
    target: str | tuple[str, Any],
    private_attr: str,
    x0: Any,
    x1: Any,
    T: Callable[[Any], float] | dict[Any, float] | None = None,
) -> float:
    """Compute the utility-weighted effect across all target states.

    This treats the target states as numerical utilities and computes the
    difference of expectations using a utility function T(V).

    Args:
        effect_func: The effect function to compute.
        bn: The Bayesian Network to use for inference.
        target: The target variable name.
        private_attr: The name of the private variable.
        x0: The baseline value of the private variable.
        x1: The modified value of the private variable.
        T: An optional utility function T(V) (callable) or a dictionary
           mapping state names to numerical values.

    Returns:
        The difference of expectations induced by the effect distribution.

    Usage:
    ```python
    # Using a custom utility function
    effect: int | float = utility_weighted_effect(TE, bn, "Y", "X", "x0", "x1", lambda x: x**2)
    ```
    """
    target_var = target if isinstance(target, str) else target[0]
    target_labels = bn.get_cpds(target_var).state_names[target_var]

    try:
        if callable(T):
            # Apply the continuous/custom function T(V) to each state
            state_values = np.array([T(label) for label in target_labels], dtype=float)
        elif isinstance(T, dict):
            # Use discrete lookup table
            state_values = np.array([T[label] for label in target_labels], dtype=float)
        else:
            # Fallback to direct casting
            state_values = np.asarray(target_labels, dtype=float)

    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(
            "utility_weighted_effect requires inherently numeric target states, a valid "
            "utility dictionary, or a valid callable function T(V) to compute expectations."
        ) from exc

    return float(
        np.dot(
            effect_distribution(effect_func, bn, target_var, private_attr, x0, x1),
            state_values,
        )
    )


def effect_distribution(
    effect_func: Callable[
        [DiscreteBayesianNetwork, tuple[str, Any], str, Any, Any], float
    ],
    bn: DiscreteBayesianNetwork,
    target: str,
    private_attr: str,
    x0: Any,
    x1: Any,
) -> np.ndarray:
    """Compute the full state-wise effect distribution.

    Args:
        effect_func: The effect function to compute (e.g., total_effect,
            natural_direct_effect, etc.)
        bn: The Bayesian Network to use for inference.
        target: The target variable name.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: The baseline value of the private variable.
        x1: The modified value of the private variable.

    Returns:
        A vector whose entries are the state-specific effect values.
    """
    target_labels = bn.get_cpds(target).state_names[target]
    effect_dist = np.zeros(len(target_labels))
    for i, target_val in enumerate(target_labels):
        effect_dist[i] = effect_func(bn, (target, target_val), private_attr, x0, x1)

    return effect_dist


def total_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: Any = None,
    x1: Any = None,
) -> float:
    """Compute the total effect on the SFM.

    TE(x0,x1,y) = P(Y @ {X:x1} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: The baseline value of the private variable.
        x1: The modified value of the private variable.

    Returns:
        The total effect.
    """
    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]

    if target_val is not None and target_val not in target_labels:
        raise ValueError("target value must be a valid state of the target variable.")

    if private_attr in bn.nodes:
        node_type = bn.nodes[private_attr].get("type")
        has_type_annotations = any("type" in bn.nodes[node] for node in bn.nodes)
        if has_type_annotations and node_type != "sensitive":
            raise ValueError(
                f"Variable '{private_attr}' is not marked as sensitive. "
                f"Found type: {node_type if node_type else 'None'}"
            )

    ve = CausalInference(bn)
    P_target_do_x1 = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        private_var=private_attr,
        private_val=x1,
    )

    P_target_do_x0 = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        private_var=private_attr,
        private_val=x0,
    )

    te_dist = P_target_do_x1 - P_target_do_x0
    te_val = te_dist[target_labels.index(target_val)]

    return te_val


def spurious_effect(
    bn: DiscreteBayesianNetwork, target: tuple[str, str], private_attr: str, x: str
) -> float:
    """Compute the spurious effect of changing the cause from x0 to x1 on the target variable.

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x: A tuple of (variable, value) representing the baseline value of the private variable.
    Returns:
        The spurious effect.
    """

    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    cause_labels = bn.get_cpds(private_attr).state_names[private_attr]
    if x not in cause_labels:
        raise ValueError("x must be a valid state of the cause variable.")

    logger.debug(f"Computing spurious effect for target={target}, private_value={x}")

    ve = CausalInference(bn)

    P_target_do_x = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        private_var=private_attr,
        private_val=x,
    )

    P_target_given_x = ve.query(
        variables=[target_var],
        evidence={private_attr: x},
        show_progress=False,
    )

    se_dist = P_target_given_x.values - P_target_do_x
    se_val = se_dist[target_val_index]

    return se_val


def total_variation(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, str],
    private_attr: str,
    x0: str,
    x1: str,
) -> float:
    """Compute the total variation.

    TV(x0,x1,y) = P(Y @ {X:x1} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: A tuple of (variable, value) representing the baseline value of the private variable.
        x1: A tuple of (variable, value) representing the modified value of the private variable.

    Returns:
        The total variation.
    """
    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    cause_labels = bn.get_cpds(private_attr).state_names[private_attr]
    if x0 not in cause_labels or x1 not in cause_labels:
        raise ValueError("x0 and x1 must be valid states of the cause variable.")

    logger.debug(
        f"Computing total variation for target={target}, private_baseline={x0}, private_mod={x1}"
    )

    ve = VariableElimination(bn)
    P_target_given_x1 = ve.query(
        variables=[target_var],
        evidence={private_attr: x1},
        show_progress=False,
    )

    P_target_given_x0 = ve.query(
        variables=[target_var],
        evidence={private_attr: x0},
        show_progress=False,
    )

    tv_dist = P_target_given_x1.values - P_target_given_x0.values
    tv_val = tv_dist[target_val_index]

    return tv_val


def natural_direct_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: Any,
    x1: Any,
) -> float:
    """Compute the natural direct effect.

    NDE(x0,x1,y) = P(Y @ {X:x1, W: W @ {X:x0}} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: The baseline value of the private variable.
        x1: The modified value of the private variable.

    Returns:
        The natural direct effect.
    """
    target_var, target_val = target
    cause_labels = bn.get_cpds(private_attr).state_names[private_attr]

    if x0 not in cause_labels or x1 not in cause_labels:
        raise ValueError("x0 and x1 must be valid states of the cause variable.")

    mediators = filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
    confounders = filter_nodes_by_type(bn.nodes(data=True, default={}), "confounder")

    ve = VariableElimination(bn)

    # P(Y_{x1, W_{x0}})
    # Equation: sum_{z, w} P(Y|x1, w, z) * P(w|x0, z) * P(z)

    # P(Z)
    factor_z = ve.query(variables=confounders, joint=True) if confounders else None

    # P(W | x0, Z)
    if mediators:
        factor_w_z_given_x0 = ve.query(
            variables=mediators + confounders, evidence={private_attr: x0}, joint=True
        )
        if confounders:
            factor_z_given_x0 = ve.query(
                variables=confounders, evidence={private_attr: x0}, joint=True
            )
            factor_w_given_x0_z = factor_w_z_given_x0 / factor_z_given_x0
        else:
            factor_w_given_x0_z = factor_w_z_given_x0
    else:
        factor_w_given_x0_z = None

    # P(Y | x1, W, Z)
    cond_vars = mediators + confounders
    if cond_vars:
        factor_y_cond_given_x1 = ve.query(
            variables=[target_var] + cond_vars, evidence={private_attr: x1}, joint=True
        )
        factor_cond_given_x1 = ve.query(
            variables=cond_vars, evidence={private_attr: x1}, joint=True
        )
        factor_y_given_x1_all = factor_y_cond_given_x1 / factor_cond_given_x1
    else:
        factor_y_given_x1_all = ve.query(
            variables=[target_var], evidence={private_attr: x1}, joint=True
        )

    cross_world_factor = factor_y_given_x1_all
    if factor_w_given_x0_z is not None:
        cross_world_factor = cross_world_factor * factor_w_given_x0_z
    if factor_z is not None:
        cross_world_factor = cross_world_factor * factor_z

    cross_world_factor.marginalize(
        [v for v in cross_world_factor.variables if v != target_var], inplace=True
    )

    first_term_val = float(cross_world_factor.get_value(**{target_var: target_val}))

    # P(Y_{x0})
    second_term_val = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=CausalInference(bn),
        target_var=target_var,
        private_var=private_attr,
        private_val=x0,
    )

    target_labels = bn.get_cpds(target_var).state_names[target_var]
    second_term_val = float(second_term_val[target_labels.index(target_val)])

    return first_term_val - second_term_val


def natural_indirect_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: Any,
    x1: Any,
) -> float:
    """Compute the natural indirect effect.

    NIE(x0,x1,y) = P(Y @ {X:x0, W: W @ {X:x1}} == y) - P(Y @ {X:x0} == y)
    """
    target_var, target_val = target
    cause_labels = bn.get_cpds(private_attr).state_names[private_attr]

    if x0 not in cause_labels or x1 not in cause_labels:
        raise ValueError("x0 and x1 must be valid states of the cause variable.")

    mediators = filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
    confounders = filter_nodes_by_type(bn.nodes(data=True, default={}), "confounder")

    ve = VariableElimination(bn)

    # P(Y_{x0, W_{x1}})
    # Equation: sum_{z, w} P(Y|x0, w, z) * P(w|x1, z) * P(z)

    # P(Z)
    factor_z = ve.query(variables=confounders, joint=True) if confounders else None

    # P(W | x1, Z)
    if mediators:
        factor_w_z_given_x1 = ve.query(
            variables=mediators + confounders, evidence={private_attr: x1}, joint=True
        )
        if confounders:
            factor_z_given_x1 = ve.query(
                variables=confounders, evidence={private_attr: x1}, joint=True
            )
            factor_w_given_x1_z = factor_w_z_given_x1 / factor_z_given_x1
        else:
            factor_w_given_x1_z = factor_w_z_given_x1
    else:
        factor_w_given_x1_z = None

    # P(Y | x0, W, Z)
    cond_vars = mediators + confounders
    if cond_vars:
        factor_y_cond_given_x0 = ve.query(
            variables=[target_var] + cond_vars, evidence={private_attr: x0}, joint=True
        )
        factor_cond_given_x0 = ve.query(
            variables=cond_vars, evidence={private_attr: x0}, joint=True
        )
        factor_y_given_x0_all = factor_y_cond_given_x0 / factor_cond_given_x0
    else:
        factor_y_given_x0_all = ve.query(
            variables=[target_var], evidence={private_attr: x0}, joint=True
        )

    cross_world_factor = factor_y_given_x0_all
    if factor_w_given_x1_z is not None:
        cross_world_factor = cross_world_factor * factor_w_given_x1_z
    if factor_z is not None:
        cross_world_factor = cross_world_factor * factor_z

    cross_world_factor.marginalize(
        [v for v in cross_world_factor.variables if v != target_var], inplace=True
    )

    first_term_val = float(cross_world_factor.get_value(**{target_var: target_val}))

    # P(Y_{x0})
    second_term_val = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=CausalInference(
            bn
        ),  # Reverting to your CausalInference class for compatibility with your helper
        target_var=target_var,
        private_var=private_attr,
        private_val=x0,
    )

    target_labels = bn.get_cpds(target_var).state_names[target_var]
    second_term_val = float(second_term_val[target_labels.index(target_val)])

    return first_term_val - second_term_val


# Aliases
TV = total_variation
TE = total_effect
SE = spurious_effect
IE = natural_indirect_effect
DE = natural_direct_effect


############################################################
############## Effect computation via SCM ##################
############################################################


def total_effect_scm(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: Any,
    x1: Any,
) -> float:
    """Compute the total effect.

    TE(x0,x1,y) = P(Y @ {X:x1} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: A tuple of (variable, value) representing the baseline value of the private variable.
        x1: A tuple of (variable, value) representing the modified value of the private variable.
    Returns:
        The total effect.
    """
    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    private_domain = bn.get_cpds(private_attr).state_names[private_attr]
    if x0 not in private_domain or x1 not in private_domain:
        raise ValueError("x0 and x1 must be valid states of the cause variable.")
    if target_val not in target_labels:
        raise ValueError("target value must be a valid state of the target variable.")

    logger.debug(
        f"Computing total effect for target={target}, private_baseline={x0}, private_mod={x1}"
    )

    ve = CausalInference(bn)

    p_target_do_x1 = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        private_var=private_attr,
        private_val=x1,
    )

    p_target_do_x0 = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        private_var=private_attr,
        private_val=x0,
    )

    te_dist = p_target_do_x1 - p_target_do_x0
    te_val = te_dist[target_val_index]

    return te_val


def spurious_effect_scm(
    bn: DiscreteBayesianNetwork, target: tuple[str, str], cause: str, x: str
) -> float:
    """Compute the spurious effect of changing the cause from x0 to x1 on the target variable.

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        cause: The name of the private variable whose effect we want to measure.
        x: A tuple of (variable, value) representing the baseline value of the private variable.
    Returns:
        The spurious effect.
    """

    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    cause_labels = bn.get_cpds(cause).state_names[cause]
    if x not in cause_labels:
        raise ValueError("x must be a valid state of the cause variable.")

    logger.debug(f"Computing spurious effect for target={target}, private_value={x}")

    ve = CausalInference(bn)

    P_target_do_x = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        private_var=cause,
        private_val=x,
    )

    P_target_given_x = ve.query(
        variables=[target_var],
        evidence={cause: x},
        show_progress=False,
    ).values[target_val_index]

    return P_target_given_x - P_target_do_x[target_val_index]


def natural_direct_effect_scm(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    cause: str,
    x0: Any,
    x1: Any,
) -> float:
    """Compute the natural direct effect via SCM graph mutations (do-calculus).

    NDE(x0,x1,y) = P(Y @ {X:x1, W: W @ {X:x0}} == y) - P(Y @ {X:x0} == y)

    We evaluate the nested counterfactual by constructing a modified Bayesian
    Network where the mediator's CPD is restricted to the baseline cause (X=x0).
    Then, we compute the interventional probability do(X=x1) on this mutated SCM.

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        cause: The name of the private variable whose effect we want to measure.
        x0: The baseline value of the private variable.
        x1: The modified value of the private variable.

    Returns:
        The natural direct effect.
    """
    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    cause_labels = bn.get_cpds(cause).state_names[cause]
    if x0 not in cause_labels or x1 not in cause_labels:
        raise ValueError("x0 and x1 must be valid states of the cause variable.")

    # Restricting to one mediator per the requested assumptions
    mediators = filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
    if len(mediators) != 1:
        raise NotImplementedError(
            "This SCM implementation currently assumes exactly one mediator."
        )
    mediator = mediators[0]

    logger.debug(
        f"Computing NDE (SCM) for target={target}, cause_baseline={x0}, cause_mod={x1}"
    )

    # NDE second term
    ve_orig = CausalInference(bn)
    p_target_do_x0 = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve_orig,
        target_var=target_var,
        private_var=cause,
        private_val=x0,
    )

    # Construct Counterfactual SCM for the nested term: P(Y @ {X:x1, W: W @ {X:x0}})
    bn_mod = bn.copy()

    # Extract the mediator's CPD and "reduce" it by permanently asserting X=x0
    # This computationally yields P(W | Z, X=x0)
    w_cpd_counterfactual = bn_mod.get_cpds(mediator).copy()
    w_cpd_counterfactual.reduce([(cause, x0)])

    # Sever the causal link between the sensitive attribute and the mediator
    bn_mod.remove_edge(cause, mediator)

    # Overwrite the original mediator mechanism with the counterfactual one
    bn_mod.remove_cpds(mediator)
    bn_mod.add_cpds(w_cpd_counterfactual)

    # Compute do(X=x1) on the nested modified SCM
    ve_mod = CausalInference(bn_mod)
    p_target_do_x1_w_x0 = _estimate_target_prob_by_adjustment(
        bn=bn_mod,
        ie=ve_mod,
        target_var=target_var,
        private_var=cause,
        private_val=x1,
    )

    nde_dist = p_target_do_x1_w_x0 - p_target_do_x0
    nde_val = nde_dist[target_val_index]

    return nde_val


def natural_indirect_effect_scm(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    cause: str,
    x0: Any,
    x1: Any,
) -> float:
    """Compute the natural indirect effect via SCM graph mutations (do-calculus).

    NIE(x0,x1,y) = P(Y @ {X:x0, W: W @ {X:x1}} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        cause: The name of the private variable whose effect we want to measure.
        x0: The baseline value of the private variable.
        x1: The modified value of the private variable.

    Returns:
        The natural indirect effect.
    """
    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    cause_labels = bn.get_cpds(cause).state_names[cause]
    if x0 not in cause_labels or x1 not in cause_labels:
        raise ValueError("x0 and x1 must be valid states of the cause variable.")

    mediators = filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
    if len(mediators) != 1:
        raise NotImplementedError(
            "This SCM implementation currently assumes exactly one mediator."
        )
    mediator = mediators[0]

    logger.debug(
        f"Computing NIE (SCM) for target={target}, cause_baseline={x0}, cause_mod={x1}"
    )

    # Compute baseline term
    ve_orig = CausalInference(bn)
    p_target_do_x0 = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve_orig,
        target_var=target_var,
        private_var=cause,
        private_val=x0,
    )

    # Construct Counterfactual SCM for the nested term
    bn_mod = bn.copy()

    w_cpd_counterfactual = bn_mod.get_cpds(mediator).copy()
    w_cpd_counterfactual.reduce([(cause, x1)])

    bn_mod.remove_edge(cause, mediator)

    bn_mod.remove_cpds(mediator)
    bn_mod.add_cpds(w_cpd_counterfactual)

    ve_mod = CausalInference(bn_mod)
    p_target_do_x0_w_x1 = _estimate_target_prob_by_adjustment(
        bn=bn_mod,
        ie=ve_mod,
        target_var=target_var,
        private_var=cause,
        private_val=x0,
    )

    nie_dist = p_target_do_x0_w_x1 - p_target_do_x0
    nie_val = nie_dist[target_val_index]

    return nie_val


def natural_direct_effect_sym(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
) -> float:
    """Compute the symmetric natural direct effect.

    NDE_sym(x0,x1,y) = 0.5 * (NDE(x0,x1,y) + NDE(x1,x0,y))

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.

    Returns:
        The symmetric natural direct effect.
    """
    private_attr_domain = bn.get_cpds(private_attr).state_names[private_attr]
    if len(private_attr_domain) != 2:
        raise ValueError(
            "Symmetric NDE computation currently assumes binary private variable."
        )

    nde_x0_x1 = natural_direct_effect(
        bn, target, private_attr, private_attr_domain[0], private_attr_domain[1]
    )
    nde_x1_x0 = natural_direct_effect(
        bn, target, private_attr, private_attr_domain[1], private_attr_domain[0]
    )

    return 0.5 * (nde_x0_x1 + nde_x1_x0)


def natural_indirect_effect_sym(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
) -> float:
    """Compute the symmetric natural indirect effect.

    NIE_sym(x0,x1,y) = 0.5 * (NIE(x0,x1,y) + NIE(x1,x0,y))

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.

    Returns:
        The symmetric natural indirect effect.
    """
    private_attr_domain = bn.get_cpds(private_attr).state_names[private_attr]
    if len(private_attr_domain) != 2:
        raise ValueError(
            "Symmetric NIE computation currently assumes binary private variable."
        )

    nie_x0_x1 = natural_indirect_effect(
        bn, target, private_attr, private_attr_domain[0], private_attr_domain[1]
    )
    nie_x1_x0 = natural_indirect_effect(
        bn, target, private_attr, private_attr_domain[1], private_attr_domain[0]
    )

    return 0.5 * (nie_x0_x1 + nie_x1_x0)


############################################################
######### Partitioned Effect Wrappers ######################
############################################################


class EffectResult:
    def __init__(
        self,
        effect_name: str,
        x0_states: list[Any],
        x1_states: list[Any],
        effect_matrix: np.ndarray,
    ):
        self.effect_name = effect_name  # e.g., "Total Effect", "Direct Effect"
        self.x0_states = x0_states
        self.x1_states = x1_states
        self.matrix = effect_matrix

    def get_effect(self, x0: Any, x1: Any) -> np.ndarray | float:
        """Get the effect value for a specific pair of states."""
        try:
            i = self.x0_states.index(x0)
            j = self.x1_states.index(x1)
            return self.matrix[:, i, j]
        except ValueError as exc:
            raise ValueError(
                f"States x0={x0} or x1={x1} not found in the respective state lists. Available x0 states: {self.x0_states}, Available x1 states: {self.x1_states}"
            ) from exc

    def mean_effect(self) -> float:
        # return np.mean(self.matrix)
        return np.sum(self.matrix) / (len(self.x0_states) * len(self.x1_states))

    def variance_effect(self) -> float:
        return np.var(self.matrix)

    def max_disparity(self) -> tuple[float, Any, Any]:
        idx = np.unravel_index(np.argmax(np.abs(self.matrix)), self.matrix.shape)
        return self.matrix[idx], self.x0_states[idx[0]], self.x1_states[idx[1]]

    def __repr__(self):
        return f"<{self.effect_name} Analysis | Mean: {self.mean_effect():.4f} | Variance: {self.variance_effect():.4f}>"

    def get_stepwise_effects(self) -> dict[str, float]:
        """Extracts the adjacent step effects: TE(x_k, x_{k+1})."""
        # np.diag with k=1 extracts the first superdiagonal
        ordered_states = self.x0_states
        stepwise_values = np.diag(self.matrix, k=1)

        steps = {}
        for i in range(len(ordered_states) - 1):
            x_k = ordered_states[i]
            x_k1 = ordered_states[i + 1]
            steps[f"{x_k} -> {x_k1}"] = stepwise_values[i]

        return steps

    def find_sign_reversals(self) -> list[str]:
        """Detects if the causal effect changes direction along the ordinal scale."""
        steps = self.get_stepwise_effects()
        values = list(steps.values())
        keys = list(steps.keys())

        reversals = []
        for i in range(len(values) - 1):
            if (
                np.sign(values[i]) != np.sign(values[i + 1])
                and values[i] != 0
                and values[i + 1] != 0
            ):
                reversals.append(f"Reversal between ({keys[i]}) and ({keys[i + 1]})")

        return reversals


def categorical_effect_full_distribution(
    effect_fn: Callable[
        [DiscreteBayesianNetwork, tuple[str, Any], str, list[Any], list[Any]],
        EffectResult,
    ],
    bn: DiscreteBayesianNetwork,
    target: str,
    private_attr: str,
    x0: list[Any],
    x1: list[Any],
) -> EffectResult:
    """Compute the full distribution of the effect for all values of the target variable.

    This is useful for understanding how the entire distribution of the target variable shifts
    in response to changes in the private variable, rather than just looking at a single
    target value. It can reveal if certain outcomes become more likely while others become
    less likely, providing a more comprehensive picture of the causal impact.

    Args:
        effect_fn: The effect function to compute (e.g., total_effect, total_variation).
        bn: The Bayesian Network to use for inference.
        target: The name of the target variable.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: The baseline values of the private variable.
        x1: The modified values of the private variable.

    Returns:
        A dictionary mapping each value of the target variable to its corresponding effect value.
    """
    target_labels = bn.get_cpds(target).state_names[target]
    effect_distribution = []
    for target_val in target_labels:
        effect_value = effect_fn(bn, (target, target_val), private_attr, x0, x1)
        effect_distribution.append((target_val, effect_value.matrix))

    distribution_result = EffectResult(
        effect_name=f"{getattr(effect_fn, '__name__', 'effect')} Distribution",
        x0_states=x0,
        x1_states=x1,
        effect_matrix=np.array([effect for _, effect in effect_distribution]),
    )
    return distribution_result


def categorical_total_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0_set: list[Any],
    x1_set: list[Any],
) -> EffectResult:
    """Compute total effect for all pairs of states in x0_set and x1_set.

    TE(x0,x1,y) = P(Y @ {X:x1} == y) - P(Y @ {X:x0} == y)

    Note:
        - If dealing with a binary variable, simply passing a list of the two states
            for x0_set and x1_set will yield the full TE matrix.
        - For multi-valued categorical variables, this allows us to compute the TE
            for multiple pairs of states, which can be useful for identifying which
            specific state changes are most impactful on the target variable
        - For multi-valued ordinal variables, this allows us to compute the TE for adjacent
            state changes (e.g., x0_set=[low, medium], x1_set=[medium, high]) to analyze
            the effect of incremental changes in the private variable. In this case, pass
            the ordered states both in x0 and x1. The stepwise effects are in the superdiagonal
            of the matrix (`np.diag(res_te.matrix, k=1)`). The end-to-end effect is on the
            top-right corner of the matrix (`res_te.matrix[0, -1]`).

    Args:
        bn: The Bayesian Network.
        target: Tuple of (variable, value) for the target.
        private_attr: The sensitive attribute variable name.
        x0_set: List of baseline states of the private variable.
        x1_set: List of modified states of the private variable.

    Returns:
        An EffectResult object containing the matrix of total effects for all pairs of states.
    """
    matrix = np.zeros((len(x0_set), len(x1_set)))
    for i, x0 in enumerate(x0_set):
        for j, x1 in enumerate(x1_set):
            te = total_effect(
                bn=bn,
                target=target,
                private_attr=private_attr,
                x0=x0,
                x1=x1,
            )
            matrix[i, j] = te

    res = EffectResult("Total Effect", x0_set, x1_set, matrix)
    return res


def categorical_total_variation(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0_set: list[Any],
    x1_set: list[Any],
) -> EffectResult:
    """Compute total variation for all pairs of states in x0_set and x1_set.

    TV(x0,x1,y) = 0.5 * sum_{y'} |P(Y=y' @ {X:x1}) - P(Y=y' @ {X:x0})|

    Args:
        bn: The Bayesian Network.
        target: Tuple of (variable, value) for the target.
        private_attr: The sensitive attribute variable name.
        x0_set: List of baseline states of the private variable.
        x1_set: List of modified states of the private variable.

    Returns:
        An EffectResult object containing the matrix of total variations for all pairs of states.
    """
    matrix = np.zeros((len(x0_set), len(x1_set)))

    pairs = list(product(x0_set, x1_set))
    for x0, x1 in pairs:
        tv = total_variation(
            bn=bn,
            target=target,
            private_attr=private_attr,
            x0=x0,
            x1=x1,
        )
        matrix[x0_set.index(x0), x1_set.index(x1)] = tv

    res = EffectResult("Total Variation", x0_set, x1_set, matrix)
    return res


def categorical_natural_direct_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: list[Any],
    x1: list[Any],
) -> EffectResult:
    """Compute the natural direct effect for categorical variables.

    NDE(x0,x1,y) = P(Y @ {X:x1, W: W @ {X:x0}} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: List of baseline values of the private variable.
        x1: List of modified values of the private variable.

    Returns:
        An EffectResult object containing the NDE value for the given pair of states.
    """
    # This implementation currently assumes exactly one mediator.
    mediators = filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
    if len(mediators) != 1:
        raise NotImplementedError(
            "This SCM implementation currently assumes exactly one mediator."
        )

    matrix = np.zeros((len(x0), len(x1)))

    pairs = list(product(x0, x1))
    for x0_val, x1_val in pairs:
        nde = natural_direct_effect(
            bn=bn,
            target=target,
            private_attr=private_attr,
            x0=x0_val,
            x1=x1_val,
        )

        matrix[x0.index(x0_val), x1.index(x1_val)] = nde

    res = EffectResult("Natural Direct Effect", x0, x1, matrix)
    return res


def categorical_natural_indirect_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: list[Any],
    x1: list[Any],
) -> EffectResult:
    """Compute the natural indirect effect for categorical variables.

    NIE(x0,x1,y) = P(Y @ {X:x0, W: W @ {X:x1}} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: List of baseline values of the private variable.
        x1: List of modified values of the private variable.

    Returns:
        An EffectResult object containing the NIE value for the given pair of states.
    """

    # This implementation currently assumes exactly one mediator.
    mediators = filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
    if len(mediators) != 1:
        raise NotImplementedError(
            "This SCM implementation currently assumes exactly one mediator."
        )

    matrix = np.zeros((len(x0), len(x1)))

    pairs = list(product(x0, x1))
    for x0_val, x1_val in pairs:
        nie = natural_indirect_effect(
            bn=bn,
            target=target,
            private_attr=private_attr,
            x0=x0_val,
            x1=x1_val,
        )

        matrix[x0.index(x0_val), x1.index(x1_val)] = nie

    res = EffectResult("Natural Indirect Effect", x0, x1, matrix)
    return res


# TODO: decomposition of effects
def decompose_indirect_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: Any,
    x1: Any,
) -> dict[str, float]:
    """Decompose the indirect effect into contributions from different mediators.

    This function is a placeholder for future implementation. The idea is to analyze how much of the indirect effect can be attributed to each mediator in the causal graph. This can provide insights into which pathways are most responsible for the observed effect and can inform targeted interventions.

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: The baseline value of the private variable.
        x1: The modified value of the private variable.

    Returns:
        A dictionary mapping each mediator to its contribution to the indirect effect.
    """
    mediators = filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
    sorted_mediators: list[str] = list(nx.topological_sort(bn.subgraph(mediators)))

    if len(sorted_mediators) == 0:
        raise ValueError("No mediators found in the Bayesian Network.")
    if len(sorted_mediators) == 1:
        # If there's only one mediator, the entire indirect effect is attributed to it
        nie = natural_indirect_effect(
            bn=bn,
            target=target,
            private_attr=private_attr,
            x0=x0,
            x1=x1,
        )
        return {sorted_mediators[0]: nie}

    contributions = {}
    for i, mediator in enumerate(sorted_mediators):
        w_A = sorted_mediators[:i]
        w_B = sorted_mediators[: i + 1]

        logger.debug(
            f"Computing contribution of mediator {mediator} with W_A={w_A} and W_B={w_B}"
        )

        nie_prec_succ = set_specific_indirect_effect(
            bn=bn,
            target=target,
            private_attr=private_attr,
            x0=x0,
            x1=x1,
            first_mediator_partition=w_A,
            second_mediator_partition=w_B,
        )

        contributions[mediator] = nie_prec_succ

    return contributions


def set_specific_indirect_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: Any,
    x1: Any,
    first_mediator_partition: list[Any],  # WA
    second_mediator_partition: list[Any],  # WB
) -> float | np.ndarray:
    """Compute the set-specific indirect effect.

    See definition 6.12 pp. 160-161 of "Causal Fairness Analysis" by Plecko D. & Bareinboim E. (2024).

    Note:
        If no order has been explicitly defined in the SFM construction, we topologically
        sort the bn.digraph and use that as the default order of the mediators for the partitioning.

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: The baseline value of the private variable.
        x1: The modified value of the private variable.
        first_mediator_partition: The partition of mediator states for the first term (W_A).
        second_mediator_partition: The partition of mediator states for the second term (W_B).
    Returns:
        The set-specific indirect effect.
    """
    ordered_mediators = list(
        nx.topological_sort(
            bn.subgraph(
                filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
            )
        )
    )
    logger.debug(
        f"""Computing set-specific indirect effect for
            target={target},
            private_baseline={x0},
            private_mod={x1},
            first_mediator_partition={first_mediator_partition},
            second_mediator_partition={second_mediator_partition},
            ordered_mediators={ordered_mediators}"""
    )

    ve = CausalInference(bn)
    y, y_val = target
    y_domain = bn.get_cpds(y).state_names[y]

    w_a = first_mediator_partition
    w_a_domains = {m: bn.get_cpds(m).state_names[m] for m in w_a}
    w_ac = [m for m in ordered_mediators if m not in first_mediator_partition]
    w_ac_domains = {m: bn.get_cpds(m).state_names[m] for m in w_ac}

    w_b = second_mediator_partition
    w_b_domains = {m: bn.get_cpds(m).state_names[m] for m in w_b}
    w_bc = [m for m in ordered_mediators if m not in second_mediator_partition]
    w_bc_domains = {m: bn.get_cpds(m).state_names[m] for m in w_bc}

    z = filter_nodes_by_type(bn.nodes(data=True, default={}), "confounder")
    z_domains = {c: bn.get_cpds(c).state_names[c] for c in z}
    z_domains_cartesian = list(product(*z_domains.values())) if z else [()]
    if z:
        p_z_factor = ve.query(variables=z, joint=True, show_progress=False)
    else:
        p_z_factor = None

    ### First term
    # TODO: create a function
    acc1 = np.zeros(len(y_domain))
    for z_val in z_domains_cartesian:
        p_z = (
            p_z_factor.get_value(**dict(zip(z_domains.keys(), z_val, strict=True)))
            if z
            else 1.0
        )

        z_evidence = dict(zip(z_domains.keys(), z_val, strict=True)) if z else {}

        # P(W_B | x1, z)
        w_b_evidence = {private_attr: x1, **z_evidence}
        p_w_b_given_x1_z = ve.query(
            variables=w_b,
            evidence=w_b_evidence,
            show_progress=False,
        )

        # P(W_BC | x0, z)
        w_bc_evidence = {private_attr: x0, **z_evidence}
        p_w_bc_given_x0_z = ve.query(
            variables=w_bc,
            evidence=w_bc_evidence,
            show_progress=False,
        )

        all_w_domains = list(w_b_domains.values()) + list(w_bc_domains.values())
        for combined_w_vals in product(*all_w_domains):
            w_b_val_tuple = combined_w_vals[: len(w_b_domains.keys())]
            w_bc_val_tuple = combined_w_vals[len(w_b_domains.keys()) :]

            w_b_evidence = dict(zip(w_b_domains.keys(), w_b_val_tuple))
            w_bc_evidence = dict(zip(w_bc_domains.keys(), w_bc_val_tuple))

            y_evidence = {
                **w_b_evidence,
                **w_bc_evidence,
                private_attr: x0,
                **z_evidence,
            }

            # P(Y | x0, W_B, W_BC, Z)
            p_y_given_x0_w_b_z = ve.query(
                variables=[y],
                evidence=y_evidence,
                show_progress=False,
            )

            # Get the scalar probabilities for the mediator states
            prob_w_b = (
                p_w_b_given_x1_z.get_value(**w_b_evidence) if p_w_b_given_x1_z else 1.0
            )
            prob_w_bc = (
                p_w_bc_given_x0_z.get_value(**w_bc_evidence)
                if p_w_bc_given_x0_z
                else 1.0
            )

            acc1 += p_y_given_x0_w_b_z.values * prob_w_b * prob_w_bc * p_z

    # Second term
    acc2 = np.zeros(len(y_domain))
    for z_val in z_domains_cartesian:
        p_z = (
            p_z_factor.get_value(**dict(zip(z_domains.keys(), z_val, strict=True)))
            if z
            else 1.0
        )

        z_evidence = dict(zip(z_domains.keys(), z_val, strict=True)) if z else {}

        # P(W_A | x1, z)
        w_a_evidence = {private_attr: x1, **z_evidence}
        p_w_a_given_x1_z = ve.query(
            variables=w_a,
            evidence=w_a_evidence,
            show_progress=False,
        )

        # P(W_AC | x0, z)
        w_ac_evidence = {private_attr: x0, **z_evidence}
        p_w_ac_given_x0_z = ve.query(
            variables=w_ac,
            evidence=w_ac_evidence,
            show_progress=False,
        )

        all_w_domains = list(w_a_domains.values()) + list(w_ac_domains.values())
        for combined_w_vals in product(*all_w_domains):
            w_a_val_tuple = combined_w_vals[: len(w_a_domains.keys())]
            w_ac_val_tuple = combined_w_vals[len(w_a_domains.keys()) :]

            w_a_evidence = dict(zip(w_a_domains.keys(), w_a_val_tuple, strict=True))
            w_ac_evidence = dict(zip(w_ac_domains.keys(), w_ac_val_tuple, strict=True))

            y_evidence = {
                **w_a_evidence,
                **w_ac_evidence,
                private_attr: x0,
                **z_evidence,
            }

            # P(Y | x0, W_A, W_AC, Z)
            p_y_given_x0_w_a_z = ve.query(
                variables=[y],
                evidence=y_evidence,
                show_progress=False,
            )

            # Get the scalar probabilities for the mediator states
            prob_w_a = (
                p_w_a_given_x1_z.get_value(**w_a_evidence) if p_w_a_given_x1_z else 1.0
            )
            prob_w_ac = (
                p_w_ac_given_x0_z.get_value(**w_ac_evidence)
                if p_w_ac_given_x0_z
                else 1.0
            )

            acc2 += p_y_given_x0_w_a_z.values * prob_w_a * prob_w_ac * p_z

    return acc1 - acc2


############################################################
################## Utility Functions #######################
############################################################


def _estimate_target_prob_by_adjustment(
    bn: DiscreteBayesianNetwork,
    ie,
    target_var: str,
    private_var: str,
    private_val: Any,
) -> np.ndarray:
    """Compute P(target_var @ {private_var: x}) by adjustment + law of total probability.

    If Z is a minimal adjustment set for (private_var, target_var), then:
    P(Y @ {X:x} == y) = sum_z P(Y | X=x, Z=z) P(Z=z)
    """
    adj_set = filter_nodes_by_type(bn.nodes(data=True, default={}), "confounder")

    # Base case: No confounders
    if not adj_set:
        return ie.query(
            variables=[target_var],
            evidence={private_var: private_val},
            show_progress=False,
            joint=True,
        ).values

    factor_z = ie.query(variables=adj_set, joint=True, show_progress=False)

    # P(Y | X=x, Z) = P(Y, Z | X=x) / P(Z | X=x)
    factor_y_z_given_x = ie.query(
        variables=[target_var] + adj_set,
        evidence={private_var: private_val},
        joint=True,
        show_progress=False,
    )
    factor_z_given_x = ie.query(
        variables=adj_set,
        evidence={private_var: private_val},
        joint=True,
        show_progress=False,
    )
    factor_y_given_x_z = factor_y_z_given_x / factor_z_given_x

    # 3. Multiply and Marginalize: sum_z P(Y | X=x, Z) * P(Z)
    adjusted_factor = factor_y_given_x_z * factor_z
    adjusted_factor.marginalize(adj_set)

    return adjusted_factor.values
