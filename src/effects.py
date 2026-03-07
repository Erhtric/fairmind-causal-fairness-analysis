from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np
from loguru import logger
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import CausalInference, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.graph import filter_nodes_by_type

############################################################
######### Effect computation via ID Expr on SFM ############
############################################################


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
    private_attr_domain = bn.get_cpds(private_attr).state_names[private_attr]

    if target_val not in target_labels:
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
        x0: A tuple of (variable, value) representing the baseline value of the private variable.
        x1: A tuple of (variable, value) representing the modified value of the private variable.
        mediator_attrs: A list of mediator variable names.
    Returns:
        The natural direct effect.
    """
    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    cause_labels = bn.get_cpds(private_attr).state_names[private_attr]

    mediators = filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
    mediators_domains = {m: bn.get_cpds(m).state_names[m] for m in mediators}
    mediators_domains_cartesian = list(product(*mediators_domains.values()))

    confounders = filter_nodes_by_type(bn.nodes(data=True, default={}), "confounder")
    confounders_domains = {c: bn.get_cpds(c).state_names[c] for c in confounders}
    confounders_domains_cartesian = list(product(*confounders_domains.values()))

    if len(confounders) > 1:
        raise NotImplementedError(
            "NDE computation with multiple confounders is not implemented yet."
        )
    if len(mediators) > 1:
        raise NotImplementedError(
            "NDE computation with multiple mediators is not implemented yet."
        )

    if x0 not in cause_labels or x1 not in cause_labels:
        raise ValueError("x0 and x1 must be valid states of the cause variable.")
    if target_val not in target_labels:
        raise ValueError("target value must be a valid state of the target variable.")

    logger.debug(
        f"Computing natural direct effect for target={target}, private_baseline={x0}, private_mod={x1}"
    )

    ve = CausalInference(bn)

    # Compute first term of NDE: sum_{z,w} P(Y | X=x1, Z=z, W=w) P(W | X=x0, Z=z) P(Z=z)
    query_vars_card = [
        len(bn.get_cpds(var).state_names[var])
        for var in [target_var] + mediators + confounders
    ]
    acc = DiscreteFactor(
        variables=[target_var] + mediators + confounders,
        cardinality=query_vars_card,
        values=np.zeros(query_vars_card),
    )
    for z_val, w_val in product(
        confounders_domains_cartesian, mediators_domains_cartesian
    ):
        z_evidence = dict(zip(confounders_domains.keys(), z_val, strict=True))
        w_evidence = dict(zip(mediators_domains.keys(), w_val, strict=True))

        evidence_first_term = {**z_evidence, **w_evidence, private_attr: x1}
        p_y_given_x1_w_z = ve.query(
            variables=[target_var],
            evidence=evidence_first_term,
            show_progress=False,
        )

        evidence_second_term = {**z_evidence, private_attr: x0}
        p_w_given_x0_z = ve.query(
            variables=mediators,
            evidence=evidence_second_term,
            show_progress=False,
        )

        p_z = ve.query(
            variables=confounders,
            show_progress=False,
        )

        acc: DiscreteFactor = acc + (p_y_given_x1_w_z * p_w_given_x0_z * p_z)

    acc.marginalize(
        variables=[v for v in acc.variables if v != target_var], inplace=True
    )
    acc.normalize(inplace=True)

    # Compute second term of NDE: P(Y | X=x0) = sum_z P(Y | X=x0, Z=z) P(Z=z)
    second_term = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        private_var=private_attr,
        private_val=x0,
    )

    nde_dist = acc.values - second_term
    nde_val = nde_dist[target_val_index]

    return nde_val


def natural_indirect_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: Any,
    x1: Any,
) -> float:
    """Compute the natural indirect effect.

    NIE(x0,x1,y) = P(Y @ {X:x0, W: W @ {X:x1}} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The name of the private variable whose effect we want to measure.
        x0: A tuple of (variable, value) representing the baseline value of the private variable.
        x1: A tuple of (variable, value) representing the modified value of the private variable.
        mediator_attrs: A list of mediator variable names.
    Returns:
        The natural indirect effect.
    """
    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    cause_labels = bn.get_cpds(private_attr).state_names[private_attr]

    mediators = filter_nodes_by_type(bn.nodes(data=True, default={}), "mediator")
    mediators_domains = {m: bn.get_cpds(m).state_names[m] for m in mediators}
    mediators_domains_cartesian = list(product(*mediators_domains.values()))

    confounders = filter_nodes_by_type(bn.nodes(data=True, default={}), "confounder")
    confounders_domains = {c: bn.get_cpds(c).state_names[c] for c in confounders}
    confounders_domains_cartesian = list(product(*confounders_domains.values()))

    if len(confounders) > 1:
        raise NotImplementedError(
            "NDE computation with multiple confounders is not implemented yet."
        )
    if len(mediators) > 1:
        raise NotImplementedError(
            "NDE computation with multiple mediators is not implemented yet."
        )

    if x0 not in cause_labels or x1 not in cause_labels:
        raise ValueError("x0 and x1 must be valid states of the cause variable.")
    if target_val not in target_labels:
        raise ValueError("target value must be a valid state of the target variable.")

    logger.debug(
        f"Computing natural indirect effect for target={target}, private_baseline={x0}, private_mod={x1}"
    )
    ve = CausalInference(bn)

    # Compute first term of NIE: sum_{z,w} P(Y | X=x0, Z=z, W=w) P(W | X=x1, Z=z) P(Z=z)
    query_vars_card = [
        len(bn.get_cpds(var).state_names[var])
        for var in [target_var] + mediators + confounders
    ]
    acc = DiscreteFactor(
        variables=[target_var] + mediators + confounders,
        cardinality=query_vars_card,
        values=np.zeros(query_vars_card),
    )
    for z_val, w_val in product(
        confounders_domains_cartesian, mediators_domains_cartesian
    ):
        z_evidence = dict(zip(confounders_domains.keys(), z_val, strict=True))
        w_evidence = dict(zip(mediators_domains.keys(), w_val, strict=True))

        evidence_first_term = {**z_evidence, **w_evidence, private_attr: x0}
        p_y_given_x0_w_z = ve.query(
            variables=[target_var],
            evidence=evidence_first_term,
            show_progress=False,
        )

        evidence_second_term = {**z_evidence, private_attr: x1}
        p_w_given_x1_z = ve.query(
            variables=mediators,
            evidence=evidence_second_term,
            show_progress=False,
        )

        p_z = ve.query(
            variables=confounders,
            show_progress=False,
        )

        acc += p_y_given_x0_w_z * p_w_given_x1_z * p_z

    acc.marginalize(
        variables=[v for v in acc.variables if v != target_var], inplace=True
    )
    acc.normalize(inplace=True)

    # Compute second term of NIE: P(Y | X=x0) = sum_z P(Y | X=x0, Z=z) P(Z=z)
    second_term = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        private_var=private_attr,
        private_val=x0,
    )

    nie_dist = acc.values - second_term
    nie_val = nie_dist[target_val_index]

    return nie_val


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

    def get_effect(self, x0: Any, x1: Any) -> float:
        """Get the effect value for a specific pair of states."""
        try:
            i = self.x0_states.index(x0)
            j = self.x1_states.index(x1)
            return self.matrix[i, j]
        except ValueError:
            raise ValueError(
                f"States x0={x0} or x1={x1} not found in the respective state lists. Available x0 states: {self.x0_states}, Available x1 states: {self.x1_states}"
            )

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

    Y must be a single variable.

    Args:
        bn: The Bayesian Network to use for inference.
        ie: A inference engine object initialized with the Bayesian Network.
        target_var: The name of the target variable.
        private_var: The name of the private variable whose effect we want to measure.
        private_val: The value of the private variable.

    Returns:
        The interventional distribution P(target_var @ {private_var: private_val}).
    """
    adj_set = ie.get_minimal_adjustment_set(private_var, target_var)
    if not adj_set:
        return ie.query(
            variables=[target_var],
            evidence={private_var: private_val},
            show_progress=False,
        ).values

    adj_set_domains = {var: bn.get_cpds(var).state_names[var] for var in adj_set}
    adj_set_domains_cartesian = list(product(*adj_set_domains.values()))
    p_z = ie.query(variables=adj_set, joint=True, show_progress=False)

    acc = np.zeros(len(bn.get_cpds(target_var).state_names[target_var]))
    for adj_values in adj_set_domains_cartesian:
        evidence = dict(zip(adj_set_domains.keys(), adj_values, strict=True))
        evidence_x = {**evidence, private_var: private_val}

        p_target_given_x_z = ie.query(
            variables=[target_var],
            evidence=evidence_x,
            show_progress=False,
        )

        p_z_val = p_z.get_value(**evidence)

        acc += p_target_given_x_z.values * p_z_val

    return acc


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
