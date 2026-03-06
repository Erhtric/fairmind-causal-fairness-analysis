from itertools import product
from typing import Any

from loguru import logger
from pgmpy.inference import CausalInference, VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from src.graph import filter_nodes_by_type

############################################################
######### Effect computation via ID Expr on SFM ############
############################################################


def _estimate_target_prob_by_adjustment(
    bn: DiscreteBayesianNetwork,
    ie,
    target_var: str,
    target_val_index: int,
    cause: str,
    x: Any,
) -> float:
    """Compute P(target_var @ {cause: x}) by adjustment + law of total probability.

    If Z is a minimal adjustment set for (cause, target_var), then:
    P(Y @ {X:x} == y) = sum_z P(Y | X=x, Z=z) P(Z=z)

    Args:
        bn: The Bayesian Network to use for inference.
        ie: A inference engine object initialized with the Bayesian Network.
        target_var: The name of the target variable.
        target_val_index: The index of the target variable's value in its CPD state names
        cause: The name of the private variable whose effect we want to measure.
        x: A tuple of (variable, value) representing the value of the private variable.

    Returns:
        The interventional probability P(target_var @ {cause: x}).
    """

    adj_set = ie.get_minimal_adjustment_set(cause, target_var)
    if not adj_set:
        return ie.query(
            variables=[target_var],
            evidence={cause: x},
            show_progress=False,
        ).values[target_val_index]

    adj_set_domains = {var: bn.get_cpds(var).state_names[var] for var in adj_set}
    adj_set_domains_cartesian = list(product(*adj_set_domains.values()))
    p_z = ie.query(variables=adj_set, joint=True, show_progress=False)

    acc = 0.0
    for adj_values in adj_set_domains_cartesian:
        evidence = dict(zip(adj_set_domains.keys(), adj_values, strict=True))
        evidence_x = {**evidence, cause: x}

        p_target_given_x_z = ie.query(
            variables=[target_var],
            evidence=evidence_x,
            show_progress=False,
        ).values[target_val_index]
        p_z_val = p_z.get_value(**evidence)
        acc += p_target_given_x_z * p_z_val

    return acc


def total_effect(
    bn: DiscreteBayesianNetwork,
    target: tuple[str, Any],
    private_attr: str,
    x0: Any,
    x1: Any,
) -> float:
    """Compute the total effect on the SFM.

    TE(x0,x1,y) = P(Y @ {X:x1} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        private_attr: The private/sensitive attribute to intervene on.
        x0: A tuple of (variable, value) representing the baseline value of the private variable.
        x1: A tuple of (variable, value) representing the modified value of the private variable.
    Returns:
        The total effect.
    """
    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    sensitive_nodes = filter_nodes_by_type(bn.nodes(data=True, default={}), "sensitive")
    if sensitive_nodes and private_attr not in sensitive_nodes:
        raise ValueError(
            f"private_attr='{private_attr}' is not marked as sensitive in the graph."
        )

    private_attr_domain = bn.get_cpds(private_attr).state_names[private_attr]

    if x0 not in private_attr_domain or x1 not in private_attr_domain:
        raise ValueError("x0 and x1 must be valid states of the private variable.")
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
        target_val_index=target_val_index,
        cause=private_attr,
        x=x1,
    )
    p_target_do_x0 = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        target_val_index=target_val_index,
        cause=private_attr,
        x=x0,
    )

    return p_target_do_x1 - p_target_do_x0


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
        target_val_index=target_val_index,
        cause=private_attr,
        x=x,
    )

    P_target_given_x = ve.query(
        variables=[target_var],
        evidence={private_attr: x},
        show_progress=False,
    ).values[target_val_index]

    return P_target_given_x - P_target_do_x


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
    ).values[target_val_index]

    P_target_given_x0 = ve.query(
        variables=[target_var],
        evidence={private_attr: x0},
        show_progress=False,
    ).values[target_val_index]

    return P_target_given_x1 - P_target_given_x0


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
        f"Computing total effect for target={target}, private_baseline={x0}, private_mod={x1}"
    )

    logger.debug(f"Confounders: {confounders}, Mediators: {mediators}")

    ve = CausalInference(bn)

    # Compute first term of NDE: sum_{z,w} P(Y | X=x1, Z=z, W=w) P(W | X=x0, Z=z) P(Z=z)
    acc = 0.0
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
        ).values[target_val_index]

        evidence_second_term = {**z_evidence, private_attr: x0}
        w_val_index = mediators_domains[mediators[0]].index(w_val[0])
        p_w_given_x0_z = ve.query(
            variables=mediators,
            evidence=evidence_second_term,
            show_progress=False,
        ).values[w_val_index]

        z_val_index = confounders_domains[confounders[0]].index(z_val[0])
        p_z = ve.query(
            variables=confounders,
            show_progress=False,
        ).values[z_val_index]

        acc += p_y_given_x1_w_z * p_w_given_x0_z * p_z

    # Compute second term of NDE: P(Y | X=x0) = sum_z P(Y | X=x0, Z=z) P(Z=z)
    second_term = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        target_val_index=target_val_index,
        cause=private_attr,
        x=x0,
    )

    logger.debug(f"NDE first term: {acc}, second term: {second_term}")

    return acc - second_term


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
        f"Computing total effect for target={target}, private_baseline={x0}, private_mod={x1}"
    )

    logger.debug(f"Confounders: {confounders}, Mediators: {mediators}")
    ve = CausalInference(bn)

    # Compute first term of NIE: sum_{z,w} P(Y | X=x0, Z=z, W=w) P(W | X=x1, Z=z) P(Z=z)
    acc = 0.0
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
        ).values[target_val_index]

        evidence_second_term = {**z_evidence, private_attr: x1}
        w_val_index = mediators_domains[mediators[0]].index(w_val[0])
        p_w_given_x1_z = ve.query(
            variables=mediators,
            evidence=evidence_second_term,
            show_progress=False,
        ).values[w_val_index]

        z_val_index = confounders_domains[confounders[0]].index(z_val[0])
        p_z = ve.query(
            variables=confounders,
            show_progress=False,
        ).values[z_val_index]

        acc += p_y_given_x0_w_z * p_w_given_x1_z * p_z

    # Compute second term of NIE: P(Y | X=x0) = sum_z P(Y | X=x0, Z=z) P(Z=z)
    second_term = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        target_val_index=target_val_index,
        cause=private_attr,
        x=x0,
    )

    logger.debug(f"NIE first term: {acc}, second term: {second_term}")

    return acc - second_term


############################################################
############## Effect computation via SCM ##################
############################################################


def total_effect_scm(
    bn: DiscreteBayesianNetwork, target: tuple[str, Any], cause: str, x0: Any, x1: Any
) -> float:
    """Compute the total effect.

    TE(x0,x1,y) = P(Y @ {X:x1} == y) - P(Y @ {X:x0} == y)

    Args:
        bn: The Bayesian Network to use for inference.
        target: A tuple of (variable, value) for the target variable and its value.
        cause: The name of the private variable whose effect we want to measure.
        x0: A tuple of (variable, value) representing the baseline value of the private variable.
        x1: A tuple of (variable, value) representing the modified value of the private variable.
    Returns:
        The total effect.
    """
    target_var, target_val = target
    target_labels = bn.get_cpds(target_var).state_names[target_var]
    target_val_index = target_labels.index(target_val)

    cause_labels = bn.get_cpds(cause).state_names[cause]
    if x0 not in cause_labels or x1 not in cause_labels:
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
        target_val_index=target_val_index,
        cause=cause,
        x=x1,
    )

    p_target_do_x0 = _estimate_target_prob_by_adjustment(
        bn=bn,
        ie=ve,
        target_var=target_var,
        target_val_index=target_val_index,
        cause=cause,
        x=x0,
    )

    return p_target_do_x1 - p_target_do_x0


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
        target_val_index=target_val_index,
        cause=cause,
        x=x,
    )

    P_target_given_x = ve.query(
        variables=[target_var],
        evidence={cause: x},
        show_progress=False,
    ).values[target_val_index]

    return P_target_given_x - P_target_do_x


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
        target_val_index=target_val_index,
        cause=cause,
        x=x0,
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
        target_val_index=target_val_index,
        cause=cause,
        x=x1,
    )

    return p_target_do_x1_w_x0 - p_target_do_x0


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
        target_val_index=target_val_index,
        cause=cause,
        x=x0,
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
        target_val_index=target_val_index,
        cause=cause,
        x=x0,
    )

    return p_target_do_x0_w_x1 - p_target_do_x0
