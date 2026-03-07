from collections.abc import Iterable
from collections import defaultdict
from typing import Optional

import networkx as nx

from src.sym.dsl import CounterfactualTerm, Variable


def filter_nodes_by_type(
    nodes: nx.classes.reportviews.NodeDataView, node_type: str
) -> list[str]:
    return [
        node_name
        for (node_name, node_data) in nodes
        if node_data.get("type") == node_type
    ]


def has_latent_variables(sfm: nx.DiGraph) -> bool:
    return any(
        node_data.get("type") == "latent" for _, node_data in sfm.nodes(data=True)
    )


def build_sfm(
    sensitive_attr: str,
    outcome_attr: str,
    confounder_attrs: list[str],
    mediator_attrs: list[str],
    latents: Optional[list[tuple[str, list[str]]] | None] = None,
) -> nx.DiGraph:
    """
    Builds a Standard Fairness Model (SFM) template.

    Args:
        sensitive_attr (str): The name of the sensitive attribute.
        outcome_attr (str): The name of the outcome attribute.
        confounder_attrs (list[str]): A list of names of confounder attributes.
        mediator_attrs (list[str]): A list of names of mediator attributes.
        latents (list[Tuple[str, list[str]]]): A list of tuples representing latent variables.
                Each tuple should contain the name of the latent variable and a list of children.
                This extends the basic SFM template to include latent variables that may confound relationships between observed variables.
    Returns:
        nx.DiGraph: A directed graph representing the SFM template.
    """
    if not isinstance(sensitive_attr, str):
        raise ValueError("Sensitive attribute must be a string.")
    if not isinstance(outcome_attr, str):
        raise ValueError("Outcome attribute must be a string.")
    if not isinstance(confounder_attrs, list) or not all(
        isinstance(attr, str) for attr in confounder_attrs
    ):
        raise ValueError("Confounder attributes must be a list of strings.")
    if not isinstance(mediator_attrs, list) or not all(
        isinstance(attr, str) for attr in mediator_attrs
    ):
        raise ValueError("Mediator attributes must be a list of strings.")

    if (
        set(confounder_attrs)
        & set(mediator_attrs)
        & set(sensitive_attr)
        & set(outcome_attr)
    ):
        raise ValueError(
            "Confounder, mediator, sensitive attribute, and outcome attribute sets must be disjoint."
        )

    sfm = nx.DiGraph()
    sfm.add_node(sensitive_attr, type="sensitive", category="endogenous")
    sfm.add_node(outcome_attr, type="outcome", category="endogenous")

    sfm.add_edge(sensitive_attr, outcome_attr)
    for confounder in confounder_attrs:
        sfm.add_node(confounder, type="confounder", category="endogenous")
        sfm.add_edge(confounder, sensitive_attr)
        sfm.add_edge(confounder, outcome_attr)
    for mediator in mediator_attrs:
        sfm.add_node(mediator, type="mediator", category="endogenous")
        sfm.add_edge(sensitive_attr, mediator)
        sfm.add_edge(mediator, outcome_attr)

    for confounder in confounder_attrs:
        for mediator in mediator_attrs:
            sfm.add_edge(confounder, mediator)

    if latents is not None:
        for latent, children in latents:
            sfm.add_node(latent, type="latent", category="latent")
            for child in children:
                sfm.add_edge(latent, child)

    return sfm


############################################################
########### Counterfactual-Calculus ############
############################################################

# TODO: wip
# TODO: to be sorted properly into a separate module due to symbolic dependencies


def get_mutilated_parents_graph(G: nx.DiGraph, interventions: set[str]) -> nx.DiGraph:
    """
    Mutilates the graph G by removing all incoming edges to variables in the intervention set X.

    Args:
        G (nx.DiGraph): The original causal graph.
        interventions (set[str]): A set of variable names that are being intervened on.

    Returns:
        nx.DiGraph: The mutilated graph.
    """
    G_bar = G.copy()
    for node in interventions:
        if node in G_bar:
            in_edges = list(G_bar.in_edges(node))
            G_bar.remove_edges_from(in_edges)
    return G_bar


def get_mutilated_children_graph(G: nx.DiGraph, interventions: set[str]) -> nx.DiGraph:
    """
    Mutilates the graph G by removing all outgoing edges from variables in the intervention set X.

    Args:
        G (nx.DiGraph): The original causal graph.
        interventions (set[str]): A set of variable names that are being intervened on.

    Returns:
        nx.DiGraph: The mutilated graph .
    """
    G_bar = G.copy()
    for node in interventions:
        if node in G_bar:
            out_edges = list(G_bar.out_edges(node))
            G_bar.remove_edges_from(out_edges)
    return G_bar


def get_counterfactual_ancestors(
    term: CounterfactualTerm | Variable | str, G: nx.DiGraph
) -> set[CounterfactualTerm]:
    """
    Computes An(Y_x), the set of counterfactual ancestors of a term.

    See Def. 2.4 of "Counterfactual Graphical Models for Causal Inference" for details.

    Args:
        term (CounterfactualTerm | Variable | str): The term for which to compute counterfactual ancestors.
            Can be a CounterfactualTerm, a Variable, or a string representing the variable name.
        G (nx.DiGraph): The original causal graph.
    """
    if isinstance(term, str):
        Y_name = term
        X_dict = {}
    elif isinstance(term, Variable):
        Y_name = term.name
        X_dict = {}
    elif isinstance(term, CounterfactualTerm):
        Y_name = term.variable.name
        X_dict = term.intervention
    X_names = {v.name for v in X_dict.keys()}

    G_bar = get_mutilated_parents_graph(G, X_names)

    ancestors_Y = nx.ancestors(G_bar, Y_name) | {Y_name}

    cf_ancestors = set()
    for W_name in ancestors_Y:
        # Intervened variables themselves are not counterfactual ancestors
        # of the outcome in their own world.
        if W_name in X_names:
            continue

        ancestors_W = nx.ancestors(G_bar, W_name) | {W_name}

        # Apply Exclusion Operator: z = x \cap An(W)_{G_{\overline{X}}}
        z_dict = {v: val for v, val in X_dict.items() if v.name in ancestors_W}

        W_var = Variable(name=W_name)
        W_term = CounterfactualTerm(variable=W_var, intervention=z_dict)
        cf_ancestors.add(W_term)

    return cf_ancestors


def construct_amwn(
    G: nx.DiGraph,
    Y_star: Iterable[CounterfactualTerm],
    bidirected_edges: Optional[list[tuple[str, str]] | None] = None,
) -> nx.DiGraph:
    """
    Algorithm 1: AMWN-CONSTRUCT(G, Y_*)

    Args:
        G: The original causal diagram.
        Y_star: A set of CounterfactualTerm objects representing the query.
        bidirected_edges: A list of tuples representing latent confounding (U_VW).

    Returns:
        G_A(Y_*): The Ancestral Multi-World Network.

    Usage:
    ```python
    G = nx.DiGraph()
    for var in ["W", "X", "Y", "Z"]:
        G.add_node(var, category="endogenous")

    G.add_edges_from([("W", "Z"), ("Z", "X"), ("X", "Y"), ("Z", "Y")])

    G.add_node("U_ZX", category="latent")
    G.add_edges_from([("U_ZX", "Z"), ("U_ZX", "X")])

    X = Variable("X")
    Y = Variable("Y")
    Z = Variable("Z")
    W = Variable("W")
    U = Variable("U_ZX")

    term_Y_xw = Y @ {X: 0, W: 0}
    term_X = X @ {}  # X factual
    term_Z = Z @ {}
    term_W = W @ {}

    Y_star: list[Unknown | CounterfactualTerm] = [
        term_Y_xw,
        term_X,
        term_Z,
        term_W,
    ]

    G_A = construct_amwn(G, Y_star)

    # Test d-separation in G_A
    nx.is_d_separator(G_A, {term_Y_xw}, {term_X}, {term_Z, term_W})

    # Find minimal d-separator
    nx.find_minimal_d_separator(G_A, {term_Y_xw}, {term_X})
    ```
    """
    if bidirected_edges is None:
        bidirected_edges = []

    G_prime = nx.DiGraph()
    nodes_in_G_prime = set()

    # Line 1: Compute An(Y_*) and add to graph
    for term in Y_star:
        nodes_in_G_prime.update(get_counterfactual_ancestors(term, G))

    for node in nodes_in_G_prime:
        G_prime.add_node(node)

    # Add directed arrows witnessing ancestrality
    for W_term in nodes_in_G_prime:
        W_name = W_term.variable.name
        Z_names = {v.name for v in W_term.intervention.keys()}

        G_bar = get_mutilated_parents_graph(G, Z_names)
        parents_W = list(G.predecessors(W_name))

        for V_name in parents_W:
            # If the parent is part of the intervention, the causal mechanism is severed
            if V_name in Z_names:
                continue

            ancestors_V = nx.ancestors(G_bar, V_name) | {V_name}
            q_dict = {
                v: val
                for v, val in W_term.intervention.items()
                if v.name in ancestors_V
            }

            V_var = Variable(name=V_name)
            V_term = CounterfactualTerm(variable=V_var, intervention=q_dict)

            if V_term in nodes_in_G_prime:
                G_prime.add_edge(V_term, W_term)

    # Group counterfactual variables by their base underlying variable
    instances_by_var = defaultdict(list)
    for node in nodes_in_G_prime:
        instances_by_var[node.variable.name].append(node)

    # Lines 2-4: For each variable V appearing more than once in G', add a latent node U_V
    for var_name, instances in instances_by_var.items():
        if len(instances) > 1:
            u_node = f"U_{var_name}"
            G_prime.add_node(u_node, category="latent")
            for inst in instances:
                G_prime.add_edge(u_node, inst)

    # Lines 5-7: For each bidirected V <--> W where V and W are in G', add U_VW
    for u, v in bidirected_edges:
        inst_U = instances_by_var.get(u, [])
        inst_V = instances_by_var.get(v, [])

        if inst_U and inst_V:
            u_node = f"U_{u}_{v}"
            G_prime.add_node(u_node, category="latent")
            for inst in inst_U:
                G_prime.add_edge(u_node, inst)
            for inst in inst_V:
                G_prime.add_edge(u_node, inst)

    return G_prime
