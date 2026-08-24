import networkx as nx
import pandas as pd
from loguru import logger
from pgmpy.estimators import (
    BIC,
    K2,
    PC,
    BDeu,
    ExpertKnowledge,
    HillClimbSearch,
)

from .graph import build_sfm


def discover_graph(
    data: pd.DataFrame,
    method: str = "hc",
    score: str = "bdeu",
    sensitive_attr: str | None = None,
    outcome_attr: str | None = None,
    fixed_edges: list[tuple[str, str]] | None = None,
    forbidden_edges: list[tuple[str, str]] | None = None,
    temporal_order: list[list[str]] | None = None,
    max_indegree: int | None = None,
    show_progress: bool = False,
) -> nx.DiGraph:
    """Learn a DAG structure from data.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset.  All columns should be categorical (or will be
        auto-discretized if numeric with >10 unique values).
    method : str
        ``"hc"`` for Hill-Climb search (score-based) or ``"pc"`` for the
        PC algorithm (constraint-based).
    score : str
        Scoring function for HC: ``"bdeu"``, ``"bic"``, or ``"k2"``.
    sensitive_attr, outcome_attr : str or None
        If both are provided and *temporal_order* is ``None``, a default
        temporal order is generated: ``[[sensitive_attr], [others...],
        [outcome_attr]]``.  This encodes the minimal causal assumption
        that the protected attribute is exogenous and the outcome is
        terminal.
    fixed_edges : list of (str, str) or None
        Edges that MUST appear in the learned graph.
    forbidden_edges : list of (str, str) or None
        Edges that MUST NOT appear.
    temporal_order : list of list of str or None
        Temporal tiers for ``ExpertKnowledge``.  Variables in earlier
        tiers cannot be caused by variables in later tiers.
    max_indegree : int or None
        Maximum number of parents per node (HC only).
    show_progress : bool
        Whether to display progress bars.

    Returns
    -------
    nx.DiGraph
        The learned directed acyclic graph.
    """
    data = data.copy()

    for col in data.columns:
        if data[col].dtype in ("int64", "float64"):
            if data[col].nunique() > 10:
                data[col] = pd.cut(
                    data[col], bins=5, include_lowest=True
                ).astype(str)

    # Default temporal order: X first, Y last, everything else in between.
    # This reliably discovers mediators (X → W → Y).  To discover
    # confounders (Z → X), pass a custom temporal_order with candidate
    # confounders in a tier before X, e.g.:
    #   temporal_order=[["age"], ["gender"], ["education"], ["income"]]
    if temporal_order is None and sensitive_attr and outcome_attr:
        others = [
            c for c in data.columns
            if c not in (sensitive_attr, outcome_attr)
        ]
        temporal_order = [
            [sensitive_attr],
            others,
            [outcome_attr],
        ]
        logger.debug(f"Default temporal order: {temporal_order}")

    expert = None
    if temporal_order or fixed_edges or forbidden_edges:
        expert = ExpertKnowledge(
            temporal_order=temporal_order,
            required_edges=fixed_edges,
            forbidden_edges=forbidden_edges,
        )

    scorer_map = {"bdeu": BDeu, "bic": BIC, "k2": K2}

    if method == "hc":
        scorer_cls = scorer_map.get(score, BDeu)
        scoring_method = scorer_cls(data)

        hc = HillClimbSearch(data)
        dag = hc.estimate(
            scoring_method=scoring_method,
            max_indegree=max_indegree,
            expert_knowledge=expert,
            show_progress=show_progress,
        )
    elif method == "pc":
        pc = PC(data)
        dag = pc.estimate(show_progress=show_progress, significance_level=0.05)
        # PC doesn't support ExpertKnowledge natively in 1.0.0,
        # so we enforce constraints post-hoc.
        if expert and (fixed_edges or forbidden_edges):
            graph = nx.DiGraph(dag)
            graph = _ensure_edges(graph, fixed_edges, forbidden_edges)
            return graph
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'hc' or 'pc'.")

    return nx.DiGraph(dag)


def _ensure_edges(
    graph: nx.DiGraph,
    fixed_edges: list[tuple[str, str]] | None,
    forbidden_edges: list[tuple[str, str]] | None,
) -> nx.DiGraph:
    """Force / forbid specific edges in a learned graph."""
    graph = graph.copy()
    if fixed_edges:
        for u, v in fixed_edges:
            if not graph.has_edge(u, v):
                if graph.has_edge(v, u):
                    graph.remove_edge(v, u)
                graph.add_edge(u, v)
    if forbidden_edges:
        for u, v in forbidden_edges:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
    return graph


def classify_roles(
    dag: nx.DiGraph,
    sensitive_attr: str,
    outcome_attr: str,
) -> dict[str, str]:
    """Classify every non-X, non-Y node in *dag* as confounder, mediator, or
    irrelevant, based on the graph topology.

    Classification rules (applied in order of priority):

    * **Confounder (Z)**: the node is an ancestor of both X and Y, or it is an
      ancestor of X and has a directed path to Y.  Intuitively, confounders
      generate a spurious association between X and Y.
    * **Mediator (W)**: the node is a descendant of X *and* an ancestor of Y
      (i.e. it lies on a directed path X -> ... -> W -> ... -> Y).
    * **Irrelevant**: everything else (nodes disconnected from the X-Y
      relationship, or pure descendants of Y).

    Parameters
    ----------
    dag : nx.DiGraph
        A learned DAG (need not carry type annotations).
    sensitive_attr : str
        Name of the protected / sensitive variable (X).
    outcome_attr : str
        Name of the outcome variable (Y).

    Returns
    -------
    dict[str, str]
        ``{node_name: role}`` where role is one of
        ``"confounder"``, ``"mediator"``, ``"irrelevant"``.
    """
    if sensitive_attr not in dag:
        raise ValueError(f"Sensitive attribute {sensitive_attr!r} not in graph.")
    if outcome_attr not in dag:
        raise ValueError(f"Outcome attribute {outcome_attr!r} not in graph.")

    ancestors_of_x = nx.ancestors(dag, sensitive_attr)
    ancestors_of_y = nx.ancestors(dag, outcome_attr)
    descendants_of_x = nx.descendants(dag, sensitive_attr)

    roles: dict[str, str] = {}

    for node in dag.nodes:
        if node in (sensitive_attr, outcome_attr):
            continue

        is_ancestor_of_x = node in ancestors_of_x
        is_ancestor_of_y = node in ancestors_of_y
        is_descendant_of_x = node in descendants_of_x
        has_path_to_y = nx.has_path(dag, node, outcome_attr)

        if is_ancestor_of_x and has_path_to_y:
            roles[node] = "confounder"
        elif is_descendant_of_x and is_ancestor_of_y:
            roles[node] = "mediator"
        else:
            roles[node] = "irrelevant"

    logger.debug(f"Role classification for X={sensitive_attr}, Y={outcome_attr}: {roles}")
    return roles


def discover_sfm(
    data: pd.DataFrame,
    sensitive_attr: str,
    outcome_attr: str,
    method: str = "hc",
    score: str = "bdeu",
    fixed_edges: list[tuple[str, str]] | None = None,
    forbidden_edges: list[tuple[str, str]] | None = None,
    temporal_order: list[list[str]] | None = None,
    max_indegree: int | None = None,
    show_progress: bool = False,
) -> tuple[nx.DiGraph, nx.DiGraph, dict[str, str]]:
    """Learn a DAG from data and build an annotated SFM.

    This is the main entry point for automatic causal graph learning.
    It combines :func:`discover_graph`, :func:`classify_roles`, and
    :func:`~src.graph.build_sfm` into a single call.

    When *temporal_order* is ``None`` (the default), a minimal causal
    ordering is generated: ``[[X], [others], [Y]]``.  This reliably
    discovers mediators (X → W → Y).  To also discover confounders
    (Z → X), pass a custom *temporal_order* with candidate confounders
    in a tier before X, e.g. ``[["age"], ["gender"], ...]``.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset (only the columns relevant to the analysis should be
        included; extra columns will be part of the learned graph).
    sensitive_attr : str
        Name of the protected / sensitive variable.
    outcome_attr : str
        Name of the outcome variable.
    method, score, fixed_edges, forbidden_edges, temporal_order,
    max_indegree, show_progress
        Forwarded to :func:`discover_graph`.

    Returns
    -------
    sfm : nx.DiGraph
        The Standard Fairness Model with type annotations on each node,
        ready for :func:`~src.model.fit_discrete_bayesian_model`.
    learned_dag : nx.DiGraph
        The raw DAG as returned by the structure-learning algorithm
        (useful for inspection / comparison with an expert graph).
    roles : dict[str, str]
        The role assigned to each variable (excluding X and Y).
    """
    cols = list(data.columns)
    if sensitive_attr not in cols:
        raise ValueError(f"Sensitive attribute {sensitive_attr!r} not in data columns.")
    if outcome_attr not in cols:
        raise ValueError(f"Outcome attribute {outcome_attr!r} not in data columns.")

    learned_dag = discover_graph(
        data=data,
        method=method,
        score=score,
        sensitive_attr=sensitive_attr,
        outcome_attr=outcome_attr,
        fixed_edges=fixed_edges,
        forbidden_edges=forbidden_edges,
        temporal_order=temporal_order,
        max_indegree=max_indegree,
        show_progress=show_progress,
    )

    if sensitive_attr not in learned_dag:
        learned_dag.add_node(sensitive_attr)
    if outcome_attr not in learned_dag:
        learned_dag.add_node(outcome_attr)

    # PC can produce cycles — break them before classifying roles
    if not nx.is_directed_acyclic_graph(learned_dag):
        logger.warning(
            "Learned graph contains cycles — breaking them by "
            "removing one edge per cycle."
        )
        while not nx.is_directed_acyclic_graph(learned_dag):
            cycle = nx.find_cycle(learned_dag)
            u, v = cycle[-1][:2]
            learned_dag.remove_edge(u, v)
            logger.debug(f"Removed edge ({u} -> {v}) to break cycle.")

    roles = classify_roles(learned_dag, sensitive_attr, outcome_attr)

    confounders = [n for n, r in roles.items() if r == "confounder"]
    mediators = [n for n, r in roles.items() if r == "mediator"]

    topo_order = list(nx.topological_sort(learned_dag))
    confounders.sort(key=lambda n: topo_order.index(n))
    mediators.sort(key=lambda n: topo_order.index(n))

    logger.info(
        f"Discovered SFM: X={sensitive_attr}, Y={outcome_attr}, "
        f"Z={confounders}, W={mediators}"
    )

    sfm = build_sfm(
        sensitive_attr=sensitive_attr,
        outcome_attr=outcome_attr,
        confounder_attrs=confounders,
        mediator_attrs=mediators,
        sorted_confounders=len(confounders) > 1,
        sorted_mediators=len(mediators) > 1,
    )

    return sfm, learned_dag, roles


def graph_similarity(
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    nodes: list[str] | None = None,
) -> dict[str, float]:
    """Compare two directed graphs.

    Parameters
    ----------
    g1 : nx.DiGraph
        Reference (ground truth) graph.
    g2 : nx.DiGraph
        Learned / predicted graph.
    nodes : list of str or None
        If given, restrict comparison to these nodes only.

    Returns
    -------
    dict
        ``shd`` (Structural Hamming Distance), ``precision``, ``recall``,
        ``f1``, ``jaccard``.
    """
    if nodes is not None:
        g1 = nx.DiGraph(g1.subgraph(nodes))
        g2 = nx.DiGraph(g2.subgraph(nodes))

    edges1 = set(g1.edges())
    edges2 = set(g2.edges())

    tp = len(edges1 & edges2)
    fp = len(edges2 - edges1)
    fn = len(edges1 - edges2)

    shd = fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "shd": shd,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "jaccard": round(jaccard, 4),
    }
