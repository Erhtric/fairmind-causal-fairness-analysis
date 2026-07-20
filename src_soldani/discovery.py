import networkx as nx
import pandas as pd
from pgmpy.estimators import HillClimbSearch, PC, BDeu, BIC, K2

from src.graph import filter_nodes_by_type


def discover_graph(
    data: pd.DataFrame,
    method: str = "hc",
    score: str = "bdeu",
    fixed_edges: list[tuple[str, str]] | None = None,
    forbidden_edges: list[tuple[str, str]] | None = None,
    max_indegree: int | None = None,
    show_progress: bool = False,
) -> nx.DiGraph:
    """Learn a causal graph structure from data.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset. All columns must be categorical (string type).
        Continuous variables MUST be discretized before calling this function.
    method : str
        "hc" for Hill-Climb search (score-based) or "pc" for PC algorithm (constraint-based).
    score : str
        Scoring method for HC: "bdeu", "bic", or "k2".
    fixed_edges : list of (str, str) or None
        Edges that MUST be present in the learned graph.
    forbidden_edges : list of (str, str) or None
        Edges that MUST NOT be present.
    max_indegree : int or None
        Maximum number of parents for any node (HC only).
    show_progress : bool
        Whether to show progress bars.

    Returns
    -------
    nx.DiGraph
        The learned directed graph.
    """
    data = data.copy()

    for col in data.columns:
        if data[col].dtype in ("int64", "float64"):
            if data[col].nunique() > 10:
                data[col] = pd.cut(
                    data[col], bins=5, include_lowest=True
                ).astype(str)

    scorer_map = {
        "bdeu": BDeu,
        "bic": BIC,
        "k2": K2,
    }

    if method == "hc":
        scorer_cls = scorer_map.get(score, BDeu)
        scoring_method = scorer_cls(data)

        hc = HillClimbSearch(data)
        dag = hc.estimate(
            scoring_method=scoring_method,
            max_indegree=max_indegree,
            show_progress=show_progress,
        )
    elif method == "pc":
        pc = PC(data)
        dag = pc.estimate(show_progress=show_progress, significance_level=0.05)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hc' or 'pc'.")

    graph = nx.DiGraph(dag)
    graph = _ensure_edges(graph, fixed_edges, forbidden_edges)
    return graph


def _ensure_edges(
    graph: nx.DiGraph,
    fixed_edges: list[tuple[str, str]] | None,
    forbidden_edges: list[tuple[str, str]] | None,
) -> nx.DiGraph:
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


def learn_sfm(
    data,
    outcome_attr: str | None = None,
    score: str = "bdeu",
    max_indegree: int | None = None,
    forbidden_edges: list[tuple[str, str]] | None = None,
) -> nx.DiGraph:
    """Learn a causal graph from data using Hill-Climb search.

    Thin wrapper around discover_graph for backward compatibility.
    The outcome_attr parameter is accepted but not used (the graph is
    learned from all columns in the data).

    Returns
    -------
    nx.DiGraph
        A directed graph with nodes annotated.
    """
    return discover_graph(
        data=data,
        method="hc",
        score=score,
        forbidden_edges=forbidden_edges,
        max_indegree=max_indegree,
        show_progress=False,
    )


def graph_similarity(
    g1: nx.DiGraph,
    g2: nx.DiGraph,
    nodes: list[str] | None = None,
) -> dict[str, float]:
    """Compare two directed graphs using standard metrics.

    Parameters
    ----------
    g1, g2 : nx.DiGraph
        Graphs to compare.
    nodes : list of str or None
        If given, only consider these nodes.

    Returns
    -------
    dict with keys:
        shd : int
            Structural Hamming Distance (edges to add/remove/reverse).
        precision : float
            TP / (TP + FP) among predicted edges.
        recall : float
            TP / (TP + FN) among true edges.
        f1 : float
        jaccard : float
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

