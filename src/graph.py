from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
from daft import PGM


def filter_nodes_by_type(
    nodes: nx.classes.reportviews.NodeDataView, node_type: str
) -> list[str]:
    return [
        node_name
        for (node_name, node_data) in nodes
        if node_data.get("type") == node_type
    ]


def build_sfm(
    sensitive_attr: str,
    outcome_attr: str,
    confounder_attrs: list[str],
    mediator_attrs: list[str],
    latents: Optional[list[tuple[str, str]] | None] = None,
) -> nx.DiGraph:
    """
    Builds a Standard Fairness Model (SFM) template.

    Args:
        sensitive_attr (str): The name of the sensitive attribute.
        outcome_attr (str): The name of the outcome attribute.
        confounder_attrs (list[str]): A list of names of confounder attributes.
        mediator_attrs (list[str]): A list of names of mediator attributes.
        latents (list[Tuple[str, str]]): A list of tuples representing latent variables.
                Each tuple should contain the name of the latent variable and a list of children.
    Returns:
        nx.DiGraph: A directed graph representing the SFM template.
    """
    if isinstance(sensitive_attr, str) == False:
        raise ValueError("Sensitive attribute must be a string.")
    if isinstance(outcome_attr, str) == False:
        raise ValueError("Outcome attribute must be a string.")
    if isinstance(confounder_attrs, list) == False or not all(
        isinstance(attr, str) for attr in confounder_attrs
    ):
        raise ValueError("Confounder attributes must be a list of strings.")
    if isinstance(mediator_attrs, list) == False or not all(
        isinstance(attr, str) for attr in mediator_attrs
    ):
        raise ValueError("Mediator attributes must be a list of strings.")

    if (
        set(confounder_attrs)
        & set(mediator_attrs)
        & set([sensitive_attr])
        & set([outcome_attr])
    ):
        raise ValueError(
            "Confounder, mediator, sensitive attribute, and outcome attribute sets must be disjoint."
        )

    sfm = nx.DiGraph()
    sfm.add_node(sensitive_attr, type="sensitive")
    sfm.add_node(outcome_attr, type="outcome")

    sfm.add_edge(sensitive_attr, outcome_attr)
    for confounder in confounder_attrs:
        sfm.add_node(confounder, type="confounder")
        sfm.add_edge(confounder, sensitive_attr)
        sfm.add_edge(confounder, outcome_attr)
    for mediator in mediator_attrs:
        sfm.add_node(mediator, type="mediator")
        sfm.add_edge(sensitive_attr, mediator)
        sfm.add_edge(mediator, outcome_attr)

    for confounder in confounder_attrs:
        for mediator in mediator_attrs:
            sfm.add_edge(confounder, mediator)

    if latents is not None:
        for latent, children in latents:
            sfm.add_node(latent, type="latent")
            for child in children:
                sfm.add_edge(latent, child)

    return sfm


def visualize_sfm(sfm: nx.DiGraph, scale_factor: float = 1.7) -> PGM:
    """
    Visualizes a Standard Fairness Model (SFM) using daft-pgm.

    Args:
        sfm (nx.DiGraph): A directed graph representing the SFM template.
        scale_factor (float): A scaling factor to adjust the size of the visualization.

    Returns:
        PGM: A daft-pgm object representing the visualized SFM.
    """
    pos = {}
    nodes = list(sfm.nodes())

    sensitive = [n for n in nodes if sfm.nodes[n].get("type") == "sensitive"]
    outcome = [n for n in nodes if sfm.nodes[n].get("type") == "outcome"]
    mediators = [n for n in nodes if sfm.nodes[n].get("type") == "mediator"]
    confounders = [n for n in nodes if sfm.nodes[n].get("type") == "confounder"]
    unobserved = [n for n in nodes if sfm.nodes[n].get("type") == "latent"]

    for i, node in enumerate(sensitive):
        pos[node] = (0, 1 - i * 0.5)

    for i, node in enumerate(outcome):
        pos[node] = (4, 1 - i * 0.5)

    for i, node in enumerate(mediators):
        # Mediators should be lined up horizontally between sensitive and outcome
        pos[node] = (4 * (i + 1) / (len(mediators) + 1), 0)

    for i, node in enumerate(confounders):
        # Confounders should be lined up horizontally between sensitive and outcome
        pos[node] = (4 * (i + 1) / (len(confounders) + 1), 2)

    # Normalize positions
    min_x = min(p[0] for p in pos.values())
    max_x = max(p[0] for p in pos.values())
    min_y = min(p[1] for p in pos.values())
    max_y = max(p[1] for p in pos.values())

    scale = scale_factor * max(
        len(sensitive), len(outcome), len(mediators), len(confounders)
    )
    pgm = PGM(shape=[scale * 2, scale * 2], node_unit=1.2)

    for node in sfm.nodes:
        if node in unobserved:
            continue  # Skip latent variables in visualization
        x = (
            (pos[node][0] - min_x) / (max_x - min_x) * scale
            if max_x != min_x
            else scale / 2
        )
        y = (
            (pos[node][1] - min_y) / (max_y - min_y) * scale
            if max_y != min_y
            else scale / 2
        )
        pgm.add_node(node, node, x, y)

    for edge in sfm.edges:
        if edge[0] in unobserved or edge[1] in unobserved:
            continue  # Skip edges involving latent variables in visualization
        pgm.add_edge(edge[0], edge[1])
    pgm.render()
    plt.show()

    return pgm
