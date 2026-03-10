import matplotlib.pyplot as plt
import networkx as nx
from daft import PGM


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

    # TODO: latent visualization is lacking

    return pgm
