import networkx as nx


def filter_nodes_by_type(
    graph_or_nodes: nx.DiGraph | nx.classes.reportviews.NodeDataView, **kwargs
) -> list[str]:
    """
    Filters nodes by node attributes.

    Args:
        graph_or_nodes: A graph or a NodeDataView.
        **kwargs: Key-value pairs to filter nodes by.

    Returns:
        A list of node names that match the specified attributes.
    """
    if isinstance(graph_or_nodes, nx.DiGraph):
        nodes = graph_or_nodes.nodes(data=True)
    else:
        nodes = graph_or_nodes

    return [
        node
        for node, data in nodes
        if all(data.get(key) == value for key, value in kwargs.items())
    ]


def build_sfm(
    sensitive_attr: str,
    outcome_attr: str,
    confounder_attrs: list[str],
    mediator_attrs: list[str],
    sorted_confounders: bool = False,
    sorted_mediators: bool = False,
    latents: list[tuple[str, list[str]]] | None = None,
) -> nx.DiGraph:
    """
    Builds a Standard Fairness Model (SFM) template.

    Args:
        sensitive_attr (str): The name of the sensitive attribute.
        outcome_attr (str): The name of the outcome attribute.
        confounder_attrs (list[str]): A list of names of confounder attributes.
        sorted_confounders (bool): Whether to sort confounder attributes in topological order.
        mediator_attrs (list[str]): A list of names of mediator attributes.
        sorted_mediators (bool): Whether to sort mediator attributes in topological order.
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
        if sorted_confounders:
            for i in range(len(confounder_attrs) - 1):
                sfm.add_edge(confounder_attrs[i], confounder_attrs[i + 1])
    for mediator in mediator_attrs:
        sfm.add_node(mediator, type="mediator", category="endogenous")
        sfm.add_edge(sensitive_attr, mediator)
        sfm.add_edge(mediator, outcome_attr)
        if sorted_mediators:
            for i in range(len(mediator_attrs) - 1):
                sfm.add_edge(mediator_attrs[i], mediator_attrs[i + 1])

    for confounder in confounder_attrs:
        for mediator in mediator_attrs:
            sfm.add_edge(confounder, mediator)

    if latents is not None:
        for latent, children in latents:
            sfm.add_node(latent, type="latent", category="latent")
            for child in children:
                sfm.add_edge(latent, child)

    return sfm
