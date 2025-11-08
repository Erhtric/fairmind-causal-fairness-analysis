import networkx as nx
import sympy


class CausalGraph:
    """
    Represents the graphical structure of an SCM.

    This class builds and validates the causal graph based on
    structural equations and provides methods for graph queries like
    finding parents, checking for cycles, and d-separation.
    """

    def __init__(
        self,
        V_names: set[str],
        U_names: set[str],
        V_sym: dict[str, sympy.Symbol],
        U_sym: dict[str, sympy.Symbol],
        F: dict[str, sympy.Expr],
    ):
        """
        Initializes the CausalGraph.

        Args:
            V_names (set[str]): Set of endogenous variable names.
            U_names (set[str]): Set of exogenous variable names.
            V_sym (dict[str, sympy.Symbol]): Map of V-names to symbols.
            U_sym (dict[str, sympy.Symbol]): Map of U-names to symbols.
            F (dict[str, sympy.Expr]): Map of endogenous variable names to their
                                       symbolic structural equations.
        """
        self.V_names = V_names
        self.U_names = U_names
        self.all_vars = V_names | U_names

        self.V_sym = V_sym
        self.U_sym = U_sym
        self.all_syms = {**V_sym, **U_sym}

        self.nx_G = self._build_graph(F)
        self._validate_graph()

    def _build_graph(self, F: dict[str, sympy.Expr]) -> nx.DiGraph:
        """Builds a networkx DiGraph from the structural equations.

        Each node is a tuple (variable_name, {type, symbol}).
        A type can either be "endo" or "exo", nor both.
        A symbol is a sympy.Symbol object of the same variable. IDs are matched
        to the ones used in the SCM's symbols.

        Args:
            F (dict[str, sympy.Expr]): The structural equations.

        Returns:
            nx.DiGraph: The networkx DiGraph representation of the SCM.
        """
        G = nx.DiGraph()

        # Add the endogneous variables (V) as nodes, along with their type and symbol
        for v_name in self.V_names:
            G.add_node(v_name, type="endo", symbol=v_name)
        # for v_sym in self.V_sym.values():
        # G.add_node(v_sym, type="endo", symbol=v_sym)

        # Add the exogenous variables (U) as nodes, along with their type and symbol
        for u_name in self.U_names:
            G.add_node(u_name, type="exo", symbol=u_name)
        # for u_sym in self.U_sym.values():
        #     G.add_node(u_sym, type="exo", symbol=u_sym)

        # Add edges based on functional dependencies in F
        for v_name, expr in F.items():
            # The parents of v_name are the free symbols in its equation
            parents_syms = expr.free_symbols
            # Get their string representations
            parents_syms = [str(p_sym) for p_sym in parents_syms]
            for p_name in parents_syms:
                G.add_edge(p_name, v_name, equation=expr)

        return G

    def _validate_graph(self):
        """Ensures the graph is a Directed Acyclic Graph (DAG)."""
        if not nx.is_directed_acyclic_graph(self.nx_G):
            cycles = list(nx.simple_cycles(self.nx_G))
            raise ValueError(f"The model's structure contains causal cycles: {cycles}")

    def get_parents(self, var_name: str) -> set[str]:
        """Returns the set of parent variable names for a given variable."""
        if var_name not in self.nx_G:
            raise ValueError(f"Variable '{var_name}' not in the graph.")
        return set(self.nx_G.predecessors(var_name))

    def get_endogenous_parents(self, var_name: str) -> set[str]:
        """Returns the set of endogenous parent variable names for a given variable."""
        if var_name not in self.nx_G:
            raise ValueError(f"Variable '{var_name}' not in the graph.")
        return set(self.nx_G.predecessors(var_name)) & self.V_names

    def get_exogenous_parents(self, var_name: str) -> set[str]:
        """Returns the set of exogenous parent variable names for a given variable."""
        if var_name not in self.nx_G:
            raise ValueError(f"Variable '{var_name}' not in the graph.")
        return set(self.nx_G.predecessors(var_name)) & self.U_names

    def get_confounded_component_for_U(self, exogenous_var_name: str) -> set[str]:
        """Returns the set of variables in the confounded component for a given exogenous variable."""
        raise NotImplementedError

    def get_markov_blanket(self, var_name: str) -> set[str]:
        """Returns the set of variables in the Markov blanket for a given variable."""
        if var_name not in self.nx_G:
            raise ValueError(f"Variable '{var_name}' not in the graph.")

        children = set(self.nx_G.successors(var_name))
        parents = set(self.nx_G.predecessors(var_name))

        parents_of_children = (
            set.union(*[self.get_parents(child) for child in children])
            if children
            else set()
        )

        return parents | parents_of_children | children

    def get_topological_sort(self) -> list[str]:
        """Returns a topological sort of all variables in the graph."""
        return list(nx.topological_sort(self.nx_G))
