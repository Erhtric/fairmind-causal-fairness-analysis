"""
Defines the DSL for Causal Inference: Variables, Interventions,
Counterfactual Terms, Events and Queries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class Variable:
    """
    Represents a random variable with a finite domain. In this simplified
    representation we allow a variable to have an identificative name and a
    domain of values.

    In this sense, it is an immutable object with an hashable name. This means that we
    allow two variables with different domains to have the same name.
    """

    name: str
    domain: tuple[Any, ...] = field(default_factory=tuple)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"{self.name}"

    def __matmul__(self, intervention: dict[Variable, Any]) -> CounterfactualTerm:
        """
        Syntax sugar for creating a counterfactual term.
        Usage: Y @ {X: 1}  ->  Y_{X=1}
        """
        return CounterfactualTerm(self, intervention)

    def __eq__(self, other: Any) -> Union[bool, Event]:
        """
        If other is a Variable, compare names for equality.
        Otherwise, return an atomic event (Syntax sugar: Y == 0).
        """
        if isinstance(other, Variable):
            return self.name == other.name

        # Note: We return an Event, not a boolean. This overrides standard equality for values.
        return Event({CounterfactualTerm(self, {}): other})

    def __lt__(self, other: Any) -> bool:
        return self.name < other.name


@dataclass(frozen=True)
class CounterfactualTerm:
    """
    Represents a counterfactual term $Y_{X=x}$, or in short $Y_{x}$.
    This is the equivalent of the Pearl's notation $Y$ in a world where ${do(X=x)}$.

    Usage:
    ```python
    Y @ {X: 1} # Y_{X=1}
    Y @ {X:1, W: W @ {X: 0}} # Y_{X=1, W_{X=0}}
    ```
    """

    variable: Variable
    intervention: dict[Variable, Union[Any, CounterfactualTerm]] = field(
        default_factory=dict
    )

    def __hash__(self):
        # Sort intervention items to ensure order independence
        items = tuple(sorted((k, v) for k, v in self.intervention.items()))
        return hash((self.variable, items))

    def __repr__(self):
        if not self.intervention:
            return self.variable.name

        intervention_str = []
        for var, val in sorted(self.intervention.items(), key=lambda x: x[0].name):
            # If val is a CounterfactualTerm, its own __repr__ handles the nesting
            intervention_str.append(f"{var.name}={val!r}")

        return f"{self.variable.name}_{{{', '.join(intervention_str)}}}"

    def __eq__(self, other: Any) -> Union[bool, Event]:
        """
        If other is a CounterfactualTerm, compare for structural equality.
        Otherwise, return an atomic event (Syntax sugar: Y == 0).
        """
        if isinstance(other, CounterfactualTerm):
            return (
                self.variable.name == other.variable.name
                and self.intervention == other.intervention
            )

        # Note: We return an Event, not a boolean. This overrides standard equality.
        return Event({self: other})

    def __ge__(self, other: Any) -> MonotonicityConstraint:
        if not isinstance(other, CounterfactualTerm):
            raise TypeError(
                "Monotonicity constraints must be between two CounterfactualTerms."
            )
        if self.variable != other.variable:
            raise ValueError("Monotonicity constraints must be on the same variable.")
        return MonotonicityConstraint(self, other, "ge")

    def __le__(self, other: Any) -> MonotonicityConstraint:
        if not isinstance(other, CounterfactualTerm):
            raise TypeError(
                "Monotonicity constraints must be between two CounterfactualTerms."
            )
        if self.variable != other.variable:
            raise ValueError("Monotonicity constraints must be on the same variable.")
        return MonotonicityConstraint(self, other, "le")

    def __lt__(self, other):
        return str(self) < str(other)


@dataclass(frozen=True)
class Event:
    """
    Represents an event, i.e. a conjunction of atomic propositions:
    e.g., {Y_{X=1}: 0, Z_{X=1, Y=0}: 1}
    An atomic proposition in this context is a counterfactual term with a value.
    Stored as a dictionary mapping counterfactual terms to their values.
    """

    assignments: dict[CounterfactualTerm, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[CounterfactualTerm, Any] = {}
        for term, value in self.assignments.items():
            if isinstance(term, Variable):
                normalized[CounterfactualTerm(term, {})] = value
                continue
            if isinstance(term, CounterfactualTerm):
                normalized[term] = value
                continue
            raise TypeError(
                "Event assignments must use Variable or CounterfactualTerm keys."
            )
        object.__setattr__(self, "assignments", normalized)

    def expand(self) -> list[Event]:
        """
        Applies the Counterfactual Unnesting Theorem (CUT) once.

        Example:
        Input: (Y_{X_z}=y)
        Output: [{Y_{X=0}=y & X_z=0}, {Y_{X=1}=y & X_z=1}, ...]

        Returns:
            A list of events

        Raises:
            ValueError: If there are contradictory assignments.
        """
        nested_term_loc = None

        for term in self.assignments.keys():
            for int_var, int_val in term.intervention.items():
                if isinstance(int_val, CounterfactualTerm):
                    nested_term_loc = (term, int_var, int_val)
                    break
                if nested_term_loc:
                    break

        if not nested_term_loc:
            return [self]

        outer_term: CounterfactualTerm = nested_term_loc[0]
        inner_var: Variable = nested_term_loc[1]
        inner_term: CounterfactualTerm = nested_term_loc[2]

        if inner_var.domain == ():
            raise ValueError("Inner variable has no domain.")

        expanded_events = []
        for val in inner_var.domain:
            new_outer_term = CounterfactualTerm(
                outer_term.variable,
                {**outer_term.intervention, inner_var: val},
            )

            new_inner_term = CounterfactualTerm(
                inner_term.variable,
                {**inner_term.intervention},
            )

            conjunction = Event(
                {new_outer_term: self.assignments[outer_term], new_inner_term: val}
            )

            expanded_events.append(conjunction)

        return expanded_events

    def __and__(self, other: Event) -> Event:
        """
        Syntax sugar for creating a conjunction of events.
        Usage: event1 & event2
        """
        new_assigments = self.assignments.copy()

        for term, value in other.assignments.items():
            if term in new_assigments and new_assigments[term] != value:
                raise ValueError(
                    f"Contradictory assignments. {term} cannot be both {new_assigments[term]} and {value}"
                )
            new_assigments[term] = value

        return Event(new_assigments)

    def __repr__(self):
        items = []
        for term, val in self.assignments.items():
            items.append(f"{term!r}={val!r}")
        return f"Event({', '.join(items)})"

    def __bool__(self):
        return bool(self.assignments)

    def __hash__(self):
        items = tuple(sorted(self.assignments.items(), key=lambda x: str(x[0])))
        return hash(items)

    def __eq__(self, other):
        if not isinstance(other, Event):
            return False
        return self.assignments == other.assignments


@dataclass(frozen=True)
class Query:
    """
    Represents a probability query P(target | evidence)

    Args:
        target: The target event
        evidence: The evidence event
    """

    target: Event
    evidence: Event = field(default_factory=Event)

    def expand(self) -> Expression:
        """Expand the query"""
        expanded_targets = self.target.expand()

        return Expression({Query(t, self.evidence): 1.0 for t in expanded_targets})

    def __repr__(self):
        if self.evidence:
            return f"P({self.target!r} | {self.evidence!r})"
        else:
            return f"P({self.target!r})"

    def __add__(self, other: Union[Query, Expression]) -> Expression:
        return Expression({self: 1.0}) + other

    def __sub__(self, other: Union[Query, Expression]) -> Expression:
        return Expression({self: 1.0}) - other

    def __mul__(self, other: Union[float, int]) -> Expression:
        return Expression({self: 1.0}) * other

    def __rmul__(self, other: Union[float, int]) -> Expression:
        return Expression({self: 1.0}) * other

    def __neg__(self) -> Expression:
        return Expression({self: -1.0})


@dataclass(frozen=True)
class Expression:
    """
    Represents a linear combination of queries (an Effect).
    Form: sum(weight_i * query_i)
    """

    terms: dict[Query, float] = field(default_factory=dict)

    def __add__(self, other: Union[Expression, Query]) -> Expression:
        new_terms = self.terms.copy()

        if isinstance(other, Query):
            # We assume equal contribution
            new_terms[other] = new_terms.get(other, 0.0) + 1.0
        elif isinstance(other, Expression):
            for q, w in other.terms.items():
                new_terms[q] = new_terms.get(q, 0.0) + w
        else:
            return NotImplemented

        # Remove terms with zero weight to keep it clean
        new_terms = {q: w for q, w in new_terms.items() if w != 0}
        return Expression(terms=new_terms)

    def __sub__(self, other: Union[Expression, Query]) -> Expression:
        return self + (-other)

    def __neg__(self) -> Expression:
        return self * -1.0

    def __mul__(self, other: Union[float, int]) -> Expression:
        if not isinstance(other, (float, int)):
            return NotImplemented
        new_terms = {q: w * other for q, w in self.terms.items()}
        return Expression(terms=new_terms)

    def __rmul__(self, other: Union[float, int]) -> Expression:
        return self.__mul__(other)

    def __repr__(self):
        parts = []
        for q, w in self.terms.items():
            sign = "+" if w >= 0 else "-"
            # scalar 1 is implicit in printing if we want cleaner output
            weight = f"{abs(w)} * " if abs(w) != 1.0 else ""
            parts.append(f"{sign} {weight}{q}")

        res = " ".join(parts).strip()
        if res.startswith("+ "):
            res = res[2:]
        return res or "0.0"


# Helper functions for creating queries
def P(target: Event, evidence: Event = None) -> Query:
    """
    Syntax sugar for creating a query.
    Usage: P(Y == 1, X == 0)

    Args:
        target: The target event
        evidence: The evidence event

    Returns:
        A query
    """
    if not isinstance(target, Event):
        raise ValueError("Target must be an Event")

    if evidence is None:
        evidence = Event()
    return Query(target, evidence)
