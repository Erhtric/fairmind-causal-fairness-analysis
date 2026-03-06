"""Symbolic TV-family fairness metrics built on top of the DSL."""

from __future__ import annotations

from .dsl import P, Variable


def symbolic_total_variation(Y: Variable, y_val, X: Variable, x0, x1):
    """Total Variation (TV): P(Y=y | X=x1) - P(Y=y | X=x0)."""
    term1 = P(Y == y_val, X == x1)
    term0 = P(Y == y_val, X == x0)
    return term1 - term0


def symbolic_total_effect(Y: Variable, y_val, X: Variable, x0, x1):
    """Total Effect (TE): P(Y_{x1}=y) - P(Y_{x0}=y)."""
    term1 = P(Y @ {X: x1} == y_val)
    term0 = P(Y @ {X: x0} == y_val)
    return term1 - term0


def symbolic_spurious_effect(Y: Variable, y_val, X: Variable, x):
    """Spurious Effect (SE): P(Y_x=y) - P(Y=y | X=x)."""
    term1 = P(Y @ {X: x} == y_val)
    term0 = P(Y == y_val, X == x)
    return term1 - term0


def symbolic_NDE(Y: Variable, y_val, X: Variable, x0, x1, W: Variable):
    """Natural Direct Effect (NDE): P(Y_{x1, W_{x0}}=y) - P(Y_{x0}=y)."""
    nested_w = W @ {X: x0}
    term1 = P(Y @ {X: x1, W: nested_w} == y_val)
    term0 = P(Y @ {X: x0} == y_val)
    return term1 - term0


def symbolic_NIE(Y: Variable, y_val, X: Variable, x0, x1, W: Variable):
    """Natural Indirect Effect (NIE): P(Y_{x0, W_{x1}}=y) - P(Y_{x0}=y)."""
    nested_w = W @ {X: x1}
    term1 = P(Y @ {X: x0, W: nested_w} == y_val)
    term0 = P(Y @ {X: x0} == y_val)
    return term1 - term0
