"""Symbolic helpers."""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from or_algos.optimization._common import DimensionMismatchError, as_1d_float_array


def evaluate_symbolic_at_point(
    expr: sp.Expr | sp.MatrixBase,
    vars: Sequence[sp.Symbol],
    x: Sequence[float],
) -> float | list[float]:
    """Evaluate a symbolic expression at a numeric point."""
    ordered_vars = tuple(vars)
    values = as_1d_float_array(x, "x")
    if len(ordered_vars) != values.size:
        raise DimensionMismatchError(
            f"Expected {len(ordered_vars)} variables but received {values.size} values."
        )
    compiled = sp.lambdify(ordered_vars, expr, modules="numpy")
    result = compiled(*values)
    if isinstance(expr, sp.MatrixBase):
        return [float(value) for value in sp.Matrix(result)]
    return float(result)
