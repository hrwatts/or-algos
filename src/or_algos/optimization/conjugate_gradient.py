"""Partial conjugate-gradient implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from or_algos.optimization._common import (
    DimensionMismatchError,
    ObjectiveAdapter,
    as_1d_float_array,
)
from or_algos.optimization.line_search import armijo, goldstein, wolfe


def pcgrad(
    f: Any,
    x0: Sequence[float],
    m: int,
    line_search: str | None = None,
    method: str | None = None,
    tolerance: float | None = None,
    max_iter: int | None = None,
    *,
    vars: Sequence[sp.Symbol] | None = None,
    grad: Any = None,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64], int]:
    """Partial conjugate-gradient method."""
    x0_array = as_1d_float_array(x0, "x0")
    if m < 0:
        raise ValueError("m must be non-negative.")

    line_search_value = "A" if line_search is None else line_search.upper()
    method_value = "F" if method is None else method.upper()
    tolerance_value = 1e-12 if tolerance is None else float(tolerance)
    max_iter_value = 50 if max_iter is None else int(max_iter)

    adapter = ObjectiveAdapter.from_inputs(f, vars=vars, grad=grad)
    n = x0_array.size

    x = np.zeros((n, m + 2), dtype=float)
    y = np.zeros((n, 2), dtype=float)
    y[:, 0] = x0_array
    y[:, 1] = y[:, 0]

    x_values = np.full((n, max_iter_value), np.nan, dtype=float)
    x_values[:, 0] = x0_array

    g = np.zeros((n, m + 2), dtype=float)
    g[:, 0] = adapter.gradient(x0_array)

    alpha = np.zeros(m + 1, dtype=float)
    beta = np.zeros(m + 1, dtype=float)

    converged = False
    iterations = 1

    while (not converged) and (iterations <= max_iter_value):
        x[:, 0] = y[:, 1]
        y[:, 0] = x[:, 0]

        d = np.zeros((n, m + 2), dtype=float)
        d[:, 0] = -adapter.gradient(x[:, 0])

        for k in range(m + 1):
            alpha[k] = _dispatch_line_search(
                line_search_value,
                f,
                x[:, k],
                d[:, k],
                vars=vars,
                grad=grad if not isinstance(f, sp.Expr) else None,
            )

            x[:, k + 1] = x[:, k] + alpha[k] * d[:, k]
            g[:, k + 1] = adapter.gradient(x[:, k + 1])

            denominator = float(g[:, k] @ g[:, k])
            if denominator == 0.0:
                beta[k] = 0.0
            elif method_value == "P":
                beta[k] = float(((g[:, k + 1] - g[:, k]) @ g[:, k + 1]) / denominator)
            else:
                beta[k] = float((g[:, k + 1] @ g[:, k + 1]) / denominator)

            x_values[:, iterations - 1] = x[:, k + 1]
            d[:, k + 1] = -g[:, k + 1] + beta[k] * d[:, k]

        y[:, 1] = x[:, m + 1]
        converged = bool(np.linalg.norm(y[:, 1] - y[:, 0]) <= tolerance_value)
        iterations += 1

    x_star = x[:, m + 1].astype(float)
    z_star = float(adapter.objective(x_star))
    valid_columns = ~np.isnan(x_values[0, :])
    x_values = x_values[:, valid_columns]

    return x_star, z_star, x_values, iterations


def _dispatch_line_search(
    line_search: str,
    f: Any,
    x: NDArray[np.float64],
    d: NDArray[np.float64],
    *,
    vars: Sequence[sp.Symbol] | None,
    grad: Any,
) -> float:
    if x.shape != d.shape:
        raise DimensionMismatchError(f"x shape {x.shape} does not match d shape {d.shape}.")
    if line_search == "G":
        return goldstein(f, x, d, vars=vars, grad=grad)
    if line_search == "W":
        return wolfe(f, x, d, vars=vars, grad=grad)
    return armijo(f, x, d, vars=vars, grad=grad)
