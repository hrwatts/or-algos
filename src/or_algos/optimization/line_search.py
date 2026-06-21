"""Line-search routines ported from MATLAB."""

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


def armijo(
    f: Any,
    x0: Sequence[float] | NDArray[np.float64],
    d: Sequence[float] | NDArray[np.float64],
    epsilon: float | None = None,
    eta: float | None = None,
    *,
    vars: Sequence[sp.Symbol] | None = None,
    grad: Any = None,
) -> float:
    """Armijo's line-search rule."""
    return _line_search_impl("A", f, x0, d, epsilon, eta, vars=vars, grad=grad)


def goldstein(
    f: Any,
    x0: Sequence[float] | NDArray[np.float64],
    d: Sequence[float] | NDArray[np.float64],
    epsilon: float | None = None,
    eta: float | None = None,
    *,
    vars: Sequence[sp.Symbol] | None = None,
    grad: Any = None,
) -> float:
    """Goldstein's line-search rule."""
    return _line_search_impl("G", f, x0, d, epsilon, eta, vars=vars, grad=grad)


def wolfe(
    f: Any,
    x0: Sequence[float] | NDArray[np.float64],
    d: Sequence[float] | NDArray[np.float64],
    epsilon: float | None = None,
    eta: float | None = None,
    *,
    vars: Sequence[sp.Symbol] | None = None,
    grad: Any = None,
) -> float:
    """Wolfe's line-search rule."""
    return _line_search_impl("W", f, x0, d, epsilon, eta, vars=vars, grad=grad)


def _line_search_impl(
    kind: str,
    f: Any,
    x0: Sequence[float] | NDArray[np.float64],
    d: Sequence[float] | NDArray[np.float64],
    epsilon: float | None,
    eta: float | None,
    *,
    vars: Sequence[sp.Symbol] | None,
    grad: Any,
) -> float:
    x0_array = as_1d_float_array(x0, "x0")
    d_array = as_1d_float_array(d, "d")
    if x0_array.shape != d_array.shape:
        raise DimensionMismatchError(
            f"x0 shape {x0_array.shape} does not match d shape {d_array.shape}."
        )

    if kind == "G":
        epsilon_value = 1.0 / 3.0 if epsilon is None else float(epsilon)
    else:
        epsilon_value = 0.5 if epsilon is None else float(epsilon)
    eta_value = 2.0 if eta is None else float(eta)

    adapter = ObjectiveAdapter.from_inputs(f, vars=vars, grad=grad)
    alpha = 1.0
    phi_0 = adapter.objective(x0_array)
    g0 = adapter.gradient(x0_array)
    phi_prime_0 = float(g0 @ d_array)

    condition_met = False
    while not condition_met:
        x_next = x0_array + alpha * d_array
        phi = adapter.objective(x_next)
        armijos_rule = float(phi_0 + epsilon_value * phi_prime_0 * alpha)
        condition_met = phi <= armijos_rule

        if not condition_met:
            if kind == "W":
                alpha = alpha / (3.0 * eta_value)
            else:
                alpha = alpha / 3.0
            x_next = x0_array + alpha * d_array
            phi = adapter.objective(x_next)
            armijos_rule = float(phi_0 + epsilon_value * phi_prime_0 * alpha)
            condition_met = phi <= armijos_rule
            continue

        if kind == "A":
            alpha = alpha * 2.0
            x_next = x0_array + alpha * d_array
            phi_eta = adapter.objective(eta_value * x_next)
            curvature_condition = float(phi_0 + epsilon_value * phi_prime_0 * eta_value * alpha)
            condition_met = phi_eta > curvature_condition
        elif kind == "G":
            alpha = alpha * 2.0
            x_next = x0_array + alpha * d_array
            phi = adapter.objective(x_next)
            goldsteins_condition = float(phi_0 + (1.0 - epsilon_value) * phi_prime_0 * alpha)
            condition_met = phi >= goldsteins_condition
        else:
            alpha = alpha * 2.0 * eta_value
            x_next = x0_array + alpha * d_array
            phi_prime = float(adapter.gradient(x_next) @ d_array)
            wolfe_condition = float((1.0 - epsilon_value) * phi_prime_0)
            condition_met = phi_prime <= wolfe_condition

    return float(alpha)
