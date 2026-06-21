"""Shared helpers for optimization routines."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray

Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]


class OptimizationError(RuntimeError):
    """Base class for optimization-specific errors."""


class DimensionMismatchError(OptimizationError, ValueError):
    """Raised when array shapes are incompatible."""


class MissingGradientError(OptimizationError, ValueError):
    """Raised when a callable objective is supplied without a gradient."""


class InfeasibleProblemError(OptimizationError):
    """Raised when a linear program is detected to be infeasible."""


class UnboundedProblemError(OptimizationError):
    """Raised when a linear program is detected to be unbounded."""


class IterationLimitError(OptimizationError):
    """Raised when an iterative method exhausts its iteration budget."""


def as_1d_float_array(value: Any, name: str) -> Array1D:
    """Convert an input into a flat float array."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise DimensionMismatchError(f"{name} must be one-dimensional after flattening.")
    return array


def as_2d_float_array(value: Any, name: str) -> Array2D:
    """Convert an input into a 2D float array."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise DimensionMismatchError(f"{name} must be two-dimensional.")
    return array


def sorted_sympy_vars(expr: sp.Expr, vars: Sequence[sp.Symbol] | None = None) -> tuple[sp.Symbol, ...]:
    """Return an explicit variable ordering."""
    if vars is not None:
        explicit = tuple(
            sp.Symbol(str(symbol)) if not isinstance(symbol, sp.Symbol) else symbol
            for symbol in vars
        )
        return explicit
    free_symbols = tuple(sorted(expr.free_symbols, key=lambda symbol: symbol.sort_key()))
    if not free_symbols:
        raise DimensionMismatchError(
            "Symbolic expressions require an explicit 'vars' sequence when no free symbols are present."
        )
    return free_symbols


@dataclass(frozen=True)
class ObjectiveAdapter:
    """Normalized objective/gradient access for symbolic expressions or callables."""

    objective: Callable[[Array1D], float]
    gradient: Callable[[Array1D], Array1D]
    vars: tuple[sp.Symbol, ...] | None

    @classmethod
    def from_inputs(
        cls,
        f: Any,
        *,
        vars: Sequence[sp.Symbol] | None = None,
        grad: Callable[[Array1D], Sequence[float]] | None = None,
    ) -> ObjectiveAdapter:
        if isinstance(f, sp.Expr):
            expr = sp.sympify(f)
            ordered_vars = sorted_sympy_vars(expr, vars)
            grad_expr = sp.Matrix([sp.diff(expr, symbol) for symbol in ordered_vars])
            objective_fn = sp.lambdify(ordered_vars, expr, modules="numpy")
            gradient_fn = sp.lambdify(ordered_vars, grad_expr, modules="numpy")

            def symbolic_objective(x: Array1D) -> float:
                checked = _check_dimension(x, len(ordered_vars), "x")
                return float(objective_fn(*checked))

            def symbolic_gradient(x: Array1D) -> Array1D:
                checked = _check_dimension(x, len(ordered_vars), "x")
                values = np.asarray(gradient_fn(*checked), dtype=float).reshape(-1)
                return values

            return cls(objective=symbolic_objective, gradient=symbolic_gradient, vars=ordered_vars)

        if not callable(f):
            raise TypeError("Objective must be a SymPy expression or a callable.")
        if grad is None:
            raise MissingGradientError("Callable objectives require a 'grad' callable.")

        def callable_objective(x: Array1D) -> float:
            values = as_1d_float_array(x, "x")
            return float(f(values))

        def callable_gradient(x: Array1D) -> Array1D:
            values = as_1d_float_array(x, "x")
            gradient_values = as_1d_float_array(grad(values), "grad(x)")
            if gradient_values.shape != values.shape:
                raise DimensionMismatchError(
                    f"Gradient shape {gradient_values.shape} does not match x shape {values.shape}."
                )
            return gradient_values

        return cls(objective=callable_objective, gradient=callable_gradient, vars=None)


def _check_dimension(x: Array1D, expected: int, name: str) -> Array1D:
    values = as_1d_float_array(x, name)
    if values.size != expected:
        raise DimensionMismatchError(
            f"Expected {expected} values for {name} but received {values.size}."
        )
    return values


def pivot_tableau(tableau: Array2D, row: int, column: int) -> None:
    """Pivot a tableau in place."""
    tableau[row, :] = tableau[row, :] / tableau[row, column]
    for index in range(tableau.shape[0]):
        if index == row:
            continue
        tableau[index, :] = tableau[index, :] - tableau[row, :] * tableau[index, column]


def basic_variable_index(tableau: Array2D, num_columns: int) -> NDArray[np.int_]:
    """Replicate the MATLAB basis-detection logic."""
    bvi = np.zeros(num_columns, dtype=int)
    for column in range(num_columns):
        rows = np.flatnonzero(tableau[:, column] != 0)
        if rows.size == 1:
            bvi[column] = int(rows[0] + 1)
    return bvi


def finite_min(values: Iterable[float]) -> tuple[float, int | None]:
    """Return the minimum finite value and its 0-based index."""
    array = np.asarray(list(values), dtype=float)
    finite = np.flatnonzero(~np.isnan(array))
    if finite.size == 0:
        return float("nan"), None
    position = int(finite[np.argmin(array[finite])])
    return float(array[position]), position
