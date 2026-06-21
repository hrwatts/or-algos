"""Tableau-style linear-programming routines ported from MATLAB."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product

import numpy as np
from numpy.typing import NDArray

from or_algos.optimization._common import (
    InfeasibleProblemError,
    as_1d_float_array,
    as_2d_float_array,
)


def bounded_simplex(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
    h: Sequence[float],
) -> tuple[NDArray[np.float64], float]:
    """Bounded-solution solver for equality-constrained bounded LPs."""
    A_array = as_2d_float_array(A, "A")
    b_array = as_1d_float_array(b, "b")
    c_array = as_1d_float_array(c, "c")
    d_array = as_1d_float_array(d, "d")
    h_array = as_1d_float_array(h, "h")

    m, n = A_array.shape
    if b_array.size != m or c_array.size != n or d_array.size != n or h_array.size != n:
        raise ValueError("Input dimensions do not agree.")
    lower_bounds = d_array
    upper_bounds = h_array
    if np.any(lower_bounds > upper_bounds):
        raise InfeasibleProblemError("Lower bounds exceed upper bounds.")

    if m > n:
        raise InfeasibleProblemError("More equality constraints than variables.")

    best_x: NDArray[np.float64] | None = None
    best_value = float("inf")
    variable_indices = tuple(range(n))

    for basis in combinations(variable_indices, m):
        basis_array = np.array(basis, dtype=int)
        nonbasis = np.array([index for index in variable_indices if index not in basis], dtype=int)
        basis_matrix = A_array[:, basis_array]
        try:
            basis_inverse_rhs = np.linalg.solve(basis_matrix, b_array[:, None])
        except np.linalg.LinAlgError:
            continue

        if nonbasis.size == 0:
            x_basis = basis_inverse_rhs[:, 0]
            if np.any(x_basis < lower_bounds[basis_array] - 1e-9) or np.any(
                x_basis > upper_bounds[basis_array] + 1e-9
            ):
                continue
            candidate = np.zeros(n, dtype=float)
            candidate[basis_array] = x_basis
            objective = float(c_array @ candidate)
            if objective < best_value:
                best_value = objective
                best_x = candidate.copy()
            continue

        assignment_matrix = np.array(list(product((0.0, 1.0), repeat=nonbasis.size)), dtype=float).T
        lower_nonbasis = lower_bounds[nonbasis][:, None]
        span_nonbasis = (upper_bounds[nonbasis] - lower_bounds[nonbasis])[:, None]
        nonbasis_values = lower_nonbasis + span_nonbasis * assignment_matrix

        rhs_matrix = b_array[:, None] - A_array[:, nonbasis] @ nonbasis_values
        try:
            basis_values = np.linalg.solve(basis_matrix, rhs_matrix)
        except np.linalg.LinAlgError:
            continue

        feasible_basis = np.all(
            (basis_values >= lower_bounds[basis_array][:, None] - 1e-9)
            & (basis_values <= upper_bounds[basis_array][:, None] + 1e-9),
            axis=0,
        )
        if not np.any(feasible_basis):
            continue

        objective_values = c_array[basis_array] @ basis_values + c_array[nonbasis] @ nonbasis_values
        feasible_objectives = np.where(feasible_basis, objective_values, np.inf)
        best_position = int(np.argmin(feasible_objectives))
        objective = float(feasible_objectives[best_position])
        if not np.isfinite(objective) or objective >= best_value:
            continue

        candidate = np.zeros(n, dtype=float)
        candidate[basis_array] = basis_values[:, best_position]
        candidate[nonbasis] = nonbasis_values[:, best_position]
        if not np.allclose(A_array @ candidate, b_array, atol=1e-8, rtol=1e-8):
            continue

        best_value = objective
        best_x = candidate.copy()

    if best_x is None:
        raise InfeasibleProblemError("Infeasible.")

    return best_x, best_value


def primal_dual(
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    c: Sequence[float],
    lam: Sequence[float],
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Solve a standard-form LP and return a compatible dual solution."""
    A_array = as_2d_float_array(A, "A")
    b_array = as_1d_float_array(b, "b")
    c_array = as_1d_float_array(c, "c")
    lam_array = as_1d_float_array(lam, "lam")

    m, n = A_array.shape
    if b_array.size != m or c_array.size != n or lam_array.size != m:
        raise ValueError("Input dimensions do not agree.")
    best_x: NDArray[np.float64] | None = None
    best_lam: NDArray[np.float64] | None = None
    best_value = float("inf")

    for basis in combinations(range(n), m):
        basis_array = np.array(basis, dtype=int)
        basis_matrix = A_array[:, basis_array]

        try:
            x_basis = np.linalg.solve(basis_matrix, b_array)
            lam_candidate = np.linalg.solve(basis_matrix.T, c_array[basis_array])
        except np.linalg.LinAlgError:
            continue

        if np.any(x_basis < -1e-9):
            continue

        reduced_costs = c_array - A_array.T @ lam_candidate
        if np.any(reduced_costs < -1e-8):
            continue

        x_candidate = np.zeros(n, dtype=float)
        x_candidate[basis_array] = x_basis

        if not np.allclose(A_array @ x_candidate, b_array, atol=1e-8, rtol=1e-8):
            continue

        objective = float(c_array @ x_candidate)
        if objective < best_value:
            best_value = objective
            best_x = x_candidate.copy()
            best_lam = lam_candidate.copy()

    if best_x is None or best_lam is None:
        raise InfeasibleProblemError("Infeasible.")

    return best_x, best_value, best_lam
