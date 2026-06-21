from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from or_algos.optimization import pcgrad


@pytest.fixture()
def pcgrad_problem() -> tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]:
    x1, x2 = sp.symbols("x1 x2")
    return (x1 - 1) ** 2 + (x2 + 2) ** 2, (x1, x2)


def test_pcgrad_returns_finite_outputs(pcgrad_problem: tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]) -> None:
    f, _ = pcgrad_problem
    x_star, z_star, x_values, iterations = pcgrad(f, [0, 0], 2, "A", "F", 1e-10, 10)
    assert x_star.shape == (2,)
    assert iterations >= 1
    assert np.all(np.isfinite(x_values))
    assert np.isfinite(z_star)


def test_pcgrad_applies_defaults_when_optional_args_omitted(
    pcgrad_problem: tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]
) -> None:
    f, _ = pcgrad_problem
    x_star, z_star, x_values, iterations = pcgrad(f, [0, 0], 2)
    assert x_star.shape == (2,)
    assert iterations >= 1
    assert np.all(np.isfinite(x_values))
    assert np.isfinite(z_star)


def test_pcgrad_converges_on_quadratic(pcgrad_problem: tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]) -> None:
    f, _ = pcgrad_problem
    x_star, z_star, _, _ = pcgrad(f, [0, 0], 2)
    assert np.allclose(x_star, [1.0, -2.0], atol=1e-6, rtol=1e-6)
    assert z_star == pytest.approx(0.0, abs=1e-10)


def test_pcgrad_callable_api_matches_symbolic_api(pcgrad_problem: tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]) -> None:
    f_expr, _ = pcgrad_problem

    def f_callable(x: np.ndarray) -> float:
        return float((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2)

    def grad_callable(x: np.ndarray) -> np.ndarray:
        return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] + 2.0)], dtype=float)

    symbolic = pcgrad(f_expr, [0, 0], 2)
    callable_result = pcgrad(f_callable, [0, 0], 2, grad=grad_callable)

    assert np.allclose(callable_result[0], symbolic[0], atol=1e-8, rtol=1e-8)
    assert callable_result[1] == pytest.approx(symbolic[1], abs=1e-12)
    assert np.allclose(callable_result[2], symbolic[2], atol=1e-8, rtol=1e-8)
    assert callable_result[3] == symbolic[3]
