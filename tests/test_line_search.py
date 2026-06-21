from __future__ import annotations

import pytest
import sympy as sp

from or_algos.optimization import armijo, goldstein, wolfe


@pytest.fixture()
def quadratic_expr() -> tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]:
    x1, x2 = sp.symbols("x1 x2")
    return x1**2 + x2**2, (x1, x2)


def test_line_search_returns_positive_step_sizes(quadratic_expr: tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]) -> None:
    f, _ = quadratic_expr
    assert armijo(f, [1, 1], [-1, -1]) > 0
    assert goldstein(f, [1, 1], [-1, -1]) > 0
    assert wolfe(f, [1, 1], [-1, -1]) > 0


def test_line_search_accepts_explicit_none_optional_args(
    quadratic_expr: tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]
) -> None:
    f, _ = quadratic_expr
    assert armijo(f, [1, 1], [-1, -1], None, None) > 0
    assert goldstein(f, [1, 1], [-1, -1], None, None) > 0
    assert wolfe(f, [1, 1], [-1, -1], None, None) > 0


def test_line_search_matches_source_quadratic_steps(
    quadratic_expr: tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]
) -> None:
    f, _ = quadratic_expr
    assert armijo(f, [1, 1], [-1, -1]) == pytest.approx(2.0)
    assert goldstein(f, [1, 1], [-1, -1]) == pytest.approx(2.0)
    assert wolfe(f, [1, 1], [-1, -1]) == pytest.approx(2.0 / 3.0)


def test_callable_api_matches_symbolic_api(quadratic_expr: tuple[sp.Expr, tuple[sp.Symbol, sp.Symbol]]) -> None:
    f_expr, _ = quadratic_expr

    def f_callable(x: list[float]) -> float:
        return x[0] ** 2 + x[1] ** 2

    def grad_callable(x: list[float]) -> list[float]:
        return [2.0 * x[0], 2.0 * x[1]]

    assert armijo(f_callable, [1, 1], [-1, -1], grad=grad_callable) == pytest.approx(
        armijo(f_expr, [1, 1], [-1, -1])
    )
    assert goldstein(f_callable, [1, 1], [-1, -1], grad=grad_callable) == pytest.approx(
        goldstein(f_expr, [1, 1], [-1, -1])
    )
    assert wolfe(f_callable, [1, 1], [-1, -1], grad=grad_callable) == pytest.approx(
        wolfe(f_expr, [1, 1], [-1, -1])
    )
