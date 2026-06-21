from __future__ import annotations

import numpy as np
import pytest

from or_algos.optimization import bounded_simplex

BASIC_BOUNDED_SIMPLEX_EXAMPLES = [
    (
        "example_1",
        np.array([[1, 0, 1, -1, 2], [0, 1, 2, 2, 1]], dtype=float),
        np.array([2, 3], dtype=float),
        np.array([2, 1, 3, -2, 10], dtype=float),
        np.array([-1, -1, -1, -1, -1], dtype=float),
        np.array([6, 9, 0, 4, 2], dtype=float),
        -2.0,
    ),
    (
        "example_2",
        np.array(
            [
                [1, -2, 1, 3, 2, -3, 4, 1, 0, 1, 2, 4, 4, -1, 0],
                [0, 2, 4, -1, 0, 1, 2, -1, 1, 2, 0, 2, 1, -2, 4],
                [2, 1, 0, 1, 2, -1, -2, 2, 1, 0, -1, -2, 1, 0, 1],
                [-2, -4, 2, 1, -2, 1, 0, 1, 2, -1, 0, 0, 2, 1, -1],
                [1, 0, 2, 4, 1, 0, 1, -1, 1, 2, 0, 1, -1, 2, -2],
            ],
            dtype=float,
        ),
        np.array([37, 30, 8, -4, 8], dtype=float),
        np.array([1, -10, 15, 20, 0, 0, 12, 3, 10, 8, 4, 12, 3, 10, -10], dtype=float),
        np.array([0, -1, -2, 0, 1, -4, -1, 4, 0, 0, -4, 0, -2, -2, 1], dtype=float),
        np.array([4, 4, 1, 5, 6, 0, 4, 10, 4, 6, 4, 6, 4, 8, 8], dtype=float),
        88.51895991332611,
    ),
    (
        "example_3",
        np.array([[1, 1], [2, 1]], dtype=float),
        np.array([6, 8], dtype=float),
        np.array([2, 0], dtype=float),
        np.array([1, 1], dtype=float),
        np.array([5, 4], dtype=float),
        4.0,
    ),
]


def test_bounded_simplex_returns_expected_small_solution() -> None:
    A = np.array([[1, 1], [2, 1]], dtype=float)
    b = np.array([6, 8], dtype=float)
    c = np.array([2, 0], dtype=float)
    d = np.array([1, 1], dtype=float)
    h = np.array([5, 4], dtype=float)

    x_star, z_star = bounded_simplex(A, b, c, d, h)

    assert x_star.shape == (2,)
    assert np.allclose(x_star, [2.0, 4.0], atol=1e-8, rtol=1e-8)
    assert z_star == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("example_name", "A", "b", "c", "d", "h", "expected_objective"),
    BASIC_BOUNDED_SIMPLEX_EXAMPLES,
    ids=[example[0] for example in BASIC_BOUNDED_SIMPLEX_EXAMPLES],
)
def test_bounded_simplex_documented_examples_are_feasible_and_recompute_objective(
    example_name: str,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    h: np.ndarray,
    expected_objective: float,
) -> None:
    del example_name
    x_star, z_star = bounded_simplex(A, b, c, d, h)
    assert np.allclose(A @ x_star, b, atol=1e-8, rtol=1e-8)
    assert np.all(x_star >= d - 1e-9)
    assert np.all(x_star <= h + 1e-9)
    assert z_star == pytest.approx(float(c @ x_star))
    assert z_star == pytest.approx(expected_objective, abs=1e-8)


def test_bounded_simplex_matches_scipy_on_examples() -> None:
    scipy = pytest.importorskip("scipy.optimize")

    for _, A, b, c, d, h, _ in BASIC_BOUNDED_SIMPLEX_EXAMPLES[:2]:
        result = scipy.linprog(
            c,
            A_eq=A,
            b_eq=b,
            bounds=list(zip(d, h, strict=True)),
            method="highs",
        )
        assert result.success
        x_star, z_star = bounded_simplex(A, b, c, d, h)
        assert np.allclose(A @ x_star, b, atol=1e-8, rtol=1e-8)
        assert np.all(x_star >= d - 1e-9)
        assert np.all(x_star <= h + 1e-9)
        assert z_star == pytest.approx(float(result.fun), abs=1e-8)
