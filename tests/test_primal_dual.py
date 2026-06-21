from __future__ import annotations

import numpy as np
import pytest

from or_algos.optimization import primal_dual

PRIMAL_DUAL_EXAMPLES = [
    (
        "example_1",
        np.array([[1, 1, 2], [2, 1, 3]], dtype=float),
        np.array([3, 5], dtype=float),
        np.array([2, 1, 4], dtype=float),
        np.array([0, 0], dtype=float),
        5.0,
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
        np.array([1, 0, -1, 2, 4], dtype=float),
        65.24,
    ),
]


def test_primal_dual_returns_expected_small_solution() -> None:
    A = np.array([[1, 1, 2], [2, 1, 3]], dtype=float)
    b = np.array([3, 5], dtype=float)
    c = np.array([2, 1, 4], dtype=float)
    lam = np.array([0, 0], dtype=float)

    x_star, z_star, lam_star = primal_dual(A, b, c, lam)

    assert x_star.shape == (3,)
    assert lam_star.shape == (2,)
    assert np.allclose(x_star, [2.0, 1.0, 0.0], atol=1e-8, rtol=1e-8)
    assert z_star == pytest.approx(5.0)
    assert np.allclose(lam_star, [0.0, 1.0], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize(
    ("example_name", "A", "b", "c", "lam", "expected_objective"),
    PRIMAL_DUAL_EXAMPLES,
    ids=[example[0] for example in PRIMAL_DUAL_EXAMPLES],
)
def test_primal_dual_documented_examples_are_feasible_and_recompute_objective(
    example_name: str,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    lam: np.ndarray,
    expected_objective: float,
) -> None:
    del example_name
    x_star, z_star, lam_star = primal_dual(A, b, c, lam)
    assert np.allclose(A @ x_star, b, atol=1e-8, rtol=1e-8)
    assert np.all(x_star >= -1e-9)
    assert np.all(A.T @ lam_star <= c + 1e-8)
    assert z_star == pytest.approx(float(c @ x_star))
    assert z_star == pytest.approx(float(b @ lam_star), abs=1e-8)
    assert z_star == pytest.approx(expected_objective, abs=1e-8)


def test_primal_dual_matches_scipy_on_examples() -> None:
    scipy = pytest.importorskip("scipy.optimize")

    for _, A, b, c, lam, _ in PRIMAL_DUAL_EXAMPLES:
        result = scipy.linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * len(c), method="highs")
        assert result.success
        x_star, z_star, lam_star = primal_dual(A, b, c, lam)
        assert np.allclose(A @ x_star, b, atol=1e-8, rtol=1e-8)
        assert np.all(x_star >= -1e-9)
        assert z_star == pytest.approx(float(result.fun), abs=1e-8)
        assert np.all(A.T @ lam_star <= c + 1e-8)
