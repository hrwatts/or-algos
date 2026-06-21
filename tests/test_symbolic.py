from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from or_algos.optimization import evaluate_symbolic_at_point
from or_algos.optimization._common import DimensionMismatchError


def test_evaluate_symbolic_at_point_returns_scalar() -> None:
    x1, x2 = sp.symbols("x1 x2")
    expr = x1**2 + x2
    value = evaluate_symbolic_at_point(expr, (x1, x2), [3, 4])
    assert value == pytest.approx(13.0)


def test_evaluate_symbolic_at_point_returns_matrix_values() -> None:
    x1, x2 = sp.symbols("x1 x2")
    expr = sp.Matrix([x1 + x2, x1 - x2])
    value = evaluate_symbolic_at_point(expr, (x1, x2), [3, 1])
    assert np.allclose(value, [4.0, 2.0])


def test_evaluate_symbolic_at_point_checks_dimensions() -> None:
    x1, x2 = sp.symbols("x1 x2")
    expr = x1 + x2
    with pytest.raises(DimensionMismatchError):
        evaluate_symbolic_at_point(expr, (x1, x2), [1])
