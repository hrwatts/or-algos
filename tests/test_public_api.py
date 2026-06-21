from __future__ import annotations

import or_algos
from or_algos import optimization


def test_public_exports_are_available() -> None:
    expected = {
        "armijo",
        "bounded_simplex",
        "evaluate_symbolic_at_point",
        "goldstein",
        "pcgrad",
        "primal_dual",
        "wolfe",
    }
    assert expected.issubset(set(or_algos.__all__))
    assert expected.issubset(set(optimization.__all__))
