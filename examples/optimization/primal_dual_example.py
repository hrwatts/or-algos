"""Example problems for primal_dual."""

from __future__ import annotations

import numpy as np

from or_algos.optimization import primal_dual


def main() -> None:
    examples = [
        (
            np.array([[1, 1, 2], [2, 1, 3]], dtype=float),
            np.array([3, 5], dtype=float),
            np.array([2, 1, 4], dtype=float),
            np.array([0, 0], dtype=float),
        ),
        (
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
        ),
    ]

    for index, (A, b, c, lam) in enumerate(examples, start=1):
        x_star, z_star, lam_star = primal_dual(A, b, c, lam)
        print(f"Example {index}")
        print("x_star =", x_star)
        print("z_star =", z_star)
        print("lam_star =", lam_star)


if __name__ == "__main__":
    main()
