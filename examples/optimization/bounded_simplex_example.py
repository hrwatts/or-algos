"""Example problems for bounded_simplex."""

from __future__ import annotations

import numpy as np

from or_algos.optimization import bounded_simplex


def main() -> None:
    examples = [
        (
            np.array([[1, 0, 1, -1, 2], [0, 1, 2, 2, 1]], dtype=float),
            np.array([2, 3], dtype=float),
            np.array([2, 1, 3, -2, 10], dtype=float),
            np.array([-1, -1, -1, -1, -1], dtype=float),
            np.array([6, 9, 0, 4, 2], dtype=float),
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
            np.array([0, -1, -2, 0, 1, -4, -1, 4, 0, 0, -4, 0, -2, -2, 1], dtype=float),
            np.array([4, 4, 1, 5, 6, 0, 4, 10, 4, 6, 4, 6, 4, 8, 8], dtype=float),
        ),
        (
            np.array([[1, 1], [2, 1]], dtype=float),
            np.array([6, 8], dtype=float),
            np.array([2, 0], dtype=float),
            np.array([1, 1], dtype=float),
            np.array([5, 4], dtype=float),
        ),
    ]

    for index, (A, b, c, d, h) in enumerate(examples, start=1):
        x_star, z_star = bounded_simplex(A, b, c, d, h)
        print(f"Example {index}")
        print("x_star =", x_star)
        print("z_star =", z_star)


if __name__ == "__main__":
    main()
