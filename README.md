# or-algos

Python implementations of small optimization routines adapted from an earlier MATLAB codebase.

## Included algorithms

- `evaluate_symbolic_at_point`: SymPy evaluation helper compatible with the MATLAB symbolic workflow
- `armijo`, `goldstein`, `wolfe`: line-search rules for symbolic or callable objectives
- `pcgrad`: partial conjugate-gradient method
- `bounded_simplex`: bounded linear-program solver for `min c^T x` with `Ax = b` and `d <= x <= h`
- `primal_dual`: standard-form linear-program solver for `min c^T x` with `Ax = b` and `x >= 0`

For the LP routines, the Python port preserves the MATLAB-compatible signatures and the same optimization problem formulations, but it does not preserve the original tableau pivot mechanics.

## Install

```bash
py -3.10 -m pip install -e .[dev]
```

## Quick usage

```python
import sympy as sp

from or_algos.optimization import pcgrad, primal_dual

x1, x2 = sp.symbols("x1 x2")
f = (x1 - 1) ** 2 + (x2 + 2) ** 2
x_star, z_star, x_values, iterations = pcgrad(f, [0, 0], 2)

A = [[1, 1, 2], [2, 1, 3]]
b = [3, 5]
c = [2, 1, 4]
lam = [0, 0]
lp_x, lp_z, lp_lam = primal_dual(A, b, c, lam)
```

## Validation scope

- The Python package is tested directly with `pytest`.
- MATLAB baseline fixtures are not included in this repository.
- Direct MATLAB parity is not established yet.
- The current validation focuses on translated examples, Python-side invariants, and optional SciPy oracle checks for the LP examples.

See [docs/algorithms.md](docs/algorithms.md), [docs/testing.md](docs/testing.md), and [docs/matlab-baseline.md](docs/matlab-baseline.md).
