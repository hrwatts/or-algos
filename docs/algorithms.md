# Algorithms

## Symbolic helpers

- `or_algos.optimization.evaluate_symbolic_at_point`

The MATLAB source relied on `symvar`, `gradient`, and `subs`. The Python port keeps a public symbolic helper and uses SymPy `lambdify` internally so repeated evaluations do not pay repeated substitution cost.

## Line search

- `or_algos.optimization.armijo`
- `or_algos.optimization.goldstein`
- `or_algos.optimization.wolfe`

These functions preserve the MATLAB positional call shape:

```python
armijo(f, x0, d, epsilon=None, eta=None)
```

They also accept keyword-only `vars=` and `grad=` hooks so Python callables can be used without forcing a symbolic objective.

## Partial conjugate gradient

- `or_algos.optimization.pcgrad`

This routine preserves the MATLAB controls:

- `m`: restart window
- `line_search`: `"A"`, `"G"`, or `"W"`
- `method`: `"F"` or `"P"`
- `tolerance`
- `max_iter`

`pcgrad` accepts either a SymPy scalar expression or a callable objective plus an explicit gradient callable.

## Linear programming

- `or_algos.optimization.bounded_simplex`
- `or_algos.optimization.primal_dual`

The MATLAB repo described these as prototype research implementations. In this Python repo, both routines preserve the MATLAB-compatible signatures and the same LP problem-level intent, but they do not preserve the original tableau pivot mechanics.

## LP implementation note

The Python LP routines use exact basis search over small dense candidate bases instead of replaying the original tableau pivot steps.

- This keeps the public call shapes intact while making the Python examples and tests solve the intended LPs reliably.
- This is a contract change at the implementation level: the signatures and solved problem classes are preserved, but step-by-step tableau behavior is not.
- Expected limits: this approach is suitable for the small educational and research examples carried over from the MATLAB repo, but it is not intended to be a production LP engine.
