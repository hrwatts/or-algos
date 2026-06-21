# Testing

## Run the suite

```bash
py -3.10 -m pytest
```

## What is covered

- ported expectations from the MATLAB optimization tests
- exact line-search checks on the canonical quadratic used for source characterization
- symbolic helper dimension checks
- `pcgrad` convergence and callable-vs-symbolic parity
- LP feasibility, objective recomputation, and optional SciPy oracle comparisons on the source example problems
- public import surface checks

## Current limitations

- MATLAB baseline fixtures are not committed in this repository.
- MATLAB parity is therefore not established yet.
- The LP tests therefore rely on source examples, invariant checks, and optional SciPy cross-checks rather than direct MATLAB output fixtures.
