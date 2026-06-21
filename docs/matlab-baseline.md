# MATLAB Baseline Status

Serialized MATLAB baseline fixtures for the optimization examples and tests are not included in this repository.

## Why it is missing

- Refreshing the baseline requires access to a licensed MATLAB environment.
- Until those fixtures are regenerated, direct parity against MATLAB outputs remains unverified.

## What was done instead

- the Python implementation was derived from the original MATLAB algorithms
- the MATLAB unit-test intent was ported into `pytest`
- the canonical optimization examples were carried over into Python example scripts and tests
- optional SciPy checks were added for the LP examples to provide an independent oracle where possible

When a licensed MATLAB environment is available, regenerate the corresponding fixtures and commit them under `fixtures/matlab_baseline/`.
