# Contributing to albireo

albireo is pre-alpha and the API is unstable. The most useful contributions at present are bug
reports with a reproducer and small, focused pull requests. For anything larger, open an issue
first to check that the change fits the recorded design.

## Development install

Python **3.12 or newer** is required (current jaxlib no longer ships cp311 wheels).

```bash
git clone https://github.com/tjayasinghe/albireo.git
cd albireo
pip install -e ".[dev]"
pre-commit install
```

JAX's x64 mode is enabled at `import albireo`, so all computation is float64. This is mandatory:
the adjoint, log-determinant, and closed-loop tolerance tests do not hold in float32. The
environment variable `ALBIREO_DISABLE_X64` turns it off for experiments, but code and tests must
be correct with x64 on.

For a GPU build, install the `jax[cuda]` wheel for the platform, following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html). The GPU runs the same
code path as the CPU, and correctness is established on the CPU first.

## Running the tests

```bash
pytest                       # whole suite, including the acceptance tests (~14 min)
pytest --no-slow             # everything except the acceptance tests (~6 min)
pytest tests/test_grids.py   # one module, seconds
```

Most of the suite is fast, but the inference tests are not. The NUTS acceptance test does real
sampling and takes a few minutes, and several closed-loop tests run MAP fits of 20–50 seconds
each. Those are marked `slow`, and `--no-slow` deselects them, which skips the expensive fixtures
as well as the assertions on them. There are also `network` and `gpu` markers with matching
`--no-network` and `--no-gpu` flags.

A bare `pytest` runs everything, so the acceptance tests are not skipped by default. In CI the
routine workflow runs `--no-slow --no-network`. The on-demand `Full` workflow runs the same
selection on the OS/Python matrix and the whole suite once with coverage.

The injection-coverage study (`scripts/m3_coverage.py`, ~100 minutes) is not part of the suite; run
it manually after changes to the sampler, the marginal likelihood, or the hyperparameter treatment.

New test modules can use the shared fixtures in `tests/conftest.py` (`rng`, `small_grid`,
`small_simulation`, `small_dataset`). The existing modules build their own datasets and should
stay that way: their closed-loop tolerances are tuned to the specific scales of those datasets.

Tests are deterministic (fixed seeds). A test that passes only intermittently is treated as a
defect in the test.

## Linting and formatting

```bash
ruff check .
ruff format .
mypy src        # optional locally; type hints are expected on public API
```

CI runs `ruff check .` and `ruff format --check .`; line length is 100. `pre-commit` runs the same
hooks, so installing it reports lint failures before CI does.

## What a pull request needs

Correctness is established against the simulator: data are generated with known injected
values and the code must recover them. The testing policy (internal/design.md §7) is:

- **Every inference feature has a closed-loop test** against simulated data with known injected
  values, asserting recovery to a stated tolerance.
- **Every linear operator has an adjoint test** (inner-product identity against
  `jax.linear_transpose`) and a gradient check against central finite differences.
- **Calibration claims are demonstrated.** Coverage and simulation-based calibration (SBC) runs
  support any statement about posterior calibration.
- **Performance work requires a benchmark.** An optimization is merged only with a recorded
  baseline in `docs/benchmarks.md`, and the resulting number goes into the same file.

In addition, public API has type hints and docstrings. New numerical results, positive and
negative, are added to `docs/benchmarks.md`, which records all results and not only the favourable
ones.

Where a parameter can leave the regime for which a build-time-static structure was built (solver
bandwidth, LSF kernel radius, eccentricity), the model returns a non-finite log-density instead of
a wrong value. New code should follow this pattern: a guard is preferred to an approximation that
fails without a symptom.

## Where the design decisions are recorded

- `internal/design.md` §2 is the **decision record**: every default, with its rationale and
  the alternative that was rejected. A change that alters a recorded default should update the row
  in the same PR and say why.
- `internal/design.md` §1 covers prior art, §5 the degeneracy policy, §8 the milestones.
- `docs/math.md` holds the equations, and §8 maps each mathematical claim to the test that
  asserts it. New mathematics belongs there, with a test in the traceability table.
- `docs/benchmarks.md` is the validation and performance record.

The degeneracy policy constrains API design: a real degeneracy is never regularized away without
notice. It is made proper with an explicit prior scale and reported in the posterior, and where only
external information can break it, the user is required to choose. For example, there is no
default light ratio.

## License

albireo is BSD 3-Clause. Contributions are accepted under the same terms. See
[`LICENSE`](LICENSE).
