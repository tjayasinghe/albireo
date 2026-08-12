# Contributing to albireo

albireo is pre-alpha and the API is unstable, so the most useful contributions right now are bug
reports with a reproducer, and small focused pull requests. If you are planning something larger,
open an issue first — the design is deliberate and it is worth checking that a change fits before
you write it.

## Development install

Python **3.12 or newer** is required (current jaxlib no longer ships cp311 wheels).

```bash
git clone https://github.com/tjayasinghe/albireo.git
cd albireo
pip install -e ".[dev]"
pre-commit install
```

JAX's x64 mode is enabled at `import albireo`, so all computation is float64. This is mandatory,
not a preference: the adjoint, log-determinant, and closed-loop tolerance tests do not hold in
float32. There is an escape hatch for experiments (`ALBIREO_DISABLE_X64`), but code and tests must
be correct with x64 on.

For a GPU build, install the `jax[cuda]` wheel for your platform following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html). CPU correctness comes
first; the GPU is an accelerator, never a separate code path.

## Running the tests

```bash
pytest                       # whole suite
pytest tests/test_grids.py   # one module, seconds
```

Most of the suite is fast, but the inference tests are not: the NUTS acceptance test does real
sampling and takes a few minutes, and several M4 closed-loop tests run MAP fits of 30–80 seconds
each. CI allows 45 minutes for the suite. The injection–coverage study
(`scripts/m3_coverage.py`, ~100 minutes) is deliberately *not* part of the suite — run it by hand
when you touch the sampler, the marginal likelihood, or the hyperparameter treatment.

Tests are deterministic (fixed seeds). A test that only passes sometimes is a bug in the test.

## Linting and formatting

```bash
ruff check .
ruff format .
mypy src        # optional locally; type hints are expected on public API
```

CI runs `ruff check .` and `ruff format --check .`; line length is 100. `pre-commit` runs the same
hooks, so installing it is the cheapest way to keep CI green.

## What a pull request needs

The simulator is the oracle. The project's test philosophy (docs/design.md §7) is that

- **every inference feature ships with a closed-loop test** against simulated data with known
  injected truth, asserting recovery to a stated tolerance;
- **every linear operator ships with an adjoint test** (inner-product identity against
  `jax.linear_transpose`) and a gradient check against central finite differences;
- **calibration claims are demonstrated, not asserted** — coverage/SBC runs back any statement
  about posterior calibration;
- **performance work is benchmark-gated**: no optimization lands without a recorded baseline in
  `docs/benchmarks.md`, and the resulting number goes into the same file.

Beyond that: public API gets type hints and docstrings; new numerical results (positive *and*
negative) get a row or a paragraph in `docs/benchmarks.md`, which is a running record rather than a
highlight reel.

A guard is better than a silent approximation. Where a parameter can leave the regime a
build-time-static structure was built for — solver bandwidth, LSF kernel radius, eccentricity —
the model returns a non-finite log-density instead of a quietly wrong number. Keep that pattern.

## Where the design decisions live

- `docs/design.md` §2 is the **decision ledger** (D1–D26): every default, with its rationale and
  the alternative that was rejected. If your change alters a recorded default, update the row in
  the same PR and say why.
- `docs/design.md` §1 covers prior art, §5 the degeneracy policy, §8 the milestones.
- `docs/math.md` holds all the equations, and §8 maps each mathematical claim to the test that
  asserts it. New maths belongs there, with a test in the traceability table.
- `docs/benchmarks.md` is the validation and performance record.

The degeneracy policy is worth restating because it constrains API design: never silently
regularize away a real degeneracy. Make it proper with an explicit prior scale, report it in the
posterior, and where only external information can break it, require the user to choose (this is
why, for example, there is no default light ratio).

## License

albireo is BSD 3-Clause. By contributing you agree that your contributions are licensed under the
same terms. See [`LICENSE`](LICENSE).
