# Disentangle an SB2 end to end

This tutorial follows the shortest complete path through albireo: simulate a double-lined
spectroscopic binary, recover its orbit with the component spectra marginalized analytically, and
compare the result with the injected truth. Every code block below is taken verbatim from
[`examples/01_sb2_end_to_end.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/01_sb2_end_to_end.py),
which ends in `assert` statements and therefore also serves as a slow smoke test of the stack.

See the [science overview](../science.md) for background and references.

```text
MarginalOrbitModel        the marginal posterior over the orbit (spectra integrated out)
  -> run_map              MAP over theta and the spectral hyperparameters (ML-II)
  -> laplace_inverse_mass the Hessian at the MAP, as the NUTS mass matrix
  -> run_nuts             sample the orbit with the hyperparameters held fixed
  -> posterior_spectra    component spectra drawn from the joint posterior
```

!!! note "Runtime"

    A few minutes on one desktop CPU with `ALBIREO_EXAMPLE_FAST=1`
 (10 epochs, 100 warmup / 150
    samples); the default size (12 epochs, 150 / 250) costs roughly 2.5 times that. NUTS
    dominates: it is a few thousand gradient evaluations, each one a banded Cholesky
    factorization in float64, so the wall time tracks whatever linear-algebra backend JAX is
    using, and a GPU build shortens the sampling phase considerably. Use the fast switch in CI.

## 1. The system and the assumed quantities

```python
GRID = ab.LogGrid.from_wavelength_range(5000.0, 5060.0, dv_kms=5.5)
P_TRUE = 6.31  # orbital period [d]
TCONJ_TRUE = 2.05  # conjunction of component 1 (nu + omega = pi/2) [d]
ECC_TRUE = 0.20
OMEGA_TRUE = 0.70  # argument of periastron of component 1 [rad]
K_TRUE = np.array([32.0, 24.0])  # (K_1, K_2) [km/s]
ELL = np.array([0.62, 0.38])  # continuum light fractions: assumed, not inferred
LSF = {"HERMES": 7.0}  # Gaussian LSF width [km/s], per instrument
SNR = 120.0  # per-pixel continuum signal-to-noise
N_EPOCHS = 10 if FAST else 12
SEED = 20260811
```

The model grid is uniform in $`\ln\lambda`$, so a Doppler shift is a pure translation and every
operator in the forward model is a convolution or a shift. 5.5 km/s per pixel over 60 Å gives
652 pixels, roughly one échelle order in the green, and enough to carry an orbit.

`ELL` is the one entry in that block that is an assumption rather than a measurement. With
constant light fractions the likelihood depends only on the products $`\ell_i d_i`$, so the
continuum light ratio and the component line depths are exactly degenerate
([`docs/math.md`](../math.md) §5.2). albireo does not estimate $`\ell`$: it is either fixed by
assumption (as here), supplied through an external photometric prior, or broken by eclipses
through per-epoch light fractions. The fit is conditional on that choice, and the recovered
depths scale as $`1/\ell_i`$ if the choice is wrong.

The solver's bandwidth is static, which is what allows the whole marginal likelihood to be
compiled inside a single `jax.jit`:

```python
V_REL_MAX = float(K_TRUE.sum()) * (1.0 + ECC_TRUE) * 1.35
```

$`(K_1 + K_2)(1 + e)`$ is the largest relative velocity the pair reaches; the factor 1.35 is
headroom. Orbits that would exceed the bound are rejected by the numpyro model with a $`-\infty`$
factor rather than mis-solved, so a prior wider than `v_rel_max_kms` costs mixing efficiency
near the bound, not correctness.

## 2. Simulate the data

The simulator pushes the injected spectra through the same operator stack the inference code
uses (shift, LSF convolution, rebin onto the native pixel grid) and then adds the observational
features the package handles.

```python
    return ab.simulate_dataset(
        GRID,
        components,
        bjd=bjd,
        instruments={"HERMES": instrument},
        light_fractions=ELL,
        orbit=orbit,
        v_bary=v_bary,
        frame="topocentric",
        gap_fraction=0.01,  # one contiguous chip gap per epoch (ivar = 0, flux unused)
        cosmic_fraction=0.002,  # a few masked cosmic hits per epoch
        seed=11,
    )
```

`frame="topocentric"` means the wavelengths are as observed: the barycentric correction is
composed inside the forward model rather than applied to the data, so nothing is resampled. The
chip gap carries unused flux values at the masked pixels, so any code path that failed to treat
`ivar == 0` as "ignore" would produce a diverging fit rather than a slightly degraded one.

## 3. The marginal model

```python
    model = ab.MarginalOrbitModel(
        GRID,
        dataset,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        v_rel_max_kms=V_REL_MAX,
    )
```

This object holds the static problem structure (rebin operators, kernels, weights, built once)
and exposes a single jit-compiled, differentiable function of the orbital parameters $`\theta`$.
The $`2 \times 652`$ component-spectrum pixels never appear as sampling parameters: at every
$`\theta`$ they are integrated out in closed form through a block-tridiagonal Cholesky
factorization ([`docs/math.md`](../math.md) §3, §4.2). NUTS therefore explores six dimensions.

The priors are ordinary numpyro distributions, one per site:

```python
PRIORS = {
    "period": dist.Normal(P_TRUE + 0.001, 0.003),
    "t_conj": dist.Normal(TCONJ_TRUE + 0.005, 0.02),
    "secosw": dist.Uniform(-1.0, 1.0),
    "sesinw": dist.Uniform(-1.0, 1.0),
    "k": dist.Uniform(jnp.array([10.0, 5.0]), jnp.array([45.0, 40.0])),
    "log_tau": dist.Normal(jnp.full(2, np.log(300.0)), 3.0),
    "log_eta": dist.Normal(jnp.full(2, np.log(5.0)), 3.0),
}
```

`period` and `t_conj` carry a photometric ephemeris, offset from the truth so that nothing is
initialized at the answer. $`(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)`$ is smooth through $`e = 0`$,
where $`\omega`$ and a time of periastron are undefined, and a uniform prior on the unit disk is a
uniform prior on $`e`$. `log_tau` and `log_eta` are the spectral-prior hyperparameters: the
curvature scale of the component spectra and the ridge that makes the unconstrained
low-frequency directions proper.

## 4. MAP, and the hyperparameters by ML-II

```python
    map_fit = ab.run_map(model.model(PRIORS), init=INIT, max_steps=300)
```

L-BFGS on numpyro's potential, in unconstrained space. Because the spectra are already
marginalized out of the likelihood, maximizing over `log_tau` and `log_eta` alongside the orbit
is ML-II (empirical Bayes) up to the weak hyperpriors. The prior curvature scale is information
the data cannot supply below the LSF width ([`docs/math.md`](../math.md) §5.1), so it is
estimated explicitly rather than left to a default.

A representative pair of lines from the run:

```text
MAP: 155 L-BFGS steps, converged=True, |grad|=6.40e-03, potential=-15040.64  [42.5 s]
  K = [31.997 23.957]  e = 0.1993  tau = [396.  366.1]  eta = [0.89 3.33]
```

`max_steps=300` raises the 200-step default. The orbital sites are determined within the first
few dozen steps; the remaining steps move along the hyperparameter directions, where the
marginal likelihood is nearly flat. How many steps that takes to cross the `tol=1e-2`
gradient-norm criterion depends on floating-point details of the linear-algebra backend, and
this fit has needed between about 150 and 215, so the example raises the cap rather than
reporting `converged=False` intermittently. That flag describes the optimizer, not the orbit:
the potential and the recovered $`K`$ agree to the printed precision either way. The MAP is used
as a starting point and a curvature estimate; the posterior is the reported result.

## 5. From the MAP to NUTS

```python
    hyper = {s: map_fit.params[s] for s in ("log_tau", "log_eta")}
    orbit_priors = {s: d for s, d in PRIORS.items() if s not in hyper}
    nuts_model = model.model(orbit_priors, fixed=hyper)
```

`fixed=` injects the ML-II values as constants instead of sampling them. The hyperparameters may
instead be left in `priors` and sampled; albireo supports both, but the reported orbital
uncertainties then include the hyperparameter uncertainty, and the chain costs noticeably more.
The empirical-Bayes route is the default because the plug-in optimism is a few percent in
coverage and is documented as such.

```python
    inverse_mass = ab.laplace_inverse_mass(nuts_model, map_fit.params)
```

The Hessian of the potential at the MAP, symmetrized, eigenvalue-floored and inverted, gives a
dense mass matrix. Without it, warmup has to discover from scratch that `period` is constrained
at the $`10^{-3}`$ level while `k` is constrained at $`10^{-1}`$, and early trajectories run into
the tree-depth cap. With it, warmup only tunes the step size. Mass adaptation defaults to off
when an explicit matrix is supplied, so that the early adaptation windows do not overwrite it
with a poor few-sample estimate.

```python
    mcmc = ab.run_nuts(
        nuts_model,
        rng_key=jax.random.PRNGKey(3),
        init=map_fit.params,
        inverse_mass_matrix=inverse_mass,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
    )
```

The example runs a single chain to stay short. For science, run at least two and check $`\hat R`$;
`run_nuts` collects `num_steps` and `diverging` as extra fields either way.

## 6. Results

```text
NUTS: 100 warmup + 150 samples x 1 chain(s), 0 divergences, 7 leapfrogs/sample  [140.7 s]

=== orbital posterior vs truth ===
  parameter      truth   post. mean         sd  rel. err      z
---------------------------------------------------------------
      P [d]    6.31000      6.31080    0.00159    0.013%   0.50
 t_conj [d]    2.05000      2.04930    0.00292    0.034%  -0.24
 K_1 [km/s]   32.00000     31.99525    0.04327    0.015%  -0.11
 K_2 [km/s]   24.00000     23.95056    0.08107    0.206%  -0.61
          e    0.20000      0.19933    0.00123    0.333%  -0.54
```

Seven leapfrog steps per sample is what a well-scaled mass matrix gives. The `z` column is the
pull, $`(\text{mean} - \text{truth})/\text{sd}`$, which should be $`\mathcal{O}(1)`$ if the
posterior is calibrated; the repository's injection-coverage study
([`docs/benchmarks.md`](../benchmarks.md)) tracks it over many injections rather than one.

The script's gate is looser than the numbers above:

```python
    k_draws = np.asarray(samples["k"])
    for i in range(2):
        rel = abs(float(k_draws[:, i].mean()) - K_TRUE[i]) / K_TRUE[i]
        assert rel < 0.02, f"K_{i + 1} off by {100 * rel:.2f}% (tolerance: 2%)"
```

## 7. The component spectra and the k = 0 degeneracy

```python
    spectra = ab.posterior_spectra(model, samples, jax.random.PRNGKey(9), num_draws=24, extra=hyper)
```

Each draw picks a posterior $`\theta`$ at random and then draws once from the conditional Gaussian
over the spectra, so the returned scatter carries both the spectral and the orbital uncertainty
rather than a bootstrap around a point estimate. `extra=hyper` supplies the sites that were
fixed during sampling.

The per-component spectra are not fully determined by the data. In Fourier space the difference
mode between the two components has information $`\propto k^2\,\mathrm{Var}_j(\Delta_j)`$,
which vanishes at $`k = 0`$: a constant added to $`d_1`$ and subtracted (light-weighted) from $`d_2`$
changes no epoch's prediction ([`docs/math.md`](../math.md) §5.1). The smooth envelope of each
component is therefore set by the prior rather than measured. The example prints both sides of
this:

```text
posterior spectra: (24, 2, 652) draws  [5.0 s]
  RMS error in line cores: component 1 0.0513, component 2 0.0838
  RMS error in the light-weighted sum (the observable): 0.0071  (the k = 0 degeneracy cancels here)
```

The light-weighted combination, which is the quantity the spectrograph recorded, is recovered an
order of magnitude better than either component alone. That difference is the degeneracy.
Absolute depths per component require eclipses or photometry, not a longer chain.

## 8. Figures

If matplotlib is importable, the script writes `sb2_rv_curve.png` (posterior orbit draws
phase-folded against the injected Keplerian) and `sb2_spectra.png` (posterior mean
$`\pm 2\sigma`$ bands against the truth) to the working directory. matplotlib is not a dependency
of albireo, and the guard is:

```python
    if importlib.util.find_spec("matplotlib") is not None:
        plot_rv_curve(samples, truth, dataset.bjd, "sb2_rv_curve.png")
        plot_spectra(spectra_np, truth, "sb2_spectra.png")
        print("\nwrote sb2_rv_curve.png and sb2_spectra.png")
```

The RV figure carries one caveat: albireo never measures a per-epoch radial velocity. There are
no RV points to plot, only the posterior over the Keplerian that the spectra imply, with tick
marks showing where the epochs constrain it.

## Run it yourself

```bash
python examples/01_sb2_end_to_end.py

# or, CI-sized:
ALBIREO_EXAMPLE_FAST=1 python examples/01_sb2_end_to_end.py
```

On Windows, `$env:ALBIREO_EXAMPLE_FAST = "1"` sets the same switch. The script exits non-zero if
$`K_1`$ or $`K_2`$ differs from the truth by more than 2%, so it can be run directly in CI.

Next: [find a companion that never shows a second set of lines](k2-scan.md).
