# Find a hidden companion with the K2 scan

A single-lined binary is a binary whose second star is not visible in the spectrum. Sometimes
the companion is dark (a neutron star, a black hole, a stripped helium core), and sometimes its
lines are a few percent deep and hidden under a bright primary. The $K_2$ scan distinguishes the
two cases, and it is the workflow behind the dormant compact-object searches.

Every code block below is taken verbatim from
[`examples/02_k2_scan.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/02_k2_scan.py),
which ends in `assert` statements and also serves as a smoke test.

See the [science overview](../science.md) for background and references.

!!! note "Runtime"

    Under ten seconds either way: about 7 s with `ALBIREO_EXAMPLE_FAST=1` (10 epochs, 4 km/s
    grid) and about 8 s at the default size (12 epochs, 2 km/s grid). There is no sampler here.
    The scan is a profile over one scalar, and each trial costs a single linear solve.

## The method

The SB1 solution is given: $P_{\rm orb}$, $T_{\rm conj}$, $e$, $\omega$ and $K_1$, all fixed. The
only remaining unknown about the putative companion's orbit is $K_2$. The companion's spectrum,
however unknown, enters the forward model linearly, so at each trial $K_2$ it is marginalized
analytically, exactly as the component spectra are in the SB2 case. That gives a detection
statistic

$$
D(K_2) = 2\left[\log p(y \mid K_2) - \log p(y \mid \text{no companion})\right]
$$

at the cost of one linear solve per grid point. It is the matched filter integrated over every
possible companion spectrum, so it needs no template library, and the recovered companion
spectrum with its pointwise uncertainty follows from the conditional Gaussian at the peak
([`docs/math.md`](../math.md) §6).

## 1. Set up the system

```python
GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
P_TRUE = 6.31  # orbital period [d]
ECC_TRUE = 0.15
OMEGA_TRUE = 0.70  # argument of periastron of the primary [rad]
T_PERI_TRUE = 2.0
K1_TRUE = 12.0  # primary semi-amplitude [km/s], known from the SB1 solution
K2_TRUE = 38.0  # companion semi-amplitude [km/s], the quantity the scan must recover
ELL = (0.9, 0.1)  # (ell_1, ell_2): assumed, not fitted; see the module docstring
LSF = {"a": 7.0}
SNR = 150.0
N_EPOCHS = 10 if FAST else 12
K2_STEP = 4.0 if FAST else 2.0
K2_GRID = np.arange(10.0, 70.0, K2_STEP)
SEED = 7
```

$K_1 = 12$ km/s against $K_2 = 38$, that is $M_1/M_2 = K_2/K_1 \approx 3.2$, with the companion
contributing 10% of the continuum. That 10% is `ELL[1]`, and it controls the interpretation of
the recovered spectrum; see [§4](#4-limits-of-the-recovered-companion-spectrum).

The spectral prior is passed explicitly rather than fitted, because the scan is a profile over
$K_2$, not a joint fit:

```python
PRIOR = ab.SmoothnessPrior(jnp.asarray([300.0, 30.0]), jnp.asarray([5.0, 5.0]))
```

The companion is given the stiffer curvature scale ($\tau_2 = 30$ against $\tau_1 = 300$). This
is a modelling choice that $D$ depends on, which is why the statistic must be calibrated
empirically rather than read off a $\chi^2$ table.

## 2. Run the scan

The SB1 solution goes in as a mapping in albireo's orbital parameterization, with `k1` supplied
separately, because the scan profiles over the companion's semi-amplitude only:

```python
    return ab.k2_scan(
        GRID,
        dataset,
        orbit=sb1_solution(),
        k1=K1_TRUE,
        k2_grid=K2_GRID,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        prior=PRIOR,
        v_rel_max_kms=V_REL_MAX,
    )
```

On the dataset with the companion injected at 38 km/s:

```text
--- companion injected at K_2 = 38 km/s  [2.4 s] ---
  peak at K_2 = 38.0 km/s   D(peak) =      26583.1
  worst scan edge          D(edge) =      22368.8
  contrast peak - edge             =       4214.3
  log p(y | no companion)          =      -1683.9
  five highest trials (K_2 : D)    = 30:25813  34:26374  38:26583  42:26363  46:25719
  companion spectrum at the peak: r = 0.982 with truth over line cores, offset +0.189, RMS about the offset 0.052
  median formal sigma on d_2: 0.068
```

The peak lands on the injected value, and the curve is smooth and single-peaked around it: the
five highest trials bracket 38 km/s symmetrically. The width of that peak, not its height,
constrains $K_2$; sharpening it requires epochs near the velocity extremes rather than more
epochs.

The absolute numbers carry no direct significance. $D(\text{peak}) \approx 2.7\times10^{4}$ is
not "$\sqrt{D}\,\sigma$" of anything: $D$ contains the companion's prior scale, and its null
distribution is estimated by injection and recovery with `albireo.simulate` rather than assumed
([`docs/math.md`](../math.md) §6). The meaningful quantities on this page are the contrasts,
peak against scan edge and, more importantly, the companion-free control below.

## 3. The companion-free control

```python
    null_dataset, _ = simulate(with_companion=False)
```

Same primary spectrum, same epochs, same noise seed, same instrument, same scan. The only
difference is that no companion was injected.

```text
--- companion-free control  [1.6 s] ---
  peak at K_2 = 10.0 km/s   D(peak) =       -464.5
  worst scan edge          D(edge) =       -464.5
  contrast peak - edge             =          0.0
  log p(y | no companion)          =      11918.9
  five highest trials (K_2 : D)    = 10:-465  14:-480  18:-485  22:-489  26:-493
  max over the whole grid: D = -464.5 (negative at every trial: the Occam term penalizes the unneeded component)
```

$D < 0$ at every trial, and this is not a tuned threshold. Both marginal likelihoods carry their
own $\tfrac12\log\det$ Occam term, so adding a marginalized component lowers the likelihood
unless coherent signal compensates for it. On companion-free data nothing compensates, and the
two-component model loses to the null everywhere on the grid, monotonically over this grid,
since a larger $K_2$ separates the components further and adds the noise-fitting freedom that
the determinant term charges for.

That baseline is what gives a positive $D$ its meaning, and it is what the script asserts:

```python
    null_max = float(np.max(null.detection))
    assert null_max < 0.0, f"companion-free scan reached D = {null_max:.1f} >= 0"
```

## 4. Limits of the recovered companion spectrum

The recovered companion spectrum is informative in a specific sense:

```text
  companion spectrum at the peak: r = 0.982 with truth over line cores, offset +0.189, RMS about the offset 0.052
```

Correlation 0.98 with the injected line pattern, and a residual offset of 0.19 in depth units.
The offset is structural and does not decrease with more data. The observable is the product
$\ell_2 d_2$, so the companion's light fraction trades exactly against its line depths
([`docs/math.md`](../math.md) §5.2). In addition, an error $\Delta$ in the bright primary's
smooth envelope maps to $-(\ell_1/\ell_2)\Delta$ in the companion, an amplification of about ten
at $\ell_2 = 0.1$, on top of the $k = 0$ indeterminacy that already leaves the envelope
prior-dominated ([`docs/math.md`](../math.md) §5.1, §6).

The pattern of the recovered lines is therefore the usable result: it identifies the companion's
spectral type, and it is what the correlation coefficient measures. Absolute depths should not
be read off it unless eclipses or photometry have pinned $\ell_2$ independently. For the same
reason `k2_scan` has no default for `light_fractions`: there is no defensible generic value, and
an implicit one would propagate into a mass.

Two further limits:

- The scan is conditional on the SB1 solution. An error in $P_{\rm orb}$ or $T_{\rm conj}$
  smears the companion's lines across epochs and depresses $D$ everywhere; a marginal detection
  should be followed by a joint refit (the `K2ScanResult` carries a ready-to-use `model` for
  that, seeded at the peak).
- $\gamma \equiv 0$ throughout. A systemic velocity is exactly degenerate with a common shift
  of all component spectra ([`docs/math.md`](../math.md) §5.3), so it is measured afterwards,
  from the disentangled spectra, outside the scan.

## 5. Figure

With matplotlib importable, the script writes `k2_scan_detection.png`, showing $D(K_2)$ for both
datasets with the injected value marked, into the working directory. matplotlib is not a
dependency:

```python
    if importlib.util.find_spec("matplotlib") is not None:
        plot_detection(injected, null, "k2_scan_detection.png")
        print("\nwrote k2_scan_detection.png")
```

## Run it yourself

```bash
python examples/02_k2_scan.py

# or, CI-sized:
ALBIREO_EXAMPLE_FAST=1 python examples/02_k2_scan.py
```

On Windows, `$env:ALBIREO_EXAMPLE_FAST = "1"` sets the same switch. The script exits non-zero
unless the peak lands on the injected $K_2$ and the companion-free control stays negative
everywhere, so it can be run directly in CI.

## From a peak to a detection claim

This tutorial finds a companion. It does not state how often noise alone would have produced the
peak it found. Two steps close that gap, both in
[`examples/05_detection_limit.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/05_detection_limit.py):

- **Marginalize $K_1$** rather than condition on the SB1 value, with `k2_scan(k1_sigma=...)`. A
  $K_1$ 10% too high took the recovered companion's line pattern from 0.96 correlation with the
  truth to 0.49 while tripling $D$ ([benchmarks](../benchmarks.md)), so the artifact reads
  as a stronger detection.
- **Calibrate the statistic** with [`albireo.detection_limit`](../api/calibrate.md), which
  resimulates this dataset through its own operators, scans hundreds of companion-free draws for
  the null distribution of $\max_{K_2} D$, and injects a ladder of light fractions for
  completeness. The output is a false-alarm probability for the peak above and a limit of the
  form "any companion contributing more than $X$% of the light would have been detected at 95%
  confidence".

A calibrated threshold and a marginalized $K_1$ do different jobs and neither replaces the
other: the null trials are drawn under whatever $K_1$ the scan assumes, so the calibration is
blind to that assumption being wrong.

Previously: [disentangle an SB2 end to end](sb2-end-to-end.md).
