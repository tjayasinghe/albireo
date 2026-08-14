# Disentangle a BLOeM SB2 end to end

BLOeM — Binarity at LOw Metallicity — is a VLT/FLAMES-GIRAFFE survey of 929 OBAF stars in
the Small Magellanic Cloud: eight fields, about 25 epochs each, an intrinsic binary
fraction above 70%, and **59 published double-lined systems whose disentangling the survey
team still lists as future work** in its July 2026 review. The spectra are public, no ESO
account is needed, and one star is about 5 MB.

That makes it the best available target for this package, and it is also the hardest of the
worked examples, because nothing about it is handed to you: the archive does not know the
survey's names, the file layout is not the one the other tutorials use, the spectra are not
normalized, and — the part that changes the *analysis*, not just the loading — **there is
no published orbit**. This page is about doing it anyway, honestly.

The executable starting point is
[`examples/06_bloem.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/06_bloem.py):

```bash
python examples/06_bloem.py 1-037
```

It stops where an analysis begins. Everything after §3 below is what comes next.

!!! warning "This page gives decisions, not results"
    The other tutorials quote the answers they get, because they run on the simulator where
    the truth is known. Here it is not: these systems have no published orbital solutions,
    which is exactly why they are interesting. So every number quoted below is one already
    measured and recorded elsewhere in this repository — the archive facts
    (`docs/design.md` D45), the loader's behaviour, and the closed-loop results from the
    simulator that justify each choice. Where a step produces a fitted result, the page
    says what to look at rather than what you will see.

## The short version

```python
import albireo as ab

star = ab.resolve_bloem("1-037")                       # name -> Gaia DR3 source id
records = ab.bloem_spectra(star, public_only=True)     # ~25 epochs, LR02
ab.download(records, "data/bloem-1-037")

ds = ab.read_dataset(
    "data/bloem-1-037/*.fits",
    instrument="GIRAFFE",
    region=(4120.0, 4300.0),     # strictly between H-delta and H-gamma; see §3
    region_pad_angstrom=40.0,
    smooth_angstrom=60.0,        # these products are NOT continuum-normalized
)
grid = ab.LogGrid.covering(ds, dv_kms=8.0, v_margin_kms=400.0, lsf_sigma_kms=20.2)
```

`resolve_bloem` and `read_dataset` need the optional extras: `pip install "albireo[io]"`.

## 1. The two steps that are not guessable

**The archive does not know the survey's names.** There is no BLOeM Phase 3 collection;
the spectra sit under `obs_collection='GIRAFFE'`, and `target_name` is the **Gaia DR3
source id**, not `1-037`. `resolve_bloem` fetches the published cross-match from VizieR,
which speaks the same TAP dialect as ESO, so the join costs no new dependency. Gaia ids are
kept as strings throughout — 809 of the 929 do not survive a float64 round trip.

**Filter on the survey programme.** A *second* programme, `115.28A9`, re-observes the same
929 stars at R = 17000 and 23000 in two other windows — 1,827 more spectra. Pooling those
with LR02 under one line-spread function would be wrong, so `bloem_spectra` defaults to
`112.25R7`. (If you find `112.25W2` in a search result, it is the arXiv v1 typo; in the
archive that is an ERIS programme.)

`public_only=True` is not politeness: sub-run `.004` releases through 2027-01-15, and the
proprietary rows would be returned and then fail to download, which is worse than not
asking for them.

## 2. The file layout, and why you do not have to know it

GIRAFFE products put the flux in `FLUX_REDUCED`, the errors in `ERR_REDUCED`, the quality
flags in `QUAL_REDUCED`, and the wavelengths in **nanometres** on an **air** scale in the
**heliocentric** frame. The FEROS files of [the real-data tutorial](real-data.md) use
`FLUX`/`ERR`, ångström, barycentric. `ab.read_dataset` reads both without being told which
is which, because it dispatches on the **IVOA utypes** rather than on column names.

Keying on UCDs instead — the obvious shortcut — is worse than the disease: UVES gives its
*sky-background* column the same UCD HARPS gives its *flux* column, so a UCD-keyed reader
silently fits the sky. Only the utype role separates them. Check what was chosen:

```python
raw = ab.read_spectrum("data/bloem-1-037/<one>.fits")
print(raw.columns, raw.wave_medium, raw.specsys, raw.err_source)
```

Two consequences of this particular collection are worth stating because they change what
you do:

**The products are not normalized and not flux-calibrated** (`CONTNORM=F`,
`FLUXCAL='UNCALIBRATED'`). `smooth_angstrom=` is therefore mandatory, not optional — it is
what runs `albireo.preprocess`'s continuum fit. The survey team's own normalized, co-added
reduction is behind a credential-gated page; assume you cannot have it.

**The epochs do not share a wavelength grid, and must not be forced onto one.** FEROS
shifts before resampling, so its 51 HR 6819 epochs agree to 0.007 km/s and can be
relabelled onto a common grid. GIRAFFE's differ by **5.3 km/s** — most of a model pixel —
so those are real wavelength solutions, not sub-pixel bookkeeping, and
`share_wavelength_grid` refuses them. albireo gives each its own rebin operator and the
cost is one operator per distinct grid, which is small next to the solve. Do not call
`share_wavelength_grid` here.

## 3. Three things you must decide

**The window.** `examples/06_bloem.py` uses 4120–4300 Å inside LR02's 3960–4571 Å: Si III
4128/4130, He I 4144, He I 4169, He II 4200. The blue edge is the load-bearing number: it
sits strictly between Hδ (4101.7) and Hγ (4340.5) so that **neither Balmer line nor its
nebular window is inside**. Check rather than trust that, because the margin is smaller
than it looks —

```python
ab.nebular_windows(wave_range=(4120.0, 4300.0), v_kms=150.0)   # ()  — nothing to model
ab.nebular_windows(wave_range=(4000.0, 4300.0), v_kms=150.0)   # ((4099.7, 4107.9),)
```

— and a window reaching only 25 Å further blue picks Hδ's ±300 km/s nebular window back
up. See §6 before you widen it.

**The line-spread function.** LR02 delivers R ≈ 6200–6300; take the value from the archive
rows themselves (`em_res_power`, which `bloem_spectra` returns and the example prints)
rather than from a paper. Then convert it correctly:

```python
lsf = ab.LSF.from_resolution(6300)     # sigma = c / (R * 2 sqrt(2 ln 2)) = 20.2 km/s
```

`R` is a **FWHM**. Using `c / R` directly is a factor of 2.35 in the kernel radius and is
the single most common way to mis-specify an instrument.

**The light fractions.** albireo will not guess these and neither should you. The
likelihood depends only on the products `ℓᵢ dᵢ`, so the continuum light ratio is *exactly*
degenerate with the line depths (`docs/math.md` §5.2) and only external information —
photometry, an SED fit, an eclipse depth — sets it. For a BLOeM SB2 with no light-curve
solution, the honest move is to pick a value, state it, and report how the answer moves
across a plausible range. Every summary albireo prints repeats the assumption back to you
for this reason.

The grid then follows from the rest:

```python
grid = ab.LogGrid.covering(ds, dv_kms=8.0, v_margin_kms=400.0, lsf_sigma_kms=lsf.sigma_kms)
```

`v_margin_kms` has to cover the SMC's ~+150 km/s systemic recession *plus* the orbit, since
albireo's model grid must clear the largest shift any component takes plus the kernel
radius.

## 4. Velocities before an orbit

Here is where a BLOeM SB2 stops resembling the other tutorials. With no published period
there is nothing to warm-start a Keplerian from, and albireo **refuses to invent one**: the
`Disentangler` façade scans conjunction *phase* at a single period and warns if you hand it
a period prior wide enough to be a search, because a phase located for the wrong period is
worse than no phase at all.

So run the free per-epoch RV table first — the mode
[`examples/09_rv_table.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/09_rv_table.py)
is built around — and get the period from *it*:

```python
import jax.numpy as jnp, numpyro.distributions as dist, numpy as np

model = ab.MarginalOrbitModel(
    grid, ds,
    light_fractions=(0.6, 0.4),
    lsf_sigma_v={"GIRAFFE": lsf.sigma_kms},
    v_rel_max_kms=400.0,
)
n = ds.n_epochs
fit = ab.run_map(
    model.model({
        "velocity": dist.Normal(0.0, 200.0).expand([2, n]).to_event(2),
        "log_tau": dist.Normal(5.7, 1.5).expand([2]).to_event(1),
        "log_eta": dist.Normal(1.6, 1.0).expand([2]).to_event(1),
    }),
    init={"velocity": jnp.asarray(v_start), "log_tau": jnp.full(2, 5.7),
          "log_eta": jnp.full(2, 1.6)},
    model_args=(model.problem,),
)
rv = np.asarray(ab.relative_velocities(fit.params["velocity"], grid))
```

!!! danger "A cold start does not work, and that is measured"
    `v_start` may not be zeros. With every epoch initialized at the same velocity the two
    components are indistinguishable, and the fit fails — measured at **122,000 nats worse**
    than the warm-started answer on the D42 fixture. What makes the mode usable is that the
    failure is loud rather than silent, so *check the potential*. Start from whatever
    external velocities you have: cross-correlation lags, or the He I 4144/4169 splitting
    measured by hand on the two most separated epochs. This is also the one place the
    façade cannot help — `Fit.free_velocities()` warm-starts from a Keplerian fit, so for
    an unsolved system the expert path above is the one that works.

Two properties of the resulting table are counter-intuitive and will mislead you if you
have not read [§7.6 of the math](../math.md#76-free-per-epoch-velocities-the-rv-table):

* It has **one arbitrary zero point per component**, not one in total, so absolute
  velocities and the systemic velocity are *not* recoverable from it. What is recoverable —
  each star's variation, the epoch-to-epoch differences, and the Wilson slope, which is the
  mass ratio — is what a period search and a mass ratio need anyway.
* **Do not read the raw Laplace diagonal as an error bar.** Each zero point is an exactly
  flat direction, so its posterior width is the prior's and every epoch inherits it: on the
  D42 fixture the raw bars came out at `120/√10 = 37.947` km/s on *every* entry against a
  real 0.059. Use `ab.relative_velocity_errors(cov, fit.unconstrained)`, or posterior
  samples of the `velocity_rel` deterministic.

Then run a periodogram on `rv` outside albireo (this package deliberately does not ship
one), and take the period into a Keplerian fit.

## 5. The Keplerian, checked against the table

With a period in hand the façade is the short path:

```python
dis = ab.Disentangler(
    ds,                                    # no grid: the façade derives it, and the budget
    components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4)],
    orbit=ab.Orbit(
        period=ab.Between(p0 * 0.98, p0 * 1.02),        # narrow: this is not a search
        k=ab.Between([20.0, 20.0], [250.0, 250.0]),     # one bound per star
    ),
    lsf={"GIRAFFE": lsf},
    dv_kms=8.0,                            # match §3; the default is the native sampling
)
print(dis.explain())                       # every derivation, including the phase scan
kep = dis.fit()
print(kep.summary())
```

`dis.grid` is the grid it chose, and it is not necessarily the one §4 built by hand — use
`dis.grid` from here on so the pieces agree.

Now use the table you already have as the model check it exists for:

```python
free = kep.free_velocities()
resid = free.keplerian_residuals(kep)      # km/s, both zero points cancel exactly
```

Compare those residuals to the *per-epoch uncertainties*, not to zero. Structure —
phase-correlated residuals, or one epoch far out — is the signature of a period that is
slightly wrong, an unmodelled third body, or line-profile variability that the Keplerian
has absorbed into `e`. On the D42 fixture a period wrong by 0.5% moved the residuals from
2.9σ to 49σ, so this check has real power.

## 6. The nebular component, before you touch the Balmer lines

BLOeM's targets sit in H II regions. Their spectra carry nebular emission that does not
move with either star and varies from night to night with seeing and slit losses, and this
is the single most important reason the survey's own disentangling is hard.

Leaving it in is not neutral, and the cost is not where you would expect. On a simulated
SB2 whose Hβ absorption carries a static nebular line
([`examples/04_nebular.py`](https://github.com/tjayasinghe/albireo/blob/main/examples/04_nebular.py)),
ignoring it costs **11.5% of the equivalent width** — which propagates into log *g*, and is
the cost the literature already describes. It wrecks the **orbit** far worse. A static line
is a component with *K* = 0, so a nebula-blind joint fit hands the emission to whichever
star can be made to move least:

| | truth | nebula-blind | with the component |
|---|---|---|---|
| K₂ [km/s] | 41.0 | **16.77 (−59.1%)** | 40.88 (−0.29%) |
| period [d] | 5.70000 | 5.87115 (**+0.171**) | 5.69986 |
| eccentricity | 0 | **0.950 — the solver's clip** | 0.0022 |

So the contamination reaches the *masses*, not only the atmospheres. Only K₁ survives it,
because 70% of the light pins it.

So if you widen the window to Hδ or Hγ, add the component and confine it:

```python
dis = ab.Disentangler(
    ds,
    components=[ab.Star("A", light=0.6), ab.Star("B", light=0.4), ab.Nebular(v_kms=v_neb)],
    orbit=ab.Orbit(period=ab.Between(p0 * 0.98, p0 * 1.02),
                   k=ab.Between([20.0, 20.0], [250.0, 250.0])),
    lsf={"GIRAFFE": lsf},
    dv_kms=8.0,
)
```

The component confines itself to the nebular line windows through the prior; at the expert
level that is `ab.nebular_windows(wave_range=(grid.wave[0], grid.wave[-1]), v_kms=v_neb)`
fed to `ab.window_profile`, and the façade assembles it for you. **The confinement is not
cosmetic**: the same fit with the nebular component left free across the whole grid lands
K₂ at +2.6% instead of −0.29%, because the freedom it was given got spent absorbing stellar
signal at wavelengths where a nebula has no lines — the failure mode the component exists
to prevent, reappearing one level up.

Two conventions come with it and neither is measured by the data: the amplitude scale (the
geometric mean is pinned to 1) and `v_kms`, which is a *placement* choice for the line
windows and must match what you pass to `nebular_windows`. For the SMC that is the systemic
recession, not zero.

!!! note "The façade will refuse this if the wavelength scale is undeclared"
    A nebular line list is a set of *absolute* wavelengths, and air against vacuum is a
    nearly constant 83 km/s — so `Disentangler` raises rather than guessing when the
    dataset does not say which it is. GIRAFFE products declare air in `TUCD1` and
    `read_dataset` carries it through, so this passes on real BLOeM data; it bites if you
    assembled the `Dataset` by hand, and the fix is `EpochData(..., medium="air")`.

## 7. The product

The disentangled spectra are not the end either — they go to an atmosphere code — so what
matters is that the uncertainty leaves with them:

```python
kep.write_spectra("bloem-1-037_spectra.fits")   # mean + band + the assumptions, as FITS
```

Read the **band**, not the mean. Between the lines, and wherever the epochs give little
leverage, the recovered spectrum is set by the smoothness prior rather than by the data,
and the band is what says so. `ab.plot_spectra(dis.grid, kep.spectra(), std=kep.std())`
draws it.

Before believing any of it, three checks:

* `fit.z_rms` — the whitened residual scatter. Not 1 means the inverse variances are not
  calibrated, and every uncertainty downstream inherits the error.
  `ab.plot_residual_zscores` shows the shape, including the lag-1 panel that catches
  correlated noise a histogram cannot.
* The light-fraction sensitivity. Re-run at the ends of your plausible range and quote the
  spread; it is a real systematic, not a rounding error.
* Whether the epochs you have can answer the question at all —
  `ab.sensitivity_forecast(dis.grid, ds, orbit=kep.theta, ...)` needs no fluxes, so you can
  ask it of the epochs you *would* request as easily as the ones you have
  ([the forecast example](https://github.com/tjayasinghe/albireo/blob/main/examples/08_forecast.py)).
  BLOeM's published multiplicity results use only the first nine epochs; the full ~25 are
  in the archive, and it is worth knowing what the extra ones buy before spending a fit on
  them.

## What this page does not claim

No orbit here has been validated against a published solution, because there are none to
validate against — that is the opportunity, and it is also the reason every number above
traces to the simulator or to the archive rather than to a BLOeM fit. If you take one of
these systems through this path and it works, that result is new.

The known limits are worth stating in the same breath. R ≈ 6300 is low for disentangling,
and 4000–4300 Å is a narrow window; the SB2 classifications are split across five
unharmonized VizieR catalogues with no single all-929 table; and the light ratio, which is
the one genuinely free choice in disentangling, is unconstrained for these targets until
somebody publishes photometry. None of those is a reason not to try. All of them belong in
the paper.
