# albireo codebase audit (read-only), 2026-09-03

Repo: `C:\Users\thari\Documents\GitHub\albireo` @ `6c41549` (main, clean, zero tags).
Nothing was modified. Test suite not run. Timings on this box are indicative only.

---

## 1. Light fractions: how they enter, and what per-pixel would cost

### 1.1 The shape is `(n_epochs, n_component)` — one scalar per component per epoch. No pixel axis anywhere.

The canonical storage is a field on `EpochGroup`:

- `src/albireo/forward.py:96` — `light: jax.Array  # (n_epochs, n_comp)`

`build_problem` accepts `(n_stellar,)` **or** `(n_stellar, n_epochs)`, broadcasts the
constant case to per-epoch, and enforces the simplex only here:

- `src/albireo/forward.py:587-593`
  ```python
  ell = np.asarray(light_fractions, dtype=np.float64)
  if ell.ndim == 1:
      ell = np.repeat(ell[:, None], n_ep, axis=1)
  if ell.shape != (n_stellar, n_ep):
      raise ValueError(f"light_fractions must be ({n_stellar},) or ({n_stellar}, {n_ep})")
  if not np.allclose(ell.sum(axis=0), 1.0, atol=1e-10):
      raise ValueError("light fractions must sum to 1 at every epoch")
  ```
- Rows are then stacked stellar → telluric (a row of ones) → nebular (the amplitude row),
  `src/albireo/forward.py:606-628`, and sliced per group at `src/albireo/forward.py:720`:
  `light=jnp.asarray(light[:, idx].T)`.

**Component order is fixed: stellar, telluric, nebular** (`forward.py:545`, and
`facade.py:938-952` `ordered_components`).

### 1.2 Where it multiplies

Forward operator, per epoch, on the **model grid before convolution and rebin**
(`src/albireo/forward.py:1337-1352`):

```python
def one_epoch(shifts_e, light_e):
    acc = jnp.zeros(d_stack.shape[1])
    for i in range(d_stack.shape[0]):
        acc = acc + light_e[i] * shift_spectrum(d_stack[i], shifts_e[i])
```
so `S_e = R K sum_i l_ie T(delta_ie)` with `l_ie` a **scalar**. Adjoint mirrors it at
`src/albireo/forward.py:1359-1367`.

Band assembly, the `(i, j)` component block of one epoch (`src/albireo/assembly.py:339-371`):

```python
def epoch_body(band, xs):
    gp, light, floor_e, frac_e = xs
    ...
            f = f * (light[i] * light[j])
```
i.e. a **single scalar multiply of the whole band image** for each `(i, j)` pair.
`light` is carried as the `lax.scan` xs at `assembly.py:392` (hoisted path) and
`assembly.py:402` (chunked path).

### 1.3 Fixed per-epoch light fractions today: yes at both levels — but the façade fixes them at *one constant vector*

| Level | Per-epoch **fixed** ℓ? | How |
|---|---|---|
| `build_problem` | **Yes** | pass `(n_stellar, n_epochs)`; simplex checked per epoch |
| `MarginalOrbitModel(light_fractions=...)` | **Yes** | forwarded verbatim to `build_problem` (`inference.py:586-591`, `light_fractions=ell`); when a `light` θ-site is present, the build value "only sets `n_stellar`" (`inference.py:526-529`) |
| `with_light_fractions` (traced) | **Yes**, and unchecked | `forward.py:890-903`; no simplex test at all |
| `θ` site `"light"` fixed via `model(priors, fixed={"light": ell})` | **Yes** | `inference.py:697-701`; note the θ-site layout for the 2-D case is **`(n_epochs, n_stellar)`** and is transposed internally |
| `ab.Disentangler` | **No** | `Star.light` is a scalar float per star (`facade.py:394-418`), summed to 1 at construction (`facade.py:877-885`). There is no per-epoch field. |

So a light-curve model's per-epoch ℓ can be used today only through the low level
(`build_problem(light_fractions=ell_2d)` or `fixed={"light": ell.T}`), or by taking
`dis.expert()` and rebuilding. The façade cannot express it.

Confirmed empirically (`.venv` python, tiny 3-epoch / 46-pixel problem):

```
(b) non-simplex (n_stellar,n_epochs) ACCEPTED; light[0] = [0.9 0.5 1.7]   # with_light_fractions
(c) build_problem refused: light fractions must sum to 1 at every epoch
(d) scalar refused:   light_fractions must have shape (2,) or (2, 3); got ()
(e) per-pixel refused: light_fractions must have shape (2,) or (2, 3); got (2, 3, 46)
```

### 1.4 Wavelength dependence inside disentangling: **none**

There is no pixel axis on `light` in `forward.py`, `assembly.py`, `inference.py` or
`likelihood.py`. `docs/math.md` §5.2 (`docs/math.md:863-881`) states the consequence
directly: the likelihood sees only the products ℓ_i d_i, and the four listed breakers are
per-epoch variation, external photometry, the positivity floor, and assumption. (The
wavelength-dependent dilution machinery — `RadiusRatio`, `ScalarDilution`, `FixedDilution`
— lives in `match.py` and is a *label-fitting* construct, not part of the disentangling
operator.)

### 1.5 What per-pixel ℓ would require structurally

The block becomes `T_i^T diag(l_i) G diag(l_j) T_j`, i.e. a row-and-column scaling of the
band image rather than a scalar. Concretely:

1. **`EpochGroup.light`** → `(n_epochs, n_comp, n_pix)`; both pytree methods
   (`forward.py:138-224`) carry it unchanged, so no aux change.
2. **`forward.build_problem`** (`forward.py:487-593`) — validation, broadcast, the
   `light_rows`/`np.vstack` assembly at `606-628` and the per-group slice at `720`.
3. **`forward.with_light_fractions`** (`forward.py:871-903`) — shape checks and the
   `jnp.concatenate([le, g.light[:, problem.n_stellar :]], axis=1)` splice at `902`.
4. **`forward._epoch_model` / `_epoch_model_adjoint`** (`forward.py:1343`, `1365`) —
   `light_e[i] * shift_spectrum(...)` becomes an elementwise `(n_pix,)` multiply. Cheap
   and exactly adjoint-preserving.
5. **`assembly.epoch_body`** (`assembly.py:371`) — the one real cost. `f = f * (light[i] *
   light[j])` becomes `f * l_i[rows][:, None] * l_j[col]`, where the column index array
   `col = q_pix[:, None] + delta + t_idx[None, :]` **already exists** two lines below
   (`assembly.py:373`), so the gather is available; it adds one `(n_pix, w_f)` gather per
   `(i, j)` pair per epoch inside the scan body, and pushes `light` from a `(nc,)` scan
   carry to an `(nc, n_pix)` one.
6. **`inference._theta_problem`** (`inference.py:697-701`) — the `ndim == 2` → transpose
   rule needs a third case, and `_THETA_SITES`/the docstring need the new layout.
7. Nothing in `solver.py`, `priors.py` or `likelihood.py` changes: the bandwidth and the
   block structure are untouched (a diagonal scaling does not widen the band).
8. Consumers that read ℓ back: `plotting.plot_light_fractions` (`plotting.py:470-520`,
   expects `(n_draws, n_epochs, n_comp)` or `(n_draws, n_comp)`), `io.write_spectra`
   (`LIGHTFR*` header cards, `io.py:1529-1532`), `forecast.sensitivity_forecast`
   (`forecast.py:718`), `todcor`/`pipeline`/`match`.

Identifiability note the plan must carry: with ℓ constant over epochs, ℓ(λ) is *still*
exactly degenerate with d_i(λ) pixel by pixel — the degeneracy gets worse, not better. A
per-pixel ℓ is only meaningful as a *fixed* input (an SED / light-curve ratio) or with a
strong parametric shape.

---

## 2. Sampling a constant light fraction under a prior

**A traced constant `light` is supported.** `MarginalOrbitModel` builds `theta` by
`numpyro.sample(name, d)` over whatever `priors` contains (`inference.py:976`), passes it
to `_theta_problem`, which treats 1-D as constant and 2-D as per-epoch:

```python
# src/albireo/inference.py:697-701
if "light" in theta:
    ell = jnp.asarray(theta["light"])
    if ell.ndim == 2:          # per-epoch, Dirichlet layout (n_epochs, n_stellar)
        ell = ell.T
    problem = with_light_fractions(problem, ell)
```

and `with_light_fractions` broadcasts 1-D over epochs (`forward.py:890-893`).

**Site name / shapes** (`inference.py:36-37`, `_THETA_SITES` at `inference.py:165-183`):

| site | shape | meaning |
|---|---|---|
| `"light"`, 1-D | `(n_stellar,)` | constant over epochs |
| `"light"`, 2-D | `(n_epochs, n_stellar)` | per epoch (transposed internally) |

Measured directly: a jitted `value_and_grad` over a 1-D traced `light` returns a finite
log-likelihood and a finite gradient (`loglike=258.892801  grad=[-14.022 -16.562]`).

**Caveats for a `Normal(0.62, 0.03)` prior:**

- A **scalar** site is refused: `with_light_fractions` requires `(n_stellar,)` or
  `(n_stellar, n_epochs)`; `()` raises
  `light_fractions must have shape (2,) or (2, 3); got ()`. You must sample a
  length-`n_stellar` vector, e.g. `dist.Normal(jnp.array([0.62, 0.38]), 0.03)`.
- **No simplex enforcement** on the traced path — by design:
  > "The simplex constraint (non-negative, sum to 1 per epoch) cannot be checked on traced
  > input and is the caller's responsibility; in the numpyro model it is guaranteed by a
  > Dirichlet prior." (`forward.py:877-880`)

  A two-component `Normal` prior on ℓ therefore samples off the simplex unless it is
  reparameterized (e.g. sample ℓ₁ and set ℓ₂ = 1 − ℓ₁ — which needs a wrapper model, not
  just an entry in `priors`, because `model()` only does `numpyro.sample` per key).
- **Test coverage is 2-D only.** `tests/test_realism.py:178`, `:207`, and the closed loop
  at `:367-447` all use `dist.Dirichlet(jnp.ones(2)).expand([n_ep])`. The 1-D constant
  path has no test.
- The façade never places a `light` prior: `Disentangler._make_specs`
  (`facade.py:1119-1193`) places only orbital sites + `log_tau`/`log_eta`
  (+ `log_nebular_amp`).

---

## 3. Time variation: the nebular amplitude, and what a rank-2 component would take

### 3.1 How the nebular per-epoch amplitude works

- **Site**: `log_nebular_amp`, shape `(n_epochs,)`, requires `nebular=True` at construction
  (`inference.py:62-75`; the refusal is `inference.py:752-759`, "it cannot be switched on
  by a θ site: rebuild the MarginalOrbitModel with nebular=True (and one more (tau, eta))").
- **Centering**: `inference.py:482-505`
  ```python
  u = jnp.atleast_1d(jnp.asarray(theta["log_nebular_amp"]))
  return jnp.exp(u - jnp.mean(u))
  ```
  geometric mean pinned to 1, because only `a_j * d_neb` is observable. Recorded as the
  `nebular_amp` deterministic (`inference.py:1023`).
- **Where it multiplies**: it *is* a light column. `with_nebular_amplitudes`
  (`forward.py:907-951`) writes it into the **last** column of the same array:
  ```python
  groups.append(replace(g, light=g.light.at[:, -1].set(col)))
  ```
  So it flows through exactly the same `light_e[i] *` in the forward operator and the same
  `f * (light[i]*light[j])` in the assembly. It is outside the stellar simplex purely by
  convention — nothing in the algebra distinguishes it.
- Façade prior: `dist.Normal(jnp.zeros(n_epochs), 0.3).to_event(1)` (`facade.py:656-658`).

### 3.2 A KOREL-style per-epoch line-strength factor on a *stellar* component: available today, no code change

`with_light_fractions` performs **no** simplex check (verified: a `[[0.9, 0.5, 1.7], ...]`
array is accepted). The θ-site path applies the same non-check. Therefore
`priors["light"] = <(n_epochs, n_stellar) positive distribution>` already gives per-epoch
per-component strength factors. Two things the plan must state:

1. The **per-component, constant-over-epochs** scale stays exactly degenerate with `d_i`
   (math.md §5.2), so the site needs the same centering discipline as
   `nebular_amplitudes` — otherwise there are `n_stellar` flat unbounded directions.
   Nothing in `inference.py` currently centers `light`.
2. The composite continuum is pinned (`z = y - r (R 1)`, `forward.py:679`), so an
   epoch-common scale on *all* components is **not** degenerate. Only the per-component
   level is.

### 3.3 A second basis vector per component (rank-2)

This is a **structural** change: it increases `n_components`. The touched surfaces:

| Surface | File:line | What changes |
|---|---|---|
| Row structure | `forward.py:606-630` (`shift_rows`/`light_rows`/`np.vstack`), `Problem.n_components` `forward.py:225-272` | one extra row per variable component, with its own shift row (= the parent's) and its own per-epoch coefficient row |
| `n_stellar` accounting | `forward.py:272-277` (`n_stellar = n_components - telluric - nebular`) | a second basis row must not be counted as a star, or every `with_light_fractions`/`with_shifts` shape check mis-sizes |
| Component ordering | `forward.py:545` docstring, `facade.py:938-952` | "stellar, telluric, nebular" becomes "stellar (+ their basis rows), telluric, nebular"; the trailing-column splices at `forward.py:902` and `forward.py:950` use `problem.n_stellar :` and `[:, -1]` and would both need re-indexing |
| SmoothnessPrior rows | `priors.py:253-345` (`tau`, `eta`, `tau_profile`, `eta_profile` all `(n_components, ...)`); length check `likelihood.py:270-285` | one more `(tau, eta)` pair, and — per the roadmap — a *shrinkage* prior on the second vector, which the current scalar+static-profile design cannot express without a new prior kind |
| Band assembly | `assembly.py:339-378` | nothing structural; `nc` grows, and cost grows as `nc²` in the epoch body and `p = nc*b_nat + nc - 1` in the bandwidth (`assembly.py:556`). A rank-2 SB2 goes nc = 2 → 4, i.e. **~4× the block work and ~2× the bandwidth** |
| `posterior_spectra` | `inference.py:1352-1397` | returns `(num_draws, n_comp, n_pix)`; consumers must know that rows `k` and `k+1` are one physical star. `draw_spectra` (`likelihood.py`) needs no change |
| Façade | `facade.py:394-418` (`Star`), `394-440` (`Telluric`/`Nebular`), `938-952`, `1055-1094` | a new component kind and a new name convention |

The roadmap already names this as Tier 3 and says why it stays tractable
(`internal/roadmap.md:638-647`): "Tier 2's nebular component is already rank-one time
variation … Line shape variation needs a second basis vector per variable component, with
a shrinkage prior and the same windowing machinery … it stays inside the linear-Gaussian
family, so the analytic marginalization survives: it is a change of basis rather than a
change of method." `internal/design.md:97` (D40) says the same.

---

## 4. The banded solver, windows, sharding, and a measured cold/warm number

### 4.1 One sequential scan over the whole grid — yes

- Rows are **interleaved by component**: band entry `BAND[q, i, k, d]` is row `q*nc + i`,
  column offset `k*nc + d` (`assembly.py:41-45`, `_band_offsets` at `assembly.py:152-156`).
- `_pack_band` gathers into `K = ceil(nc*n_pix / B)` dense blocks via
  `jax.lax.map(blocks_at, jnp.arange(k_blocks))` (`assembly.py:512`) — parallel, fine.
- `block_cholesky` is the sequential part: `jax.lax.scan(step, l0, (d[1:], e))`
  (`solver.py:262`), with `solve_lower` (`solver.py:279`) and `solve_upper`
  (`solver.py:296`, `reverse=True`) likewise one scan each. So **the factorization is one
  strictly sequential sweep of depth K over the whole grid.**
- Epoch loops are separate `lax.scan`s per group, accumulating into one shared band tensor
  (`assembly.py:562-565`: `for g in problem.groups: band = _epoch_band_scan(...)`).

### 4.2 Disjoint wavelength windows: **the precision does not become block-diagonal, and the code scans the gap**

- There is exactly one model grid, `LogGrid.covering` spans `min(wave)` to `max(wave)` over
  the whole dataset plus margin (`grids.py:257-269`). A gap becomes interior grid pixels
  with **zero data weight**, not absent pixels.
- The data term vanishes there, but `_add_prior_band` (`assembly.py:454-464`) adds the
  pentadiagonal `D2ᵀ diag(t) D2 + diag(e)` across **every** pixel including the gap, so the
  band stays connected. The precision is therefore *nearly* block-diagonal (coupling only
  through the smoothness prior, half-bandwidth 2 per component) but not exactly so, and
  nothing detects or exploits it.
- Cost of the gap is real: it is `nc × n_gap_pixels` extra rows in `K`, at full block width
  `B ≥ p = nc*b_nat + nc - 1` — i.e. the gap costs the *same per-pixel price as data*.
- The workaround in practice has been **separate fits per window** (HR 6819:
  `docs/benchmarks.md:977` "two independent windows"; `:562` "Two disjoint windows").

### 4.3 Structural opportunity

`docs/math.md:611-616` already names it as unbuilt:

> "Two orthogonal levers batch this further: `vmap` over independent wavelength chunks
> (echelle orders and natural mask gaps make chunks exactly independent; otherwise chunking
> is an explicit, benchmarked approximation) and `vmap` over systems (survey mode)."

- **Window `vmap`**: with W windows the total flops are unchanged minus the gap pixels, but
  the sequential depth of `block_cholesky` drops from K to K/W and the W scans batch. This
  requires a *multi-segment grid* abstraction (`LogGrid` is a single `x0, dx, n`,
  `grids.py:167-189`), which is the real work — the solver itself would need only a leading
  batch axis. The exactness caveat in math.md ("otherwise chunking is an explicit,
  benchmarked approximation") applies whenever the windows are *not* separated by more than
  the prior's half-bandwidth 2 + kernel radius.
- **Multi-device sharding**: **no code exists.** `grep` for
  `sharding|pmap|device_put|jax.devices|mesh|NamedSharding|shard_map` across `src/albireo`
  returns only `jnp.meshgrid`/`np.meshgrid` in `todcor.py`. Everything runs on the default
  device. The pipeline parallelizes across *stars* by spawning OS processes
  (`pipeline.py`, D58), not across devices.
- **Memory batching does exist** and is the closest analogue: `_epoch_chunk_default`
  (`assembly.py:159-183`) with `_GP_HOIST_BYTES` / `_GP_CHUNK_BYTES`, dividing the budget
  by `n_groups` because "the group loop is unrolled into a single jit graph".

### 4.4 Persistent JAX compilation cache: **not configured anywhere**

`grep -rn "compilation_cache|jax_compilation|persistent_cache|jax.config.update|enable_x64"`
over `src/albireo/*.py`, `scripts/*.py`, `docs/*.md` returns exactly one hit:

- `src/albireo/__init__.py:19` — `jax.config.update("jax_enable_x64", True)`

No `jax_compilation_cache_dir`, no `jax.experimental.compilation_cache`. Nothing in
`.github/workflows`, `pyproject.toml`, or `conftest.py` either.

### 4.5 Measured: cold vs warm façade fit on the packaged example

`.venv/Scripts/python.exe`, `ab.load_example("sb2_sim")` + the README `Disentangler` +
`.fit()`, one process, second call builds a **fresh** `Disentangler` of identical shape:

```
import albireo:                1.22 s
jax backend: cpu   devices: [CpuDevice(id=0)]
grid: 1074 px   half-bandwidth: 127        (nc = 2 → p = 255, n = 2148, K = 9 blocks)
COLD facade fit:              96.24 s
WARM facade fit (same process, fresh Disentangler):  92.18 s
ratio cold/warm:               1.04x
```

An earlier run of the identical script on the same box gave **113.14 s** cold, so run-to-run
scatter is ~15% (other jobs were running). **Reading: XLA compilation is ≈ 4 s of a ≈ 96 s
fit, about 4%.** A persistent compilation cache would therefore buy at most ~4% on this
example — the fit is dominated by the conjunction scan plus up to 300 L-BFGS steps
(`facade.py:1317`, `max_steps: int = 300`), not by compiling. That number will grow in
absolute terms at survey scale but the ratio is the fact to plan against.

---

## 5. Façade refusals and gaps (exact strings)

`albireo.facade` is the **only** module marked experimental (`facade.py:3`):
> "**Experimental.** The vocabulary defined here may change;
> :class:`~albireo.inference.MarginalOrbitModel` and the functions around it are the
> supported surface, and :meth:`Disentangler.expert` returns that surface directly."

### 5.1 Silently not exposed (no error — they simply have no vocabulary)

`Disentangler._make_specs` (`facade.py:1119-1193`) places only `period`, `t_conj`, `k`,
`secosw`/`sesinw` (+ `_out` variants), `velocity`, `log_tau`, `log_eta`,
`log_nebular_amp`. The documented gap list is `facade.py:1200-1202`:

> "Features the declarative interface does not expose (jitter, AR(1) noise, inferred
> light fractions, inferred LSF widths) are reached by modifying the returned
> dictionaries and calling `MarginalOrbitModel` directly."

and `docs/quickstart.md:164-167` repeats it with "per-epoch jitter" named explicitly.

| Feature | Façade | Low level |
|---|---|---|
| `log_jitter` | not exposed | **yes** — `forward.with_jitter` (`forward.py:954`), θ site, scalar or per-epoch |
| `ar1_phi` | not exposed | **yes** — `forward.with_ar1` (`forward.py:1120`), needs `MarginalOrbitModel(ar1=True)` |
| inferred `light` | not exposed | **yes** — θ site, 1-D or 2-D (§2) |
| per-epoch **fixed** light | **not expressible** (`Star.light` is one float) | **yes** — `build_problem(light_fractions=(n_stellar, n_epochs))` |
| inferred `lsf_sigma` | not exposed | **yes** — θ site, one entry per anchor / un-anchored instrument |
| `lsf_h3` | **no field at all**, deliberately | reaches only `build_problem`; **`MarginalOrbitModel.__init__` has no `lsf_h3` parameter** (`inference.py:568-583`), so the θ site `lsf_h3` can only be used on a model whose `problem` was built by hand |
| `response` | not exposed | **yes** — `forward.with_response` (`forward.py:1203`), θ site |
| per-pixel prior profiles | only via `Nebular` | **yes** — `SmoothnessPrior(tau_profile=, eta_profile=)` |
| light **ratio** as a fitted quantity | never; declared and reported under "Assumed, not measured" (`facade.py:1270-1312`) | only as the `light` θ site |
| period **search** | never; warns (below) | `ab.find_period` (Lomb–Scargle, `rvorbit.py:70`) operates on a `VelocityTable`, not on the disentangling likelihood |

The `LSF` h3 note, verbatim (`facade.py:331-336`):
> "Gauss-Hermite skewness (``h3``) has no field here. It reaches the kernel only through
> :func:`albireo.build_problem` and not through the model class built by this module, so a
> field would have been accepted and then discarded. Use :meth:`Disentangler.expert` to
> declare it."

The `Smoothness` note (`facade.py:329-331`): "Declare some margin if the width is to be
inferred through the low-level API."

### 5.2 Hard refusals — exact strings

**Hierarchical triple** — `facade.py:895-901`, `NotImplementedError`:
> "hierarchical triples are not in the façade's v1 vocabulary. The model supports them (the
> period_out/t_conj_out/k_out sites), so build one through albireo.MarginalOrbitModel
> directly: Disentangler.expert() on a two-star declaration gives you the triple to start
> from."

**Eccentricity lower bound** — `facade.py:2566-2574`, `ValueError`:
> "orbit.ecc lower bound must be 0; got {lo}. The sampled parameters are (sqrt(e)cos w,
> sqrt(e)sin w), in which a lower bound on e is an annulus rather than a box, so it cannot
> be expressed as a prior here. Use ecc=ab.Between(0.0, hi) and read the posterior, or
> ecc=ab.Fixed(e) with omega to hold it."

Related: `facade.py:2562` `"orbit.ecc upper bound {hi} exceeds ecc_max={ecc_max}"`;
`facade.py:2546-2550` `"a Fixed non-zero eccentricity needs omega as well: e and omega
together are one point in the (sqrt(e)cos w, sqrt(e)sin w) plane. Pass
omega=ab.Fixed(radians)."`; `facade.py:2585` `"orbit.ecc must be Fixed or Between; got ..."`.

**Light fractions** — `facade.py:872-885`:
> "every star's light fraction must be finite and positive; got {listed}. A component
> contributing no light is not a component: remove it."

> "the star light fractions must sum to 1; {listed} sums to {total:g}. This is an assumption
> the data cannot check: with constant light fractions the likelihood sees only l_i * d_i,
> so every recovered depth scales as 1/l_i."

**Orbit vs velocities** — `facade.py:886-892`:
> "declare exactly one of orbit= (a Keplerian to fit) and velocities= (the per-epoch
> velocities you measured, for a system whose orbit is not known yet). They are
> alternatives: a free velocity table replaces the orbit entirely, and the model rejects
> Keplerian sites alongside it."

**Period search (warning, `RuntimeWarning`)** — `facade.py:1410-1418`, fires when
`(hi - lo)/lo > 0.2`:
> "the period prior spans {lo:g} to {hi:g} d, which is {pct} of its own lower bound. The
> conjunction scan resolves *phase* at one period; it is not a period search, and it runs at
> the prior's midpoint. Narrow the prior or run a periodogram first, or the fit starts from
> a phase located for the wrong period."

Also `Orbit.period` docstring, `facade.py:490-493`: "This declaration does not perform a
period search: it scans conjunction phase at a single period."

**Wide `t_conj` (warning)** — `facade.py:1164-1173`, when width > 0.2 × period:
> "orbit.t_conj was declared as a range {width:g} d wide, which is {pct} of a period, so the
> conjunction scan is skipped and L-BFGS starts from its midpoint. … Leave t_conj at its
> default 'scan' to locate it first."

**Anchored (wavelength-dependent) LSF in a scan** — `facade.py:1263-1268`:
> "{what}() takes one line-spread width per instrument, but {anchored} declared
> wavelength-dependent widths. Passing them would silently use only the first anchor.
> Declare a single representative sigma for the scan, or use albireo.k2_scan directly."

**Scan preconditions** — `facade.py:1176-1181` (`Scanned` k in a fit declaration);
`facade.py:1451-1455` ("a K2 scan holds the primary's orbit fixed, so {missing} must be
declared Fixed(...)"); `facade.py:1476-1482` ("a K2 scan searches over a companion's
semi-amplitude within a *known* SB1 orbit …"); `facade.py:1493-1496` ("a K2 scan is the
two-component workflow; this declaration has {n} stars."); `facade.py:1499-1506`.

**`Sampled` without a bound** — `facade.py:255-261`:
> "Sampled(...) needs upper_bound= when it is used for a quantity the velocity budget is
> derived from (a semi-amplitude). Give the largest value the prior can realistically reach;
> the starting value is not that, and sizing the solver from it silently truncates the prior
> against a guard."

**Bare tuple** — `facade.py:277-281`: "A bare tuple is deliberately not accepted: (5.5, 6.5)
reads as either a range or a two-component vector."

**Mixed spec kinds in a vector site** — `facade.py:2526-2530`.

**Undeclared wavelength medium with Telluric/Nebular** — `facade.py:909-915` and again at
`facade.py:2137-2141` for label matching: "The difference is a nearly constant 83 km/s."

**Mode mismatches on `Fit`** — `facade.py:1735-1739` (`.orbit()` on a velocity fit),
`facade.py:1775-1778` (`velocity_errors()` on a Keplerian fit), `facade.py:1960-1964`
(`sample()` on a velocity fit), `facade.py:2075` ("keplerian_residuals() applies to a
free-velocity fit").

---

## 6. SB3 / triples

**Low level: fully supported.** Sites, all-or-nothing (`inference.py:164`):

```python
_OUTER_SITES = ("period_out", "t_conj_out", "secosw_out", "sesinw_out", "k_out")
```
`_has_outer_orbit` raises `"outer orbit needs all of {…}; missing {…}"`
(`inference.py:200-206`). Semantics (`inference.py:260-283`):

> "the outer center-of-mass velocity (semi-amplitude `k_out[0]`, argument `omega_out`,
> conjunction convention on the inner pair's center of mass) is added to every inner
> component, and one tertiary row is appended with semi-amplitude `k_out[1]` and argument
> `omega_out + pi`."

with `"k_out must have exactly two entries (K_inner_com, K_tertiary)"` (`inference.py:278`).
Nested blocks are built by the suffixed `_kepler_block` (`inference.py:217-228`), so the
outer Keplerian reuses the same `(secosw, sesinw)` disk factor —
`numpyro.deterministic("ecc_out", …)`, `"omega_out"`, `numpyro.factor("ecc_disk_out", …)`
at `inference.py:989-996`. `orbit_parameters` returns an `"outer"` key
(`inference.py:245-258`). Budget guidance: "for an SB3 add the outer orbit's
`(K_AB + K_C)(1 + e_out)`" (`inference.py:543-545`).

**Note**: the tertiary is a *stellar row*, so `MarginalOrbitModel(light_fractions=...)` must
be given three fractions and the prior three `(tau, eta)` pairs; the mismatch error is
`"theta implies {n} stellar components (len(k) + tertiary) but the model was built with
{n_stellar} light fractions"` (`inference.py:690-695`).

**Façade: refused** (§5.2).

**Tests** — all in `tests/test_realism.py` (module docstring line 4 names it as an M4 gate):
- `:103` `test_sb3_velocities_match_hand_composed` — parameterization units vs a hand
  composition of `radial_velocity`.
- `:121` `test_outer_sites_all_or_none`.
- `:128` `test_k_out_needs_two_entries`.
- `:230-274` `test_guards_reject_wide_lsf_and_outer_disk` — builds a 3-component model and
  checks the outer disk factor rejects `e_out > 1`; also checks the light-fraction count
  mismatch raises.
- `:283-329` fixture `sb3_fit` (14 epochs, `P_in`=6.31 d, `P_out`=47.3 d, ℓ=(0.5,0.3,0.2)),
  `:331` `test_sb3_map_recovers_inner_and_outer` (`k_out` to rtol 0.02, `period_out` to
  2e-3), `:341` `test_sb3_spectra_recovered`. Both `@pytest.mark.slow`.
- `tests/test_facade.py:316` `test_a_hierarchical_triple_says_it_is_not_in_v1`.

No SB3 coverage in `test_inference.py`, `test_assembly.py`, `test_todcor.py`,
`test_pipeline.py`, `test_scan.py`, or the CLI.

---

## 7. Release readiness

**Version `0.1.0.dev0`**, single source `src/albireo/__init__.py:189`, hatchling-derived in
`pyproject.toml:89-90`. Duplicated in `CITATION.cff:16`, `codemeta.json:6`,
`docs/citing.md:22`, two notebook outputs, `internal/design.md:76`. **All eight agree
today**, but only `CITATION.cff` is guarded (by `tests/test_metadata.py:41-52` *and*
`release.yml:61-64`); `codemeta.json` and `docs/citing.md` are unguarded, and
`internal/releasing.md:90-92` doesn't mention them — a live hazard because `codemeta.json`
prefills ASCL, hence ADS.

**Zero git tags.** `git tag --list` is empty; 72 commits, first `1690fab` 2026-08-11.
`internal/releasing.md:7` claims "nothing has ever been pushed", but `origin/main` is in
sync with local `main` — the doc's step-2 ordering advice is stale.

**PyPI publishing: trusted publishing (OIDC), no token.** No `secrets.*` in any workflow.

- `.github/workflows/release.yml:91-93`: "Trusted publishing: PyPI verifies this workflow's
  OIDC identity, so there is no API token to store or leak."
- `release.yml:94-95`: `permissions:` / `id-token: write`
- `release.yml:96-97`: `environment: name: ${{ (… inputs.test_pypi) && 'testpypi' || 'pypi' }}`
- `release.yml:105-109`: TestPyPI leg, `pypa/gh-action-pypi-publish@release/v1` with
  `repository-url: https://test.pypi.org/legacy/`
- `release.yml:111-113`: PyPI leg, **no `with:` block at all** — no `password:`.

**Every checkbox in `internal/releasing.md` is unchecked (23 × `- [ ]`, zero `- [x]`).**

### Unfinished items

| Item | Where | Status |
|---|---|---|
| ORCID | `CITATION.cff:10` (TODO w/ `0000-0000-0000-0000` template), `paper/paper.md:14`, **`codemeta.json` has no `@id` and no TODO** | missing in 3 files |
| Affiliation | `paper/paper.md:19` — **`  - name: TODO`, a literal rendered value**; `CITATION.cff` no key; `pyproject.toml:17-18` name+email only; `codemeta.json` none | missing, one live placeholder |
| Zenodo DOI | `CITATION.cff:21` `#   doi: 10.5281/zenodo.XXXXXXX` (commented out), `date-released` also commented at `:20`; `codemeta.json` no `identifier`; no DOI badge | none; **webhook must be linked before the first tag** (`internal/releasing.md:72-74`) |
| PyPI | `README.md:114` "albireo is not yet on PyPI"; name verified free 2026-08-11, fallback `albireo-spectra` (`internal/releasing.md:31-32`) | unregistered; **two** trusted publishers needed (pypi + testpypi environments), `internal/releasing.md:77-86` |
| GitHub Pages | deploy gated on `vars.PAGES_ENABLED == 'true'` (`docs.yml:72`); `README.md:4` and `mkdocs.yml:3` already point at `https://tjayasinghe.github.io/albireo/` | not enabled → badge is a 404 |
| Discussions | `internal/releasing.md:55` | not enabled. `.github/` contains **only** `workflows/` — no issue templates, `SECURITY.md`, or `CODE_OF_CONDUCT.md` |
| Version tag | `internal/releasing.md:23-25` — `0.1.0` vs `0.1.0.dev0` undecided; `CHANGELOG.md:12` still `## [Unreleased]` | none |
| paper.bib TODOs | `:54` verify DOI (gonzalez2006), `:83` verify DOI + authors (shenar2020), `:106` set jax `version` (**no `version` field at all**), `:139-140` optax; `optax2020` url is the org root not the repo | 4 open |
| paper.md gating note | `paper/paper.md:26-29` "the package is pre-alpha and the JOSS checklist is the M5 deliverable"; `date: 11 August 2026` is the first-commit date | open |
| M5 GPU gate | `paper/paper.md:2-3` keeps "GPU-accelerated" out of the title until it closes — but that exact phrase is the description in `CITATION.cff:4`, `codemeta.json:5`, `pyproject.toml:11`, `mkdocs.yml:2` | open (see §10) |
| pyproject URLs | `:81-83` only `Repository` + `Issues` — no Documentation/Homepage/Changelog; no `Typing :: Typed` classifier despite shipping `py.typed` | gaps |
| `codemeta.json:7` | `"developmentStatus": "wip"` | will be indexed as WIP |

**JOSS clock**: `internal/releasing.md:9-14` — six months of public history "begins at the
first push, not at the first commit". History starts 2026-08-11 ⇒ earliest submission
≈ **February 2027**.

### Workflows

| File | Name | Triggers | Jobs / matrix |
|---|---|---|---|
| `ci.yml` | CI | push/PR to main (`paths-ignore: **.md, docs/**, paper/**`), dispatch | one job `check`, **ubuntu-latest, py3.13 only**, no matrix. ruff → mypy (`continue-on-error`) → bare-install guard → `pytest --no-slow --no-network` |
| `full.yml` | Full | **`workflow_dispatch` only** (3 bool inputs, default true) | `matrix` (ubuntu+windows × 3.12/3.13 = 4, `fail-fast: false`), `gates` (slow + coverage, 90 min; "a bare `pytest` measures 36m25s"), `examples` (11 scripts, `ALBIREO_EXAMPLE_FAST=1`; 03_hr6819 and 06_bloem excluded) |
| `docs.yml` | Docs | push/PR to main (no paths-ignore, deliberate), dispatch | `build` (`mkdocs build --strict`, upload-pages-artifact@v4) + `deploy` (gated on `PAGES_ENABLED`) |
| `release.yml` | Release | push `tags: ["v*"]`; dispatch with `test_pypi` **default true** | `build` (uv build, tag/CITATION verify **skipped on dispatch**, wheel smoke test) + `publish` (OIDC) |

Three risks worth carrying into the plan:
1. **`release.yml` runs no tests.** `internal/releasing.md:27-28`: "nothing between a tag and
   PyPI checks the science."
2. The version/CITATION guard is `if: startsWith(github.ref, 'refs/tags/')`
   (`release.yml:49`) — the TestPyPI rehearsal validates nothing about version agreement,
   and the guard never reads `codemeta.json`.
3. A dispatch with `test_pypi: false` on a branch is a **silent green no-op**: neither
   publish step's `if:` matches.

---

## 8. Public API surface

- **`__all__` has 181 names** (`src/albireo/__init__.py:234-415`): 55 capitalized
  (classes/constants), 126 lowercase (functions). Not alphabetically sorted overall
  (constants first, then classes, then functions).
- **Lazy re-exports** via `__getattr__` (`__init__.py:218-227`): `_IO_EXPORTS` (5 names —
  `RawSpectrum`, `read_dataset`, `read_spectrum`, `to_epoch`, `write_spectra`) and
  `_PLOT_EXPORTS` (13 `plot_*` names), imported on first use because astropy/matplotlib are
  optional. `__dir__` unions them (`__init__.py:230-231`).
- **x64 is forced at import**: `__init__.py:16-19`, opt-out via `ALBIREO_DISABLE_X64`.

**Experimental markers — exactly one module:**

- `src/albireo/facade.py:3` — "**Experimental.** The vocabulary defined here may change; …"
- Echoed in `docs/quickstart.md:167` and `CHANGELOG.md:399`.

Everything else (`inference`, `forward`, `likelihood`, `assembly`, `solver`, `forecast`,
`todcor`, `match`, `library`, `pipeline`, `rvorbit`, `archive`, `handoff`) carries **no**
stability marker — the facade docstring designates `MarginalOrbitModel` "and the functions
around it" as *the supported surface*.

**Deprecation machinery: none.** `grep -rn "DeprecationWarning|deprecated|FutureWarning|
PendingDeprecation" src/albireo/*.py` returns zero hits. There is no `__deprecated__`, no
alias shim, no version-gated removal policy. Warnings used are `RuntimeWarning` (façade
guidance) and `UserWarning` (io/preprocess assumptions).

---

## 9. Data model

### `EpochData` (`src/albireo/data.py:146-153`)

```python
wave: np.ndarray          # native, strictly increasing, Angstrom, never resampled
flux: np.ndarray          # continuum-normalized
ivar: np.ndarray          # 0 == masked
bjd: float                # BJD_TDB at mid-exposure
v_bary: float = 0.0
instrument: str = "default"
mask: np.ndarray | None = None      # True == GOOD, folded in by effective_ivar only
medium: str | None = None           # "air" | "vacuum" | None (undeclared)
```

| Question | Answer |
|---|---|
| per-pixel mask / quality | **Yes**, two ways: `ivar == 0` (the canonical mask) and the optional boolean `mask` (True = good). "``mask`` is optional … It is folded into the weights in one place, :attr:`EpochData.effective_ivar`, which is what downstream code consumes; nothing else in albireo reads ``mask``" (`data.py:13-16`). Non-finite flux is legal where `ivar == 0`. |
| per-epoch resolving power | **No.** No `R` / `lsf_sigma` field. LSF is **per instrument**, supplied to `build_problem(lsf_sigma_v={...})`. `RawSpectrum` *does* carry `resolving_power` and `lsf_sigma_kms` (`io.py:241`, `:281-290`), but `to_epoch` drops it. |
| order / window identifier | **No.** Only `instrument`. Grouping is derived: `forward._epoch_groups` (`forward.py:402`) groups by `(instrument, native wavelength array)`. |
| medium | **Yes**, `medium: str | None`; `Dataset` refuses a mixture, and undeclared ≠ air (`data.py:104-119`). |
| v_bary | **Yes**, scalar km/s per epoch; composed inside the forward model, never applied to `wave` (`data.py:26-33`). |
| instrument | **Yes**, string key into the LSF/response tables. |
| frame | On `Dataset`, not the epoch: `"topocentric"` (default) or `"barycentric"` (`data.py:304-305`). |

### specutils interop: **none, and deliberately deferred**

`grep -rn "specutils|Spectrum1D|SpectrumList"` over `src/`, `docs/`, `pyproject.toml` →
zero hits. `internal/roadmap.md:649-653` (Tier 3):

> "**specutils interoperability.** Accept and return `Spectrum` objects at the boundary; do
> not adopt them internally. The reason for the asymmetry is concrete: `SpectrumCollection`
> requires equal-length spectra and rejects per-epoch metadata differences, so it cannot
> represent a ragged multi-epoch multi-instrument dataset, which is albireo's central input."

### `read_dataset` (`src/albireo/io.py:1429-1491`)

`read_raw_spectra` + `dataset_from_raw`. Takes a glob, a directory, or an iterable of paths;
`instrument=`, `frame=`, `medium=`, `sort_by_time=`, `read_kwargs=`, plus `**epoch_kwargs`
forwarded to `to_epoch` (`region`, `region_pad_angstrom`, `normalize_continuum`,
`smooth_angstrom`, `ivar_scaling`, `mask`, `tellurics`, `spike_threshold`).

**Formats**: FITS only — a binary table (tried first) or a 1-D WCS image
(`CRVAL1`/`CDELT1`), `io.py:1030-1040`. No ASCII/ECSV/HDF5 reader on input (`write_ascii`,
`write_spectra`, `write_gssp`, `write_ispec` are output-only).

**Header conventions** (`io.py:961-1120`): `INSTRUME`; `SPECSYS` → frame (raw value kept as
`specsys`; heliocentric aliased to barycentric); BJD_TDB from `TMID`/`MJD-OBS`+`EXPTIME/2`
with `TELAPSE` vetoing the midpoint fallback; `SPEC_RES`/`SPECRES`/`RESOLUTI` for `R`
(a bare `R` card is deliberately **not** consulted, `io.py:992-994`); `CUNIT1` or the column
unit for the wavelength scale; `CONTNORM`; `TUCD1`/`TUTYPn` for air vs vacuum.
Column selection **dispatches on IVOA utypes** (`TUTYPn`, matched on the suffix after
`Data.`), with UCD + name as fallback — because UCDs are ambiguous (UVES labels its sky
column identically to HARPS flux; only `BackgroundModel.Value` vs `FluxAxis.Value`
separates them), `io.py:13`, `:124-135`, `:403-500`.

### Multi-order echelle: **merged, then trimmed — there is no per-order model**

- `to_epoch(region=(lo, hi))`: "a full echelle spectrum has many more pixels than a
  disentangling run needs, and the cost of the solve grows with the pixel count"
  (`io.py:1150-1153`).
- `preprocess.select_region` (`preprocess.py:598-604`): "This function reduces a full
  echelle spectrum to the region to be modelled; interior removals go through
  `mask_ranges`."
- `preprocess.mask_flux_gaps` (`preprocess.py:791-810`) exists precisely because merged
  spectra carry detector gaps as flat runs of zeros at full weight (the HARPS 32.9 Å hole).
- `preprocess.share_wavelength_grid` (`preprocess.py:861-873`) collapses per-exposure grids
  into one, because one operator group per exposure was measured at "several times the
  memory".
- `docs/math.md:1609-1612` states the policy for the TODCOR path: "multi-order spectra are
  pooled through their declared weights into one chi-square rather than combined through the
  per-order maximum-likelihood product of Zucker (2003) … an `ivar` declared per order (as
  the readers do), or splitting the dataset by order, covers that case."

So: **one merged, monotone `wave` array per epoch**. Per-order work is done by splitting the
dataset (separate fits) or by declaring per-order `ivar`, not by any order axis in the model.

---

## 10. Throughput facts recorded in `docs/benchmarks.md`

Note for quoting: benchmarks.md contains **no** literal "D49"/"D50"/"D58"/"M5" strings — the
D-numbers live in `CHANGELOG.md` and `internal/design.md`. Use the section headings below.

### 10.1 Wall per gradient vs model pixels (the "D49" pass)

Heading: **`## Second speedup pass: the assembly's reverse pass, and four dead ends (2026-08-15)`**
(`docs/benchmarks.md:1962`), table under **`### What the two changes bought`** (`:2059`).
Machine (`:1964-1969`): "AMD Ryzen 9 9950X3D desktop, 16 cores / 32 threads, 32 GB, Windows
11, CPU only, float64 … 66.8 GB/s triad, 1188 GFLOP/s fp64 dgemm at n = 2000."

| n (model px) | eval before | eval after | ×    | ∇ before | ∇ after | ×    |
|---|---|---|---|---|---|---|
| 31,734 | 2.71 s | 2.16 s | 1.25 | 10.07 s | 5.52 s | **1.82** |
| 74,322 | 6.45 s | 5.24 s | 1.23 | 27.98 s | 15.25 s | **1.83** |
| 135,052 | 12.64 s | 10.15 s | 1.25 | 53.02 s | 29.09 s | **1.82** |
| **203,440 (design target)** | 19.11 s | **15.67 s** | 1.22 | 80.23 s | **45.88 s** | **1.75** |

Gradient/eval ratio fell 4.2× → 2.9× (`:2072-2075`). Stage split at row 0 (`:2079-2084`):
band assembly 1.94→1.38 s, block Cholesky 0.93→0.72 s, whole marginal 2.97→2.20 s,
gradient-in-velocities 10.23→5.19 s. Caveat `:2086`: whole-marginal repeats vary ~10%.

Earlier ladders on the **"laptop"** (not comparable in absolute terms):
`### Design-target ladder …` (`:279`) 203,497 px → 149.7 s / 729.9 s;
`### Ladder, before → after …` (`:689`) 203,440 → 26.0 s / 111.3 s;
memory pass (`:800`) 203,440 → 22.16 s / 87.43 s.

### 10.2 NUTS walls

| Target | Wall | Draws / chains | Source |
|---|---|---|---|
| **Packaged `sb2_sim`** via `fit.sample(seed=0)` | **601.4 s** | 1000 draws total; façade defaults `num_warmup=500, num_samples=500, num_chains=2` (`facade.py:1921-1923`), sequential CPU | `docs/tutorials/showcase.ipynb` cell `c977090e`; `scripts/build_showcase_notebook.py:10` |
| Tutorial SB2 (its own sim, not `sb2_sim`) | **140.7 s** | 100 warmup + 150 samples × 1 chain, 0 divergences | `docs/tutorials/sb2-end-to-end.md:204` |
| Gate problem, full pipeline | MAP 40 s + Laplace 23 s + **NUTS 102 s** ≈ **3 min** | 150 + 250, 0 divergences, 6.5 leapfrogs | `benchmarks.md:90-100`, `### The MAP → Laplace → NUTS pipeline …` (`:84`) |
| Gate, numpyro default warmup | **>35 min, still in warmup** | unit mass matrix | `benchmarks.md:86-89` |
| Injection–coverage study | **101 min** (~4.2 min/injection) | 24 injections × (150+250); 0.4% divergent | `benchmarks.md:135-141` |
| **AI Phe** | **no NUTS run exists** — MAP only; "Shift-and-add again took 0.07 s against albireo's 11 s" | — | `benchmarks.md:599`, `### AI Phoenicis …` (`:537`) |
| **HR 6819** | **no NUTS run exists, by decision**: "The next step is not NUTS. Sampling would return the same optimistic width around the same systematics-limited point." Opt-in only behind `ALBIREO_HR6819_NUTS=1` (300+300×2) | — | `benchmarks.md:1013-1014`; `examples/03_hr6819_real_data.py:275-277` |

HR 6819 L-BFGS walls, for scale: 120 steps / 2820 s (`:975`); AR(1) probe path 7,948–8,369 s
per window at 53–56 s/step (`:1261`); AR(1) **band** path 868 s at 5.8 s/step — 9.6×
end-to-end (`:1441`); wide window 200 steps / 2,447 s (`:1460`); fitted σ(λ) 5,650 s
(`:1530`); +h₃ 10,789 s (`:1591`).

### 10.3 Pipeline per-star walls (the "D58" pass)

Heading: **`## The pipeline in worker processes (2026-09-01)`** (`benchmarks.md:2752`). Same
9950X3D desktop. Workload: 8 simulated stars × 8 epochs × 725 native px, S/N 120, label
stage and figures **off**.

| workers | threads/worker | batch wall [s] | mean per star [s] | speedup |
|---|---|---|---|---|
| 1 (in-process) | 32 | 132.2 | 16.5 | 1.00× |
| 2 | 16 | 101.4 | 24.0 | 1.30× |
| 4 | 8 | 66.7 | 29.9 | **1.98×** |
| 8 | 4 | 54.0 | 43.5 | **2.45×** |
| 8 | 32 (cap off) | 54.7 | 43.0 | 2.42× |

"The thread cap made no measurable difference" (`:2785`). `jobs="auto"` = `cpu_count // 4`
= 8 here (`:2795`). **Not measured**: the label stage, "the expensive part of a full run
(~50 s against ~30 s of disentangling on this star in fast mode)" (`:2800`). Restated in
`docs/api/pipeline.md:61-64` and `docs/tutorials/pipeline.md:144`.

### 10.4 The M5 GPU gate: **open**

Headings: `## Scale, benchmarks, release readiness (2026-08-11)` (`:229`) →
`### GPU projection (stated as projection, not measurement)` (`:295`) →
**`### First real GPU run (2026-08-14): the path works, the gate does not close`** (`:310`).

Gate text (`:231-237`): "2×10⁵ px / 50 epochs samples in minutes on one GPU … the CUDA path
works and scales as predicted, but the gate does not close on the consumer card available
here, for two measured reasons: 16 GB of device memory against a gradient that needs
18.24 GB, and fp64 at 1/50 of fp32."

Hardware: RTX 5070 Ti, 16 GB, WSL2, jax 0.11.0 cuda12 (`:312-315`).

| n (model px) | native px/epoch | eval | ∇ eval |
|---|---|---|---|
| 4,877 | 1,702 | 0.232 s | 0.236 s |
| 9,526 | 3,457 | 0.447 s | 0.461 s |
| 18,221 | 6,965 | 0.915 s | 0.871 s |
| 31,734 | 13,062 | **OOM** (7.42 GiB single request, 13.8 GiB free) | — |

4096³ matmul: **fp32 39,023 GF/s vs fp64 783 GF/s — a 50× penalty** (`:340-348`).
Requirement now stated as "**≥ 24–40 GB of device memory and a 1:2 fp64 ratio, i.e. A100 or
H100 class**" (`:350-356`). What *did* close: portability — "albireo's graph compiles and
runs correctly under CUDA with no code change" (`:358-361`). Revised projection (`:741-744`):
design-target gradient ~0.5–1.5 s on one A100-class device, converged posterior in tens of
minutes.

### 10.5 Machine labelling caveat (matters for the plan's quoting)

benchmarks.md treats "Windows 11 laptop" and the 9950X3D desktop as **two different
machines** and forbids reading absolute numbers across the boundary (`:1966-1969`,
`:2319-2321`, `:2403-2407`). Both are described as 32 GB / Windows 11, and the D50 stack line
(`:2242`) says "Windows 11 Pro build 26200" — the same as this box. If the "laptop" tables
were in fact taken on the same desktop, the 1.46× "residual clean-machine gap" reasoning
(`:2298-2300`) and "fd3 moved 12% across that hardware change" (`:2125`) rest on a mislabel.
Nothing in the repo settles it either way.

### 10.6 Three-way vs fd3 and shift-and-add (authoritative re-run)

Heading: **`## Re-run: all three codes, one machine, and the wall that would not reproduce
(2026-08-16)`** (`:2233`); accuracy `### Accuracy first: twelve values, twelve exact
reproductions` (`:2245`), walls `### The walls, one protocol` (`:2254`).

| | recorded (earlier laptop) | min | median |
|---|---|---|---|
| shift-and-add, 7 sweeps | 0.018 s | **0.0263 s** | 0.0267 s |
| albireo (jitted steady state; cold compile+first call 0.495 s) | 0.182 s | **0.0591 s** | 0.0625 s |
| fd3, `OMP_NUM_THREADS=1` | — | **0.0636 s** | 0.0640 s |
| fd3, environment as found | 0.111 s | 0.1042 s | 0.1113 s |

Accuracy reproduced exactly: albireo 0.0118/0.0093 and 0.0165/0.0116; fd3 0.1767/0.0198 and
0.2597/0.0223; shift-and-add 0.0317/0.0248 and 0.0849/0.0302 (`:2248-2250`) — **albireo ~2×
more accurate, and the only one returning a posterior.** Both original anomalies were
measurement artifacts: the harness heated the heap (`:2275`) and fd3's BLAS spun 32 threads
("1938% CPU … 1.95 s of user time inside a 0.10 s wall", `:2305`).

---

## 11. The forecast API, and a Gaia DR4 RVS forecast today

### Signatures

```python
# src/albireo/forecast.py:85
def plan_epochs(
    template: EpochData | InstrumentSpec,
    bjd,
    *,
    snr: float | None = None,
    v_bary=0.0,
    instrument: str | None = None,
    medium: str | None = None,
) -> tuple[EpochData, ...]
```

```python
# src/albireo/forecast.py:714
def sensitivity_forecast(
    grid: LogGrid,
    dataset: Dataset,
    *,
    light_fractions,
    lsf_sigma_v: Mapping[str, float | Sequence[float]],
    prior: SmoothnessPrior,
    orbit: Mapping | OrbitParams | None = None,
    velocities=None,
    baseline=None,
    lsf_anchors_angstrom=None, lsf_h3=None, response_coeffs=None,
    telluric=False, nebular=False, nebular_v_kms=0.0, nebular_amplitudes=None,
    jitter=None, ar1_phi=None,
    region=None, region_floor=0.1,
    n_modes=4, mode_iterations=120, mode_seed=0,
    feature_scales_kms=None, n_scales=48, penalty_threshold=2.0,
    block_size=None, half_bandwidth=None,
) -> SensitivityForecast
```

### What a forecast needs — and why it needs no fluxes

`forecast.py:1-15`:
> "Only the epoch times (through the velocities, hence the shifts), the per-pixel weights,
> the masks, the line-spread functions, the light fractions, the response and the prior
> enter it, and each of these is known for an observation that has not been taken."

`tests/test_forecast.py` verifies this by overwriting every flux and requiring a
bit-identical result. Planned epochs carry `flux = 1.0` exactly (`forecast.py:171`).

Exactly one of `orbit` / `velocities` is required. `orbit` may be a θ mapping or an
`ab.OrbitParams` "as tabulated in a paper" (`forecast.py:768-775`).

**The orbit is not forecast** (`forecast.py:37-40`): "an error bar on `K_2` requires the
component line depths, which are not known before the data exist." The outputs are
`component_std`, worst-determined modes, `p_eff`, `information_nats`, `blind_fraction`.

### Gaia DR4 RVS (961 px, 846–870 nm, R ≈ 11500, ~20–40 transits with known times)

Everything needed exists today. Sketch:

```python
wave = np.linspace(8460.0, 8700.0, 961)   # Angstrom, VACUUM; 0.25 A step = 8.74 km/s @ 8580 A
spec = ab.InstrumentSpec(wave=wave, sigma_v_lsf=ab.LSF.from_resolution(11_500).sigma_kms,
                         snr=<per-transit S/N>)            # -> 11.07 km/s
plan = ab.plan_epochs(spec, bjd=<known transit times>, v_bary=<astropy per date>,
                      instrument="RVS", medium="vacuum")
ds   = ab.Dataset(plan, frame="barycentric")
grid = ab.LogGrid.covering(ds, dv_kms=8.7,                  # <= the 8.74 km/s native step
                           v_margin_kms=(K1+K2)*(1+e), lsf_sigma_kms=11.07)
fc   = ab.sensitivity_forecast(grid, ds, light_fractions=(l1, l2),
                               lsf_sigma_v={"RVS": 11.07},
                               prior=ab.SmoothnessPrior(tau=..., eta=...),
                               orbit=ab.OrbitParams(period=..., t_peri=..., ecc=...,
                                                    omega=..., k=(K1, K2)))
```

Five things the plan should state:

1. `InstrumentSpec` (`simulate.py:66-92`) is not named, so `plan_epochs` gives it
   `instrument="default"` and `medium=None` unless overridden — pass `instrument="RVS"` and
   `medium="vacuum"` (RVS is a vacuum scale; `grids.py:71-78` puts the air/vacuum difference
   at ~83 km/s, larger than most K's).
2. `snr` is the only noise handle (`ivar = snr**2` uniform, `forecast.py:149`). Use
   `jitter=` to express doubt about an ETC number — "exposure-time-calculator inverse
   variances are typically optimistic" (`forecast.py:47-49`).
3. `v_bary` defaults to 0, "a convention, not a measurement" (`forecast.py:122-126`) — it
   only matters with a telluric/nebular column, which RVS does not need.
4. The `prior` **is part of the forecast** (`forecast.py:760-767`): a looser prior reports a
   degeneracy the fit will not have. There is no RVS-calibrated `(tau, eta)` in the repo, so
   this is an assumption to state.
5. **D47 finding to carry**: math.md's own `Var_j(Delta)` proxy ranks designs *incorrectly*;
   `_separation_diagnostics` records why the exact covariance is computed alongside it rather
   than in place of it (`forecast.py:31-35`).

Related, from memory and confirmed in the repo: Gaia DR3 RVS is unusable (rest-frame spectra,
no hot stars, HR 6819 absent); DR4 `rvs_epoch_spectrum` is the target, released 2026-12-02.
