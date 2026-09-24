# Synthetic spectral libraries

A published grid of synthetic spectra (BOSZ, POLLUX, PHOENIX) reduced to the four
quantities a label fit needs: the node labels, the normalized flux at each node, the
continuum at each node, and the wavelength scale on which they are defined. albireo reads
grids computed elsewhere and carries their citations; it contains no line list and no
radiative transfer.

Two requirements are stricter than usual, because the alternative in each case fails
without warning.

**The wavelength medium is a required field with no default.** Air and vacuum wavelengths
differ by about 83 km/s across the optical, the same order as the orbital semi-amplitudes
albireo measures, so a library on the wrong scale gives a confident wrong answer rather
than a slightly worse fit. Upstream documentation is not a reliable source: BOSZ
2017 was vacuum throughout and BOSZ 2024 is air above 200 nm, under the same name.
`line_core_medium` therefore measures the convention from the spectra themselves, and the
ingest paths use it to verify a declaration rather than to supply one.

**Interpolation is in flux, never in model atmospheres.** On the 250 K / 0.5 dex spacing
that BOSZ uses, Mészáros & Allende Prieto (2013) measured 0.19% scatter when interpolating
atmospheres, against 0.051% when interpolating fluxes linearly and 0.031% with a cubic,
while a Payne-style neural emulator reaches about 0.1%. On a well-sampled grid the
differentiable cubic used here is therefore the more accurate option, with no training cost
and no weights to host. Whether this holds for a particular grid is measured by
`crossval_library`.

`library_interpolator` selects its method from the grid's geometry: a separable
Catmull-Rom cubic on a complete axis product, and barycentric interpolation over a Delaunay
triangulation when physical limits have removed corners of the grid, as they have for every
public OB library. The cubic reproduces a node bit-for-bit, because a node lands on weights
that are exactly 1 and 0. The simplex path reproduces one to rounding, because its weights
are an affine transform of the point: the error is machine epsilon times the spread of the
spectra across the simplex, at the ulp level for a library whose neighbouring nodes are
alike. Either is far below anything the data can distinguish, so the warm-start node scan in
[`albireo.match`](match.md) and the continuous fit can be compared on the same footing.

They are not equivalent for the fit that follows. The simplex interpolant is
piecewise linear, and on a lattice with a piece removed the triangulation is arbitrary, so
the objective has kinks along simplex faces where a gradient method stops: on a perfect
spectrum drawn from the BOSZ grid itself, the label fit under the simplex path ended 20 to
150 K from the truth with a chi-square above the truth's, while the cubic path recovers it.
BOSZ lacks exactly one model in its FGK box, (5750 K, log g 3.0, [M/H] −0.75), and one gap
costs the grid its box structure. `ingest_bosz` therefore fills a missing node that two
published neighbours bracket along one axis by linear interpolation between them
(metallicity first, then gravity, then temperature), records the node and the rule in
`meta["filled_nodes"]`, and the summary and the label report name it; a node with no
neighbours across it, a corner, is still dropped. A fit within one grid step of a filled
node rests partly on that interpolation.

## The library's own resolving power

A published grid is not an intrinsic spectrum: every BOSZ entry in the registry is the
R = 20,000 file, so each line already carries a Gaussian of
$`\sigma_{\mathrm{lib}} = c/(R\, 2\sqrt{2\ln 2}) = 6.37`$ km/s. A comparison that convolves such
a template with the whole instrument profile broadens it twice, and the fitted *v* sin *i*
absorbs the difference; on the D65 Gaia RVS benchmark that put *v* sin *i* at the floor of its
prior on 6 of 24 components. `SpectralLibrary.resolving_power` therefore reports the value,
and the label fit applies only the quadrature width
$`\sqrt{\sigma_{\mathrm{inst}}^2 - \sigma_{\mathrm{lib}}^2}`$ wherever it convolves a template
([§9.2a](../math.md#92a-comparing-in-the-epoch-space)), refusing a library at or below the
instrument's resolving power.

The property reads `meta["resolving_power"]`, the key to set on a hand-built library, and
otherwise `meta["resolution"]`, which `ingest_bosz` records:

| source | `resolving_power` |
|---|---|
| `fetch_library` / `ingest_bosz`, all four `bosz2024-*` entries | 20,000, carried through the cache, `wave_range=`, `sliced`, `in_medium` and `resampled_to` |
| `pollux-ob-smc24`, built by hand (`ingest_pollux` raises) | `None` unless the builder sets `meta["resolving_power"]`, which it should |
| `albireo.simulate.synthetic_library` and any library declaring neither key | `None`: an intrinsic grid |

A value that is not a finite positive number raises rather than being read as intrinsic.

## Obtaining a grid

`fetch_library` builds a named library on first use and caches it, so the download occurs
once per machine:

```python
import albireo as ab

ab.library_names()
# ['bosz2024-fgk-r20000', 'bosz2024-fgk-rvs', 'bosz2024-hot-r20000',
#  'bosz2024-hot-rvs', 'pollux-ob-smc24']

ab.library_info("bosz2024-fgk-r20000")["licence"]   # look before you download
library = ab.fetch_library("bosz2024-fgk-r20000")   # ~621 MB once, ~51 MB cached
```

The cache lives under `albireo.examples.cache_dir()`, and `$ALBIREO_DATA_DIR` redirects
it, which is also how a shared or pre-populated directory is used on a cluster. Narrowing
the band with `wave_range=` is supported; widening it is refused, because the registered
band is what was downloaded and returning a narrower band than requested would be a silent
failure.

| name | grid | coverage | band | nodes |
|---|---|---|---|---|
| `bosz2024-fgk-r20000` | BOSZ 2024, MARCS, R = 20,000 | Teff 4000–7000 K (250 K), log g 3–5 (0.5), [M/H] −1→+0.5 | 4000–7000 Å | 455, one filled |
| `bosz2024-fgk-rvs` | the same nodes | as above | 8350–8850 Å, the Gaia RVS window | 455, one filled |
| `bosz2024-hot-r20000` | BOSZ 2024, MARCS + ATLAS9, R = 20,000 | Teff 7000–10,000 K (250 K), log g 3.5–5 (0.5), [M/H] −1→+0.5 | 4000–7000 Å | 364, all published |
| `bosz2024-hot-rvs` | the same nodes | as above | 8350–8850 Å, the Gaia RVS window | 364, all published |
| `pollux-ob-smc24` | POLLUX, CMFGEN, non-LTE | Teff 23–55 kK, log g 2.5–4.5, [M/H] = −0.73 fixed | 3850–4650 Å | 915 |

Both upstream grids, BOSZ 2024 (Mészáros et al. 2024) and POLLUX (Palacios et al. 2010),
are CC BY 4.0, and `library_info` carries the citation each one requires.

**BOSZ builds automatically; POLLUX does not.** BOSZ's URLs on MAST are deterministic, so
`ingest_bosz` constructs them, downloads the shards in parallel, keeps the raw files so that
another band can be cut without a new download, and slices to the registered band. POLLUX
serves its collections through a form that posts to `/download/`, so there is no stable URL
to fetch; `ingest_pollux` reports this and stops rather than parsing a file format that has
not been examined.

**Integrity of a cached build.** A cached build is verified on every load against a digest
taken over the arrays, so it is reproducible across machines and independent of how the
`.npz` was compressed; two users can compare `meta["content_sha256"]` to confirm they built
the same library. The assembled spectra have their medium measured and checked against the
registry's declaration, and a disagreement raises rather than being reconciled. A
registry-level pin published under a DOI is not yet in place. BOSZ was recomputed on
2025-09-25 to correct its hydrogen lines and OH⁺ strength without a change of name, so the
build date is recorded in `meta["retrieved"]`, and an older build is a different
calculation.

## Above 7000 K: the hot box

`bosz2024-hot-rvs` and `bosz2024-hot-r20000` carry Teff 7000 to 10,000 K in 250 K steps,
log g 3.5 to 5.0 in 0.5 dex, and [M/H] −1.0 to +0.5 in 0.25 dex. All 364 nodes are
published at the composition the registry pins, checked against the archive's own index on
2026-09-10, so nothing is filled or dropped and the cubic applies without the filled-node
caveat. The temperature axis continues the FGK box's 250 K
spacing rather than reaching the step changes BOSZ's grid has at 4000 and 12,000 K, which
is what keeps the Catmull-Rom weights, written in their uniform-parameter form, valid over
the whole range. The gravity floor at 3.5 is a main-sequence floor, since an A or early-F
dwarf sits between log g 3.9 and 4.3; it also keeps the MARCS part of the box
plane-parallel, the geometry ATLAS9 uses, so the spherical-to-plane-parallel step inside
the FGK box's log g axis does not occur in this one.

The box is not a continuation of the FGK library, and five of its properties bear on what
a velocity or a label taken from it means.

**A seam between two model-atmosphere codes.** MARCS supplies the box to 8000 K and ATLAS9
from 8250 K, on different solar abundance scales (Grevesse et al. 2007 below, Asplund et
al. 2005 above). Mészáros et al. (2024) measure the two families as differing by about half
a percent in flux between 350 and 1000 nm at 8000 K, and give no recommendation on
interpolating across them; the overlap spectra exist to quantify the difference, not to
blend it. A cubic stencil spanning 7750 to 8500 K mixes the two codes into one tangent, so
a temperature within one grid step of 8125 K rests on that. A smooth flux offset divides
out of the normalized spectrum and lands in `log_continuum`, so the seam should be expected
to reach the light ratio, which `rvs_light_ratio` takes from `log_continuum`, before it
reaches the velocities. The half a percent is a flux difference over 350 to 1000 nm and is
not a measurement in the RVS band or after normalization; the 7500 to 8000 K overlap
publishes both families at the same parameters, so it can be measured directly.

**The Ca II triplet gives way to the Paschen series.** Between 5000 and 10,000 K the
triplet's line depths fall from 0.66–0.72 to 0.34–0.47 while Paschen 12, 14 and 16 deepen
to 0.33, 0.21 and 0.10. Five Paschen members lie inside albireo's `RVS_BAND` of 8460 to
8700 Å, at 8467, 8502, 8545, 8598 and 8665 Å in air, and two of them are blends: P15 at
8545.4 Å sits 3.3 Å from Ca II 8542.09, and P13 at 8665.0 Å sits 2.9 Å from Ca II 8662.14.
Above roughly 8000 K a velocity measured in this band is therefore a hydrogen-line velocity
with a calcium blend rather than a metal-line velocity, and hydrogen lines are broad and
pressure sensitive, so the velocity and the gravity are correlated in a way they are not
for an FGK star.

**Rotation defeats the RVS resolution.** `RVS_RESOLVING_POWER` is 11,500, which is 26.1
km/s FWHM, or 11.1 km/s of Gaussian $`\sigma`$. A field A star rotates at 100 to 200 km/s,
four to eight times that width. At $`v \sin i = 150`$ km/s the rotational half-width at
8542 Å is 4.3 Å, so the profile spans 8.5 Å and the 3.3 Å separation of the Ca II and
Paschen pair disappears inside it. `_draw_vsini` in
[`albireo.population`](gaia.md#populations) draws rotation above the Kraft break from a
log-normal around 40 km/s, which is an F-star rate; for A stars it is optimistic, and a
population drawn with it understates the difficulty of this box.

**LTE, unquantified rather than small.** Every BOSZ 2024 spectrum assumes LTE. The paper
stops the grid at 16,000 K for want of NLTE in ATLAS9 and gives no estimate of the LTE
error below that, and it does not discuss the Ca II triplet. For cool stars the 3D non-LTE
radial-velocity correction on the triplet reaches about 0.25 km/s typically and 1 km/s in
outliers at RVS resolution (Lagae, Amarsi & Lind 2025), a floor under any LTE template; the
equivalent number for A stars does not exist in the literature the registry cites. The
ATLAS9 half is also untested against data: the paper's CALSPEC comparison reaches only
9420 K, and its one quantified statement about the hot models is model against model, a 10
to 15 per cent flux change from BOSZ 2017 near the Paschen jump above 10,000 K, which
affects `log_continuum` at the top of the box more than the normalized flux. Every ATLAS9
node in the grid is published, so that half is the complete one; completeness is not
reliability.

**The recomputation date is not settled for these shards.** BOSZ 2024 was recomputed on
2025-09-25 to correct the hydrogen lines and the OH⁺ band strength. The archive's shards
for this box carry Last-Modified dates in April and May 2025, before that, while their
directories were rewritten in late September. The correction replaced an eight-level
hydrogen atom, which carries no transition from $`n \geq 9`$, and $`n \geq 9`$ is the
crowded end of the Paschen series inside this band; the MAST README scopes the defect to
wavelengths above 5.8 μm, which is hard to reconcile with that. Check a build on the high
Paschen members before trusting the band above 8000 K.

Both entries request the same 364 shards, 532 MB, and a machine that builds one and then
the other pays the network once. The built caches are 4 MB for the RVS slice and 41 MB for
the optical, and the raw shards are retained unless `ingest_bosz(..., keep_raw=False)` is
used.

```python
library = ab.fetch_library("bosz2024-hot-rvs")   # ~532 MB once, ~4 MB cached
ab.library_info("bosz2024-hot-rvs")["caveats"]   # the same statements, beside the numbers
```

One run uses one library, so a benchmark or a pipeline configuration selects the box that
matches its sample: `python scripts/gaia_rvs_benchmark.py --library bosz2024-hot-rvs`
admits the catalogue systems above 7000 K and leaves the cool ones out.

The theory is in [§9.3](../math.md#93-interpolation-and-why-not-an-emulator-yet).

Background and references: [science overview](../science.md).

::: albireo.library
