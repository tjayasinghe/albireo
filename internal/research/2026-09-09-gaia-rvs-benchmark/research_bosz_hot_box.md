# Extending the BOSZ 2024 library above 7000 K for the Gaia RVS band

Research note, 2026-09-10. No file under `src/`, `tests/` or git was touched, and no spectrum
file was downloaded. What was fetched: the fourteen metallicity directory indexes under
`https://archive.stsci.edu/hlsps/bosz/bosz2024/r20000/`, 43 HTTP HEAD requests for exact file
sizes, the paper and the MAST README. Sizes and node existence below are measurements against
the archive as it stood on 2026-09-10, not quotations of documentation.

Outputs in this directory: `nodes_hot.csv` (2917 published nodes at Teff >= 7000 K),
`nodes_all.csv` (all 6628 nodes at the fixed composition, all temperatures), `head_sizes.csv`
(the HEAD probes), `gaia_rows.json` and `two_systems.json` (task 2), `harvest.py` and
`check_boxes.py` (the two scripts).

## 1. Published coverage above 7000 K

The archive serves an Apache index per metallicity directory, so no probing was needed for
existence. Every line naming a file with `a+0.00_c+0.00_v2_r20000_resam` was kept, which is
exactly the composition `_BOSZ_FIXED` pins. That gives 6628 nodes over all temperatures and
all fourteen metallicities, of which 2917 have Teff >= 7000 K. `nodes_hot.csv` holds them as
`teff, logg, mh, atmosphere, bytes, last_modified`.

**Temperature axis.** 53 values, in three regimes: 100 K below 4000 K, 250 K from 4250 to
12000 K, and 500 K from 12500 to 16000 K. The step change sits between 12000 and 12500 K and
does not coincide with the model-atmosphere boundary. The grid ends at 16000 K. This matches
Table 2 of the paper.

**Model-atmosphere families.** ATLAS9 (`ap`) starts at 7500 K and MARCS (`ms`, `mp`) ends at
8000 K, so 7500, 7750 and 8000 K are a genuine three-node overlap in which both families are
published for the same physical parameters. Below 7500 K there is no `ap` file and above
8000 K there is no `mp` or `ms` file. `_bosz_atmosphere` resolves the overlap toward MARCS,
which is a deliberate choice recorded in its docstring.

**Gravity axis, measured, union over all fourteen metallicities:**

| Teff | family | log g published |
|---|---|---|
| 7000 K | MARCS `ms` | +1.0 to +3.0 |
| 7000 K | MARCS `mp` | +3.5 to +5.0 |
| 7250 to 8000 K | MARCS `ms` | +2.0 to +3.0 |
| 7250 to 8000 K | MARCS `mp` | +3.5 to +5.0 |
| 7500 to 12000 K | ATLAS9 `ap` | +2.0 to +5.0 |
| 12500 to 16000 K | ATLAS9 `ap` | +3.0 to +5.0 |

The step is 0.5 dex everywhere. The upper limit is +5.0 above 4000 K, not the +5.5 the MAST
HLSP page prints for three of its rows; +5.5 exists only below 4000 K, and HEAD requests on
`mp_t5500_g+5.5`, `mp_t6000_g+5.5` and `mp_t7500_g+5.5` return 404. Table 2 of the paper is
right and the MAST page is wrong on this point.

**Composition and microturbulence** (paper, Table 1 and Sect. 2.1, confirmed against the
fourteen directories): [M/H] from -2.50 to +0.75 in 0.25 dex (14 values), [alpha/M] from
-0.25 to +0.50 in 0.25 (4), [C/M] from -0.75 to +0.50 in 0.25 (6), microturbulence 0, 1, 2
and 4 km/s. 336 unique compositions.

**Holes above 7000 K.** At the fixed composition there are exactly six incomplete
(family, Teff, log g) nodes above 7000 K, and every one of them is MARCS:

| node | metallicities published |
|---|---|
| `ms` 7000 K, log g +1.0 | 1 of 14 (only [M/H] +0.00) |
| `ms` 7000 K, log g +1.5 | 13 of 14 (missing -1.25) |
| `ms` 7500 K, log g +2.0 | 13 of 14 (missing -2.00) |
| `ms` 7750 K, log g +2.0 | 12 of 14 (missing -1.25, -1.00) |
| `ms` 8000 K, log g +2.0 | 9 of 14 (missing -1.50, -1.00, -0.75, -0.25, +0.00) |
| `mp` 7750 K, log g +4.0 | 13 of 14 (missing -1.25) |

**Every ATLAS9 node in the entire grid is published at this composition, with no holes at
all.** That is the measured version of the remark in Sect. 2.2 of the paper that the ATLAS9
grid is complete over its range, and it means the hot half of any proposed box is free of the
bracketing question entirely. The only holes that can land inside a plausible hot box are the
`ms` log g +2.0 group and the single `mp` 7750 K, log g +4.0, [M/H] -1.25 node.

## 2. The two systems the D62 Gaia run left out

Reproduced by running `query_gaia_sb2(16)` and `from_gaia_sb2(rows, seed=0,
library=fetch_library("bosz2024-fgk-rvs"), release="dr3")`, which is exactly what
`scripts/gaia_rvs_benchmark.py --gaia-sb2 16 --seed 0` does. The library came from the
existing cache, so nothing was downloaded. `inside_library` compares against
`library.bounds`, which is Teff 4000 to 7000 K, log g 3.0 to 5.0, [M/H] -1.0 to +0.5.

**gaia-34564217466900480** (SB2, P = 12.2122 d, K1 = 62.59, K2 = 66.41 km/s, e = 0.029,
G = 7.59, G_RVS = 7.13, 10 transits). The `binary_masses` row of the catalogue has m1 = 1.763
and m2 = 1.871 solar, so `_complete` puts the more massive star first and the components come
out in the order below.

| | primary | secondary |
|---|---|---|
| mass (solar) | 1.871 | 1.763 |
| radius (solar) | 1.885 | 1.749 |
| Teff (K) | **8618.7** | **7636.3** |
| log g | **4.1594** | **4.1987** |
| [M/H] | 0.0 | 0.0 |
| vsini (km/s) | 7.08 | 6.57 |
| light fraction in the RVS band | 0.6074 | 0.3926 |

F2 / F1 = 0.6462.

**gaia-49181812242533888** (SB2, P = 4.5141 d, K1 = 71.41, K2 = 71.59 km/s, e = 0.0059,
G = 8.52, G_RVS = 7.99, 18 transits, m1 = 1.524, m2 = 1.520 solar).

| | primary | secondary |
|---|---|---|
| mass (solar) | 1.524 | 1.520 |
| radius (solar) | 1.690 | 1.688 |
| Teff (K) | **7065.0** | **7057.7** |
| log g | **4.1652** | **4.1650** |
| [M/H] | 0.0 | 0.0 |
| vsini (km/s) | 14.51 | 14.49 |
| light fraction in the RVS band | 0.5012 | 0.4988 |

F2 / F1 = 0.9952.

Three things follow.

The second system misses the box by 65 K and 58 K. It is a pair of near-identical early-F
stars that the 7000 K ceiling excludes by less than one grid step. Adding a single Teff node
at 7250 K to the FGK box would admit it (490 nodes rather than 455, one more filled node at
5750 K, 671 MB), which is the cheapest possible fix for one of the two but does nothing for
the other and would force every existing user to rebuild the FGK library.

Both metallicities are 0.0 because `from_gaia_sb2` hard-codes `mh=0.0`; the archive row
carries no abundance. Any [M/H] axis for a hot library is therefore driven by what the
library should support, not by what these two rows need.

Both light ratios came from the Planck fallback in `_complete`, not from the library, because
`_ContinuumRatio.inside` is false for labels outside the box. Inside a hot library they would
instead come from the interpolated `log_continuum`, and would change. That is a second, quiet
consequence of the leave-out: the declared light fractions of the population for hot systems
are currently a Rayleigh-Jeans approximation at 858 nm rather than a model continuum ratio.

Neither system is fast-rotating, because both periods are under the 15-day synchronisation
threshold in `_draw_vsini` and the resulting equatorial velocities are small. The rotation
caveat in Sect. 5 is about the wider late-A population, not about these two.

## 3. The proposed box

The test applied to each candidate is the one `ingest_bosz` actually performs: for every
(Teff, log g, [M/H]) in the box, ask `_bosz_atmosphere` which family it would request, then
look that exact file up in the harvested listing. A node the archive does not publish is then
classified the way `_bracketing_neighbours` would classify it, trying metallicity, then
gravity, then temperature. `check_boxes.py` does this and its output is reproduced below.

### Recommended: Teff 7000 to 10000 K, log g +3.5 to +5.0, [M/H] -1.0 to +0.5

364 nodes. **Every node is published.** No node needs filling, none is dropped, and the grid
is a complete box, so the cubic interpolant applies with no caveat about a filled node.

Steps: Teff 250 K throughout (13 values, 7000 to 10000), log g 0.5 dex throughout (4 values),
[M/H] 0.25 dex throughout (7 values). The Teff step is uniform across the whole axis, which
matters because `_axis_weights` uses Catmull-Rom in its uniform-parameter form: the tangent at
a node is estimated from the two neighbours as if they were equally spaced, so a non-uniform
axis would lose linear reproduction at the step change.

Why log g stops at +3.5 rather than +3.0. Both left-out systems sit at log g 4.16 to 4.20,
and a late-A or early-F main-sequence star sits between 3.9 and 4.3, so 3.5 is already a
generous floor. Stopping there buys two things. First, the box is 364 nodes rather than 455,
a fifth cheaper. Second, and more important, the MARCS part of the box is then entirely `mp`,
plane-parallel, which is also the geometry ATLAS9 uses, so the existing caveat about MARCS
switching from spherical to plane-parallel inside the log g axis does not apply to this
library at all, and the geometry is uniform across the family boundary as well.

The variant with log g down to +3.0 is also complete, 455 nodes, 666 MB. It matches the log g
axis of the FGK box exactly, which is worth something if a user moves between the two
libraries. Its cost is that at log g +3.0 the requested family is `ms` (MARCS spherical) below
8000 K and `ap` (ATLAS9 plane-parallel) above, so that one row of the box changes both code
and geometry along the temperature axis. Recommend it only if the subgiant coverage is wanted.

The variant with log g down to +2.5 is complete too, 546 nodes, 798 MB. **Do not go below
+2.5.** A box reaching log g +2.0 loses its box structure: at [M/H] -1.00 the nodes
(7750 K, +2.0) and (8000 K, +2.0) are both absent and adjacent, so neither can be bracketed
along temperature and neither has a published metallicity neighbour on the low side within the
box, and `ingest_bosz` drops them both. Three further nodes at (8000 K, +2.0, [M/H] -0.75,
-0.25, +0.00) would be filled along temperature. Two drops are enough to send the whole grid
to the barycentric fallback, whose kinks cost the label fit 20 to 150 K.

The [M/H] axis is the one the FGK box uses. Extending it down to -1.50 or -2.00 stays usable
but is no longer clean: the node (7750 K, log g +4.0, [M/H] -1.25) is unpublished and would be
filled by linear interpolation along metallicity, so the box gains a `known_gaps` entry.

### The registry entry

```python
# The hot box is one uniform 250 K axis from the ceiling of the FGK box to 10,000 K, so the
# Catmull-Rom weights, which assume equal node spacing, stay valid across the whole range.
# Verified against the archive listing on 2026-09-10: every one of these 364 nodes is
# published at a+0.00, c+0.00, v2, r20000, with no hole to fill and none to drop.
# log g starts at 3.5 so that the MARCS half is plane-parallel throughout, the same geometry
# ATLAS9 uses, which removes the spherical-to-plane-parallel step of the FGK box from this one.
_BOSZ_HOT_AXES: dict[str, Any] = {
    "teff": [float(t) for t in range(7000, 10001, 250)],
    "logg": [3.5, 4.0, 4.5, 5.0],
    "mh": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5],
}

_BOSZ_HOT_CAVEATS = (
    "The box crosses the MARCS/ATLAS9 boundary between 8000 and 8250 K. Below it the "
    "spectra come from MARCS atmospheres on the Grevesse et al. (2007) solar scale; above "
    "it from ATLAS-APOGEE ATLAS9 on the Asplund et al. (2005) scale. Meszaros et al. "
    "(2024, Sect. 3.4) measure the two families as differing by about half a percent in "
    "flux between 350 and 1000 nm at 8000 K, and offer no recommendation on interpolating "
    "across the seam. A cubic stencil spanning 7750 to 8500 K mixes the two codes into one "
    "tangent, so a temperature within one grid step of 8125 K rests on that assumption.",
    "The atmospheres either side of the seam were built at different microturbulence: "
    "1 km/s for MARCS plane-parallel, 2 km/s for ATLAS9, while the synthesis on both is "
    "fixed here at 2 km/s. The inconsistency between structure and synthesis is therefore "
    "present below 8000 K and absent above it.",
    "Everything is LTE. The paper stops the grid at 16,000 K for want of NLTE in ATLAS9 "
    "and gives no estimate of the LTE error below it, so for the A stars in this box the "
    "size of that error is unquantified rather than small.",
    "Above about 8000 K the Ca II triplet weakens and the Paschen series carries the band. "
    "Five Paschen members lie in 8460-8700 Angstrom, and P15 at 8545 A and P13 at 8665 A "
    "sit within 3.3 A of Ca II 8542 and 8662. A velocity measured from this band above "
    "8000 K is a hydrogen-line velocity with a calcium blend, not a metal-line velocity.",
    "Meszaros et al. (2024, Sect. 4.1) report BOSZ 2017 to 2024 flux changes of 10 to 15 "
    "percent near the Paschen jump at 850 to 900 nm above 10,000 K. The normalized flux is "
    "far less affected than the continuum, but log_continuum in this band is "
    "version-sensitive at the top of the box, and it is log_continuum that sets the "
    "light ratio.",
    "Gaia publishes RVS spectra on the vacuum scale and this library is air. Convert with "
    "SpectralLibrary.in_medium('vacuum') before comparing the two.",
)

"bosz2024-hot-rvs": _Library(
    name="bosz2024-hot-rvs",
    description=(
        "BOSZ 2024 (MARCS + ATLAS9) late-A to early-F in the Gaia RVS band, "
        "R = 20,000, 8350-8850 Angstrom"
    ),
    source="bosz2024",
    version="1",
    wave_range=(8350.0, 8850.0),
    medium="air",
    label_names=("teff", "logg", "mh"),
    axes=_BOSZ_HOT_AXES,
    fixed=_BOSZ_FIXED,
    licence="CC BY 4.0",
    citation="Meszaros et al. 2024, A&A 688, A197 (arXiv:2407.10872)",
    doi="10.17909/T95G68",
    upstream_note=_BOSZ_RECOMPUTE_NOTE,
    caveats=_BOSZ_HOT_CAVEATS,
    known_gaps=(),
    download_mb=535.0,
    cache_mb=4.5,
),
```

The optical companion is the same entry with `wave_range=(4000.0, 7000.0)`, the last caveat
dropped, `cache_mb=42.0`, and `name="bosz2024-hot-r20000"`. The two share their raw shards,
so building the second after the first costs nothing to download.

**No change to `_bosz_atmosphere` or `ingest_bosz` is required.** The existing dispatch
already returns `mp` for 7000 to 8000 K at log g >= 3.5 and `ap` above 8000 K, and
`_BOSZ_FIXED` is reusable unchanged.

### Alternative reaching the hot end of the grid

The grid ends at 16000 K, but its temperature axis is 250 K to 12000 K and 500 K above, so a
single uniform axis cannot span both. Two clean choices exist.

**Stop at 12000 K on the 250 K axis.** Teff 7000 to 12000 (21 values), log g +3.5 to +5.0,
[M/H] -1.0 to +0.5: 588 nodes, every one published, complete box, 863 MB. This is the hottest
box that keeps the 250 K spacing the FGK library uses, and therefore the only one to which the
Meszaros and Allende Prieto (2013) interpolation figures, 0.051 percent linear and 0.031
percent cubic, transfer without an argument by analogy.

**Reach 16000 K by taking every other node.** Teff 7000 to 16000 in 500 K steps (19 values),
log g +3.5 to +5.0, [M/H] -1.0 to +0.5: 532 nodes, every one published, complete box, 790 MB.
The ATLAS9 part allows this without qualification, since every `ap` node in the grid is
published; the constraint is not the archive but the axis. With log g down to +3.0 it is 665
nodes and 988 MB, also complete, and +3.0 is in any case the floor above 12000 K, so a box
reaching 16000 K cannot go below it without losing its rectangular shape.

The price of the 500 K axis is that it is twice the spacing on which the 2013 interpolation
error was measured, so those numbers do not carry over and `crossval_library` would have to be
run before the accuracy is claimed. If the target is A and late-B components, the 12000 K box
covers B8 and later and is the better trade.

An ATLAS9-only box was also checked, since it would remove the seam entirely: Teff 7500 to
12000 at 250 K, log g +2.0 to +5.0, [M/H] -1.0 to +0.5 is 931 nodes with every node published.
It is clean but it needs `_bosz_atmosphere` changed to prefer `ap` in the 7500 to 8000 K
overlap, it leaves a 500 K hole between the FGK ceiling at 7000 K and its own floor at 7500 K,
and it still would not admit gaia-49181812242533888 at 7065 K. Mentioned for completeness, not
recommended.

## 4. Download and cache sizes

**Per node, download.** The 455 nodes of the FGK box are 622.7 MB on this machine, 1.3716 MB
per node, and the 645 MB the registry declares over-states it by 4 percent. Above 7000 K the
shards are larger: 43 HEAD requests spread over Teff 7250 to 16000 K and [M/H] -1.0, 0.0, +0.5
give a mean of **1.475 MB**, running from 1.409 MB at (7250 K, [M/H] -1.0) to 1.539 MB at
(14000 K, [M/H] +0.5). Size grows with both temperature and metallicity. Summing the per-file
sizes of the archive index over a whole box agrees with 1.475 MB per node to about 1 percent,
and reproduces the measured 622.7 MB of the FGK box as 621 MB, so the index sums below are
trustworthy.

**Per node, cache.** The built `.npz` stores two float32 arrays per node over the samples of
the band. The shared R = 20,000 wavelength grid has 300,000 points, uniform in ln lambda; the
4000 to 7000 Angstrom optical slice takes **25,982** of them and the 8350 to 8850 RVS slice
takes **2700**. Measured on the existing builds: **11.09 kB per node** in the RVS band and
**111.7 kB per node** in the optical, a compression of about 1.95 over the raw float32.
Re-compressing the RVS build by temperature gives 12.05, 11.02 and 12.41 kB per node at 4000,
5500 and 7000 K, so the figure is flat in temperature to about 10 percent and extrapolating it
to 10,000 K is safe.

| box | nodes | download | RVS cache | optical cache |
|---|---|---|---|---|
| FGK, as registered | 455 | 621 MB (registry says 645) | 5.05 MB (says 16) | 50.8 MB (says 95) |
| A, 7000-8750 K, log g 3.5-5.0 | 224 | 327 MB | 2.5 MB | 25.0 MB |
| A2, 7000-8750 K, log g 3.0-5.0 | 280 | 409 MB | 3.1 MB | 31.3 MB |
| **B, 7000-10000 K, log g 3.5-5.0** | **364** | **532 MB** | **4.0 MB** | **40.7 MB** |
| B2, 7000-10000 K, log g 3.0-5.0 | 455 | 666 MB | 5.0 MB | 50.8 MB |
| B3, 7000-10000 K, log g 2.5-5.0 | 546 | 798 MB | 6.1 MB | 61.0 MB |
| C, 7000-12000 K, log g 3.5-5.0 | 588 | 863 MB | 6.5 MB | 65.7 MB |
| D, 7000-16000 K at 500 K, log g 3.5-5.0 | 532 | 790 MB | 5.9 MB | 59.4 MB |
| D2, 7000-16000 K at 500 K, log g 3.0-5.0 | 665 | 988 MB | 7.4 MB | 74.3 MB |

Download is once per box and shared between the two bands, because `ingest_bosz` keeps the raw
shards. A user who builds both the RVS and the optical hot library pays 532 MB of network and
45 MB of `.npz`, on top of 532 MB of retained raw shards unless `keep_raw=False`.

The `cache_mb` the registry declares for the two existing entries is two to three times the
built file. That is a separate correction worth making while the hot entries are added, so
that the new numbers are not the only honest ones in the table.

## 5. Physics caveats the docs should state

**The Ca II triplet gives way to the Paschen series.** Measured on BOSZ spectra in the earlier
research note, the line depths of the triplet fall from 0.66 to 0.72 at 5000 K to 0.34 to 0.47
at 10000 K, while Paschen 12, 14 and 16 deepen to 0.33, 0.21 and 0.10. Five Paschen members
lie inside the `RVS_BAND` of albireo, 8460 to 8700 Angstrom, at 8467, 8502, 8545, 8598 and
8665 Angstrom in air, and the series limit at 8204 Angstrom means they crowd toward the blue
edge. P15 at 8545.4 is 3.3 Angstrom from Ca II 8542.09 and P13 at 8665.0 is 2.9 Angstrom from
Ca II 8662.14. At R = 11,500 the resolution element at 8542 Angstrom is 0.74 Angstrom, so at
zero rotation they are separable, but the centroid of a blended pair is not the centroid of
either line. The practical statement for the docs is that above roughly 8000 K a velocity
measured in the RVS band is a hydrogen-line velocity with a calcium blend, and hydrogen lines
are broad, pressure-sensitive and gravity-sensitive, so the velocity and the gravity are
correlated in a way they are not for an FGK star.

**Rotation defeats the resolution.** `RVS_RESOLVING_POWER` is 11,500, which is
299792.458 / 11500 = 26.07 km/s FWHM, or 11.07 km/s of Gaussian sigma. A field A star rotates
at 100 to 200 km/s, so the rotational profile is four to eight times the instrumental width
and the spectrum retains essentially no sharp feature. At vsini 150 km/s the rotational
half-width at 8542 Angstrom is 8542 x 150 / 299792.458 = 4.27 Angstrom, so the profile spans
8.5 Angstrom, and the 3.3 Angstrom Ca II to Paschen separation vanishes inside it. Three
consequences: the two systems in Sect. 2 are not a fair sample of the difficulty, because both
periods are under the 15-day threshold in `_draw_vsini` and their tidally synchronised
rotations are 6.6 to 14.5 km/s; the log-normal around 40 km/s that `_draw_vsini` uses above
the Kraft break for longer periods is an F-star rate, not an A-star rate, so a hot population
drawn from it will be optimistic; and the vsini of the label fit will absorb whatever it can,
which on AI Phe was already shown to be the parameter that soaks up a mis-specified likelihood.

**LTE, and the hydrogen recomputation.** All BOSZ 2024 spectra assume LTE. The paper stops the
grid at 16,000 K explicitly for want of NLTE in ATLAS9 and points to Bohlin et al. (2022)
above that, but it gives no estimate of the LTE error anywhere below 16,000 K, and offers no
discussion of the Ca II triplet at all. For cool stars the 3D non-LTE radial-velocity
correction on the triplet reaches about 0.25 km/s typically and 1 km/s in outliers at RVS
resolution (Lagae, Amarsi and Lind 2025), which is a floor under any LTE template; for A stars
the equivalent number does not exist in the literature the registry cites, so the honest
caveat is that the LTE error above 7000 K is unquantified, not that it is small. The existing
`_BOSZ_RECOMPUTE_NOTE` carries the 2025-09-25 recomputation of the hydrogen lines and the OH+
band strength, and it is more load-bearing for a hot library than for the FGK one: the
recomputation replaced an eight-level hydrogen atom, and an eight-level atom has no
transitions from n >= 9, which is exactly the crowded end of the Paschen series inside the RVS
band. The README scopes the defect to wavelengths above 5.8 microns, which is hard to reconcile
with that, so any hot build should be checked for the high Paschen members before it is
trusted. On this machine the cached FGK shards match the current Content-Length of the archive
byte for byte, so the local cache is the recomputed calculation, and hot shards HEAD as
Last-Modified 2025-04-03 to 2025-05-22 while their directories were rewritten 2025-09-24 to
2025-10-03.

**The ATLAS9 half is unvalidated against data.** The paper reports no observational test of
its hot models: the CALSPEC comparison in its Sect. 4.2 reaches only 9420 K, and the one
quantified change for the hot models is model against model, the 10 to 15 percent flux
difference from BOSZ 2017 near the Paschen jump above 10,000 K. It says nothing about
convection, the mixing length or overshoot anywhere. Against that, the ATLAS9 grid is the
complete half: the 9165 non-converged models the paper counts are all MARCS, and the
measurement in Sect. 1 confirms zero missing ATLAS9 nodes at this composition. Completeness
and reliability are not the same property, and the docs should say which one is established.

**The seam.** The paper measures the two families as differing by about half a percent in flux
between 350 and 1000 nm at 8000 K and solar composition, growing to 5 to 10 percent at 3500
and 4000 K, and it never claims the families are continuous or advises interpolating across
them; the overlap spectra exist to quantify the difference between the two types of model
atmosphere, not as a blending aid. The solar abundance reference also changes at the seam,
Grevesse et al. (2007) for MARCS and Asplund et al. (2005) for ATLAS9. Because a smooth flux
offset divides out of the normalized spectrum and lands almost entirely in `log_continuum`,
the seam should be expected to matter most for the light ratio, which `rvs_light_ratio` takes
from `log_continuum`, and least for the velocities. That expectation is testable without new
physics: the 7500, 7750 and 8000 K overlap publishes both families for the same parameters, so
building `ms`/`mp` and `ap` at those three temperatures and differencing them over 8350 to
8850 Angstrom would measure the seam directly, at a cost of 3 x 4 x 7 = 84 extra shards, about
124 MB. Until that is done the caveat should state the half a percent of the paper and say it
was measured in flux over 350 to 1000 nm, not in the RVS band and not after normalization.
