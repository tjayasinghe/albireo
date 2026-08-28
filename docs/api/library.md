# Synthetic spectral libraries

A published grid of synthetic spectra — BOSZ, POLLUX, PHOENIX — reduced to the four things a
label fit needs: the node labels, the normalized flux at each node, the continuum at each
node, and the wavelength scale those live on. albireo reads grids other people computed and
cites them; there is no line list here and no radiative transfer.

Two things about this module are stricter than they look, and both are there because the
alternative fails silently.

**The wavelength medium is a required field with no default.** Air and vacuum differ by about
83 km/s across the optical — the same order as the orbital semi-amplitudes albireo exists to
measure — so a library on the wrong scale does not produce a slightly worse fit, it produces a
confident wrong answer. Nor is the upstream documentation a reliable source: BOSZ 2017 was
vacuum throughout and BOSZ 2024 is air above 200 nm, *under the same name*. `line_core_medium`
therefore measures the convention from the spectra themselves, and the ingest paths use it to
verify a declaration rather than to supply one.

**Interpolation is in flux, never in model atmospheres.** On the 250 K / 0.5 dex spacing BOSZ
actually uses, Mészáros & Allende Prieto (2013) measured 0.19% scatter interpolating
atmospheres against 0.051% interpolating fluxes linearly and 0.031% with a cubic — while a
Payne-style neural emulator reaches about 0.1%. On a well-sampled grid the differentiable
cubic here is therefore not a compromise but the more accurate option, at no training cost and
with no weights to host. Whether that holds for a *particular* grid is an empirical question,
and `crossval_library` is the measurement that answers it rather than an argument about it.

`library_interpolator` picks its method from the grid's geometry: a separable Catmull-Rom cubic
on a complete axis product, and barycentric interpolation over a Delaunay triangulation when
physics has cut the corners off the grid, as it has for every public OB library. Both reproduce
a node exactly — bit-for-bit, not to a tolerance — which is what lets the warm-start node scan
in [`albireo.match`](match.md) and the continuous fit be compared on the same footing.

## Getting a grid

`fetch_library` builds a named library on first use and caches it, so the download is paid
once per machine:

```python
import albireo as ab

ab.library_names()
# ['bosz2024-fgk-r20000', 'bosz2024-fgk-rvs', 'pollux-ob-smc24']

ab.library_info("bosz2024-fgk-r20000")["licence"]   # look before you download
library = ab.fetch_library("bosz2024-fgk-r20000")   # ~645 MB once, ~95 MB cached
```

The cache lives under `albireo.examples.cache_dir()`, and `$ALBIREO_DATA_DIR` redirects it —
which is also how a shared or pre-populated directory is pointed at on a cluster. Narrowing
the band with `wave_range=` is free; widening it is refused, because the band is what was
downloaded and quietly returning less than was asked for is worse than an error.

| name | grid | coverage | band | nodes |
|---|---|---|---|---|
| `bosz2024-fgk-r20000` | BOSZ 2024, MARCS, R = 20,000 | Teff 4000–7000 K (250 K), log g 3–5 (0.5), [M/H] −1→+0.5 | 4000–7000 Å | 455 |
| `bosz2024-fgk-rvs` | the same nodes | as above | 8350–8850 Å, the Gaia RVS window | 455 |
| `pollux-ob-smc24` | POLLUX, CMFGEN, non-LTE | Teff 23–55 kK, log g 2.5–4.5, [M/H] = −0.73 fixed | 3850–4650 Å | 915 |

Both upstream grids are CC BY 4.0, and `library_info` carries the citation each one
obliges you to give.

**BOSZ builds automatically; POLLUX does not.** BOSZ's URLs on MAST are deterministic, so
`ingest_bosz` constructs them, downloads the shards in parallel, keeps the raw files so
re-cutting another band costs nothing, and slices to the registered band. POLLUX serves its
collections through a form that posts to `/download/`, so there is no stable URL to fetch:
`ingest_pollux` says so and stops, rather than shipping a parser written against a file format
nobody has looked at.

**What is guaranteed about the bytes.** A cached build is verified on every load against a
digest taken over the *arrays*, so it is reproducible across machines and independent of how
the `.npz` happened to compress — two people can compare `meta["content_sha256"]` to confirm
they built the same library. The assembled spectra have their medium measured and checked
against the registry's declaration, and a disagreement raises rather than being reconciled.
What is not yet in place is a registry-level pin published under a DOI. Note also that BOSZ
was silently recomputed on 2025-09-25 to fix its hydrogen lines and OH⁺ strength, so the build
date is recorded in `meta["retrieved"]` and anything older is a different calculation.

The theory is [§9.3](../math.md#93-interpolation-and-why-not-an-emulator-yet); the decision
record is D52.

::: albireo.library
