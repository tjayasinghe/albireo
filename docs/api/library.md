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

The theory is [§9.3](../math.md#93-interpolation-and-why-not-an-emulator-yet); the decision
record is D52.

::: albireo.library
