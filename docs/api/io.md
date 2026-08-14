# Reading spectra

The one module that needs astropy — install it with `pip install "albireo[io]"`. Everything
else in the package works on arrays you already hold in memory.

## How a column is identified

Thirteen real ESO Phase 3 spectra across seven instruments were read column by column to
settle this, and **no two collections agree on anything except the IVOA utypes**. The flux
column is `FLUX` on HARPS, `FLUX_REDUCED` on GIRAFFE, and both at once on X-shooter and
some UVES products. The extension is `SPECTRUM` except in the Gaia-ESO release, which uses
`phase3spectrum`. Wavelengths arrive as `angstrom`, `Angstrom` or `nm`.

So the reader dispatches on `TUTYPn` — the data model's name for what a column *is* — and
uses names only as a last resort. The distinction is not cosmetic:

- **UCDs cannot key the flux.** UVES labels its sky-background column
  `phot.flux.density;em.wl;stat.uncalib`, which is byte-identical to the UCD on the HARPS
  *flux* column. A UCD-keyed reader hands the solver the sky. The utype separates them
  (`BackgroundModel.Value` against `FluxAxis.Value`); nothing else does.
- **Where two fluxes exist, the calibrated one wins**, marked `meta.main`, and its error
  must come from the same namespace. Pairing X-shooter's calibrated flux (erg/cm²/s/Å) with
  its raw error (adu) is wrong by the whole flux calibration, and both arrays are finite and
  positive, so nothing downstream complains.
- **Air vs vacuum is declared only in the spectral axis's own UCD** — `em.wl;obs.atmos` is
  air, bare `em.wl` is vacuum. No file carries an `AIR`/`VACUUM` keyword, and the
  human-readable column comments contradict each other between collections. It reaches
  [`EpochData.medium`][albireo.data.EpochData], which is worth 83 km/s.
- Utypes are matched by their suffix after `Data.`, because the namespace prefix is `spec:`
  in one standard version, `Spectrum.` in another and `eso:` on ESO's own reduced columns —
  and because ESPRESSO and GIRAFFE misspell `Accuracy` as `Accurancy`. The namespace itself
  is *not* used to rank: it looks like it marks the raw column, but XShootU products put the
  science flux in `eso:` and a derived telluric-corrected column in `spec:`.
- `SpectralAxis` does not mean wavelength — the same utype carries frequency and energy
  axes, separated only by the UCD. One that says `em.freq` is refused, not converted.

## What counts as a measurement

A pixel gets zero weight when the file says it holds nothing: a nonzero quality flag, a
non-finite flux, or — where there is a real error array — a non-finite, zero or negative
uncertainty. A zero error is not infinite precision; it is how these pipelines write
"nothing here". Flagged pixels are also kept out of the continuum fit, since a flagged
cosmic pulls the upper envelope up and propagates into every line depth.

**A flag whose convention cannot be read is ignored rather than guessed at.** The standard
says zero is good, but UVES_SQUAD's `STATUS` runs `{-5, 1}` and never takes the value 0 —
taken at face value it condemns every pixel of all 467 products in that collection. A
quality column in which zero never appears is therefore dropped with a warning, and columns
named only `MASK`/`FLAG` are not read at all, since those names carry no agreed polarity
(albireo's own `EpochData.mask` uses `True` = *good*, the opposite convention). Losing a
mask is recoverable; inverting one keeps exactly the pixels the file rejected.

One case is deliberately not covered, because no generic rule can be: UVES pads the ends of
its merged spectra with `flux = 0.0, err = 1.0` exactly, which nothing distinguishes from a
genuine measurement, and those products carry no quality column. Trim the ends.

::: albireo.io
