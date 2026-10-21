# Reading spectra

This is the only module that requires astropy, which is installed with
`pip install -e ".[io]"`. The rest of the package operates on arrays already held in
memory.

## Column identification

Thirteen ESO Phase 3 spectra from seven instruments were read column by column to settle
the identification rule, and no two collections agree on anything except the IVOA utypes.
The flux column is `FLUX` on HARPS, `FLUX_REDUCED` on GIRAFFE, and both on X-shooter and
some UVES products. The extension is `SPECTRUM` except in the Gaia-ESO release, which uses
`phase3spectrum`. Wavelength units are given as `angstrom`, `Angstrom` or `nm`.

The reader therefore dispatches on `TUTYPn`, the data model's statement of each column's
role, and uses column names only as a last resort. This choice has the following
consequences:

- **UCDs cannot identify the flux.** UVES labels its sky-background column
  `phot.flux.density;em.wl;stat.uncalib`, byte-identical to the UCD on the HARPS flux
  column. A UCD-keyed reader would pass the sky to the solver. The utype separates them
  (`BackgroundModel.Value` against `FluxAxis.Value`). No other field does.
- **Calibrated flux takes precedence.** Where two fluxes exist, the calibrated one, marked
  `meta.main`, is used, and its error must come from the same namespace. Pairing
  X-shooter's calibrated flux (erg/cm²/s/Å) with its raw error (adu) is wrong by the whole
  flux calibration, and since both arrays are finite and positive, nothing downstream
  detects it.
- **Air versus vacuum is declared only in the spectral axis's own UCD.** `em.wl;obs.atmos`
  is air and bare `em.wl` is vacuum. No file has an `AIR`/`VACUUM` keyword, and the
  human-readable column comments contradict each other between collections. The value is
  passed to [`EpochData.medium`][albireo.data.EpochData]. The difference between the two
  scales is 83 km/s.
- **Utypes are matched by their suffix after `Data.`.** The namespace prefix is `spec:` in
  one standard version, `Spectrum.` in another and `eso:` on ESO's own reduced columns, and
  ESPRESSO and GIRAFFE misspell `Accuracy` as `Accurancy`. The namespace is not used to
  rank columns: it appears to mark the raw column, but XShootU products put the science
  flux in `eso:` and a derived telluric-corrected column in `spec:`.
- **`SpectralAxis` does not imply wavelength.** The same utype labels frequency and energy
  axes, distinguished only by the UCD. An axis whose UCD is `em.freq` is rejected rather
  than converted.

## Pixel weights and quality flags

A pixel receives zero weight when the file marks it as empty: a nonzero quality flag, a
non-finite flux, or, where a real error array exists, a non-finite, zero or negative
uncertainty. A zero error does not denote infinite precision. It is the convention these
pipelines use for a missing value. Flagged pixels are also excluded from the continuum
fit, since a flagged cosmic ray raises the upper envelope and propagates into every line
depth.

A flag whose convention cannot be determined is ignored rather than guessed. The standard
defines zero as good, but UVES_SQUAD's `STATUS` takes the values `{-5, 1}` and never 0.
Read under the standard convention, it would mark every pixel of all 467 products in that
collection as bad. A quality column in which zero never appears is therefore dropped with a
warning, and columns named only `MASK`/`FLAG` are not read, since those names have no
agreed polarity (albireo's own `EpochData.mask` uses `True` for good, the opposite
convention). A lost mask is recoverable. An inverted mask keeps exactly the pixels the file
rejected.

One case is not covered, because no generic rule can cover it. UVES pads the ends of its
merged spectra with `flux = 0.0, err = 1.0` exactly, which is indistinguishable from a
measurement, and those products have no quality column. The ends must be trimmed by the
caller.

## Image spectra and older headers

A one-dimensional image spectrum takes its wavelengths from `CRVAL1`, `CDELT1` and
`CRPIX1`. Three logarithmic conventions are read:

- IRAF's `DC-FLAG = 1`: the axis is the base-10 logarithm of the wavelength.
- A `CTYPE1` ending in `-LOG` (FITS WCS, Greisen et al. 2006): `CRVAL1` is a wavelength, and
  $`\lambda = \mathrm{CRVAL1}\, \exp[\mathrm{CDELT1}\,(p - \mathrm{CRPIX1}) / \mathrm{CRVAL1}]`$
  at pixel $`p`$.
- `LOG` in `CTYPE1` without a base, as the Mercator HERMES pipeline writes for natural
  logarithms: base 10 where that gives wavelengths between 100 Å and 10⁶ Å, and the natural
  logarithm otherwise, with a warning.

`DC-FLAG = -1`, which records no dispersion solution, raises an error, and so does an axis
that gives no wavelengths between 100 Å and 10⁶ Å in any reading; `wave_scale=` corrects a
wrong unit. Samples without a finite wavelength, such as the padding of a table, are left out
with a warning.

A `DATE-OBS` that holds only a date is combined with the first of `TIME-OBS`, `UTSTART`,
`UT-START` and `UT` that gives the time as `hh:mm:ss`. Where none does, or the time is not
valid, the reader raises an error, and `bjd=` must then give the time. Read as midnight,
the epoch could be up to 24 hours wrong.
The pre-1999 form `DD/MM/YY` is read as 19YY. Site coordinates may be numbers or
sexagesimal strings and are read as east-positive degrees, so a header that writes a
west-positive longitude needs `bjd=` and `v_bary=`. An elevation that cannot be read is taken
as 0 m with a warning, which changes the BJD by less than 20 μs and the barycentric velocity
by less than 0.4 m/s for a site below 5 km. Sexagesimal `RA` and `DEC` strings are read in
hours and degrees.

`read_raw_spectra` and `read_dataset` take the `instrument` and `frame` of `read_kwargs`, the
`[read]` table of a pipeline configuration, as overrides for every file. Given the keyword
arguments of `to_epoch` (`epoch_kwargs=`, which `read_dataset` passes), each spectrum is cut
to the padded region as soon as it is read, so that one full-band file at a time is held in
memory; the epochs are identical bit for bit.

Background and references: [science overview](../science.md).

::: albireo.io
