# APF exposures

**Experimental.** The names, the defaults and the reader options may change.

This module extracts the exposures of the Automated Planet Finder (Lick Observatory) that the
California Planet Search pipeline delivers: one 79 x 4608 image of counts per exposure,
`r<night>.<number>.fits`, with the wavelength solution in a separate file
(`apf_wave_2022.fits`). [`read_spectrum`][albireo.io.read_spectrum] recognises the exposures
and calls [`extract`][albireo.apf.extract] with the options of its `apf=` argument, of which
the solution is the one that must be given:

```python
import albireo as ab

raw = ab.read_spectrum("rabc.123.fits", apf={"wavelengths": "apf_wave_2022.fits"})
```

Every exposure's zero point and line-spread width are measured from its own telluric lines
([`albireo.telluric`](telluric.md)). The 2022 solution is distorted within each order on
exposures of later years; [`refine_wavelengths`][albireo.apf.refine_wavelengths], or the
command `albireo apf-wavelengths`, refits it to one exposure of a line-rich star, and the
refined file is then passed in its place. The [tutorial](../tutorials/apf-pepsi.md) gives
the whole procedure, from a directory of exposures to orbits.

::: albireo.apf
