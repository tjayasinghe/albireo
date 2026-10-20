# Telluric lines

**Experimental.** The names and the packaged line list may change.

Telluric lines are at rest in the topocentric frame, so their apparent positions in a
spectrum measure the error of its wavelength scale, and their widths measure the
line-spread function. This module provides a list of 183 O2 and H2O lines between 6276 and
7410 Å (air), a model spectrum built from it, and the two measurements. The APF reader
([`albireo.apf`](apf.md)) uses them for the zero point and the line-spread width of every
exposure. They apply to any continuum-normalised spectrum in the topocentric frame that
covers the O2 gamma or B band or the H2O bands near 7200 Å.

The shift returned by [`telluric_shift`][albireo.telluric.telluric_shift] is the correction:
the wavelengths multiplied by `1 + v / c` put the telluric lines at their rest positions, so
lines that appear displaced to the red by 2 km/s give `v = -2` km/s.

::: albireo.telluric
