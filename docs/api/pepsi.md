# PEPSI products

**Experimental.** The names and the default resolving power may change.

This module reads the products of the PEPSI Spectroscopic Data Systems pipeline (Large
Binocular Telescope). [`read_spectrum`][albireo.io.read_spectrum] and
[`read_dataset`][albireo.io.read_dataset] recognise them by `INSTRUME = 'PEPSI'` and call
[`read_pepsi`][albireo.pepsi.read_pepsi], so a directory of PEPSI products is read like any
other set of spectra. The product files are named after their kind
(`pepsib.20240101.001.dxt.bwl`) rather than `.fits`; a directory passed to `read_dataset`
contributes them together with its FITS files.

A PEPSI directory usually holds both arms of every exposure. With a `region`, the files that
do not reach it are left out by `min_coverage` (`read_dataset(..., min_coverage=0.5)`), and
the pipeline does the same for the files that do not reach the declared region at all. The
[tutorial](../tutorials/apf-pepsi.md) gives a complete example.

::: albireo.pepsi
