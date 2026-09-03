# Results and plotting

Saving a fit, converting it for arviz, exporting the disentangled spectra, and producing
the figures.

`albireo.plotting` requires matplotlib and `plot_corner` requires arviz; both are installed
by `pip install -e ".[plots]"`. Writing FITS or ECSV requires astropy (`albireo[io]`);
`write_ascii` requires only NumPy.

Background and references: [science overview](../science.md).

::: albireo.results

::: albireo.plotting
