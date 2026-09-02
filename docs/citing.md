# Citing albireo

If albireo contributed to a result, please cite it. Software citations are the record
that makes maintenance of research software visible and fundable.

!!! warning "Pre-release"

    albireo has not had a tagged release, so there is no archive DOI and no methods paper
    yet. Until both exist, cite the repository and state the version or commit used. This
    page will be updated with the DOI at the first release and with the paper when it
    appears.

## What to cite today

Cite the repository and state the version (or the exact commit):

```bibtex
@software{albireo,
  author  = {Jayasinghe, Tharindu},
  title   = {albireo: Bayesian spectral disentangling of spectroscopic binaries},
  url     = {https://github.com/tjayasinghe/albireo},
  version = {0.1.0.dev0},
  year    = {2026}
}
```

`albireo.__version__` returns the version string, and
[`CITATION.cff`](https://github.com/tjayasinghe/albireo/blob/main/CITATION.cff) carries the
same metadata in machine-readable form; GitHub renders it as a "Cite this repository"
button, and most reference managers import it directly.

## In an AAS journal

AAS journals collect software credit through the `\software{}` macro, which places the
dependencies in the machine-readable record as well as in the text:

```latex
\software{albireo \citep{albireo},
          JAX \citep{jax2018github},
          NumPyro \citep{phan2019composable},
          NumPy \citep{harris2020array},
          SciPy \citep{virtanen2020scipy}}
```

## Dependencies to cite

albireo is a scientific layer over the following packages. Please cite them alongside it:

| Package | Reference |
|---|---|
| JAX | Bradbury et al. (2018), [github.com/jax-ml/jax](https://github.com/jax-ml/jax) |
| NumPyro | Phan, Pradhan & Jankowiak (2019), arXiv:1912.11554; Bingham et al. (2019), JMLR, 20, 28 |
| NumPy | Harris et al. (2020), Nature, 585, 357 |
| SciPy | Virtanen et al. (2020), Nature Methods, 17, 261 |
| optax | DeepMind et al. (2020), [github.com/google-deepmind/optax](https://github.com/google-deepmind/optax) |
| astropy (with `albireo.io`) | Astropy Collaboration (2013, 2018, 2022) |
| ArviZ (with the results layer) | Kumar et al. (2019), JOSS, 4, 1143 |
| matplotlib (with `albireo.plotting`) | Hunter (2007), Computing in Science & Engineering, 9, 90 |

The [scientific background](science.md#references) page gives the full references with
ADS links.

## Citing the methods

When describing spectral disentangling itself rather than this implementation, the
foundational references are Simon & Sturm (1994) for the wavelength-space formulation
that albireo follows and Hadrava (1995) for the Fourier-domain formulation. The likelihood
albireo evaluates and the analytic marginalization of the component spectra are written
out in [Mathematical foundations](math.md). The epoch-velocity mode implements the
two-dimensional correlation of Zucker & Mazeh (1994) with the maximum-likelihood errors of
Zucker (2003), and the label-fitting mode follows the binary mode of GSSP (Tkachenko 2015)
and reads the BOSZ (Bohlin et al. 2017; Mészáros et al. 2024) and POLLUX (Palacios et al.
2010) grids, each of which carries its own citation requirement recorded in
`albireo.library_info`. See the [scientific background](science.md) for the complete list.
