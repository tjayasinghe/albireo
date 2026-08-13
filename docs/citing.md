# Citing albireo

If albireo contributed to a result, please cite it. Software citations are what make
maintenance fundable and are the only signal that a package is worth keeping alive.

!!! warning "Pre-release"

    albireo has not had its first tagged release yet, so there is no archive DOI and no
    methods paper to cite. Until both exist, cite the repository. This page will be updated
    with the DOI at the first release and with the paper when it appears; the version you
    used should always be recorded either way.

## What to cite today

Cite the repository and state the version (or, better, the exact commit) you ran:

```bibtex
@software{albireo,
  author  = {Jayasinghe, Tharindu},
  title   = {albireo: Bayesian spectral disentangling of spectroscopic binaries},
  url     = {https://github.com/tjayasinghe/albireo},
  version = {0.1.0.dev0},
  year    = {2026}
}
```

`albireo.__version__` gives you the version string, and
[`CITATION.cff`](https://github.com/tjayasinghe/albireo/blob/main/CITATION.cff) carries the
same metadata in a machine-readable form — GitHub renders it as a "Cite this repository"
button, and most reference managers can import it directly.

## In an AAS journal

AAS journals collect software credit through the `\software{}` macro, which is what puts
your dependencies into the machine-readable record rather than only into your prose:

```latex
\software{albireo \citep{albireo},
          JAX \citep{jax2018github},
          NumPyro \citep{phan2019composable},
          NumPy \citep{harris2020array},
          SciPy \citep{virtanen2020scipy}}
```

## Please also cite what albireo is built on

albireo is a thin scientific layer over other people's infrastructure, and that
infrastructure is chronically under-credited. If you cite albireo, cite these too:

| Package | Cite |
|---|---|
| **JAX** | Bradbury et al. (2018), [github.com/jax-ml/jax](https://github.com/jax-ml/jax) |
| **NumPyro** | Phan, Pradhan & Jankowiak (2019), *arXiv:1912.11554*; Bingham et al. (2019), *JMLR* 20(28) |
| **NumPy** | Harris et al. (2020), *Nature* 585, 357 |
| **SciPy** | Virtanen et al. (2020), *Nature Methods* 17, 261 |
| **optax** | DeepMind et al. (2020), [github.com/google-deepmind/optax](https://github.com/google-deepmind/optax) |
| **astropy** (if you used `albireo.io`) | Astropy Collaboration (2013, 2018, 2022) |
| **arviz** (if you used the results layer) | Kumar et al. (2019), *JOSS* 4(33), 1143 |

## Citing the method rather than the code

If you are describing spectral disentangling itself rather than this implementation, the
foundational references are Simon & Sturm (1994) for the wavelength-space linear-inversion
formulation that albireo follows, and Hadrava (1995) for the Fourier-domain formulation. The
specific likelihood albireo evaluates, and the analytic marginalization of the component
spectra, are written out in [Mathematical foundations](math.md), which also records where
the treatment departs from both.
