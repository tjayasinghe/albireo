"""Persisting and exporting fits.

A converged fit is expensive — the HR 6819 runs are hours — and until this module existed
the only thing that left the process was printed text. Three jobs:

**Save and load.** :func:`save_fit` and :func:`load_fit` round-trip the result objects
through ``.npz`` with a JSON header. The format is deliberately dull: every result type is
"flat arrays plus a few scalars", so the reader reconstructs by calling the constructor with
keywords and there is no pickling and no pytree aux-data serialization to keep in step with
the classes. NumPy is the only hard dependency that can write a container — netCDF would
pull in xarray and h5netcdf, and FITS would pull in astropy — so ``.npz`` it is.

**arviz.** :func:`to_inference_data` hands a NUTS run to arviz, which is where the
convergence diagnostics, the plotting, and the on-disk netCDF format the rest of the
Bayesian ecosystem reads all live. albireo does not reimplement any of that.

**Export.** :func:`write_ascii` writes the disentangled spectra as plain text with no
optional dependency at all. :func:`albireo.io.write_spectra` writes FITS and ECSV and needs
astropy. The disentangled spectrum plus its uncertainty band is the product; it has to be
able to leave.

Loaded results are plain data. A :class:`~albireo.likelihood.MarginalResult` read back from
disk is no longer differentiable in the orbital parameters, and — unless it was saved with
``precision=True`` — no longer carries the posterior precision, so it cannot generate new
draws. That is the intended trade: what you save is the answer, not the machinery.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "load_fit",
    "save_fit",
    "to_inference_data",
    "write_ascii",
]

# Bumped only for a change that an older reader could not understand. The reader checks it
# and refuses rather than silently misreading a future file.
_FORMAT_VERSION = 1

_HEADER_KEY = "__albireo__"


def _require_arviz():
    try:
        import arviz
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the bare-install job
        raise ModuleNotFoundError(
            "albireo.results.to_inference_data needs arviz, which is an optional "
            'dependency. Install it with `pip install "albireo[plots]"` or '
            "`pip install arviz`."
        ) from exc
    return arviz


def _as_numpy(value):
    """JAX array, NumPy array, or Python scalar -> NumPy array."""
    return np.asarray(value)


def _flatten_mapping(prefix: str, mapping: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {f"{prefix}/{name}": _as_numpy(value) for name, value in mapping.items()}


def _unflatten_mapping(prefix: str, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    head = f"{prefix}/"
    return {key[len(head) :]: value for key, value in arrays.items() if key.startswith(head)}


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def save_fit(result, path, *, precision: bool = False, compress: bool = True) -> Path:
    """Write a fit result to ``path`` as ``.npz``.

    Parameters
    ----------
    result
        A :class:`~albireo.inference.MAPResult`, :class:`~albireo.scan.K2ScanResult`, or
        :class:`~albireo.likelihood.MarginalResult`.
    precision
        :class:`~albireo.likelihood.MarginalResult` only. If True, store the posterior
        precision blocks so that the loaded result can still draw spectra and compute
        pointwise uncertainties. This is the large object — at survey scale the blocks run
        to gigabytes — so the default stores the posterior mean and, if it can be computed,
        the pointwise standard deviation instead.
    compress
        Use ``np.savez_compressed``. Spectra compress well; turn it off for speed on very
        large arrays.

    Returns
    -------
    pathlib.Path
        The path written, with the ``.npz`` suffix applied if it was missing.

    Notes
    -----
    A :class:`~albireo.scan.K2ScanResult` carries the
    :class:`~albireo.inference.MarginalOrbitModel` it was produced by; that model holds the
    dataset and JAX-traced structure and is *not* saved. Loading gives back a result whose
    ``model`` is None. Rebuild it from the same dataset if you need to continue.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    kind = type(result).__name__
    writer = _SAVERS.get(kind)
    if writer is None:
        raise TypeError(
            f"albireo.results.save_fit does not know how to save {kind!r}. "
            f"Supported: {', '.join(sorted(_SAVERS))}."
        )

    from albireo import __version__

    header: dict[str, Any] = {
        "type": kind,
        "format_version": _FORMAT_VERSION,
        "albireo_version": __version__,
    }
    arrays: dict[str, np.ndarray] = {}
    writer(result, header, arrays, precision=precision)

    arrays[_HEADER_KEY] = np.array(json.dumps(header))
    path.parent.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    # numpy's stubs give these a second positional parameter `allow_pickle: bool`, so a
    # `**kwargs` splat of arrays is not expressible in the signature even though it is
    # exactly the documented calling convention (each keyword names an array).
    save(path, **arrays)  # type: ignore[arg-type]
    return path


def load_fit(path):
    """Read a fit written by :func:`save_fit`.

    Returns an object of the same class it was saved from, with the caveats in the module
    docstring: it is plain data, not a live JAX computation.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if _HEADER_KEY not in data:
            raise ValueError(
                f"{path} is not an albireo fit file (no {_HEADER_KEY!r} header). "
                "It may be a plain .npz written by something else."
            )
        header = json.loads(str(data[_HEADER_KEY]))
        arrays = {key: data[key] for key in data.files if key != _HEADER_KEY}

    version = header.get("format_version")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"{path} is albireo fit-format version {version}, but this albireo reads "
            f"version {_FORMAT_VERSION}. Install the version of albireo that wrote it "
            f"(the file records {header.get('albireo_version')!r})."
        )

    kind = header.get("type")
    reader = _LOADERS.get(kind)
    if reader is None:
        raise ValueError(f"{path} records an unknown result type {kind!r}.")
    return reader(header, arrays)


# -- MAPResult --------------------------------------------------------------


def _save_map(result, header, arrays, **_):
    header["scalars"] = {
        "potential": float(result.potential),
        "grad_norm": float(result.grad_norm),
        "converged": bool(result.converged),
        "num_steps": int(result.num_steps),
    }
    header["params"] = sorted(result.params)
    header["unconstrained"] = sorted(result.unconstrained)
    arrays.update(_flatten_mapping("params", result.params))
    arrays.update(_flatten_mapping("unconstrained", result.unconstrained))


def _load_map(header, arrays):
    from albireo.inference import MAPResult

    return MAPResult(
        params=_unflatten_mapping("params", arrays),
        unconstrained=_unflatten_mapping("unconstrained", arrays),
        **header["scalars"],
    )


# -- K2ScanResult -----------------------------------------------------------

_K2_ARRAYS = (
    "k2_grid",
    "log_likelihood",
    "detection",
    "primary",
    "primary_std",
    "companion",
    "companion_std",
)
# The K_1 quadrature (scan.k2_scan(k1_sigma=...)). Optional on read so that a file
# written before the marginalization existed still loads; always written.
_K2_OPTIONAL_ARRAYS = (
    "k1_grid",
    "k1_log_weights",
    "log_likelihood_grid",
    "log_likelihood_null_grid",
)


def _save_k2(result, header, arrays, **_):
    header["scalars"] = {
        "log_likelihood_null": float(result.log_likelihood_null),
        "k2_peak": float(result.k2_peak),
        "k1_peak": float(result.k1_peak),
    }
    for name in _K2_ARRAYS:
        arrays[name] = _as_numpy(getattr(result, name))
    for name in _K2_OPTIONAL_ARRAYS:
        value = getattr(result, name)
        if value is not None:
            arrays[name] = _as_numpy(value)


def _load_k2(header, arrays):
    from albireo.scan import K2ScanResult

    return K2ScanResult(
        **{name: arrays[name] for name in _K2_ARRAYS},
        **{name: arrays[name] for name in _K2_OPTIONAL_ARRAYS if name in arrays},
        **header["scalars"],
        model=None,
    )


# -- MarginalResult ---------------------------------------------------------


def _save_marginal(result, header, arrays, *, precision: bool = False):
    header["scalars"] = {
        "n_components": int(result.n_components),
        "n_pixels": int(result.n_pixels),
    }
    arrays["log_likelihood"] = _as_numpy(result.log_likelihood)
    arrays["d_hat"] = _as_numpy(result.d_hat)

    if precision:
        arrays["precision_diag"] = _as_numpy(result.precision.diag)
        arrays["precision_lower"] = _as_numpy(result.precision.lower)
        header["precision_n"] = int(result.precision.n)
        header["has_precision"] = True
    else:
        header["has_precision"] = False
        # One Takahashi sweep now saves the user from discovering at read time that the
        # uncertainty band is the one thing they did not keep.
        from albireo.likelihood import spectra_std

        arrays["d_std"] = _as_numpy(spectra_std(result))


def _load_marginal(header, arrays):
    from albireo.likelihood import MarginalResult
    from albireo.solver import BlockTridiagonal

    precision = None
    if header.get("has_precision"):
        precision = BlockTridiagonal(
            diag=arrays["precision_diag"],
            lower=arrays["precision_lower"],
            n=header["precision_n"],
        )
    result = MarginalResult(
        log_likelihood=arrays["log_likelihood"],
        d_hat=arrays["d_hat"],
        precision=precision,
        **header["scalars"],
    )
    if "d_std" in arrays:
        # Attached rather than stored on the class: MarginalResult is a registered pytree
        # and a fifth field would change its flatten signature. `object.__setattr__`
        # because the dataclass is frozen.
        object.__setattr__(result, "d_std", arrays["d_std"])
    return result


_SAVERS = {
    "MAPResult": _save_map,
    "K2ScanResult": _save_k2,
    "MarginalResult": _save_marginal,
}
_LOADERS = {
    "MAPResult": _load_map,
    "K2ScanResult": _load_k2,
    "MarginalResult": _load_marginal,
}


# ---------------------------------------------------------------------------
# arviz
# ---------------------------------------------------------------------------


def to_inference_data(mcmc, *, coords=None, dims=None, component_names=None):
    """Convert a NUTS run to arviz's inference-data container.

    This is the bridge to the rest of the Bayesian Python ecosystem: R-hat and effective
    sample size, trace and pair plots, and ``.to_netcdf()`` for on-disk storage.

    The exact type returned is whatever the installed arviz builds — its own
    ``InferenceData`` up to arviz 0.x, an xarray ``DataTree`` from arviz 1.0 onwards. Both
    expose the ``.posterior`` group and the plotting entry points, so code that reads
    groups rather than checking the class works across the change.

    Parameters
    ----------
    mcmc
        The :class:`numpyro.infer.MCMC` returned by :func:`albireo.inference.run_nuts`.
    coords, dims
        Passed through to arviz. By default the vector-valued sites are labelled by
        component (``k[K_1]``, ``k[K_2]``, ...) rather than by integer index, which is the
        difference between a readable summary table and a cryptic one.
    component_names
        Labels for the stellar components; defaults to ``K_1, K_2, ...`` sized from the
        posterior itself.

    Notes
    -----
    albireo's likelihood enters the numpyro model as a
    :func:`numpyro.factor`, not as an observed site, so arviz cannot extract a pointwise
    log-likelihood group and none is requested. That means the information criteria that
    need pointwise values (LOO, WAIC) are not available from this object — which is
    correct rather than unfortunate, since the component spectra have been marginalized
    out and there is no per-observation factorization left to point at.
    """
    az = _require_arviz()

    samples = mcmc.get_samples()
    if dims is None or coords is None:
        vector_sites = {"k": "component", "k_out": "component_out"}
        auto_coords: dict[str, list[str]] = {}
        auto_dims: dict[str, list[str]] = {}
        for site, dim in vector_sites.items():
            values = samples.get(site)
            if values is None or np.ndim(values) < 2:
                continue
            size = int(np.shape(values)[-1])
            labels = list(component_names) if component_names else []
            if len(labels) != size:
                labels = [f"K_{i + 1}" for i in range(size)]
            auto_coords[dim] = labels
            auto_dims[site] = [dim]
        coords = auto_coords if coords is None else coords
        dims = auto_dims if dims is None else dims

    try:
        return az.from_numpyro(mcmc, coords=coords, dims=dims, log_likelihood=False)
    except Exception:
        # `run_nuts` passes the Problem pytree as a traced model argument, and arviz
        # inspects model args to pick up constant data. If that inspection trips over the
        # pytree, fall back to the posterior itself, which is all we actually need.
        posterior = mcmc.get_samples(group_by_chain=True)
        stats = {
            name: np.asarray(value)
            for name, value in getattr(mcmc, "_states", {}).get("adapt_state", {}).items()
        }
        return az.from_dict(
            posterior={k: np.asarray(v) for k, v in posterior.items()},
            sample_stats=stats or None,
            coords=coords,
            dims=dims,
        )


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def write_ascii(path, grid, d_hat, std=None, *, component: int | None = None, header: str = ""):
    """Write disentangled spectra as plain text. No optional dependency.

    Columns are ``wavelength``, ``flux``, and — when ``std`` is given — ``flux_err``. The
    flux written is ``1 + d``, the normalized component spectrum, not the deviation ``d``:
    that is what an atmosphere-fitting code expects to read.

    Parameters
    ----------
    path
        Output path. With more than one component and ``component=None``, one file per
        component is written with ``_1``, ``_2``, ... inserted before the suffix, and the
        list of paths is returned.
    grid
        The :class:`~albireo.grids.LogGrid` the spectra live on.
    d_hat
        Deviation spectra, shape ``(n_comp, n_pix)`` or ``(n_pix,)``.
    std
        Matching pointwise standard deviations, e.g. from
        :func:`albireo.likelihood.spectra_std`.
    component
        Write only this component (0-based).
    header
        Extra text prepended to the comment header.

    Returns
    -------
    pathlib.Path or list[pathlib.Path]

    Notes
    -----
    The recovered quantity is the *light-weighted* contribution ``l_i * d_i``; the split
    between the light fraction and the deviation depth is set by the light fractions used
    in the fit. If those were assumed rather than inferred, the line depths written here
    inherit that assumption. See ``docs/math.md`` §5.2.
    """
    d_hat = np.atleast_2d(np.asarray(d_hat))
    std_arr = None if std is None else np.atleast_2d(np.asarray(std))
    wave = np.asarray(grid.wave)

    if d_hat.shape[-1] != wave.size:
        raise ValueError(
            f"d_hat has {d_hat.shape[-1]} pixels but the grid has {wave.size}; "
            "they must be the spectra and the grid from the same fit."
        )
    if std_arr is not None and std_arr.shape != d_hat.shape:
        raise ValueError(f"std has shape {std_arr.shape}, expected {d_hat.shape}")

    indices = range(d_hat.shape[0]) if component is None else [component]
    path = Path(path)
    written = []
    multiple = len(list(indices)) > 1
    for i in indices:
        out = path if not multiple else path.with_name(f"{path.stem}_{i + 1}{path.suffix}")
        columns = [wave, 1.0 + d_hat[i]]
        names = "wavelength flux"
        if std_arr is not None:
            columns.append(std_arr[i])
            names += " flux_err"
        text = f"{header}\n" if header else ""
        text += f"component {i + 1} of {d_hat.shape[0]}, normalized flux (1 + d)\n{names}"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, np.column_stack(columns), header=text, fmt="%.10g")
        written.append(out)
    return written[0] if not multiple else written
