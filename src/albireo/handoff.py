"""Writers for the atmosphere codes that consume disentangled spectra.

albireo is the front half of a pipeline. The back half — GSSP, iSpec, Korg.jl, PySME —
turns a component spectrum into effective temperature, surface gravity and abundances, and
albireo has no business reimplementing any of it. What it owes those codes is a file they
read without the user hand-editing anything.

**The formats are not interchangeable, and the differences are the kind that fail
silently.** Both are recorded here against their primary sources rather than remembered:

* **GSSP** (Tkachenko 2015, A&A 581, A129, Appendix B — which *is* the manual; there is no
  separate document and no source repository) takes "a two-column ASCII file, where the
  first and the second columns refer to wavelength (in [Angstrom], linear scale) and
  normalized flux". Two columns. There is no error column, no S/N entry, and no weighting
  entry anywhere in its configuration files. And it requires an **equidistant** wavelength
  scale, because "the step width in wavelength that will be used for the calculation of
  synthetic spectra is computed from the observations" — so a log-wavelength grid dumped
  as-is does not merely look odd, it sets GSSP's synthetic step from the first pixel pair.
* **iSpec** (Blanco-Cuaresma et al. 2014) takes tab-separated text with one header line and
  exactly three columns, ``waveobs``/``flux``/``err``, wavelength in **nanometres**, and the
  error as an absolute 1-sigma in the same units as the flux. The reader drops line 1
  positionally and fixes the column *order*, so the names are cosmetic and the order is not.

That GSSP has nowhere to put a per-pixel uncertainty is the single most consequential fact
in this module, and it is why :func:`export_draws` exists. The disentangling posterior
cannot reach an atmospheric parameter through a file format that has no room for it; it can
only reach it by fitting many spectra. See :func:`export_draws` for what that does and does
not buy.

Neither writer converts between air and vacuum. iSpec ships ``air_to_vacuum`` /
``vacuum_to_air`` as explicit user steps and does no conversion on read; albireo does the
same here, for the reason recorded in D43 — the offset is a nearly constant 83 km/s, the
same order as the orbits being measured, so guessing it is worse than declining to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "export_draws",
    "write_gssp",
    "write_ispec",
]

# iSpec discards any pixel whose error is <= 0 ("N fluxes have been discarded because their
# ERRORS are negative or zero") rather than down-weighting it, and deactivates error use
# entirely if all of them are. A disentangled component's posterior sd legitimately
# approaches zero where the prior dominates or the component carries no light, so an
# unfloored export quietly deletes exactly the pixels the band was describing.
_ISPEC_ERR_FLOOR = 1e-8


def _components(d_hat, std, component):
    """Normalize (d_hat, std) to 2-D and pick the components to write."""
    d_hat = np.atleast_2d(np.asarray(d_hat, dtype=float))
    if std is None:
        std_arr = None
    else:
        std_arr = np.atleast_2d(np.asarray(std, dtype=float))
        if std_arr.shape != d_hat.shape:
            raise ValueError(f"std has shape {std_arr.shape}, expected {d_hat.shape}")
    indices = range(d_hat.shape[0]) if component is None else [component]
    return d_hat, std_arr, list(indices)


def _paths(path, indices, n_comp):
    """One path per component, with ``_1``, ``_2`` … inserted when there is more than one."""
    path = Path(path)
    if len(indices) == 1 and n_comp == 1:
        return [path]
    return [path.with_name(f"{path.stem}_{i + 1}{path.suffix}") for i in indices]


def _linear_grid(wave, step_angstrom):
    """The equidistant grid GSSP requires, and the step it will infer from it."""
    lo, hi = float(wave[0]), float(wave[-1])
    if step_angstrom is None:
        # The median spacing of the source grid: finer than that oversamples a spectrum
        # that has no information at the added scales, coarser throws resolution away.
        step_angstrom = float(np.median(np.diff(wave)))
    if not step_angstrom > 0:
        raise ValueError(f"step_angstrom must be positive, got {step_angstrom}")
    n = int(np.floor((hi - lo) / step_angstrom)) + 1
    if n < 2:
        raise ValueError(
            f"a step of {step_angstrom} A leaves {n} pixel(s) across "
            f"[{lo:.3f}, {hi:.3f}] A; the grid would not be a spectrum"
        )
    return lo + step_angstrom * np.arange(n)


def write_gssp(path, grid, d_hat, *, component=None, step_angstrom=None, dilute=False):
    """Write component spectra as the two-column ASCII GSSP reads.

    Parameters
    ----------
    path
        Output path. With more than one component and ``component=None``, one file per
        component is written with ``_1``, ``_2``, … inserted before the suffix.
    grid
        The :class:`~albireo.grids.LogGrid` the spectra were solved on.
    d_hat
        Deviation spectra, shape ``(n_comp, n_pix)`` or ``(n_pix,)``. Written as flux
        ``1 + d``, the normalized component spectrum.
    component
        Write only this component (0-based).
    step_angstrom
        Wavelength step of the written grid. Defaults to the median spacing of ``grid``.
        **The written grid is always equidistant**, whatever ``grid`` is: GSSP infers the
        step of its synthetic spectra from the observation, so a log-wavelength grid must
        be resampled rather than dumped.
    dilute
        Kept for the caller to state intent explicitly; ``True`` is a no-op that documents
        that the spectrum being written is still light-diluted, which is what GSSP's
        ``dilution_flag adjust`` mode expects for a disentangled component. albireo never
        undilutes on its own — the light fraction is an assumption of the fit, not a
        measurement (``docs/math.md`` §5.2) — so this flag changes no numbers. It exists
        because writing the wrong one into a GSSP configuration is a real error and the
        export is where a user thinks about it.

    Returns
    -------
    pathlib.Path or list[pathlib.Path]

    Notes
    -----
    Resampling is linear interpolation onto the equidistant grid. That correlates
    neighbouring pixels, which is exactly why albireo refuses to do it to *observations*
    (D4) — but this is an export of a model quantity to a code that requires the spacing,
    and the alternative is a file GSSP mis-steps. Applied identically to every draw in
    :func:`export_draws`, so draws stay comparable to each other.

    GSSP has **no error column** (Tkachenko 2015, Appendix B.2): its configuration files
    contain no error path, no S/N and no weighting, and its own quoted uncertainties come
    from chi-square on the fit residuals. The posterior band therefore cannot be handed to
    it directly. :func:`export_draws` is the route.
    """
    d_hat, _, indices = _components(d_hat, None, component)
    wave = np.asarray(grid.wave, dtype=float)
    if d_hat.shape[-1] != wave.size:
        raise ValueError(
            f"d_hat has {d_hat.shape[-1]} pixels but the grid has {wave.size}; "
            "they must be the spectra and the grid from the same fit."
        )
    out_wave = _linear_grid(wave, step_angstrom)
    written = []
    for target, i in zip(_paths(path, indices, d_hat.shape[0]), indices, strict=True):
        flux = np.interp(out_wave, wave, 1.0 + d_hat[i])
        # No header: GSSP's reader takes two numeric columns and nothing else.
        np.savetxt(target, np.column_stack([out_wave, flux]), fmt="%.8f %.8f")
        written.append(target)
    return written[0] if len(written) == 1 else written


def write_ispec(path, grid, d_hat, std=None, *, component=None, err_floor=_ISPEC_ERR_FLOOR):
    """Write component spectra as the tab-separated text iSpec reads.

    Parameters
    ----------
    path
        Output path, as in :func:`write_gssp`.
    grid
        The :class:`~albireo.grids.LogGrid` the spectra were solved on.
    d_hat
        Deviation spectra. Written as flux ``1 + d``.
    std
        Pointwise posterior standard deviations with the same shape, e.g. from
        :func:`albireo.likelihood.spectra_std`. Written to the ``err`` column as an
        absolute 1-sigma in flux units, which is what iSpec means by it. When omitted the
        column is written at ``err_floor`` — the column is not optional, and a two-column
        file falls into an undocumented legacy parser rather than failing.
    component
        Write only this component (0-based).
    err_floor
        Smallest error written. iSpec *discards* pixels with ``err <= 0`` instead of
        down-weighting them, so a posterior sd that has gone to zero would silently remove
        those pixels from the fit.

    Returns
    -------
    pathlib.Path or list[pathlib.Path]

    Notes
    -----
    **Wavelengths are written in nanometres**, because iSpec's plain-text path performs no
    unit conversion and its whole internal scale — including the atomic line lists — is nm.
    An Angstrom value written here lands a factor of ten outside every model grid, which is
    the most likely silent failure of this module and is regression-tested.

    The grid is written as-is: iSpec imposes no equidistance requirement, and resampling it
    would correlate the noise for nothing.

    iSpec *does* use the error column in its reported parameter uncertainties, but it
    weights by ``sqrt(1/err)`` rather than ``1/err**2`` — a deliberate hand-calibration in
    its own source. The returned ``errors['teff']`` is therefore not a Gaussian propagation
    of this band and does not scale linearly with it. It is the within-fit error of one
    spectrum. Do not add it in quadrature to the spread from :func:`export_draws` without
    reading that function's notes: the two overlap.
    """
    d_hat, std_arr, indices = _components(d_hat, std, component)
    wave_nm = np.asarray(grid.wave, dtype=float) / 10.0
    if d_hat.shape[-1] != wave_nm.size:
        raise ValueError(
            f"d_hat has {d_hat.shape[-1]} pixels but the grid has {wave_nm.size}; "
            "they must be the spectra and the grid from the same fit."
        )
    written = []
    for target, i in zip(_paths(path, indices, d_hat.shape[0]), indices, strict=True):
        flux = 1.0 + d_hat[i]
        if std_arr is None:
            err = np.full_like(flux, err_floor)
        else:
            err = np.maximum(std_arr[i], err_floor)
        rows = "\n".join(
            f"{w:.10f}\t{f:.10f}\t{e:.10f}" for w, f, e in zip(wave_nm, flux, err, strict=True)
        )
        # One header line, and no trailing newline: iSpec drops line 1 positionally, and a
        # final empty line splits to a 1-tuple, which breaks the primary parser and drops
        # the file into a legacy fallback that reads it wrongly rather than refusing it.
        Path(target).write_text(f"waveobs\tflux\terr\n{rows}", encoding="utf-8")
        written.append(Path(target))
    return written[0] if len(written) == 1 else written


def export_draws(directory, grid, draws, *, format="gssp", prefix="draw", **kwargs):
    """Write ``N`` posterior draws as ``N`` fittable spectra, one set per draw.

    This is how a disentangling uncertainty reaches an effective temperature. Fit all ``N``
    exported spectra with the same atmosphere code, the same grid and the same settings,
    and the **spread** of the resulting parameters is the contribution of the disentangling
    posterior — the term the literature currently drops (Mahy et al. 2020, §3.1: the
    uncertainties from the normalization procedure "are not taken into account in the global
    uncertainties on the presented properties"; Pavlovski, Southworth & Tamajo 2018:
    "propagation of uncertainties through this process is difficult so must be tackled
    numerically").

    Parameters
    ----------
    directory
        Output directory, created if absent.
    grid
        The :class:`~albireo.grids.LogGrid` the draws live on.
    draws
        Shape ``(n_draws, n_comp, n_pix)``, from
        :func:`albireo.likelihood.draw_spectra`.
    format
        ``"gssp"`` or ``"ispec"``.
    prefix
        Filename stem. Files are ``{prefix}_{draw:04d}_{component}.{ext}``.
    **kwargs
        Passed to the underlying writer.

    Returns
    -------
    list[list[pathlib.Path]]
        Outer index is the draw, inner index the component.

    Notes
    -----
    **The draw index is the point, and it is preserved in the filename.** The draws from
    :func:`~albireo.likelihood.draw_spectra` are ``d_hat + L^-T z`` on the *stacked* vector
    over all components, so draw *i* of component A and draw *i* of component B come from
    one sample of the joint posterior. Fitting them as a pair and plotting *T*\\ :sub:`eff,A`
    against *T*\\ :sub:`eff,B` per draw shows the correlation between the two stars; pooling
    the draws per component throws it away.

    That jointness is also what separates this from the established practice it resembles.
    Kiran et al. (2016, §3.5) added "artificial Gaussian noise with sigma = sigma_c" to a
    disentangled profile, refitted 500 times, and took the scatter — the same loop, but with
    draws that assume the error is independent from pixel to pixel. Disentangling error is
    not: it has a genuine low-frequency null space (Pavlovski & Hensberge 2011), which is
    the part that moves a continuum and therefore a temperature. The loop is old; the draws
    are new. Cite them.

    **What the spread does not contain**, and what a tutorial using it must say:

    * The atmosphere code's own model error — grid coarseness, LTE assumptions, line-list
      quality. That is outside albireo's posterior entirely and is unaffected by the draws.
    * Anything albireo conditions on rather than marginalizes. The **light fractions** are
      assumed, not inferred, and the marginal likelihood is flat in them under constant
      light (see ``scripts/m5_light_ratio_demo.py``), which is precisely the systematic
      Pavlovski & Hensberge identify as dominant. The draw spread is silent about it.
    * Double counting is easy here: iSpec's own ``errors['teff']`` is a within-draw fit
      error computed from the ``err`` column, so adding it in quadrature to a spread that
      already came from the same posterior counts part of it twice.

    ``N = 100`` is a defensible production number: the relative standard error of a sample
    standard deviation is ``1/sqrt(2(N-1))``, i.e. 7% at 100 and 12.7% at 32. Below ~32 the
    spread is too noisy to quote. Check also that the atmosphere grid step is smaller than
    the spread being measured — if every draw lands in one grid cell the answer is zero for
    a reason that has nothing to do with the data.
    """
    writers = {"gssp": write_gssp, "ispec": write_ispec}
    if format not in writers:
        raise ValueError(f"format must be one of {sorted(writers)}, got {format!r}")
    writer = writers[format]
    suffix = ".dat" if format == "gssp" else ".txt"

    draws = np.asarray(draws)
    if draws.ndim != 3:
        raise ValueError(
            f"draws must have shape (n_draws, n_comp, n_pix), got {draws.shape}; "
            "pass the array from albireo.likelihood.draw_spectra unchanged"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    out: list[list[Path]] = []
    for k, draw in enumerate(draws):
        per_draw = []
        for i in range(draw.shape[0]):
            target = directory / f"{prefix}_{k:04d}_{i + 1}{suffix}"
            per_draw.append(Path(writer(target, grid, draw[i], component=None, **kwargs)))
        out.append(per_draw)
    return out
