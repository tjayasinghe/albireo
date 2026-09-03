"""Writers for the atmosphere codes that consume disentangled spectra.

**Experimental.** The writers follow the file formats the atmosphere codes accept,
and change when those do.

albireo produces component spectra; codes such as GSSP, iSpec, Korg.jl and PySME turn a
component spectrum into effective temperature, surface gravity and abundances. This module
writes the input files those codes read, without further hand-editing. The two formats are
not interchangeable and their differences produce no error message, so both are recorded here
against their primary sources.

GSSP (Tkachenko 2015, Appendix B, which serves as the manual: there is no separate document
and no source repository) takes a two-column ASCII file whose first and second columns are
the wavelength (in Angstrom, on a linear scale) and the normalized flux. There is no error
column, no S/N entry and no weighting entry anywhere in its configuration files. The
wavelength scale must be equidistant, because the step width used for the calculation of the
synthetic spectra is computed from the observation: a log-wavelength grid written as-is sets
GSSP's synthetic step from the first pixel pair.

iSpec (Blanco-Cuaresma et al. 2014) takes tab-separated text with one header line and exactly
three columns, ``waveobs``, ``flux`` and ``err``, with the wavelength in nanometres and the
error as an absolute 1-sigma in the same units as the flux. The reader drops line 1
positionally and fixes the column order, so the column names are cosmetic and the order is
not.

GSSP accepts no per-pixel uncertainty, so the disentangling posterior can reach an
atmospheric parameter only through repeated fits of posterior draws; see
:func:`export_draws`.

Neither writer converts between air and vacuum. iSpec provides ``air_to_vacuum`` and
``vacuum_to_air`` as explicit user steps and does no conversion on read; albireo does the same
here, because the offset is a nearly constant 83 km/s, the same order as the orbits
being measured.

References
----------
Tkachenko, A. 2015, A&A, 581, A129
Blanco-Cuaresma, S. et al. 2014, A&A, 569, A111
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "export_draws",
    "write_gssp",
    "write_ispec",
]

# iSpec discards any pixel whose error is <= 0 rather than down-weighting it, and deactivates
# the use of errors entirely if every error is non-positive. A disentangled component's
# posterior standard deviation approaches zero where the prior dominates or the component
# carries no light, so an unfloored export would remove exactly the pixels the band describes.
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
    """The equidistant grid GSSP requires, and the step it infers from it."""
    lo, hi = float(wave[0]), float(wave[-1])
    if step_angstrom is None:
        # The median spacing of the source grid: a finer step oversamples a spectrum that
        # carries no information at the added scales, a coarser one discards resolution.
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
        Wavelength step of the written grid. Defaults to the median spacing of ``grid``. The
        written grid is always equidistant, whatever ``grid`` is: GSSP infers the step of its
        synthetic spectra from the observation, so a log-wavelength grid must be resampled
        rather than written unchanged.
    dilute
        Records the caller's intent; ``True`` is a no-op stating that the spectrum written is
        still light-diluted, which is what GSSP's ``dilution_flag adjust`` mode expects for a
        disentangled component. albireo never removes the dilution, since the light fraction
        is an assumption of the fit rather than a measurement (``docs/math.md`` §5.2), so this
        flag changes no numbers. It exists because the corresponding GSSP configuration
        setting must be chosen at export time.

    Returns
    -------
    pathlib.Path or list[pathlib.Path]

    Notes
    -----
    Resampling is linear interpolation onto the equidistant grid. It correlates neighbouring
    pixels, which is why albireo does not resample observations; here it is applied to a
    model quantity for a code that requires the spacing, and an unresampled file would set
    GSSP's synthetic step incorrectly. The same interpolation is applied to every draw in
    :func:`export_draws`, so draws remain comparable to each other.

    GSSP has no error column (Tkachenko 2015, Appendix B.2): its configuration files contain
    no error path, no S/N and no weighting, and its quoted uncertainties come from chi-square
    on the fit residuals. The posterior band therefore cannot be passed to it directly;
    :func:`export_draws` is the route.

    References
    ----------
    Tkachenko, A. 2015, A&A, 581, A129
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
        column is written at ``err_floor``: the column is not optional, and a two-column file
        is read by an undocumented legacy parser instead of being rejected.
    component
        Write only this component (0-based).
    err_floor
        Smallest error written. iSpec discards pixels with ``err <= 0`` instead of
        down-weighting them, so a posterior standard deviation that has reached zero would
        remove those pixels from the fit without any message.

    Returns
    -------
    pathlib.Path or list[pathlib.Path]

    Notes
    -----
    Wavelengths are written in nanometres. iSpec's plain-text path performs no unit
    conversion and its internal scale, including the atomic line lists, is nm. An Angstrom
    value written here would land a factor of ten outside every model grid; the unit is
    regression-tested.

    The grid is written as-is: iSpec imposes no equidistance requirement, and resampling it
    would correlate the noise to no purpose.

    iSpec does use the error column in its reported parameter uncertainties, but it weights
    by ``sqrt(1/err)`` rather than ``1/err**2``, a hand-calibration in its own source. The
    returned ``errors['teff']`` is therefore not a Gaussian propagation of this band and does
    not scale linearly with it; it is the within-fit error of one spectrum. It should not be
    added in quadrature to the spread from :func:`export_draws`, whose notes describe the
    overlap between the two.

    References
    ----------
    Blanco-Cuaresma, S. et al. 2014, A&A, 569, A111
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

    This is the route by which a disentangling uncertainty reaches an effective temperature.
    All ``N`` exported spectra are fitted with the same atmosphere code, the same grid and the
    same settings, and the spread of the resulting parameters is the contribution of the
    disentangling posterior. That term is usually omitted: Mahy et al. (2020, §3.1) state that
    the uncertainties from the normalization procedure are not included in the properties they
    present, and Pavlovski, Southworth & Tamajo (2018) note that propagating uncertainties
    through disentangling is difficult and must be tackled numerically.

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
    The draw index is preserved in the filename. The draws from
    :func:`~albireo.likelihood.draw_spectra` are ``d_hat + L^-T z`` on the stacked vector over
    all components, so draw *i* of component A and draw *i* of component B come from one
    sample of the joint posterior. Fitting them as a pair and plotting *T*\\ :sub:`eff,A`
    against *T*\\ :sub:`eff,B` per draw shows the correlation between the two stars; pooling
    the draws per component discards it.

    That jointness distinguishes this procedure from the established practice it resembles.
    Kiran et al. (2016, §3.5) added artificial Gaussian noise of sigma = sigma_c to a
    disentangled profile, refitted it 500 times and took the scatter. Those draws assume the
    error is independent from pixel to pixel. Disentangling error is not: it has a
    low-frequency null space (Pavlovski & Hensberge 2010), which is the part that moves a
    continuum and therefore a temperature.

    The spread does not contain the following, which a report using it should state:

    * The atmosphere code's own model error: grid coarseness, LTE assumptions, line-list
      quality. That lies outside albireo's posterior and is unaffected by the draws.
    * Anything albireo conditions on rather than marginalizes. The light fractions are
      assumed, not inferred, and the marginal likelihood is flat in them under constant light
      (see ``scripts/m5_light_ratio_demo.py``), which is the systematic Pavlovski & Hensberge
      (2010) identify as dominant. The draw spread carries no information about it.
    * iSpec's own ``errors['teff']``, a within-draw fit error computed from the ``err``
      column. Adding it in quadrature to a spread that came from the same posterior counts
      part of that posterior twice.

    ``N = 100`` is a reasonable production value: the relative standard error of a sample
    standard deviation is ``1/sqrt(2(N-1))``, i.e. 7% at 100 and 12.7% at 32. Below about 32
    the spread is too noisy to quote. The atmosphere grid step should also be smaller than the
    spread being measured; if every draw lands in one grid cell the spread is zero for a
    reason unrelated to the data.

    References
    ----------
    Kiran, E. et al. 2016, A&A, 587, A127
    Mahy, L. et al. 2020, A&A, 634, A118
    Pavlovski, K. & Hensberge, H. 2010, in ASP Conf. Ser. 435, Binaries - Key to
    Comprehension of the Universe, 207
    Pavlovski, K., Southworth, J. & Tamajo, E. 2018, MNRAS, 481, 3129
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
