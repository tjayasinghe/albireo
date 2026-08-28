"""albireo: GPU-accelerated Bayesian spectral disentangling of spectroscopic binaries.

albireo requires float64 — adjoint identities, log-determinants, and sub-km/s velocity
work are not reliable in float32 — so importing the package enables JAX x64 mode.
Set the environment variable ``ALBIREO_DISABLE_X64`` (before import) to opt out for
experiments; the test suite will not pass without x64.
"""

import os

if not os.environ.get("ALBIREO_DISABLE_X64"):
    import jax

    jax.config.update("jax_enable_x64", True)

from albireo.archive import (
    ArchiveRecord,
    BloemTarget,
    bloem_catalogue,
    bloem_spectra,
    download,
    query,
    resolve_bloem,
    spectra_query,
)
from albireo.calibrate import DetectionLimit, detection_limit
from albireo.data import Dataset, EpochData
from albireo.examples import clear_example_cache, example_info, example_names, load_example
from albireo.facade import (
    LSF,
    Between,
    Disentangler,
    Fit,
    Fixed,
    Known,
    Nebular,
    Orbit,
    Posterior,
    Sampled,
    Scanned,
    Smoothness,
    Star,
    Telluric,
)
from albireo.forecast import SensitivityForecast, plan_epochs, sensitivity_forecast
from albireo.forward import (
    Problem,
    build_problem,
    data_residual_zscores,
    with_ar1,
    with_data,
    with_jitter,
    with_light_fractions,
    with_lsf,
    with_nebular_amplitudes,
    with_response,
    with_shifts,
    with_velocities,
)
from albireo.grids import C_KMS, LogGrid, air_to_vacuum, log_doppler_shift, vacuum_to_air
from albireo.handoff import export_draws, write_gssp, write_ispec
from albireo.inference import (
    MAPResult,
    MarginalOrbitModel,
    keplerian_residuals,
    laplace_inverse_mass,
    nebular_amplitudes,
    orbit_parameters,
    orbit_velocities,
    posterior_spectra,
    relative_velocities,
    relative_velocity_errors,
    run_map,
    run_nuts,
)
from albireo.kepler import radial_velocity, solve_kepler, t_peri_from_t_conj, true_anomaly
from albireo.library import (
    BoxInterpolator,
    SimplexInterpolator,
    SpectralLibrary,
    clear_library_cache,
    crossval_library,
    fetch_library,
    ingest_bosz,
    ingest_pollux,
    library_info,
    library_interpolator,
    library_names,
    line_core_medium,
    load_library,
    save_library,
)
from albireo.likelihood import (
    MarginalResult,
    draw_spectra,
    marginal_loglikelihood,
    spectra_std,
)
from albireo.match import (
    FixedDilution,
    LabelMatch,
    RadiusRatio,
    ScalarDilution,
    StarLabels,
    match_labels,
    refit_draws,
)
from albireo.operators import (
    InterpOperator,
    RebinOperator,
    bin_edges_from_centers,
    convolve_spectrum,
    convolve_varying,
    convolve_varying_adjoint,
    gauss_hermite_kernel_traced,
    gaussian_kernel,
    gaussian_kernel_traced,
    gaussian_lsf_profiles,
    interp_operator,
    lsf_anchor_tables,
    rebin_operator,
    rotational_kernel,
    rotational_kernel_traced,
    rotational_radius_for,
    shift_spectrum,
    shift_spectrum_adjoint,
)
from albireo.preprocess import (
    TELLURIC_BANDS,
    der_snr_sigma,
    estimate_ivar,
    fit_continuum,
    mask_flux_gaps,
    mask_ranges,
    mask_spikes,
    mask_tellurics,
    normalize,
    select_region,
    share_wavelength_grid,
)
from albireo.priors import (
    NEBULAR_LINES,
    SmoothnessPrior,
    nebular_windows,
    window_profile,
)
from albireo.results import load_fit, save_fit, to_inference_data, write_ascii
from albireo.scan import K2ScanResult, k2_scan
from albireo.simulate import (
    InstrumentSpec,
    OrbitParams,
    SimulationTruth,
    resimulate,
    simulate_dataset,
    synthetic_deviation_spectrum,
    synthetic_nebular_spectrum,
    synthetic_telluric_spectrum,
)

__version__ = "0.1.0.dev0"

# albireo.io is the one module that needs astropy, so it is imported on first use rather
# than at package import: `albireo.read_dataset(...)` stays discoverable, and installs
# without the [io] extra keep working for everyone who already has arrays in memory.
_IO_EXPORTS = frozenset(
    {"RawSpectrum", "read_dataset", "read_spectrum", "to_epoch", "write_spectra"}
)

# albireo.plotting needs matplotlib (and arviz, for the corner plot), which are optional
# for the same reason astropy is: a fit that runs on a headless cluster node should not
# have to install a plotting stack. Same lazy treatment, so `albireo.plot_spectra` stays
# discoverable and raises an actionable error rather than an ImportError from inside a
# figure.
_PLOT_EXPORTS = frozenset(
    {
        "plot_corner",
        "plot_detection",
        "plot_detection_limit",
        "plot_forecast",
        "plot_light_fractions",
        "plot_lsf",
        "plot_phase_fold",
        "plot_residual_zscores",
        "plot_rv_curve",
        "plot_spectra",
    }
)


def __getattr__(name: str):
    if name in _IO_EXPORTS:
        from albireo import io

        return getattr(io, name)
    if name in _PLOT_EXPORTS:
        from albireo import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | _IO_EXPORTS | _PLOT_EXPORTS)


__all__ = [
    "C_KMS",
    "LSF",
    "NEBULAR_LINES",
    "TELLURIC_BANDS",
    "ArchiveRecord",
    "Between",
    "BloemTarget",
    "BoxInterpolator",
    "Dataset",
    "DetectionLimit",
    "Disentangler",
    "EpochData",
    "Fit",
    "Fixed",
    "FixedDilution",
    "InstrumentSpec",
    "InterpOperator",
    "K2ScanResult",
    "Known",
    "LabelMatch",
    "LogGrid",
    "MAPResult",
    "MarginalOrbitModel",
    "MarginalResult",
    "Nebular",
    "Orbit",
    "OrbitParams",
    "Posterior",
    "Problem",
    "RadiusRatio",
    "RawSpectrum",
    "RebinOperator",
    "Sampled",
    "ScalarDilution",
    "Scanned",
    "SensitivityForecast",
    "SimplexInterpolator",
    "SimulationTruth",
    "Smoothness",
    "SmoothnessPrior",
    "SpectralLibrary",
    "Star",
    "StarLabels",
    "Telluric",
    "air_to_vacuum",
    "bin_edges_from_centers",
    "bloem_catalogue",
    "bloem_spectra",
    "build_problem",
    "clear_example_cache",
    "clear_library_cache",
    "convolve_spectrum",
    "convolve_varying",
    "convolve_varying_adjoint",
    "crossval_library",
    "data_residual_zscores",
    "der_snr_sigma",
    "detection_limit",
    "download",
    "draw_spectra",
    "estimate_ivar",
    "example_info",
    "example_names",
    "export_draws",
    "fetch_library",
    "fit_continuum",
    "gauss_hermite_kernel_traced",
    "gaussian_kernel",
    "gaussian_kernel_traced",
    "gaussian_lsf_profiles",
    "ingest_bosz",
    "ingest_pollux",
    "interp_operator",
    "k2_scan",
    "keplerian_residuals",
    "laplace_inverse_mass",
    "library_info",
    "library_interpolator",
    "library_names",
    "line_core_medium",
    "load_example",
    "load_fit",
    "load_library",
    "log_doppler_shift",
    "lsf_anchor_tables",
    "marginal_loglikelihood",
    "mask_flux_gaps",
    "mask_ranges",
    "mask_spikes",
    "mask_tellurics",
    "match_labels",
    "nebular_amplitudes",
    "nebular_windows",
    "normalize",
    "orbit_parameters",
    "orbit_velocities",
    "plan_epochs",
    "plot_corner",
    "plot_detection",
    "plot_detection_limit",
    "plot_forecast",
    "plot_light_fractions",
    "plot_lsf",
    "plot_phase_fold",
    "plot_residual_zscores",
    "plot_rv_curve",
    "plot_spectra",
    "posterior_spectra",
    "query",
    "radial_velocity",
    "read_dataset",
    "read_spectrum",
    "rebin_operator",
    "refit_draws",
    "relative_velocities",
    "relative_velocity_errors",
    "resimulate",
    "resolve_bloem",
    "rotational_kernel",
    "rotational_kernel_traced",
    "rotational_radius_for",
    "run_map",
    "run_nuts",
    "save_fit",
    "save_library",
    "select_region",
    "sensitivity_forecast",
    "share_wavelength_grid",
    "shift_spectrum",
    "shift_spectrum_adjoint",
    "simulate_dataset",
    "solve_kepler",
    "spectra_query",
    "spectra_std",
    "synthetic_deviation_spectrum",
    "synthetic_nebular_spectrum",
    "synthetic_telluric_spectrum",
    "t_peri_from_t_conj",
    "to_epoch",
    "to_inference_data",
    "true_anomaly",
    "vacuum_to_air",
    "window_profile",
    "with_ar1",
    "with_data",
    "with_jitter",
    "with_light_fractions",
    "with_lsf",
    "with_nebular_amplitudes",
    "with_response",
    "with_shifts",
    "with_velocities",
    "write_ascii",
    "write_gssp",
    "write_ispec",
    "write_spectra",
]
