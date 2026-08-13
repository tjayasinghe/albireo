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

from albireo.data import Dataset, EpochData
from albireo.forward import (
    Problem,
    build_problem,
    data_residual_zscores,
    with_ar1,
    with_jitter,
    with_light_fractions,
    with_lsf,
    with_response,
    with_velocities,
)
from albireo.grids import C_KMS, LogGrid, log_doppler_shift
from albireo.inference import (
    MAPResult,
    MarginalOrbitModel,
    laplace_inverse_mass,
    orbit_parameters,
    orbit_velocities,
    posterior_spectra,
    run_map,
    run_nuts,
)
from albireo.kepler import radial_velocity, solve_kepler, t_peri_from_t_conj, true_anomaly
from albireo.likelihood import (
    MarginalResult,
    draw_spectra,
    marginal_loglikelihood,
    spectra_std,
)
from albireo.operators import (
    InterpOperator,
    RebinOperator,
    bin_edges_from_centers,
    convolve_spectrum,
    convolve_varying,
    convolve_varying_adjoint,
    gaussian_kernel,
    gaussian_kernel_traced,
    gaussian_lsf_profiles,
    interp_operator,
    lsf_anchor_tables,
    rebin_operator,
    shift_spectrum,
    shift_spectrum_adjoint,
)
from albireo.preprocess import (
    TELLURIC_BANDS,
    der_snr_sigma,
    estimate_ivar,
    fit_continuum,
    mask_ranges,
    mask_spikes,
    mask_tellurics,
    normalize,
    select_region,
    share_wavelength_grid,
)
from albireo.priors import SmoothnessPrior
from albireo.scan import K2ScanResult, k2_scan
from albireo.simulate import (
    InstrumentSpec,
    OrbitParams,
    SimulationTruth,
    simulate_dataset,
    synthetic_deviation_spectrum,
    synthetic_telluric_spectrum,
)

__version__ = "0.1.0.dev0"

# albireo.io is the one module that needs astropy, so it is imported on first use rather
# than at package import: `albireo.read_dataset(...)` stays discoverable, and installs
# without the [io] extra keep working for everyone who already has arrays in memory.
_IO_EXPORTS = frozenset({"RawSpectrum", "read_dataset", "read_spectrum", "to_epoch"})


def __getattr__(name: str):
    if name in _IO_EXPORTS:
        from albireo import io

        return getattr(io, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | _IO_EXPORTS)


__all__ = [
    "C_KMS",
    "TELLURIC_BANDS",
    "Dataset",
    "EpochData",
    "InstrumentSpec",
    "InterpOperator",
    "K2ScanResult",
    "LogGrid",
    "MAPResult",
    "MarginalOrbitModel",
    "MarginalResult",
    "OrbitParams",
    "Problem",
    "RawSpectrum",
    "RebinOperator",
    "SimulationTruth",
    "SmoothnessPrior",
    "__version__",
    "bin_edges_from_centers",
    "build_problem",
    "convolve_spectrum",
    "convolve_varying",
    "convolve_varying_adjoint",
    "data_residual_zscores",
    "der_snr_sigma",
    "draw_spectra",
    "estimate_ivar",
    "fit_continuum",
    "gaussian_kernel",
    "gaussian_kernel_traced",
    "gaussian_lsf_profiles",
    "interp_operator",
    "k2_scan",
    "laplace_inverse_mass",
    "log_doppler_shift",
    "lsf_anchor_tables",
    "marginal_loglikelihood",
    "mask_ranges",
    "mask_spikes",
    "mask_tellurics",
    "normalize",
    "orbit_parameters",
    "orbit_velocities",
    "posterior_spectra",
    "radial_velocity",
    "read_dataset",
    "read_spectrum",
    "rebin_operator",
    "run_map",
    "run_nuts",
    "select_region",
    "share_wavelength_grid",
    "shift_spectrum",
    "shift_spectrum_adjoint",
    "simulate_dataset",
    "solve_kepler",
    "spectra_std",
    "synthetic_deviation_spectrum",
    "synthetic_telluric_spectrum",
    "t_peri_from_t_conj",
    "to_epoch",
    "true_anomaly",
    "with_ar1",
    "with_jitter",
    "with_light_fractions",
    "with_lsf",
    "with_response",
    "with_velocities",
]
