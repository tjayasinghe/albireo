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
from albireo.forward import Problem, build_problem, with_velocities
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
    gaussian_kernel,
    interp_operator,
    rebin_operator,
    shift_spectrum,
    shift_spectrum_adjoint,
)
from albireo.priors import SmoothnessPrior
from albireo.simulate import (
    InstrumentSpec,
    OrbitParams,
    SimulationTruth,
    simulate_dataset,
    synthetic_deviation_spectrum,
    synthetic_telluric_spectrum,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "C_KMS",
    "Dataset",
    "EpochData",
    "InstrumentSpec",
    "InterpOperator",
    "LogGrid",
    "MAPResult",
    "MarginalOrbitModel",
    "MarginalResult",
    "OrbitParams",
    "Problem",
    "RebinOperator",
    "SimulationTruth",
    "SmoothnessPrior",
    "__version__",
    "bin_edges_from_centers",
    "build_problem",
    "convolve_spectrum",
    "draw_spectra",
    "gaussian_kernel",
    "interp_operator",
    "laplace_inverse_mass",
    "log_doppler_shift",
    "marginal_loglikelihood",
    "orbit_parameters",
    "orbit_velocities",
    "posterior_spectra",
    "radial_velocity",
    "rebin_operator",
    "run_map",
    "run_nuts",
    "shift_spectrum",
    "shift_spectrum_adjoint",
    "simulate_dataset",
    "solve_kepler",
    "spectra_std",
    "synthetic_deviation_spectrum",
    "synthetic_telluric_spectrum",
    "t_peri_from_t_conj",
    "true_anomaly",
    "with_velocities",
]
