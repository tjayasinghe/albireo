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

from albireo.grids import C_KMS, LogGrid, log_doppler_shift
from albireo.operators import (
    InterpOperator,
    RebinOperator,
    bin_edges_from_centers,
    interp_operator,
    rebin_operator,
    shift_spectrum,
    shift_spectrum_adjoint,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "C_KMS",
    "InterpOperator",
    "LogGrid",
    "RebinOperator",
    "__version__",
    "bin_edges_from_centers",
    "interp_operator",
    "log_doppler_shift",
    "rebin_operator",
    "shift_spectrum",
    "shift_spectrum_adjoint",
]
