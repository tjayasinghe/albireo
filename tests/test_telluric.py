"""Closed-loop MAP test with a telluric component (topocentric frame).

Mirrors the M3 gate configuration of ``tests/test_inference.py`` but adds a third,
static telluric component to the model. The telluric column is appended *last* by
:func:`albireo.forward.build_problem`, so ``d_hat[2]`` is the telluric spectrum and
the ``log_tau``/``log_eta`` sites carry batch shape (3,).

In the topocentric frame the stellar shifts are ``xi(v_star) - xi(v_bary)`` while the
telluric shift is zero, so the star-vs-telluric relative velocity picks up the full
barycentric motion on top of the orbit — hence the enlarged ``v_rel_max_kms`` budget.
"""

import numpy as np
import numpyro.distributions as dist
import pytest

import albireo as ab
from albireo.inference import MarginalOrbitModel, run_map
from albireo.kepler import t_peri_from_t_conj
from albireo.simulate import InstrumentSpec, OrbitParams, simulate_dataset
from albireo.simulate import synthetic_deviation_spectrum as synth_spectrum
from albireo.simulate import synthetic_telluric_spectrum as synth_telluric

# Every test here reads the same module-scoped MAP fit, so the whole module is one
# acceptance gate: deselecting it with -m "not slow" skips the fit itself, not just the
# assertions on it.
pytestmark = pytest.mark.slow

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
P_TRUE, TCONJ_TRUE, ECC_TRUE, OMEGA_TRUE = 6.31, 2.05, 0.2, 0.7
K_TRUE = np.array([30.0, 22.0])
ELL = np.array([0.62, 0.38])
LSF = {"inst": 7.0}
N_EP = 12
N_COMP = 3  # 2 stellar + 1 telluric
# Orbit (K_1 + K_2)(1 + e) ~ 62 km/s, plus |v_bary| <= 25 km/s of star-vs-telluric
# relative motion in the topocentric frame, plus headroom for the prior on k.
V_REL_MAX = 110.0

PRIORS = {
    "period": dist.Normal(P_TRUE + 0.001, 0.003),
    "t_conj": dist.Normal(TCONJ_TRUE + 0.005, 0.02),
    "secosw": dist.Uniform(-1.0, 1.0),
    "sesinw": dist.Uniform(-1.0, 1.0),
    "k": dist.Uniform(np.array([10.0, 5.0]), np.array([45.0, 40.0])),
    "log_tau": dist.Normal(np.full(N_COMP, np.log(300.0)), 3.0),
    "log_eta": dist.Normal(np.full(N_COMP, np.log(5.0)), 3.0),
}
INIT = {
    "period": P_TRUE + 0.001,
    "t_conj": TCONJ_TRUE + 0.005,
    "secosw": np.sqrt(0.15) * np.cos(0.5),
    "sesinw": np.sqrt(0.15) * np.sin(0.5),
    "k": np.array([27.0, 25.0]),
    "log_tau": np.full(N_COMP, np.log(300.0)),
    "log_eta": np.full(N_COMP, np.log(5.0)),
}

_THETA_SITES = ("period", "t_conj", "secosw", "sesinw", "k", "log_tau", "log_eta")


def _edge_mask(frac: float = 0.05) -> np.ndarray:
    """True away from the outer ``frac`` of the grid (shift/convolution edge effects)."""
    lo = int(np.ceil(frac * GRID.n))
    mask = np.zeros(GRID.n, dtype=bool)
    mask[lo : GRID.n - lo] = True
    return mask


@pytest.fixture(scope="module")
def telluric_fit():
    """Simulate a telluric-contaminated SB2 dataset and run MAP/ML-II once."""
    rng = np.random.default_rng(42)
    comps = [
        synth_spectrum(GRID, n_lines=30, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=1),
        synth_spectrum(GRID, n_lines=25, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=2),
    ]
    # Narrow lines clustered in bands, but wide enough to be *representable* on a
    # 5.5 km/s grid seen through a 7 km/s LSF (the default 1.5-4 km/s widths are
    # sub-pixel: convolving the truth with the LSF alone already costs rms 0.20), and
    # shallow enough not to pile up against the -0.95 saturation clip, whose
    # flat-bottomed corners the curvature prior cannot represent.
    tell = synth_telluric(GRID, depth_range=(0.02, 0.4), sigma_v_range=(6.0, 12.0), seed=3)
    t_peri = float(t_peri_from_t_conj(TCONJ_TRUE, period=P_TRUE, ecc=ECC_TRUE, omega=OMEGA_TRUE))
    orbit = OrbitParams(
        period=P_TRUE, t_peri=t_peri, ecc=ECC_TRUE, omega=OMEGA_TRUE, k=tuple(K_TRUE)
    )
    bjd = np.sort(rng.uniform(0.0, 2.9 * P_TRUE, N_EP))
    v_bary = rng.uniform(-25.0, 25.0, N_EP)
    spec = InstrumentSpec(wave=np.arange(5003.0, 5042.0, 0.11), sigma_v_lsf=7.0, snr=130.0)
    ds, truth = simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ELL,
        orbit=orbit,
        v_bary=v_bary,
        frame="topocentric",
        telluric=tell,
        gap_fraction=0.01,
        cosmic_fraction=0.002,
        seed=11,
    )
    model = MarginalOrbitModel(
        GRID,
        ds,
        light_fractions=ELL,
        lsf_sigma_v=LSF,
        v_rel_max_kms=V_REL_MAX,
        telluric=True,
    )
    fit = run_map(model.model(PRIORS), init=INIT, max_steps=200)
    return truth, model, fit


@pytest.fixture(scope="module")
def map_spectra(telluric_fit):
    """Conditional posterior-mean spectra at the MAP orbit, shape (3, n_pix)."""
    _, model, fit = telluric_fit
    theta_map = {s: fit.params[s] for s in _THETA_SITES}
    return np.asarray(model.marginal(theta_map).d_hat)


def test_map_recovers_orbit_with_telluric(telluric_fit):
    """K_1, K_2 within 1% with a third (telluric) component in the model."""
    _, _, fit = telluric_fit
    k_map = np.asarray(fit.params["k"])
    for i in range(2):
        rel_err = abs(k_map[i] - K_TRUE[i]) / K_TRUE[i]
        assert rel_err < 0.01, f"K_{i + 1} off by {100 * rel_err:.2f}% (target: < 1%)"
    np.testing.assert_allclose(float(fit.params["ecc"]), ECC_TRUE, atol=0.03)
    np.testing.assert_allclose(float(fit.params["omega"]), OMEGA_TRUE, atol=0.05)
    assert np.isfinite(fit.potential)


def test_telluric_spectrum_recovered(telluric_fit, map_spectra):
    """The telluric component (d_hat[2], appended last) is recovered in its bands."""
    truth, _, _ = telluric_fit
    truth_tell = np.asarray(truth.telluric)
    tell_hat = map_spectra[2]
    assert map_spectra.shape == (N_COMP, GRID.n)
    core = (truth_tell < -0.05) & _edge_mask()
    assert core.sum() > 20, "telluric core mask is degenerate"
    rms = float(np.sqrt(np.mean((tell_hat[core] - truth_tell[core]) ** 2)))
    corr = float(np.corrcoef(tell_hat[core], truth_tell[core])[0, 1])
    assert rms < 0.05, f"telluric RMS {rms:.4f}"
    assert corr > 0.98, f"telluric correlation {corr:.4f}"


def test_stellar_combination_recovered(telluric_fit, map_spectra):
    """The observable light-weighted stellar combination is recovered.

    Two k = 0 additive indeterminacies are in play here (``docs/math.md`` §5.1,
    benchmarks.md M2 lesson 1). Constant light fractions make the *individual* stellar
    components indeterminate, so only the light-weighted combination is observable —
    as in ``test_inference.py``. The telluric component adds a second, exact one: since
    the light fractions sum to 1 and a constant is shift-invariant, adding ``a`` to the
    telluric while subtracting ``a`` from every stellar component leaves every epoch's
    prediction unchanged. That single scalar is fixed only by the ridge ``eta``, which
    ML-II sets weakly (it is honestly unconstrained), so it is removed before the
    residual is measured — and its cancellation across the two blocks is asserted
    directly, which is the sharper statement.
    """
    truth, _, _ = telluric_fit
    truth_d = np.stack([np.asarray(c) for c in truth.components])
    truth_comb = ELL @ truth_d
    comb_hat = ELL @ map_spectra[:2]
    interior = _edge_mask()
    core = (truth_comb < -0.05) & interior
    assert core.sum() > 20, "stellar core mask is degenerate"

    offset_stellar = float(np.mean((comb_hat - truth_comb)[interior]))
    offset_tell = float(np.mean((map_spectra[2] - np.asarray(truth.telluric))[interior]))
    assert abs(offset_stellar + offset_tell) < 0.005, (
        f"offsets do not cancel: stellar {offset_stellar:+.4f}, telluric {offset_tell:+.4f}"
    )

    resid = comb_hat[core] - truth_comb[core]
    assert float(np.sqrt(np.mean(resid**2))) < 0.04
    rms = float(np.sqrt(np.mean((resid - offset_stellar) ** 2)))
    corr = float(np.corrcoef(comb_hat[core], truth_comb[core])[0, 1])
    assert rms < 0.02, f"stellar combination RMS {rms:.4f} (offset removed)"
    assert corr > 0.99, f"stellar combination correlation {corr:.4f}"
