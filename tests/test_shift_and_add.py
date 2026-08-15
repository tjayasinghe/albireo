"""The clean-room shift-and-add reference implementation (roadmap Tier 2 item 9).

`scripts/shift_and_add.py` is not part of the package — it exists so the benchmark page can
compare albireo against the technique the field actually uses, on identical data. That makes
it exactly the kind of code that can be quietly wrong: nothing downstream of it fails if the
recurrence is subtly not the published one, the numbers just come out different and get
written into a comparison table.

So these tests check it against the *paper*, not against itself. The sharpest is
`test_the_published_smearing_law_is_reproduced`: González & Levato (2006) §2.3 derive that
the residual is not annihilated but diffused, by a Gaussian of sqrt(2m)·sigma_d after m
sweeps. Reproducing a law the implementation was not fitted to is the strongest available
evidence that the iteration coded here is the iteration in the paper.
"""

from __future__ import annotations

import numpy as np
import pytest

import albireo as ab
from albireo.operators import shift_spectrum
from scripts.shift_and_add import _shift, disentangle, smearing_sigma_pix

GRID = ab.LogGrid.from_wavelength_range(4400.0, 4460.0, dv_kms=4.0)
LIGHT = np.array([0.62, 0.38])


def _components(seed=1):
    return np.stack(
        [
            np.asarray(
                ab.synthetic_deviation_spectrum(
                    GRID, n_lines=14, depth_range=(0.15, 0.6), sigma_v_range=(9.0, 18.0), seed=s
                )
            )
            for s in (seed, seed + 1)
        ]
    )


def _observe(comps, shifts_pix, noise=0.0, seed=0):
    """Build composite deviations the way albireo's forward model does: sum l_i * T(v_i) d_i."""
    n_ep = shifts_pix.shape[1]
    out = np.zeros((n_ep, GRID.n))
    for j in range(n_ep):
        for i in range(2):
            out[j] += LIGHT[i] * np.asarray(shift_spectrum(comps[i], float(shifts_pix[i, j])))
    if noise:
        out = out + np.random.default_rng(seed).normal(0.0, noise, out.shape)
    return out


def _shifts(n_ep=16, amp=(6.0, -10.0), seed=3):
    phase = np.linspace(0.0, 2.0 * np.pi, n_ep, endpoint=False)
    return np.stack([amp[0] * np.sin(phase), amp[1] * np.sin(phase)])


def test_it_recovers_the_light_weighted_components_without_noise():
    """The fixed point is l_i * d_i, because that is what the composite contains."""
    comps = _components()
    shifts = _shifts()
    obs = _observe(comps, shifts)
    rec = disentangle(obs, shifts, n_iter=12)

    interior = slice(40, -40)
    for i in range(2):
        target = LIGHT[i] * comps[i][interior]
        got = rec[i][interior]
        # Mean-aligned: the DC level is a fixed point of the iteration (module docstring).
        err = (got - target) - (got - target).mean()
        assert np.sqrt(np.mean(err**2)) < 0.02, f"component {i + 1} RMS {np.sqrt(np.mean(err**2))}"


def test_dividing_by_the_light_fraction_is_what_recovers_the_truth():
    """The conversion most likely to be got wrong silently in a comparison table."""
    comps = _components()
    shifts = _shifts()
    rec = disentangle(_observe(comps, shifts), shifts, n_iter=12)
    interior = slice(40, -40)

    undiluted = rec[0][interior] / LIGHT[0]
    raw = rec[0][interior]
    truth = comps[0][interior]
    err_ok = np.std(undiluted - truth)
    err_bad = np.std(raw - truth)
    assert err_ok < err_bad / 2.0, (
        "dividing by the light fraction must materially improve agreement with the undiluted "
        f"truth; got {err_ok:.4f} against {err_bad:.4f}"
    )


def test_the_first_primary_estimate_is_the_rest_frame_coadd():
    """B_0 = 0 and A is never seeded, so A_1 falls out as the plain co-add.

    González & Levato: "the starting primary spectrum A_0 is not needed at all, since A_1 is
    computed from B_0."
    """
    comps = _components()
    shifts = _shifts()
    obs = _observe(comps, shifts)
    rec = disentangle(obs, shifts, n_iter=1)

    coadd = np.mean(
        [np.asarray(shift_spectrum(obs[j], -shifts[0, j])) for j in range(shifts.shape[1])],
        axis=0,
    )
    np.testing.assert_allclose(rec[0], coadd, atol=1e-12)


def test_the_published_smearing_law_is_reproduced():
    """González & Levato §2.3: after m sweeps the residual is smeared by sqrt(2m)*sigma_d.

    Tested on a *delta-function* residual so the smearing is directly measurable: seed the
    companion with a single spike, run the error recursion, and measure the width of what
    comes back. The law is a prediction the implementation was not fitted to.
    """
    n_ep = 24
    shifts = _shifts(n_ep=n_ep, amp=(8.0, -13.0))
    n_pix = GRID.n
    centre = n_pix // 2

    # Propagate a pure error: zero data, a spiked companion. The recursion is then exactly
    # the paper's Delta A_{j+1} = < Delta A_j(x - d_i + d_k) >.
    obs = np.zeros((n_ep, n_pix))
    spike = np.zeros(n_pix)
    spike[centre] = 1.0

    def width_after(m):
        comp = np.zeros((2, n_pix))
        comp[1] = spike
        # Drive the recursion by hand so the seeded error is the only thing present.
        va, vb = shifts[0], shifts[1]
        for _ in range(m):
            for this, other, v_this, v_other in ((0, 1, va, vb), (1, 0, vb, va)):
                acc = np.zeros(n_pix)
                for j in range(n_ep):
                    acc += _shift(obs[j], -v_this[j]) - _shift(comp[other], v_other[j] - v_this[j])
                comp[this] = acc / n_ep
        x = np.arange(n_pix) - centre
        p = np.abs(comp[1])
        if p.sum() <= 0:
            return 0.0
        p = p / p.sum()
        return float(np.sqrt(np.sum(p * x**2)))

    sigma_d = (shifts[1] - shifts[0]).std()
    for m in (2, 4):
        measured = width_after(m)
        predicted = np.sqrt(2.0 * m) * sigma_d
        assert 0.45 < measured / predicted < 1.7, (
            f"after {m} sweeps the residual width was {measured:.2f} px against the paper's "
            f"predicted {predicted:.2f} px (ratio {measured / predicted:.2f})"
        )
    assert smearing_sigma_pix(shifts, 4) == pytest.approx(np.sqrt(8.0) * sigma_d)


def test_the_dc_level_is_a_fixed_point_the_method_cannot_determine():
    """The rigorous limit: the per-mode convergence factor has modulus 1 at zero frequency.

    Adding a constant to one component and subtracting it from the other leaves the composite
    unchanged, so no iteration can distinguish them. This is the same low-frequency null space
    albireo's prior addresses and fd3 exhibits.
    """
    comps = _components()
    shifts = _shifts()

    # Move a constant from one component to the other. The composite is untouched, so no
    # amount of iterating can tell the two truths apart.
    delta = 0.05
    tweaked = comps.copy()
    tweaked[0] = tweaked[0] + delta / LIGHT[0]
    tweaked[1] = tweaked[1] - delta / LIGHT[1]
    # Interior only, and the exception is the interesting part: shifting zero-pads, so within
    # a shift of the boundary the two constants do not cancel and the finite window breaks the
    # degeneracy. That is the same margin effect D47 found dominating the forecast's
    # worst-determined mode. In the interior the exchange is exact.
    edge = int(np.ceil(np.abs(shifts).max())) + 2
    np.testing.assert_allclose(
        _observe(comps, shifts)[:, edge:-edge],
        _observe(tweaked, shifts)[:, edge:-edge],
        atol=1e-12,
        err_msg="in the interior the exchange must leave the composite spectra identical",
    )

    rec = disentangle(_observe(comps, shifts), shifts, n_iter=10)
    interior = slice(40, -40)
    # The shape is recovered regardless; only the level is undetermined.
    for i in range(2):
        err = rec[i][interior] - LIGHT[i] * comps[i][interior]
        assert np.std(err) < 0.02, f"shape must be recovered even so: {np.std(err)}"


def test_it_survives_noise_and_still_beats_doing_nothing():
    comps = _components()
    shifts = _shifts()
    obs = _observe(comps, shifts, noise=0.01, seed=5)
    rec = disentangle(obs, shifts, n_iter=8)
    interior = slice(40, -40)
    for i in range(2):
        target = LIGHT[i] * comps[i][interior]
        err = (rec[i][interior] - target) - (rec[i][interior] - target).mean()
        assert np.sqrt(np.mean(err**2)) < 0.03


def test_weights_are_accepted_in_both_shapes():
    """Permitted by the method ("any combination algorithm... weights"), formula unpublished."""
    comps = _components()
    shifts = _shifts()
    obs = _observe(comps, shifts)
    per_epoch = np.ones(shifts.shape[1])
    per_pixel = np.ones((shifts.shape[1], GRID.n))
    a = disentangle(obs, shifts, n_iter=4)
    b = disentangle(obs, shifts, n_iter=4, weights=per_epoch)
    c = disentangle(obs, shifts, n_iter=4, weights=per_pixel)
    np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(a, c, atol=1e-12)


def test_masking_an_epoch_by_weight_actually_removes_it():
    """Decisive for the nebular figure: this method CAN mask, so a comparison that assumes
    it cannot would be unfair to it."""
    comps = _components()
    shifts = _shifts()
    obs = _observe(comps, shifts)
    obs[3] += 5.0  # a ruined epoch
    w = np.ones(shifts.shape[1])
    w[3] = 0.0
    clean = disentangle(obs, shifts, n_iter=8, weights=w)
    dirty = disentangle(obs, shifts, n_iter=8)
    interior = slice(40, -40)
    err_clean = np.std((clean[0] - LIGHT[0] * comps[0])[interior])
    err_dirty = np.std((dirty[0] - LIGHT[0] * comps[0])[interior])
    assert err_clean < err_dirty, f"zero weight must exclude the epoch: {err_clean} vs {err_dirty}"


def test_it_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="deviations must be"):
        disentangle(np.zeros(10), np.zeros((2, 5)))
    with pytest.raises(ValueError, match="shifts_pix must be"):
        disentangle(np.zeros((5, 10)), np.zeros((2, 4)))
