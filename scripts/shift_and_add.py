"""Shift-and-add disentangling: a clean-room reference implementation.

Provenance. The implementation follows the published algorithm, González & Levato (2006),
A&A 448, 283, §2.1 Eqs. (1)-(2) and §2.3, with the identical recurrence restated
independently by Quintero et al. (2020). No source code was consulted. The widely used
existing implementation (the one behind the LB-1 and HR 6819 companion identifications)
carries no license file, so it cannot be copied into an open-source package; nothing here was
derived from reading it. Where the papers do not specify a detail, the choice made below is
marked as ours and is not attributed to the method.

The module supports the benchmark record, ``docs/benchmarks.md``, which compares albireo
against the technique in widest use on identical data. It is not part of the albireo package
and is not exported.

The iteration, with ``x = ln lambda`` and velocities in units of *c*::

    A_j(x) = < S_i(x + v_a,i) - B_{j-1}(x - v_b,i + v_a,i) >_i
    B_j(x) = < S_i(x + v_b,i) - A_j    (x - v_a,i + v_b,i) >_i

Two properties of that recurrence are easy to get wrong.

The iteration is Gauss-Seidel, not Jacobi. The second line uses ``A_j``, the estimate
produced by the first line in the same sweep, not ``A_{j-1}``. The Jacobi variant is a
different algorithm with a different convergence rate.

The observation is shifted into the component's rest frame, and the companion is then pushed
to the relative velocity it has in that frame. González & Levato state it in prose: "The
first step is to shift the spectrum B to match the lines of the secondary component in the
i-th spectrum. Then we compute the differences S_i(x) - B(x - v_b,i), which correspond to the
i-th observed spectrum but now contain only spectral features of the primary."

Initialization is B = 0, with A not seeded: "adopting a flat spectrum for the secondary
component (the weakest) works well as a starting point... the starting primary spectrum A_0
is not needed at all, since A_1 is computed from B_0." The first primary estimate is
therefore the plain rest-frame co-add.

The published convergence statement is an iteration count, not a threshold: "the residuals of
the secondary lines still present in the A spectrum are reduced approximately by a factor
1/n... rarely more than 5-7 iterations are needed." There is no published stopping rule, so
``tol`` below is ours and is off by default.

The error diffuses rather than vanishing. §2.3 gives the error recursion
``DeltaA_{j+1}(x) = < DeltaA_j(x - d_i + d_k) >_{i,k}`` with ``d_i = v_b,i - v_a,i``: each
sweep convolves the residual with ``f(x) = n^-2 sum_ik delta(x - d_i + d_k)``. When the
relative velocities cover their range densely, after *m* sweeps the residual has been smeared
by a Gaussian of ``sigma = sqrt(2m) * sigma_d``. That prediction is checked against this
implementation in ``tests/test_shift_and_add.py``, which is the available evidence that the
recurrence here is the one in the paper.

The same analysis states the method's hard limit. In Fourier space the per-mode factor has
modulus exactly 1 at zero frequency, so the DC level of each component is a fixed point:
shift-and-add cannot determine it from constant-light data. This is the low-frequency null
space of ``docs/math.md`` §5.1, and the one fd3 exhibits: three independent methods, one
degeneracy.

Choices the papers leave open, made here and flagged as ours:

* Interpolation. González & Levato used IRAF's ``dopcor``; no kernel is specified anywhere.
  This module uses albireo's own linear shift operator, so that sharing the interpolation
  between the two codes isolates the algorithmic difference instead of measuring two
  interpolators against each other.
* Weights. The paper permits "any combination algorithm... weights or some rejection
  algorithm" but publishes no formula. The default here is the plain mean, which is the
  published default; ``weights`` accepts a per-epoch/per-pixel array.
* Convergence tolerance, as above.
* No clipping to the continuum. Some applications force the disentangled spectra below the
  continuum; no source states the exact form, so it is omitted rather than guessed.

References
----------
González, J. F. & Levato, H. 2006, A&A, 448, 283
"""

from __future__ import annotations

import numpy as np

__all__ = ["disentangle", "smearing_sigma_pix"]


def _shift(flux: np.ndarray, shift_pix: float) -> np.ndarray:
    """Sample ``flux`` at ``p - shift_pix`` by linear interpolation, zero outside.

    Matches :func:`albireo.operators.shift_spectrum`, so both codes in the benchmark carry
    the same interpolation error and the comparison is of the algorithms.
    """
    n = flux.size
    p = np.arange(n) - shift_pix
    i = np.floor(p).astype(int)
    f = p - i
    lo = np.where((i >= 0) & (i < n), np.take(flux, np.clip(i, 0, n - 1)), 0.0)
    hi = np.where((i + 1 >= 0) & (i + 1 < n), np.take(flux, np.clip(i + 1, 0, n - 1)), 0.0)
    return (1.0 - f) * lo + f * hi


def disentangle(
    deviations: np.ndarray,
    shifts_pix: np.ndarray,
    *,
    n_iter: int = 7,
    weights: np.ndarray | None = None,
    small_n_correction: bool = False,
    tol: float | None = None,
    return_history: bool = False,
):
    """Recover two component spectra by the González & Levato (2006) iteration.

    Parameters
    ----------
    deviations
        Observed composite spectra as deviations from the continuum, shape
        ``(n_epochs, n_pix)``: ``flux - 1`` on a uniform log-wavelength grid. The deviation
        convention makes zero-padding at the edges correct, since outside the spectrum the
        deviation is zero, which is the continuum.
    shifts_pix
        Per-component per-epoch shifts in pixels, shape ``(2, n_epochs)``. On a uniform
        ``ln lambda`` grid a velocity is a constant pixel shift, ``ξ(v) / dx``.
    n_iter
        Number of sweeps. The paper quotes 5-7.
    weights
        Optional co-add weights, ``(n_epochs,)`` or ``(n_epochs, n_pix)``. Permitted by the
        method ("any combination algorithm... weights or some rejection algorithm") but with
        no published formula, so the default is the plain mean.
    small_n_correction
        Apply the published ``n/(n-1)`` rescaling of ``B`` before the second ``A`` update,
        which the paper suggests for samples of 2-3 spectra.
    tol
        Stop when the largest change in either component falls below this. Ours: the paper
        publishes an iteration count and no threshold. ``None`` runs ``n_iter`` sweeps.
    return_history
        Also return the per-sweep max change, for convergence studies.

    Returns
    -------
    numpy.ndarray
        Shape ``(2, n_pix)``, the recovered deviations. The values are light-weighted: the
        fixed point of the iteration is ``l_i * d_i``, because that is what the composite
        spectrum contains. Divide by the light fractions before comparing with undiluted
        component spectra.

    Notes
    -----
    The DC level of each output is not determined by the data (see the module docstring);
    with ``B = 0`` as the start, the split of the common continuum between the two components
    is a property of the initialization rather than a measurement.

    References
    ----------
    González, J. F. & Levato, H. 2006, A&A, 448, 283
    """
    deviations = np.asarray(deviations, dtype=float)
    shifts_pix = np.asarray(shifts_pix, dtype=float)
    if deviations.ndim != 2:
        raise ValueError(f"deviations must be (n_epochs, n_pix), got {deviations.shape}")
    n_ep, n_pix = deviations.shape
    if shifts_pix.shape != (2, n_ep):
        raise ValueError(f"shifts_pix must be (2, {n_ep}), got {shifts_pix.shape}")

    if weights is None:
        w = np.ones((n_ep, 1))
    else:
        w = np.asarray(weights, dtype=float)
        w = w[:, None] if w.ndim == 1 else w
        if w.shape[0] != n_ep:
            raise ValueError(f"weights must have {n_ep} epochs, got {w.shape}")
    wsum = np.broadcast_to(w, (n_ep, n_pix)).sum(axis=0)
    wsum = np.where(wsum > 0.0, wsum, 1.0)

    va, vb = shifts_pix[0], shifts_pix[1]
    comp = np.zeros((2, n_pix))  # B_0 = 0; A is never seeded (A_1 falls out of Eq. 1)
    history = []

    for sweep in range(n_iter):
        previous = comp.copy()
        # Eq. (1): A from the current B, then Eq. (2): B from the A just produced.
        for this, other, v_this, v_other in ((0, 1, va, vb), (1, 0, vb, va)):
            acc = np.zeros(n_pix)
            for j in range(n_ep):
                # S_i(x + v_this): sampling at x+v is a shift by -v.
                obs = _shift(deviations[j], -v_this[j])
                # other(x - v_other + v_this): a shift by (v_other - v_this).
                contam = _shift(comp[other], v_other[j] - v_this[j])
                acc += np.broadcast_to(w, (n_ep, n_pix))[j] * (obs - contam)
            comp[this] = acc / wsum
            if small_n_correction and sweep == 0 and this == 1:
                comp[1] *= n_ep / (n_ep - 1.0)

        change = float(np.abs(comp - previous).max())
        history.append(change)
        if tol is not None and change < tol:
            break

    return (comp, history) if return_history else comp


def smearing_sigma_pix(shifts_pix: np.ndarray, n_sweeps: int) -> float:
    """The residual smearing scale the paper predicts, in pixels.

    González & Levato §2.3: after *m* sweeps the residual has been convolved with a Gaussian
    of ``sigma = sqrt(2m) · sigma_d``, where ``sigma_d`` is the standard deviation of the
    relative velocities ``d_i = v_b,i - v_a,i``. Exposed so that the prediction can be tested.
    """
    shifts_pix = np.asarray(shifts_pix, dtype=float)
    d = shifts_pix[1] - shifts_pix[0]
    return float(np.sqrt(2.0 * n_sweeps) * d.std())
