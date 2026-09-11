# The pinned velocity table of gaia-45788547559850496 (oracle tier, D62 run)

Diagnosis of 2026-09-10 (D63) from the archived products; nothing was re-run. Three failures
compound, and only the third is a code defect.

1. **The disentangling diverged.** It stopped at the 100-step cap with a gradient norm of
   7.2e5, a residual z-score rms of 45.0 (every other oracle star of the run has 1.0), a
   period 0.128 d from a Gaussian prior of width 5.6e-4 d (228 prior sigma), tau for A driven
   to 2e19, and component spectra correlating with the injected ones at 0.067 (A) and 0.015
   (B): A's recovered equivalent width in the metal window is 33 times the truth, B's is
   negative. The semi-amplitudes look right only because the oracle tier declares K as
   15-percent Gaussians. The same star under the same seed in the first run (diagonal
   noise) fitted cleanly (z-score rms 0.96, period 5.609267 against 5.609275); the only
   difference reaching this fit is the AR(1) noise model declared in D62 (investigated
   separately, D63).
2. **The label fit disowned its result and was used anyway.** It could not beat either
   null (chi-square 1.15e7 against a no-template null of 7.59e6), pinned vsini at 150 km/s
   and [M/H] at -1.0 for both components, and pinned component A's frame offset at
   -149.9997 km/s, 3e-4 km/s from the bound of its 15-trial scan over +-150 km/s. The
   pipeline's `_templates` adopted that number as A's template zero point regardless.
3. **The correlation stage searched a window that could not contain the truth, and a
   fine-pass defect reported a position it never evaluated.** `Fit.measure_velocities`
   builds one shared window `(fitted.min() - 40, fitted.max() + 40)` from the fitted
   Keplerian without a systemic velocity (-118.8 to +117.6 km/s here) and composes each
   template's zero point afterwards, so A's reachable reported velocities were -268.8 to
   -32.4 km/s while its true velocities ran from -22.4 to +96.2 (the two zero points differ
   by 247.4 km/s, which is not a possible pair for one binary). A's coarse minimum sat on
   the top coarse node in every epoch (`at_edge`); the fine pass then walked its 11-pixel
   window upward four times without a clamp, applying its update after the last evaluation,
   so `shift = fine_start + pos` added a position from the window starting at 49 px to a
   start of 54 px and reported 64 px, five pixels beyond the last evaluated point; `_refine`
   had returned `ok = False` with the minimum on the window boundary. The reported
   42.000288 km/s is v(64 px) composed with the pinned zero point, exact to eleven figures,
   identical in all fifteen epochs because every step is deterministic and epoch
   independent; B's five distinct values are integer pixel offsets from its own zero
   point. Every diagnostic in the row (chi2, light, dchi2, r2, the curvature the sigma
   comes from) was evaluated at 59 px, so the rows are internally inconsistent by 15 km/s;
   r2 runs -18 to -26 (the two templates fit 23 times worse than no template), the
   detection statistic dchi2 (2100 to 9000) never trips the "weakly detected" guard because
   it measures the rise when one component is removed from a model already far worse than
   the null, eight epochs got sigma_A near 4e6 km/s and seven a non-positive-definite
   Hessian and `nan` sigmas (the same seven flagged "blended").

The run's own gates held: `good` false in all fifteen rows, `n_usable` 0, four flags, the
orbit stage skipped, the truth block's velocity statistics null. What was written to disk is
fifteen rows of plausible-looking velocities in `velocities.rv`; a consumer of that file
through `velocities = "<file>"` (which reads BJD plus one column per component and never
looks at the flags) would have taken them as data.

**The rest of the oracle tier.** Across the other 32 oracle runs of both populations
(1140 epochs) there is exactly one `at_edge` flag (mixed-0015, 1 of 27, which still gave 26
usable epochs and an orbit) and no non-finite uncertainty; the pathology is confined to
this star. The discriminators were all present before the velocity stage ran: spectra
correlations 0.07 and 0.02 against 0.29 to 0.998 elsewhere, z-score rms 45 against 1.0, the
two zero points disagreeing by 247 km/s against 0.1 to 61 km/s elsewhere. `converged` is
false for almost every star (the MAP stops at the cap), so it discriminates nothing.

**Changes proposed** (C1, C2 and C4 implemented in D63; C3 and C5 pending the period-search
edits of pipeline.py):

- C1 `todcor._run`: report the position at which the chi-square was evaluated
  (`evaluated_start + pos`), clamp the fine window inside the requested range, and flag
  `at_edge` when the fine minimum is still on the window boundary after the last attempt.
- C2 `todcor._run`: write `nan` into the velocity and sigma of an unmeasured epoch, keeping
  the diagnostics, so the file reads as failed where it is.
- C3 `pipeline._templates`: refuse a zero point the label stage disowned (the match beats
  neither null; a fitted offset within one trial step of the scan's bound; the components'
  offsets differing by more than the velocity budget on a Keplerian fit), leaving the table
  differential with one gamma per component as on the no-library route.
- C4 `Fit.measure_velocities`: one window per template, shifted by its zero point relative
  to the median, and a refusal by name when a component's window cannot contain its own
  fitted Keplerian velocities.
- C5 `pipeline._assess_table`: a `failed` status when the median r2 is negative or no epoch
  is usable, and a marked file rather than plausible rows. Upstream, a z-score rms far above
  one after the disentangling should stop the star before the label and velocity stages run.
