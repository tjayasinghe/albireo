# Why one field star failed at the optimizer's initialisation (D63, 2026-09-11)

`field5/mixed-0018__orbit` (a 3.99-day circular near-twin pair, 71 epochs) ended twice in
numpyro's "Cannot find valid initial parameters" inside `run_map`. Reproduced from the
population file (index 18, seed 17, the survivor index; the manifest record matches to every
digit) with the pipeline's own stages up to the declaration, then the pre-fit sequence of
`Disentangler.fit` run step by step. Scripts: the session scratchpad `wp-t/`.

**The site is `t_conj`, and the mechanism is a boundary, not a guard.** After the scans
`init["t_conj"]` was 2456881.970524587 and the prior handed to the optimizer was
`Uniform(2456877.9841521564, 2456881.970524587)`: the start equalled the upper bound bit for
bit. numpyro's interval constraint is closed, so the prior density is finite there, but the
unconstraining transform of the bound is infinite, so the potential numpyro evaluates is
infinite. Every factor added in D63 is satisfied at that start: `log_tau - log_eta` is 4.09
against the bound of 30 (the prior-amplitude shift moves both by the same amount, so the
bound cannot be reached that way; here it took tau from 300 to 1200 and eta from 5 to 20),
the chi-square sign guard does not fire (marginal +200,310), the eccentricity disk and the
bandwidth guard hold, and the constrained log density is finite (+200,295).

**How the start reached the edge.** The phase scan's 41-point grid put the conjunction at
2456879.977. The first semi-amplitude scan, whose level-0 phase grid holds `t_conj + P/2`
exactly (the antipode the 41-point grid cannot sample: with `linspace(0, P, 41,
endpoint=False)` the antipode of any trial lies midway between two trials), found the
antipode 672 nats better and moved the start there; on a near-twin pair the two mirrors are
almost equally good and the finer grid decides. The prior-amplitude profile then asked for a
factor 0.5 on both stars, the second scan pass gained exactly 0.0 (its best was 24 nats below
its start and the start was kept), and the final phase scan, which runs only when the last
scan's gain is positive, did not run. The one-period window was then built around the phase
scan's best while the start was half a period on: on the edge, and by an exact binary
coincidence (`period / 8` is exact, so `best + 4 (P/8) == best + P/2` as doubles) rather
than just outside it, where the message would have been the same. Any adopted phase above
`scan.best + P/2` falls outside such a window; this star landed on the one point where the
density is finite and the transform is not.

**The fix, measured.** Centre the window on the conjunction the start carries. On the failing
configuration `initialize_model` then succeeds at potential -199,893 with the unconstrained
coordinate at zero. Folding the start one period down, or nudging it inside the edge, both
initialise but leave it in the flat tail of the logit transform (unconstrained coordinates
of -22.9 and +13.8, where the gradient in `t_conj` is suppressed by about `exp(-|z|)`), so
L-BFGS would sit pinned at the edge; centring is the change. `Fit.sample` built the same
window from the phase scan and now centres it on the fitted conjunction.

**Second-order.** The 41-point phase scan cannot sample its own antipode, and on this star
the antipode was better by 672 nats. An even number of trials, or a second pass offset by
half a grid step, would let it. In the D62 run of the same star the fit had started from
the phase scan's best and settled on the exchange-swapped assignment (K 52.9 and 51.4 against
51.1 and 53.0). This star is the only failure of its kind across the archived runs.
