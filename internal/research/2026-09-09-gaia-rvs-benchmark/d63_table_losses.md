# Why the post-fit velocity table lost a component on three field systems (D63, 2026-09-11)

The third run's tables (300 L-BFGS steps, ML-II at its optimum) lost a component on three
field systems while the disentangling's own semi-amplitudes stayed right. Each dataset was
re-simulated (the manifest record reproduced to four figures), the archived disentangled
components rebuilt as correlation templates with the run's zero points, and `todcor` run
five ways: the third run's templates at the declared light (which reproduces the archived
table to 1e-7 km/s), at a free "global" amplitude, and at the true light; and the second
run's templates (100 steps, tau near its start) at the declared and the free amplitude.
Every table was then passed through `reassign_by_orbit` as the pipeline does and a
Keplerian fitted at the true period. Scripts and per-star JSON: the session scratchpad
`wp-q/`.

**mixed-0004 (K 22.65 / 42.34 km/s, P 116.9 d, e 0.72, secondary 5 percent of the light):
the exchange step, not the light and not the shrinkage.** Before the exchange the primary is
measured at 0.65 to 0.68 km/s rms with K_A between -5.2 and -5.7 percent in all five
configurations; the secondary is never measured (rms 30 km/s, detection statistic below 25
in six to nine epochs, K_B 79 to 90 percent off whatever the light or templates).
`reassign_by_orbit` then swaps 18 or 19 of the 80 epochs on the relative velocity alone,
which with a noise-draw v_B carrying a formal error under 3 km/s exceeds three sigma
routinely; the swap moves a well-measured primary velocity into the B column, the primary's
rms rises to 8.4 km/s, and K_A lands at -55.8 percent (third run), +13.8 (second run's
templates) or -1.7 (the second run's archived table), a coin flip over which epochs were
swapped. The tell is in `velocities.rv`: nineteen rows carry `light_A = 0.0538`, the
secondary's fraction, because the exchange moves the light row with the velocity; under
held amplitudes at fractions a factor 18 apart the two correlation peaks are not equivalent
solutions. The table's own Keplerian has a reduced chi-square near 1100 after the exchange
and 35 to 49 before it.

**mixed-0008 (K 27.5 / 44.5 km/s, P 2.09 d, secondary 11 percent of the light): the
shrinkage, with the free amplitude a partial repair.** The second run's templates recover
the secondary under both light modes (+1.6 and +0.6 percent); the third run's lose it under
the declared fraction (-89.1) and recover it halfway under a free amplitude (-24.9). The
templates are what changed: B lost a quarter of its metal-window depth as tau_B went from
396 to 1980 (A a fifth, tau_A 808 to 35966), its equivalent width against the injected
component fell from 0.52 to 0.36 of the truth, and v_B carries a median offset of -8.2 km/s
with a scatter of 7.7 against -1.4 and 3.1 with the second run's templates. The free
amplitude moves l_B from the declared 0.1425 to 0.1855, the direction that compensates a
shallower template; holding the light at the TRUE fraction (0.1119) is the worst of the
three (-100 percent, the largest scatter). What a shrunk template needs is not the true
light but a larger amplitude than the true one: the light fraction and the template scale
are not the same quantity once ML-II has shrunk the posterior mean.

**mixed-0015 (K 12.0 / 21.9 km/s, P 553 d, e 0.48, secondary 7 percent): a declared light
wrong by a factor 1.7 and a zero point 46 km/s off.** The templates are identical between
the runs (tau did not move: 71231 to 71092 and 293 to 288), so smoothing plays no part. The
light stage measured l_B = 0.124 against 0.073; the label fit placed B's rest frame at
+47.3 km/s against a systemic velocity of +0.98, made against a disentangled secondary that
correlates with the injected one at 0.31 and keeps 11 percent of its equivalent width; with
two absolute templates the shared systemic velocity leaves that 46 km/s in every B row, and
the reported K_B (0.0, 10.7 or 21.4) is whatever one exchanged epoch decides. The primary is
stable at -17 percent throughout, the disentangling's own bias carried into the table.

**Line depths** (standard deviation of the normalised flux over 8552 to 8654 A, third run
over second run): mixed-0004 B 0.905; mixed-0008 A 0.797, B 0.753; mixed-0015 A and B
1.000. Where the fitted smoothness moved, the spectrum lost depth; where it did not, the
two runs' components agree to five decimals.

**What the stage now does (D63).** The exchange by the orbit runs only when the two light
fractions are within a factor 3 of each other, and the table as measured is written beside
the delivered one (`velocities_unexchanged.rv`) whenever an epoch was exchanged. The
amplitudes are fitted freely when any component's smoothness precision moved more than a
factor 10 from its start, with a flag saying the light column is then a template scale and
not a fraction. A label fit that learned nothing about a component's frame offset no longer
pins that component's zero point; its velocities stay differential with their own systemic
velocity. Left for later: gating individual swaps on the detection statistic; a scale per
template with the ratio held at the declared light, reported as a template diagnostic; and
treating a table whose own Keplerian has a reduced chi-square in the tens as having no
usable semi-amplitude rather than a flagged one. The third run's tables predate these
changes.
