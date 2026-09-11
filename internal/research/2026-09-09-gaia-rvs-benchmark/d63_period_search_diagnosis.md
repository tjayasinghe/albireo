# Why the blind period search fails on sparse Gaia-like velocity tables

Diagnosis of `albireo.rvorbit.find_period` and of the candidate decision in
`albireo.pipeline._bootstrap` / `_orbit_over_candidates`, run on the oracle-tier
velocity tables of the D62 benchmark (`gaia4`, 14 systems, and `field3`, 19 systems).
Numpy and scipy only; nothing under `src/` was modified and no JAX code was run.

## How the numbers were produced, and what they were checked against

`find_period`, `fit_rv_orbit`, `reassign_by_orbit` and `_orbit_over_candidates` were
transcribed into numpy in `rvlib.py`, with the same parameterization, the same starting
values, the same 24-point conjunction scan, the same bounds and an exact analytic
Jacobian in place of the JAX one. The transcription was verified four ways
(`validate.py`):

* the semi-amplitudes from the orbit fitted at the true period on every oracle table
  reproduce the `k_A_rel` and `k_B_rel` columns of `rows.csv` to four decimals on 31 of
  the 32 usable tables (the exception is `mixed-0010`, a P = 310 d system whose fit is
  nearly degenerate). This is the load-bearing check: those columns are the pipeline's
  own `fit_rv_orbit` run on these same tables. The reduced chi-square also matches, for
  instance 7.70 against the flag "orbit from the table: reduced chi-square 7.7" recorded
  for `gaia-7289006178185856`;
* on the blind table of `gaia-50868531799454720` the transcribed `find_period` returns
  the same six peaks as the ones recorded in that star's `summary.txt`
  (3.80351, 3.1822, 5.737, 3.0834, 1.2051 d), with the top peak agreeing to five
  decimals. This one is a consistency check rather than a reproduction: the `.rv` file in
  a blind directory is the final table measured against the disentangled templates, not
  the bootstrap table measured against library templates that the recorded periodogram
  was computed from, and the two differ slightly (`v_A` spans 1.274 to 112.847 km/s in
  the written file against 2.562 to 111.284 in the bootstrap block). The peaks whose
  powers are close therefore come out in a different order;
* the analytic Jacobian agrees with a central difference to 1e-8 relative in every
  column;
* the hand-written floating-mean GLS agrees with `astropy.timeseries.LombScargle`
  (`fit_mean=True`) and with `scipy.signal.lombscargle(weights=..., floating_mean=True)`
  to 1.4e-9 in power.

One thing that was suspected and is not true: the `max_nfev=200` budget that
`fit_rv_orbit` passes to `scipy.optimize.least_squares` does not bind. Over all 640
candidate fits used below, raising it to 5000 improved the chi-square by more than 0.1%
in exactly zero cases.

One data caveat: the oracle table of `gaia-45788547559850496` has zero usable epochs.
Every epoch is flagged `at_edge` with a NaN uncertainty and the velocities are pinned at
42.000288 and 91.415314 km/s, so the oracle TODCOR search there failed outright. That
system is excluded, leaving 13 Gaia-like and 19 field systems, 32 in total. Note that
the oracle tables are cleaner than the blind ones, so the current method scores better
here (7/13 and 16/19) than the blind benchmark recorded (3/14 and 12/19). The failure
modes are the same.

---

## 1. The current method, per system

Ranks are of the true period among all peaks of the current periodogram that are
distinct by 2% in period, ordered by power, searched 600 deep. `chi2/dof chosen` is the
winning orbit of the pipeline's own candidate loop; `chi2/dof at P_true` is the orbit
fitted from the true period. "Lost at candidate generation" means no candidate offered
to the orbit fit lay within 2% of the truth; "lost at the decision" means one did and
the chi-square preferred something else.

### 1.1 Gaia-like population (`gaia4`)

| system | P_true (d) | e | K1+K2 | n_ep | rank P | rank P/2 | rank 2P | top 6 | P chosen (d) | chi2/dof chosen | chi2/dof at P_true | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaia-7289006178185856 | 0.5440 | 0.21 | 261 | 10 | 54 | 18 | 88 | no | 0.2141 | 4.2 | 7.7 | lost at the decision |
| gaia-17170287112790656 | 0.5786 | 0.00 | 296 | 11 | 15 | 23 | 68 | no | 0.5786 | 0.8 | 0.8 | recovered |
| gaia-51574864941234048 | 0.7234 | 0.13 | 28 | 14 | 15 | 82 | 57 | no | 1.3473 | 3.0 | 2.4 | lost at candidate generation |
| gaia-40041022325608704 | 2.8086 | 0.01 | 205 | 12 | 1 | 20 | 10 | yes | 2.8086 | 0.7 | 0.7 | recovered |
| gaia-45354171749336960 | 3.8268 | 0.00 | 150 | 23 | 1 | 59 | 18 | yes | 3.8268 | 0.9 | 0.9 | recovered |
| gaia-56716765427964800 | 5.3561 | 0.09 | 128 | 10 | 103 | 26 | 172 | no | 3.0935 | 19.4 | 0.5 | lost at candidate generation |
| gaia-45788547559850496 | 5.6093 | 0.01 | 152 | 0/15 | - | - | - | - | - | - | - | table unusable |
| gaia-37608387208382848 | 6.1037 | 0.03 | 137 | 15 | 1 | 211 | 123 | yes | 6.1032 | 1.0 | 1.0 | recovered |
| gaia-56861694804053120 | 6.8142 | 0.42 | 144 | 15 | 1 | 28 | 149 | yes | 6.8142 | 1.2 | 1.2 | recovered |
| gaia-47620265212420096 | 7.6752 | 0.00 | 53 | 13 | 1 | 49 | 93 | yes | 7.6747 | 1.3 | 1.3 | recovered |
| gaia-50868531799454720 | 8.8642 | 0.05 | 108 | 16 | 9 | 39 | 32 | no | 3.0829 | 72.1 | 0.6 | lost at candidate generation |
| gaia-51884824140205824 | 17.7539 | 0.48 | 103 | 22 | 9 | 47 | 180 | no | 2.9889 | 647.3 | 2.9 | lost at candidate generation |
| gaia-53290511099783296 | 18.8885 | 0.41 | 69 | 16 | 2 | 159 | 32 | yes | 18.8880 | 0.7 | 0.7 | recovered |
| gaia-53680017390963072 | 29.6049 | 0.47 | 199 | 17 | 25 | 192 | 179 | no | 39.3127 | 47.0 | 0.6 | lost at candidate generation |

Recovered 7 of 13. The true period is in the top six peaks in 6 of 13, and taking P/2 or
2P into account adds nothing (still 6 of 13). In five of the six failures the truth was
never a candidate.

### 1.2 Field population (`field3`)

| system | P_true (d) | e | K1+K2 | n_ep | rank P | rank P/2 | rank 2P | top 6 | P chosen (d) | chi2/dof chosen | chi2/dof at P_true | outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed-0003 | 1.3226 | 0.00 | 273 | 50 | 1 | 148 | 2 | yes | 1.3226 | 0.9 | 0.9 | recovered |
| mixed-0008 | 2.0895 | 0.00 | 72 | 38 | 1 | 34 | 78 | yes | 2.0895 | 4.5 | 4.5 | recovered |
| mixed-0002 | 3.5735 | 0.00 | 193 | 28 | 1 | 54 | 5 | yes | 3.5735 | 0.9 | 0.9 | recovered |
| mixed-0014 | 3.9498 | 0.00 | 129 | 45 | 1 | 36 | 236 | yes | 3.9498 | 0.8 | 0.8 | recovered |
| mixed-0018 | 3.9864 | 0.00 | 104 | 70 | 1 | 8 | 112 | yes | 3.9863 | 1.1 | 1.0 | recovered |
| mixed-0000 | 5.0242 | 0.00 | 137 | 45 | 1 | 9 | 66 | yes | 5.0242 | 0.8 | 0.8 | recovered |
| mixed-0009 | 6.4820 | 0.00 | 134 | 34 | 1 | 90 | 169 | yes | 6.4820 | 1.0 | 1.0 | recovered |
| mixed-0012 | 7.3746 | 0.00 | 119 | 98 | 1 | 129 | 107 | yes | 7.3746 | 1.1 | 1.1 | recovered |
| mixed-0006 | 19.1633 | 0.17 | 96 | 53 | 1 | 160 | 197 | yes | 19.1635 | 0.9 | 0.9 | recovered |
| mixed-0016 | 23.6586 | 0.64 | 120 | 49 | 1 | 23 | 235 | yes | 23.6580 | 2.6 | 2.6 | recovered |
| mixed-0007 | 45.3942 | 0.50 | 68 | 38 | 1 | 32 | 184 | yes | 45.3813 | 3.3 | 3.3 | recovered |
| mixed-0004 | 116.8901 | 0.72 | 65 | 80 | 1 | 167 | 67 | yes | 116.6981 | 1080.5 | 1156.7 | recovered |
| mixed-0011 | 133.2333 | 0.39 | 63 | 24 | 2 | 191 | 192 | yes | 133.1421 | 1.2 | 1.2 | recovered |
| mixed-0013 | 221.9622 | 0.66 | 60 | 43 | 1 | 22 | >600 | yes | 222.2207 | 1.2 | 1.2 | recovered |
| mixed-0017 | 308.6882 | 0.67 | 51 | 22 | 46 | >600 | >600 | no | 3.8605 | 16.8 | 0.8 | lost at candidate generation |
| mixed-0010 | 310.0558 | 0.70 | 43 | 62 | >600 | 1 | >600 | no | 311.7403 | 4.8 | 5.6 | recovered |
| mixed-0019 | 330.2300 | 0.58 | 45 | 11 | >600 | >600 | >600 | no | 9.4220 | 0.6 | 6.0 | lost at candidate generation |
| mixed-0015 | 552.5478 | 0.48 | 34 | 26 | 1 | >600 | >600 | yes | 552.2049 | 8.0 | 8.0 | recovered |
| mixed-0005 | 844.3146 | 0.62 | 40 | 29 | >600 | >600 | >600 | no | 827.0872 | 0.3 | 0.3 | lost at candidate generation |

Recovered 16 of 19. The true period is in the top six peaks in 15 of 19, and 16 of 19
counting P/2 or 2P. All three failures are candidate-generation failures. `mixed-0005`
is a boundary case: the chosen 827.087 d is 2.04% from the truth and the two fits are
indistinguishable in chi-square (0.3 per degree of freedom either way), so it is counted
as a miss only because the tolerance is 2%.

Note the two rows where the chosen orbit fits *better* than the truth
(`gaia-7289006178185856`, 4.2 against 7.7; `mixed-0019`, 0.6 against 6.0). These are the
only two systems in the whole set where the periodogram and the chi-square agree on the
wrong answer, and both have tables the true Keplerian fits badly (a 100 km/s vsini pair
at P = 0.54 d, and an 11-epoch table at P = 330 d).

---

## 2. Alternatives

All counts are over the same tables, with the truth's rank measured in the same greedy
2%-distinct peak list.

### 2a-b. Which periodogram (grid 10 per 1/T)

| periodogram | Gaia: rank 1 / top 6 / top 20 | Field: rank 1 / top 6 / top 20 |
|---|---|---|
| current, `scipy.signal.lombscargle` of `(y - weighted mean) * sqrt(w)` | 5 / 6 / 10 | 14 / 15 / 15 |
| classical LS of `y - weighted mean`, no `sqrt(w)` scaling | 5 / 7 / 10 | 14 / 15 / 15 |
| floating-mean GLS, equal weights | 9 / 11 / 12 | 13 / 15 / 15 |
| floating-mean GLS, weights `1/sigma^2` (Zechmeister & Kurster 2009) | 9 / 10 / 12 | 13 / 15 / 15 |
| two components jointly, one zero point each (summed chi-square drop) | 9 / 10 / 12 | 15 / 15 / 15 |
| mean of the two components' independent GLS powers | 9 / 10 / 12 | 14 / 14 / 14 |
| component A alone, floating-mean GLS | 9 / 10 / 12 | 15 / 16 / 16 |
| floating mean plus first harmonic, weighted | 8 / 12 / 12 | 13 / 15 / 15 |
| floating-mean GLS of `|v_A - v_B|` (swap invariant) | 0 / 1 / 4 | 4 / 7 / 8 |

Allowing P/2 or 2P instead of P changes the GLS row to 9 / 10 / 13 (Gaia) and
14 / 16 / 16 (field). Both extra systems enter at P/2: `gaia-7289006178185856` at rank 15
against 32 for P itself, and `mixed-0010` at rank 1 against no peak within 2% of P at
all. The swap-invariant row is the only one for which the halving matters structurally,
as expected, and it recovers most of its ground once P/2 is allowed: 6 / 6 / 10 (Gaia)
and 13 / 16 / 16 (field).

The point-wise `y * sqrt(w)` scaling is a genuine error but not the one that costs
periods here. It fits a sinusoid whose amplitude is modulated by `sigma_i` rather than
performing a weighted fit of a constant-amplitude sinusoid, and because the weighted
mean is removed before the scaling the transformed series no longer has zero mean.
Measured over the 32 tables, `sqrt(w)` spans a factor 1.09 to 2.27 (median 1.40 in the
Gaia set, 1.54 in the field set), and the residual mean of the transformed series is
0.1% to 7% of its rms (median 0.7%). Removing the scaling alone changes nothing:
5 / 13 and 14 / 19 truths at rank 1, exactly as before. Adding the floating mean alone,
with equal weights, moves the Gaia count from 5 to 9. The floating mean is the whole
effect; the weights and the scaling are second order on these tables.

Using the two components jointly instead of their difference is worth almost nothing.
In the Gaia set the counts are identical. In the field set the joint statistic gains one
rank-1 system over the difference (15 against 13), and component A on its own does as
well or better (15 / 16 / 16). The difference series is not what is losing the period.

### 2c. The two-harmonic periodogram on the eccentric systems

| system | e | n_ep | GLS rank | 2-harmonic rank | GLS rank (P/2) | 2-harmonic rank (P/2) |
|---|---|---|---|---|---|---|
| gaia-56861694804053120 | 0.42 | 15 | 1 | 1 | 7 | 2 |
| gaia-51884824140205824 | 0.48 | 22 | 1 | 1 | 62 | 49 |
| gaia-53290511099783296 | 0.41 | 16 | 10 | 2 | 158 | 161 |
| gaia-53680017390963072 | 0.47 | 17 | 10 | 1 | 129 | 117 |
| mixed-0016 | 0.64 | 49 | 1 | 1 | 26 | 27 |
| mixed-0007 | 0.50 | 38 | 1 | 1 | 30 | 38 |
| mixed-0004 | 0.72 | 80 | 1 | 1 | 169 | 11 |
| mixed-0011 | 0.39 | 24 | 2 | 1 | 194 | 125 |
| mixed-0013 | 0.66 | 43 | 1 | 1 | 20 | 77 |
| mixed-0017 | 0.67 | 22 | >600 | 1 | 126 | 57 |
| mixed-0010 | 0.70 | 62 | >600 | >600 | 1 | >600 |
| mixed-0019 | 0.58 | 11 | >600 | 78 | >600 | 182 |
| mixed-0015 | 0.48 | 26 | 3 | >600 | >600 | >600 |
| mixed-0005 | 0.62 | 29 | >600 | >600 | >600 | 7 |

On the ten systems with e > 0.3 and P shorter than a quarter of the baseline the second
harmonic never hurts and helps twice in the Gaia set (rank 10 to 2, rank 10 to 1) and
twice in the field set (rank 2 to 1, rank >600 to 1). Of the four field systems with
`P > 0.15 T` it is better on one, the same on two and worse on one (`mixed-0015`,
rank 3 to >600) at this grid density, and that is a grid effect (see 2d), not a model
effect. It is also slightly worse on the near-circular systems, where the second
harmonic is pure freedom: `gaia-17170287112790656` (e = 0) drops from rank 1 to 4. The
right use of it is as a second candidate source, not a replacement.

### 2d. Frequency grid density

| periodogram | grid | median points | Gaia rank 1 / top 6 / top 20 | Field rank 1 / top 6 / top 20 |
|---|---|---|---|---|
| current | 10 per 1/T | 119166 / 129725 | 5 / 6 / 10 | 14 / 15 / 15 |
| current | 20 per 1/T | 238331 / 259449 | 5 / 6 / 10 | 14 / 16 / 16 |
| current | 50 per 1/T | 595826 / 648622 | 5 / 6 / 10 | 14 / 16 / 16 |
| GLS | 10 per 1/T | 119166 / 129725 | 9 / 10 / 12 | 13 / 15 / 15 |
| GLS | 20 per 1/T | 238331 / 259449 | 9 / 10 / 12 | 13 / 16 / 16 |
| GLS | 50 per 1/T | 595826 / 648622 | 9 / 10 / 12 | 13 / 16 / 16 |
| GLS, two components jointly | 10 / 20 / 50 per 1/T | as above | 9 / 10 / 12 at all three | 15, 14, 14 rank 1 |
| two-harmonic | 10 per 1/T | 119166 / 129725 | 8 / 12 / 12 | 13 / 15 / 15 |
| two-harmonic | 20 per 1/T | 238331 / 259449 | 10 / 12 / 12 | 15 / 17 / 18 |
| two-harmonic | 50 per 1/T | 595826 / 648622 | 9 / 12 / 12 | 18 / 18 / 18 |

For the single-sinusoid periodograms 10 per 1/T is enough: no count changes in the Gaia
set at any density, and the one change in the field set (top 6 from 15 to 16) is a
long-period system. Ten per 1/T is not enough for the two-harmonic periodogram, where
the field count at rank 1 rises from 13 to 15 to 18 as the grid is refined.

The reason is arithmetic, not statistics. On a grid uniform in frequency with a step
`df = 1 / (N_os T)`, the step in period is `dP / P = P df = P / (N_os T)`. With
`N_os = 10` that is 1.6% at `P = 0.16 T` and 4.7% at `P = 0.47 T`, so for the four field
systems with `P > 0.15 T` the grid cannot represent the true period to the 2% the
benchmark asks for. That is why their rank reads `>600` at 10 per 1/T. The Keplerian
fit hides most of this, because its bounds are `0.5 P0` to `2 P0` and a start 2 to 5%
away converges to the truth: `mixed-0010` is recovered from a peak 2.23% off, and
`mixed-0005` converges to 827.09 d, 2.04% from 844.31 d, from a peak 2.44% off. The
nearest GLS local maximum to the truth is within 0.12% of it for every system with
`P / T < 0.02`, within 1.7% for `P / T < 0.1`, and 2.2 to 2.4% for `P / T > 0.15`.

The 2% distinctness rule itself is fine. Replacing it by the conventional resolution
rule, peaks separated by more than one frequency element `1 / T`, changes no outcome at
all (11/13 and 16/19 either way). At short period the resolution rule keeps far more
candidates than the 2% rule and simply fills the top twenty with near-duplicates of the
same alias family: for `gaia-7289006178185856` the truth moves from rank 32 to rank 57.

### 2e. The decision rule, over the top 20 GLS candidates

`GLS rank` is the rank of the truth on the periodogram; `expected null max` is
`1 - M^(-2/(n-3))` with `M = T (f_max - f_min)` independent frequencies, the power a
pure-noise series of `n` points is expected to reach somewhere on this grid. `chi2 gap`
is the eccentric chi-square at the truth minus the eccentric chi-square at the best
alias, so a negative number means the truth wins and by how much.

#### Gaia-like

| system | P_true (d) | e | n_ep | GLS rank | power at P_true | best power | expected null max | eccentric chi2 | circular chi2 | BIC | chi2 gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaia-7289006178185856 | 0.5440 | 0.21 | 10 | 32 | 0.745 | 0.973 | 0.932 | 0.375 d | 0.331 d | 0.375 d | n/a |
| gaia-17170287112790656 | 0.5786 | 0.00 | 11 | 1 | 0.981 | 0.981 | 0.902 | truth | truth | truth | -240.7 |
| gaia-51574864941234048 | 0.7234 | 0.13 | 14 | 1 | 0.970 | 0.977 | 0.818 | 4.241 d | 0.461 d | 4.241 d | +4.4 |
| gaia-40041022325608704 | 2.8086 | 0.01 | 12 | 1 | 0.971 | 0.971 | 0.870 | truth | truth | truth | -196.5 |
| gaia-45354171749336960 | 3.8268 | 0.00 | 23 | 1 | 0.994 | 0.994 | 0.610 | truth | truth | truth | -47292.1 |
| gaia-56716765427964800 | 5.3561 | 0.09 | 10 | 3 | 0.989 | 0.991 | 0.930 | truth | truth | truth | -51.6 |
| gaia-37608387208382848 | 6.1037 | 0.03 | 15 | 1 | 0.997 | 0.997 | 0.786 | truth | truth | truth | -226.0 |
| gaia-56861694804053120 | 6.8142 | 0.42 | 15 | 1 | 0.972 | 0.972 | 0.794 | truth | 3.191 d | truth | -16069.6 |
| gaia-47620265212420096 | 7.6752 | 0.00 | 13 | 1 | 1.000 | 1.000 | 0.838 | truth | truth | truth | -839.3 |
| gaia-50868531799454720 | 8.8642 | 0.05 | 16 | 1 | 0.997 | 0.998 | 0.764 | truth | truth | truth | -1655.0 |
| gaia-51884824140205824 | 17.7539 | 0.48 | 22 | 1 | 0.866 | 0.880 | 0.632 | truth | truth | truth | -40386.5 |
| gaia-53290511099783296 | 18.8885 | 0.41 | 16 | 10 | 0.820 | 0.874 | 0.762 | truth | 0.566 d | truth | -202.9 |
| gaia-53680017390963072 | 29.6049 | 0.47 | 17 | 10 | 0.817 | 0.937 | 0.740 | truth | 8.511 d | truth | -1763.2 |

#### Field

| system | P_true (d) | e | n_ep | GLS rank | power at P_true | best power | expected null max | eccentric chi2 | circular chi2 | BIC | chi2 gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed-0003 | 1.3226 | 0.00 | 50 | 1 | 0.998 | 0.998 | 0.330 | truth | truth | truth | -11173.7 |
| mixed-0008 | 2.0895 | 0.00 | 38 | 1 | 0.990 | 0.990 | 0.419 | truth | truth | truth | -1698.9 |
| mixed-0002 | 3.5735 | 0.00 | 28 | 1 | 0.992 | 0.992 | 0.530 | truth | truth | truth | -5641.4 |
| mixed-0014 | 3.9498 | 0.00 | 45 | 1 | 0.998 | 0.998 | 0.364 | truth | truth | truth | -63317.1 |
| mixed-0018 | 3.9864 | 0.00 | 70 | 1 | 0.998 | 0.998 | 0.247 | truth | truth | truth | -163821.1 |
| mixed-0000 | 5.0242 | 0.00 | 45 | 1 | 1.000 | 1.000 | 0.364 | truth | truth | truth | -84702.8 |
| mixed-0009 | 6.4820 | 0.00 | 34 | 1 | 0.992 | 0.992 | 0.454 | truth | truth | truth | -5975.9 |
| mixed-0012 | 7.3746 | 0.00 | 98 | 1 | 0.997 | 0.997 | 0.181 | truth | truth | truth | -85741.1 |
| mixed-0006 | 19.1633 | 0.17 | 53 | 1 | 0.975 | 0.975 | 0.316 | truth | truth | truth | -15242.4 |
| mixed-0016 | 23.6586 | 0.64 | 49 | 1 | 0.680 | 0.680 | 0.338 | truth | 26.390 d | truth | -12651.1 |
| mixed-0007 | 45.3942 | 0.50 | 38 | 1 | 0.689 | 0.689 | 0.419 | truth | 2.227 d | truth | -442.3 |
| mixed-0004 | 116.8901 | 0.72 | 80 | 1 | 0.580 | 0.597 | 0.218 | truth | truth | truth | -112978.7 |
| mixed-0011 | 133.2333 | 0.39 | 24 | 2 | 0.800 | 0.892 | 0.588 | truth | 6.151 d | truth | -636.7 |
| mixed-0013 | 221.9622 | 0.66 | 43 | 1 | 0.706 | 0.706 | 0.377 | truth | 3.655 d | truth | -1845.9 |
| mixed-0017 | 308.6882 | 0.67 | 22 | >600 | 0.597 | 0.883 | 0.630 | 67.787 d | 67.933 d | 67.787 d | n/a |
| mixed-0010 | 310.0558 | 0.70 | 62 | >600 | 0.418 | 0.482 | 0.275 | truth | 0.636 d | truth | -152.4 |
| mixed-0019 | 330.2300 | 0.58 | 11 | >600 | 0.930 | 0.998 | 0.904 | 25.547 d | 25.538 d | 25.547 d | +12.1 |
| mixed-0015 | 552.5478 | 0.48 | 26 | 3 | 0.657 | 0.704 | 0.554 | truth | truth | truth | -368.8 |
| mixed-0005 | 844.3146 | 0.62 | 29 | >600 | 0.727 | 0.841 | 0.515 | 827.087 d | 815.042 d | 827.087 d | n/a |

| decision statistic over the top 20 GLS candidates | Gaia | Field |
|---|---|---|
| the truth is reachable (some candidate's fit lands within 2% of it) | 12 / 13 | 17 / 19 |
| eccentric chi-square picks it | 11 / 13 | 16 / 19 |
| circular chi-square picks it | 8 / 13 | 11 / 19 |
| BIC over both model orders picks it | 11 / 13 | 16 / 19 |
| same, candidates extended by each peak's half and double | 11 / 13 (reachable 13) | 16 / 19 (reachable 17) |
| the same three statistics on the *current* periodogram's top 20 (eccentric / circular / BIC) | 9 / 7 / 9 of 13 | 16 / 12 / 16 of 19 |

Beaten on the periodogram against beaten on the orbit fit, over both populations:

| | GLS top 20 | current periodogram top 20 |
|---|---|---|
| an alias has more power than the truth | 10 of 32 | 13 of 32 |
| the truth is unreachable from any of the 20 candidates | 3 of 32 | 5 of 32 |
| an alias has a lower eccentric chi-square than the truth, given the truth is reachable | 2 of 29 | 2 of 27 |

### 2f. End to end: what each change is worth

Every configuration below runs the pipeline's own loop (eccentric fit at each candidate,
`reassign_by_orbit`, lowest chi-square wins), changing only the candidate list.

| configuration | Gaia | Field | total |
|---|---|---|---|
| A. current periodogram, top 6, plus the swap-invariant branch (the pipeline today) | 7 / 13 | 16 / 19 | 23 / 32 |
| B. current periodogram, top 6, no swap-invariant branch | 6 / 13 | 16 / 19 | 22 / 32 |
| C. current periodogram, top 20, no swap-invariant branch | 9 / 13 | 16 / 19 | 25 / 32 |
| D. GLS, top 6, no swap-invariant branch | 10 / 13 | 16 / 19 | 26 / 32 |
| E. GLS, top 20, no swap-invariant branch | 11 / 13 | 16 / 19 | 27 / 32 |
| F. GLS, top 20, plus the swap-invariant branch | 11 / 13 | 17 / 19 | 28 / 32 |

Adding the two-harmonic periodogram as a second candidate source, and varying the grid
it is computed on:

| configuration | Gaia | Field | total | median periodogram cost |
|---|---|---|---|---|
| P. GLS at 10 per 1/T, top 20 | 11 / 13 | 16 / 19 | 27 / 32 | 0.08 s, 0.26 s |
| Q. P plus two-harmonic at 10 per 1/T, top 20 | 11 / 13 | 17 / 19 | 28 / 32 | 0.45 s, 1.22 s |
| R. GLS at 20 per 1/T plus two-harmonic at 50 per 1/T, top 20 each | 11 / 13 | 17 / 19 | 28 / 32 | 2.03 s, 5.27 s |
| G. Q plus the swap-invariant branch (recommended) | 11 / 13 | 17 / 19 | 28 / 32 | 0.53 s, 1.66 s |

Configuration G fits a median of 37 distinct candidates in 0.8 to 1.0 s. Note that the
refined grid of configuration R buys nothing end to end even though it is worth five
systems on the rank statistic: the Keplerian fit refines the period from a start a few
per cent away, so grid coarseness costs ranks, not periods.

The four residual misses of configuration G are `gaia-7289006178185856` and `mixed-0019`,
where the truth fits the table worse than the alias that wins and no decision rule on
that table can help; `mixed-0005`, where the chosen 827.09 d is 2.04% from the truth and
the two fits are indistinguishable; and `gaia-51574864941234048` (P = 0.723 d,
K1 + K2 = 28.5 km/s, 14 epochs), where the truth is rank 1 on the periodogram and loses
the chi-square comparison by 4.4. That last one is the only remaining genuine decision
failure in either population.

---

## 3. Conclusions

### (i) Where the period is lost, and why

It is lost at candidate generation, not at the decision. Of the nine systems the current
route misses on these tables, eight never had a candidate within 2% of the truth and
only one (`gaia-7289006178185856`) had one and lost it to a better-fitting alias. The
same asymmetry holds when the candidate list is widened: with twenty GLS candidates an
alias outranks the truth on the periodogram in 10 of 32 systems but outfits it in only 2
of the 29 in which the truth is reachable.

The dominant cause is that `find_period` computes a classical Lomb-Scargle periodogram
with the offset held at zero. `scipy.signal.lombscargle` fits `a cos(wt) + b sin(wt)`
and nothing else, so the model has no constant term. That is harmless for dense, evenly
spread sampling, where `<cos wt>` and `<sin wt>` are near zero at every frequency of
interest. It is not harmless for a Gaia scanning-law table: 10 to 25 epochs fall into
about 11 visibility windows separated by hundreds of days, the sampling window has a
large mean at most frequencies, and the constant the model cannot fit is absorbed into
the sinusoid. The spurious power that follows is what buries the truth. In the five Gaia
candidate-generation failures the true period ranks 9, 9, 15, 25 and 103 among the
distinct peaks; on a floating-mean periodogram of the same series the same five ranks
are 1, 1, 1, 10 and 3.

The other two suspects were tested and are not the cause. Removing the point-wise
`y * sqrt(w)` scaling, which is separately wrong (it fits a sinusoid of amplitude
proportional to `sigma_i` and leaves a non-zero mean in the transformed series, measured
at 0.1 to 7% of its rms), changes the rank-1 count by zero in both populations. Refining
the frequency grid from 10 to 50 per 1/T changes the rank-1 count by zero for every
single-sinusoid periodogram in both populations.

The second cause is the size of the candidate list. Six peaks is too few even for the
current periodogram: taking twenty instead of six raises the Gaia recovery from 7 to 9
of 13 with no other change.

A third, smaller cause bites only at `P > 0.15 T`. The grid step in period is
`dP / P = P / (N_os T)`, which at `N_os = 10` exceeds the 2% recovery tolerance for
`P > 0.2 T`, and the single-sinusoid peak of a strongly eccentric long-period orbit is
in any case displaced by 2 to 3% from the truth. The Keplerian fit rescues most of these
because its period bounds are `0.5 P0` to `2 P0`.

### (ii) Which periodogram, and how many candidates

The floating-mean, weighted generalized Lomb-Scargle periodogram of Zechmeister and
Kurster (2009) on the same difference series, with twenty candidates. That puts the
truth at rank 1 in 9 of 13 Gaia and 13 of 19 field systems, in the top six in 10 and 15,
and in the top twenty in 12 and 15; allowing P/2 or 2P, the top twenty covers 13 of 13
and 16 of 19. End to end it lifts the recovery from 23 of 32 to 27 with the twenty GLS
peaks alone and to 28 with the swap-invariant branch kept, and the two parts of the
change are separable: the periodogram is worth about three systems and the widened list
about two.

Using both components jointly instead of their difference is not worth doing. It is
identical in the Gaia set and gains one rank-1 system in the field set, where component
A on its own does as well. The swap-invariant branch is worth keeping: it costs eight
extra fits and rescues one field system (`mixed-0017`) that no other candidate source
reaches.

The one alternative that adds genuinely new information is the two-harmonic
floating-mean periodogram, and only for eccentric orbits. It moves two Gaia systems at
e = 0.41 and 0.47 from rank 10 to rank 2 and 1, and on a grid of 50 per 1/T it puts 18
of 19 field truths at rank 1 against 13 for the single sinusoid. It is slightly worse on
circular systems, so it should be a second candidate source unioned with the first, not
a replacement. Unioned at the same 10 per 1/T grid it rescues one further field system
and brings the whole route to 28 of 32; the finer grid, which is worth five systems on
the rank statistic, is worth nothing on the outcome, because the Keplerian fit refines
the period from a start a few per cent away.

### (iii) How much the table chi-square can discriminate

More than the periodogram, by a wide margin, and enough to carry a list of twenty
candidates. Over both populations, when the truth was reachable from some candidate the
eccentric chi-square picked it in 27 of 29 cases, and the two losses were by 4.4 and
12.1 in chi-square. When it won it usually won by hundreds to more than 10^5. That is
the argument for generating many candidates and letting the fit decide, rather than
trying to sharpen the peak.

The model order matters. Holding the orbit circular costs 3 systems in the Gaia set and
5 in the field set, and it fails exactly on the eccentric ones, where a circular fit at
the true period is a poorer description than an eccentric fit at an alias. A
BIC-penalised choice between the circular and the eccentric fit changes no decision in
either population: the chi-square differences involved are two to five orders of
magnitude larger than the `2 ln N` penalty of about 7 that the two eccentricity
parameters carry.

The false-alarm arithmetic explains why the periodogram cannot be trusted and why the
chi-square can. With `T (f_max - f_min)` of about 12,000 independent frequencies, a
series of `n` points reaches a maximum GLS power of about `1 - M^(-2/(n-3))` on noise
alone: 0.93 at n = 10, 0.82 at n = 14, 0.76 at n = 16, 0.63 at n = 22, 0.25 at n = 70.
The measured power at the true period in the Gaia set runs from 0.75 to 1.00. For the
10-epoch tables the true peak therefore stands only hundredths above what pure noise
produces somewhere on the grid, and a sinusoid explaining 90% of the variance is
unremarkable. The chi-square of a Keplerian, by contrast, uses the shape of the curve
and all `2n` velocities at once, and the same tables separate the truth from the best
alias by hundreds.

The limit of this is visible in the two systems where both statistics agree on the wrong
answer. `gaia-7289006178185856` (P = 0.544 d, 10 epochs, vsini near 100 km/s) fits
better at 0.375 d than at the truth, and `mixed-0019` (P = 330 d, 11 epochs) fits better
at 25.5 d. When a table has 10 or 11 epochs and its true Keplerian leaves a reduced
chi-square of 6 to 8, no decision rule on that table can find the period, and the honest
output is an ambiguity flag rather than a period.

### (iv) Recommended changes

In `albireo.rvorbit.find_period`:

1. Replace the `scipy.signal.lombscargle` call with a floating-mean, weighted GLS. With
   `W = w / sum(w)`, `Y = sum(W y)`, `YY = sum(W (y - Y)^2)` and, at each angular
   frequency `w = 2 pi f`,

   ```
   C  = sum(W cos)          S  = sum(W sin)
   YC = sum(W y cos) - Y C  YS = sum(W y sin) - Y S
   CC = sum(W cos^2) - C^2  SS = sum(W sin^2) - S^2  CS = sum(W cos sin) - C S
   p  = (SS YC^2 + CC YS^2 - 2 CS YC YS) / (YY (CC SS - CS^2))
   ```

   That is about twenty-five lines of numpy, needs no new dependency and no change to
   the declared `scipy>=1.11` floor. Recent scipy offers the same thing directly as
   `lombscargle(t, y, 2*pi*f, weights=w/w.sum(), floating_mean=True, normalize=True)`;
   the installed 1.18 has it and it agrees with the hand-written version to 1e-9, but
   the declared floor does not, so writing it out is the safer choice. Drop the
   `y * sqrt(w)` scaling: the weights belong in `W`, not in the data.

2. Keep the grid uniform in frequency and keep `10` per `1/T`. It was tested at 10, 20
   and 50 for every periodogram: no count changed for the single-sinusoid ones, and
   although refining it is worth five systems on the two-harmonic rank statistic in the
   field set, it changed no end-to-end outcome (configurations Q and R above are both
   28 of 32, at four times the cost). Record the arithmetic in the docstring anyway,
   because it governs how precisely a returned peak can be quoted: the step in period is
   `dP/P = P/(N_os T)`, which at `N_os = 10` is 4.7% at `P = 0.47 T`, so a peak reported
   near a third of the baseline is only good to a few per cent and the Keplerian fit,
   not the grid, is what pins it down.

3. Keep the 2% distinctness rule. The conventional alternative, one frequency resolution
   element `1/T`, was tested and changed no outcome in either population while making
   the short-period lists worse by filling them with near-duplicates.

4. Return twenty peaks rather than six. Change `if len(peaks) >= 6` to a parameter
   `n_peaks=20` and return `aliases` of length `n_peaks - 1`. Restrict the greedy loop
   to the local maxima of the power array first: an accepted peak is always the highest
   point in its own exclusion window, so the accepted list is unchanged, but the loop
   then walks a few thousand local maxima instead of 120,000 grid points, each against
   every peak already accepted.

   Leave the default period range alone. Raising the floor from twice the shortest gap
   (0.148 d, set by the 106.5 minute pair spacing) to 0.5 d was tested: it moves the
   truth of `gaia-7289006178185856` from rank 32 to rank 2 and changes nothing else in
   either population, and it would buy that at the cost of an astrophysical assumption
   the search should not make.

5. Add `n_harmonics=1` and, for `n_harmonics=2`, fit
   `c + a1 cos(wt) + b1 sin(wt) + a2 cos(2wt) + b2 sin(2wt)` by weighted least squares
   and report `1 - chi2/chi2_null`. Five parameters against `n` points is safe down to
   about ten epochs on this evidence. It is not a replacement for `n_harmonics=1`: it is
   worse on the circular systems, so both lists are needed.

In `albireo.pipeline._bootstrap`:

6. Build the candidate list as the union of three sources, all on the same 10 per 1/T
   grid: the top twenty peaks of the floating-mean GLS of `v_A - v_B`; the top twenty
   peaks of the two-harmonic periodogram of the same series; and, unchanged, the first
   four peaks of the swap-invariant periodogram together with their doubles. That is 48
   starting periods, which the existing 2% test deduplicates to a median of 37. Measured
   end to end this is configuration G above, 28 of 32 against the current 23 of 32, at
   0.5 to 1.7 s of periodogram and about 1 s of fitting per system. Do not extend the
   list by each peak's half and double: it raises the number of systems whose truth is
   reachable from 12 to 13 in the Gaia set and changes no decision, because every extra
   candidate is also one more chance for an alias to win.

7. Leave the decision as it is: fit an eccentric Keplerian at every candidate and keep
   the lowest chi-square. Do not add a circular fit or a BIC; both were tested and
   neither helps, and the circular fit alone is much worse.

8. Deduplicate on the *fitted* period rather than on the starting period, and widen the
   ambiguity flag. The present flag fires only on the single runner-up and only within
   `delta chi2 < 9`. With twenty or more candidates the useful statement is the number
   of distinct fitted periods within, say, `delta chi2 < 25` of the winner, since a
   bootstrap that hands a `+-3%` prior to the disentangling is unrecoverable if it is
   wrong.

9. Note in the docstring that a table of 10 or 11 epochs whose true Keplerian leaves a
   reduced chi-square above about 5 cannot be searched at all, and that `period =
   "search"` on such a table should be reported as a failure rather than as a period.

The cost of all of this is small. On these tables the GLS at 10 per 1/T takes 0.08 s per
Gaia-like system and 0.26 s per field system, the three periodograms of configuration G
together 0.53 and 1.66 s, and the 37 Keplerian fits that follow 0.8 and 1.0 s, against
the several minutes the disentangling that follows already takes.

---

## Files

All scripts and outputs are in this directory.

| file | what it is |
|---|---|
| `rvlib.py` | the numpy transcription: reader, Kepler model and analytic Jacobian, periodograms, peak list, orbit fit, re-assignment, candidate loop |
| `validate.py` | the four checks against the recorded pipeline output, astropy and scipy |
| `task1.py`, `task1.json` | Task 1, the current method per system |
| `task2.py`, `task2.json` | Task 2a-d, nine periodograms at three grid densities |
| `task2e.py`, `task2e_gls.json`, `task2e_current.json` | Task 2e, the decision rule over the top 20 |
| `task2f.py`, `task2f.json` | period-range restriction and nearest-peak offsets |
| `task2g.py`, `task2g.json` | the 2% distinctness rule against a `1/T` resolution rule |
| `task2h.py`, `task2h.json` | the proposed candidate generator, end to end |
| `ablation.py`, `ablation.json` | configurations A to F |
| `report.py`, `report.txt` | the tables above |
| `weights.json` | the `sqrt(w)` spread and the leaked mean per table |
