# albireo on Gaia RVS double-lined binaries: Gaia DR3 orbits (D63)

Generated 2026-09-11T12:01:49+00:00 by albireo 0.1.0.dev0 on Windows 11, AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, 32 threads. Product `dr4-epoch` (DR4 epoch-spectrum sampling: 0.025 nm, 961 samples from 846.0 to 870.0 nm); library `bosz2024-fgk-rvs`; cadence `scanning-law`; resolving power `nominal`; model grid 3.0 km/s; K prior 2.0 to 250.0 km/s; eccentricity to 0.9; 300 disentangling and 80 label steps; noise model AR(1) with the delivered grid's lag-one correlation; seed 0.

Every system was simulated once with albireo's Gaia RVS model (photon noise per detector pixel at the S/N that follows from G_RVS, one transit per epoch, delivered on the archive grid with its correlated noise) and run through the pipeline under each tier; the numbers below come from the pipeline's own comparison against the injected truth. A pull is a difference over the quoted error; a calibrated error gives a pull rms near one.

## The population

```
14 systems (4 eclipsing), sources Gaia DR3 nss_two_body_orbit
  P [d]        quartiles 3.06 / 5.86 / 8.57
  e            quartiles 0.0133 / 0.0683 / 0.358
  K1 + K2 [km/s] quartiles 105 / 141 / 187
  q = M2/M1    quartiles 0.932 / 0.964 / 0.978
  F2/F1 (RVS)  quartiles 0.758 / 0.853 / 0.957
  Teff1 [K]    quartiles 5.82e+03 / 6.16e+03 / 6.73e+03
  G_RVS        quartiles 7.38 / 8.51 / 9.7
  transits     quartiles 12.2 / 15 / 16
```

14 systems drawn, 46 star runs planned over the tiers oracle, eclipsing, orbit, blind.

![the population](figures/population.png)

## Tiers

| tier | period | conjunction | elements | light fractions | label priors |
|---|---|---|---|---|---|
| oracle | known (width 0.0001) | known | known | truth | narrow |
| eclipsing | known (width 0.0001) | known | free | truth | narrow |
| orbit | known (width 0.0001) | scan | free | measure | wide |
| blind | search | scan | free | measure | wide |

## Recovery by tier

Median and the 16th to 84th percentile range over the systems that completed.

| quantity | oracle | eclipsing | orbit | blind |
|---|---|---|---|---|
| completed / planned | 14 / 14 | 4 / 4 | 14 / 14 | 14 / 14 |
| abs(dK1 / K1) [%], disentangling | 0.59 (0.198 to 3.01) | 5.85 (2.3 to 51.5) | 1.02 (0.356 to 17.3) | 5.73 (0.444 to 44) |
| abs(dK2 / K2) [%], disentangling | 0.495 (0.323 to 3.64) | 9.05 (2.84 to 39.2) | 2.16 (0.228 to 58.2) | 6.3 (1.09 to 56) |
| K1 within 1%, disentangling | 64% | 25% | 50% | 36% |
| K2 within 1%, disentangling | 79% | 0% | 43% | 14% |
| K1 within 5%, disentangling | 93% | 50% | 71% | 50% |
| K2 within 5%, disentangling | 93% | 50% | 79% | 50% |
| abs(dK1 / K1) [%], velocity table | 0.527 (0.226 to 2.79) | 4.79 (1.08 to 31) | 0.591 (0.328 to 6.57) | 1.39 (0.261 to 32.4) |
| abs(dK2 / K2) [%], velocity table | 1.01 (0.214 to 2.79) | 7.84 (2.42 to 56.2) | 0.898 (0.289 to 21.9) | 2.22 (0.196 to 58.8) |
| K1 within 1%, velocity table | 64% | 25% | 64% | 43% |
| K1 within 5%, velocity table | 86% | 50% | 79% | 57% |
| K2 within 5%, velocity table | 86% | 50% | 79% | 57% |
| K1 pull rms, velocity table | 1.37 | 2.25 | 3 | 1.13 |
| K2 pull rms, velocity table | 6.97 | 22.7 | 19.5 | 1.45 |
| K1 pull within 1, velocity table | 71% | 25% | 43% | 64% |
| K2 pull within 2, velocity table | 86% | 25% | 64% | 79% |
| components recovered in the other order | 0% | 0% | 7% | 14% |
| abs(dP / P) from the table | 4.52e-06 (9.38e-07 to 5.93e-05) | 1.59e-05 (9.26e-06 to 7.97e-05) | 1.12e-05 (3.38e-06 to 7.81e-05) | 0.0385 (3.46e-06 to 0.885) |
| abs(de) | 0.00341 (0.000783 to 0.0109) | 0.043 (0.00808 to 0.229) | 0.00896 (0.000553 to 0.0541) | 0.0745 (0.00072 to 0.598) |
| abs(d omega) [deg] | 0.818 (0.0379 to 15.3) | 25.1 (8.7 to 41.6) | 4.11 (1.18 to 108) | 63 (1.26 to 169) |
| abs(d t_conj) [phase] | 0.00266 (0.000116 to 0.0104) | 0.0114 (0.00221 to 0.257) | 0.00285 (0.00101 to 0.285) | 0.171 (0.00148 to 0.386) |
| abs(d gamma) A [km/s] | 0.22 (0.0916 to 0.48) | 2.15 (0.266 to 6.81) | 0.263 (0.114 to 2.33) | 0.466 (0.0922 to 1.28) |
| epoch velocity rms, A [km/s] | 1.11 (0.454 to 5.07) | 4.64 (2.45 to 77.8) | 1.1 (0.47 to 8.48) | 4.26 (0.442 to 44.3) |
| epoch velocity rms, B [km/s] | 1.97 (0.387 to 9.2) | 14.1 (4.09 to 64.6) | 2.2 (0.459 to 41.7) | 9.22 (0.796 to 77.5) |
| epoch velocity pull rms, A | 1.12 (0.954 to 1.68) | 0.985 (0.824 to 1.38) | 1.28 (0.863 to 2.48) | 3.67 (0.932 to 25.9) |
| epoch velocity pull rms, B | 0.965 (0.697 to 1.71) | 1.23 (0.955 to 11.1) | 1.12 (0.907 to 4.8) | 3.59 (1.06 to 26.4) |
| secondary detected per epoch (dchi2 > 25) | 1 (1 to 1) | 1 (0.905 to 1) | 1 (1 to 1) | 1 (1 to 1) |
| spectrum correlation, A | 0.893 (0.66 to 0.946) | 0.864 (0.444 to 0.965) | 0.858 (0.608 to 0.945) | 0.895 (0.644 to 0.962) |
| spectrum correlation, B | 0.887 (0.54 to 0.95) | 0.861 (0.676 to 0.906) | 0.859 (0.541 to 0.95) | 0.803 (0.506 to 0.918) |
| spectrum pull rms, A | 3.02 (1.13 to 5.14) | 2.2 (1.38 to 6.98) | 3.88 (1.69 to 5.22) | 3.2 (1.17 to 5.42) |
| EW ratio, metal windows, A | 1.21 (0.992 to 1.37) | 0.611 (0.0269 to 1.43) | 0.793 (0.0536 to 1.35) | 0.859 (0.57 to 1.26) |
| EW ratio, Ca II triplet, A | 1.05 (0.955 to 1.1) | 0.93 (0.379 to 1.17) | 0.93 (0.657 to 1.1) | 0.978 (0.843 to 1.17) |
| EW ratio, metal windows, B | 0.73 (0.0387 to 0.997) | 1.19 (0.093 to 2.13) | 1.24 (0.052 to 2.29) | 1.02 (0.154 to 1.38) |
| abs(dTeff) A [K] | 132 (12.2 to 203) | 258 (141 to 272) | 136 (53.6 to 228) | 186 (31 to 282) |
| abs(dTeff) B [K] | 108 (63.4 to 228) | 166 (47.7 to 272) | 328 (67.5 to 782) | 320 (122 to 626) |
| abs(dlog g) A | 0.15 (0.0709 to 0.286) | 0.243 (0.165 to 0.298) | 0.239 (0.0847 to 0.768) | 0.191 (0.0751 to 0.697) |
| abs(dvsini) A [km/s] | 6.98 (2.11 to 15) | 14.2 (5.85 to 23) | 10.9 (3.74 to 16.2) | 11.7 (5.11 to 19) |
| abs(dlight) from the label fit, A | 0.0108 (0.00154 to 0.0479) | 0.0484 (0.0294 to 0.292) | 0.0248 (0.005 to 0.169) | 0.0243 (0.00356 to 0.217) |
| abs(dlight) as declared, A | 0 (0 to 0) | 0 (0 to 0) | 0.0378 (0.027 to 0.205) | 0.0378 (0.027 to 0.205) |
| residual z rms | 0.986 (0.973 to 0.991) | 0.992 (0.99 to 1) | 0.986 (0.973 to 0.999) | 0.991 (0.982 to 1.01) |
| period search recovered within 2% | n/a | n/a | n/a | 50% of 14 |
| wall per star [s] | 403 (336 to 474) | 580 (521 to 1.96e+03) | 654 (495 to 900) | 671 (617 to 772) |

![k_recovery](figures/k_recovery.png)

Relative error of the semi-amplitudes against G_RVS, from the Keplerian of the disentangling (above) and from the orbit fitted to the velocity table (below); the dotted and dashed lines mark 1% and 5%.

![k_vs_transits_separation](figures/k_vs_transits_separation.png)

The semi-amplitudes of the disentangling against the number of transits and against the largest velocity separation in units of the line-spread FWHM (26 km/s).

![pulls](figures/pulls.png)

Calibration: the semi-amplitude pulls against a unit normal, and the per-system rms of the epoch-velocity pulls against one.

![velocities_spectra_labels](figures/velocities_spectra_labels.png)

Epoch-velocity precision against the per-epoch S/N, the recovered primary spectrum's correlation with the truth against the combined S/N, and the primary's temperature error.

![period_search](figures/period_search.png)

The blind tier's period search.

## Epoch velocities

One velocity per component per epoch, measured by TODCOR against the disentangled components with the label fit's zero points, is a product of every star run (`velocities.rv` and `velocities.csv` in its directory). `velocities.csv` in this directory gathers them all with the injected velocity of each epoch, one row per system, tier, epoch and component; the statistics below pool the usable epochs of every system in the tier.

| quantity | oracle | eclipsing | orbit | blind |
|---|---|---|---|---|
| systems, epochs measured | 14, 215 | 4, 50 | 14, 215 | 14, 215 |
| epochs usable | 96% | 98% | 96% | 97% |
| rms residual, A [km/s] | 3.02 | 61.9 | 40.2 | 34.8 |
| rms residual, B [km/s] | 8.96 | 45.6 | 32.3 | 43.2 |
| median abs residual, A [km/s] | 0.644 | 3.1 | 0.589 | 0.957 |
| median abs residual, B [km/s] | 0.786 | 7.12 | 0.833 | 1.52 |
| median quoted error, A [km/s] | 0.51 | 3.16 | 0.593 | 0.511 |
| median quoted error, B [km/s] | 1.41 | 5 | 1.2 | 1.41 |
| pull rms, A | 4.03 | 1.12 | 5.95 | 19.3 |
| pull rms, B | 2.92 | 8.64 | 6.32 | 19.7 |
| within 3 sigma, A | 92% | 100% | 86% | 70% |
| within 3 sigma, B | 91% | 82% | 83% | 69% |

![rv_curves_oracle](figures/rv_curves_oracle.png)

![rv_curves_eclipsing](figures/rv_curves_eclipsing.png)

![rv_curves_orbit](figures/rv_curves_orbit.png)

![rv_curves_blind](figures/rv_curves_blind.png)

Phase-folded measured velocities (circles the primary, squares the secondary; filled where the table calls the epoch usable, open where it flags it) against the injected orbit, one panel per system, per tier. A pair recovered in the other order is drawn against the exchanged truth and marked.

## Failures and flags

- **oracle**: 0 failed, 0 not run.
    - 4x labels learned nothing about logg_A, logg_B, teff_A, teff_B (posterior width >= 80% of the prior)
    - 1x labels learned nothing about teff_B (posterior width >= 80% of the prior)
    - 1x B is weakly detected (delta chi2 < 25) in 2 epoch(s)
    - 1x 1 of 25 epochs had their components re-assigned by the disentangling's orbit
    - 1x 2 of 25 epochs unusable in the velocity table (2 blended, 0 at the search edge)
    - 1x labels learned nothing about logg_A, teff_A (posterior width >= 80% of the prior)
    - 1x labels learned nothing about teff_B, v_B, vsini_B (posterior width >= 80% of the prior)
    - 1x the label fit measures a light fraction of 1.00 for 'A' against the declared 0.60
    - 1x 3 of 14 epochs unusable in the velocity table (0 blended, 3 at the search edge)
    - 1x B is weakly detected (delta chi2 < 25) in 7 epoch(s)
    - 1x 1 of 24 epochs had their components re-assigned by the disentangling's orbit
    - 1x 2 of 24 epochs unusable in the velocity table (2 blended, 0 at the search edge)
- **eclipsing**: 0 failed, 0 not run.
    - 1x the semi-amplitude scan moved K_A from 132.9 to 245.0 km/s, K_B from 106.4 to 12.1 km/s (+36 nats)
    - 1x labels learned nothing about teff_B, v_A, vsini_A (posterior width >= 80% of the prior)
    - 1x the label fit measures a light fraction of 0.00 for 'A' against the declared 0.52
    - 1x 1 of 10 epochs unusable in the velocity table (0 blended, 1 at the search edge)
    - 1x A is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 1x K_A from the velocity table (60.88 km/s) disagrees with the disentangling (246.30) by 185.42
    - 1x K_B from the velocity table (4.91 km/s) disagrees with the disentangling (49.00) by 44.09
    - 1x B is weakly detected (delta chi2 < 25) in 2 epoch(s)
    - 1x labels learned nothing about logg_A, logg_B, teff_A, teff_B (posterior width >= 80% of the prior)
- **orbit**: 0 failed, 0 not run.
    - 14x light fractions measured by correlation against library templates rather than declared
    - 2x orbit from the table
    - 1x the label fit measures a light fraction of 0.13 for 'A' against the declared 0.54
    - 1x labels
    - 1x template zero points refused, so the velocities stay differential (one gamma per component in the orbit fit)
    - 1x A is weakly detected (delta chi2 < 25) in 2 epoch(s)
    - 1x velocities are differential
    - 1x no candidate orbit lies inside the declared ranges; the lowest chi-square one is taken as it stands
    - 1x the semi-amplitude scan moved K_B from 51.6 to 219.2 km/s (+62 nats)
    - 1x the label fit measures a light fraction of 0.96 for 'A' against the declared 0.60
    - 1x B is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 1x 1 of 25 epochs had their components re-assigned by the disentangling's orbit
- **blind**: 0 failed, 0 not run.
    - 17x light fractions measured by correlation against library templates rather than declared
    - 10x bootstrap
    - 6x orbit from the table
    - 2x template zero points refused, so the velocities stay differential (one gamma per component in the orbit fit)
    - 2x velocities are differential
    - 2x components recovered in the other order
    - 1x the label fit measures a light fraction of 0.15 for 'A' against the declared 0.54
    - 1x labels
    - 1x 1 of 10 epochs unusable in the velocity table (0 blended, 1 at the search edge)
    - 1x the lowest chi-square candidate orbit (P 0.3098 d, K [313.4, 0.0] km/s, e 0.66) lies outside the declared ranges (K up to 250, e up to 0.9) and was set aside
    - 1x labels learned nothing about vsini_B (posterior width >= 80% of the prior)
    - 1x the label fit measures a light fraction of 0.97 for 'A' against the declared 0.60

## Files

- `rows.csv`: one line per star run with the truth and every metric.
- `velocities.csv`: every measured epoch velocity with its error, flags, the injected velocity and the pull.
- `summary.json`: the per-tier statistics tabulated above.
- `population.json`, `manifest.json`: the systems and the settings.
- one directory per star with the pipeline's `summary.txt`, `result.json`, tables, spectra and figures.
