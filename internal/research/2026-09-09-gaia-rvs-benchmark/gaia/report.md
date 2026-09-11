# albireo on Gaia RVS double-lined binaries: Gaia DR3 orbits (D62)

Generated 2026-09-10T18:17:42+00:00 by albireo 0.1.0.dev0 on Windows 11, AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, 32 threads. Product `dr4-epoch` (DR4 epoch-spectrum sampling: 0.025 nm, 961 samples from 846.0 to 870.0 nm); library `bosz2024-fgk-rvs`; cadence `scanning-law`; resolving power `nominal`; model grid 3.0 km/s; K prior 2.0 to 250.0 km/s; eccentricity to 0.9; 100 disentangling and 80 label steps; noise model AR(1) with the delivered grid's lag-one correlation; seed 0.

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
| abs(dK1 / K1) [%], disentangling | 0.453 (0.23 to 3.04) | 2.96 (1.38 to 6.44) | 3.47 (0.316 to 34.4) | 24.1 (3.8 to 110) |
| abs(dK2 / K2) [%], disentangling | 0.358 (0.0323 to 1.06) | 13.8 (5.19 to 21.6) | 4.23 (0.565 to 88.2) | 24 (3.46 to 150) |
| K1 within 1%, disentangling | 79% | 25% | 29% | 7% |
| K2 within 1%, disentangling | 71% | 0% | 21% | 0% |
| K1 within 5%, disentangling | 93% | 75% | 64% | 21% |
| K2 within 5%, disentangling | 100% | 25% | 50% | 21% |
| abs(dK1 / K1) [%], velocity table | 0.62 (0.309 to 7.26) | 1.88 (1.01 to 5.35) | 0.908 (0.178 to 25.1) | 27.2 (1.46 to 86.1) |
| abs(dK2 / K2) [%], velocity table | 0.774 (0.123 to 1.88) | 11.3 (4.38 to 19.3) | 1.91 (0.092 to 82.2) | 25.4 (2.24 to 282) |
| K1 within 1%, velocity table | 69% | 25% | 50% | 21% |
| K1 within 5%, velocity table | 77% | 75% | 71% | 21% |
| K2 within 5%, velocity table | 92% | 25% | 71% | 21% |
| K1 pull rms, velocity table | 1.62 | 4.39 | 6.32 | 5.4 |
| K2 pull rms, velocity table | 0.835 | 12.5 | 26.7 | 5.27 |
| K1 pull within 1, velocity table | 62% | 25% | 50% | 29% |
| K2 pull within 2, velocity table | 100% | 0% | 71% | 43% |
| components recovered in the other order | 0% | 0% | 7% | 14% |
| abs(dP / P) from the table | 1.66e-05 (1.18e-06 to 3.15e-05) | 1.08e-05 (6.69e-06 to 3.89e-05) | 2.51e-05 (2.85e-06 to 6.47e-05) | 0.453 (0.00616 to 0.828) |
| abs(de) | 0.00468 (0.000428 to 0.0275) | 0.0113 (0.00578 to 0.0936) | 0.011 (0.00134 to 0.0555) | 0.204 (0.0069 to 0.699) |
| abs(d omega) [deg] | 0.734 (0.635 to 29.7) | 24.5 (11.1 to 37.9) | 4.68 (0.725 to 174) | 27.7 (22.1 to 148) |
| abs(d t_conj) [phase] | 0.00224 (0.000397 to 0.0211) | 0.00652 (0.00237 to 0.02) | 0.00461 (0.00211 to 0.242) | 0.29 (0.029 to 0.398) |
| abs(d gamma) A [km/s] | 0.209 (0.0769 to 1.43) | 2.45 (1.39 to 2.59) | 0.254 (0.0476 to 1.01) | 0.45 (0.0615 to 9.35) |
| epoch velocity rms, A [km/s] | 1.22 (0.471 to 3.33) | 3.51 (1.7 to 9.86) | 1.95 (0.361 to 8.05) | 20.2 (0.583 to 50) |
| epoch velocity rms, B [km/s] | 2.85 (0.371 to 4.76) | 13.5 (4.96 to 22.1) | 2.99 (0.518 to 46.1) | 31.5 (1.41 to 133) |
| epoch velocity pull rms, A | 1.14 (0.942 to 3.16) | 2.36 (1.04 to 5.73) | 1.49 (0.946 to 6.52) | 15.9 (1.17 to 47.7) |
| epoch velocity pull rms, B | 0.936 (0.871 to 1.9) | 2.79 (1.63 to 8.41) | 1.32 (1.13 to 14.5) | 17.3 (1.33 to 62.5) |
| secondary detected per epoch (dchi2 > 25) | 1 (1 to 1) | 1 (0.953 to 1) | 1 (1 to 1) | 1 (1 to 1) |
| spectrum correlation, A | 0.92 (0.61 to 0.947) | 0.821 (0.765 to 0.902) | 0.863 (0.661 to 0.942) | 0.908 (0.666 to 0.967) |
| spectrum correlation, B | 0.857 (0.547 to 0.95) | 0.807 (0.664 to 0.836) | 0.634 (0.533 to 0.905) | 0.612 (0.458 to 0.926) |
| spectrum pull rms, A | 2.82 (1.13 to 4.58) | 0.712 (0.639 to 1.24) | 2.76 (0.901 to 4.12) | 3.6 (1.74 to 5.03) |
| EW ratio, metal windows, A | 1.18 (0.153 to 1.59) | 1.1 (1.03 to 1.37) | 1.08 (0.455 to 1.3) | 0.894 (-0.136 to 1.3) |
| EW ratio, Ca II triplet, A | 1.05 (0.818 to 1.12) | 1.03 (0.969 to 1.12) | 0.978 (0.826 to 1.29) | 0.919 (0.588 to 1.28) |
| EW ratio, metal windows, B | 0.745 (0.125 to 1.97) | 0.59 (0.155 to 0.937) | 0.845 (-0.0285 to 1.71) | 1.08 (0.0044 to 2.35) |
| abs(dTeff) A [K] | 196 (44.6 to 248) | 27.5 (11.5 to 133) | 166 (63 to 260) | 155 (59.2 to 363) |
| abs(dTeff) B [K] | 122 (61.6 to 265) | 73.9 (47.5 to 175) | 260 (105 to 621) | 432 (130 to 652) |
| abs(dlog g) A | 0.162 (0.0711 to 0.273) | 0.197 (0.165 to 0.254) | 0.16 (0.0758 to 0.408) | 0.239 (0.0841 to 0.523) |
| abs(dvsini) A [km/s] | 7.03 (3.22 to 21.7) | 1.78 (0.958 to 4.32) | 7.47 (2.44 to 20.7) | 8.57 (0.514 to 26.1) |
| abs(dlight) from the label fit, A | 0.00951 (0.00188 to 0.0381) | 0.0293 (0.0166 to 0.0348) | 0.0161 (0.0051 to 0.127) | 0.0199 (0.00665 to 0.23) |
| abs(dlight) as declared, A | 0 (0 to 0) | 0 (0 to 0) | 0.0378 (0.027 to 0.205) | 0.0378 (0.027 to 0.205) |
| residual z rms | 0.984 (0.971 to 0.992) | 0.966 (0.962 to 0.969) | 0.971 (0.966 to 0.99) | 1 (0.98 to 1.04) |
| period search recovered within 2% | n/a | n/a | n/a | 21% of 14 |
| wall per star [s] | 413 (374 to 500) | 609 (583 to 639) | 581 (518 to 651) | 551 (363 to 653) |

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
| epochs usable | 90% | 100% | 98% | 98% |
| rms residual, A [km/s] | 18.5 | 6.9 | 27.5 | 46.5 |
| rms residual, B [km/s] | 17.8 | 15.1 | 38.2 | 68.2 |
| median abs residual, A [km/s] | 0.614 | 1.59 | 0.654 | 2.16 |
| median abs residual, B [km/s] | 0.776 | 8.67 | 0.931 | 2.23 |
| median quoted error, A [km/s] | 0.695 | 1.61 | 0.532 | 0.719 |
| median quoted error, B [km/s] | 0.928 | 2.33 | 1.06 | 1.11 |
| pull rms, A | 2.43 | 3.92 | 13.7 | 33 |
| pull rms, B | 2.36 | 6.26 | 15.3 | 38 |
| within 3 sigma, A | 93% | 78% | 81% | 59% |
| within 3 sigma, B | 95% | 56% | 79% | 57% |

![rv_curves_oracle](figures/rv_curves_oracle.png)

![rv_curves_eclipsing](figures/rv_curves_eclipsing.png)

![rv_curves_orbit](figures/rv_curves_orbit.png)

![rv_curves_blind](figures/rv_curves_blind.png)

Phase-folded measured velocities (circles the primary, squares the secondary; filled where the table calls the epoch usable, open where it flags it) against the injected orbit, one panel per system, per tier. A pair recovered in the other order is drawn against the exchanged truth and marked.

## Failures and flags

- **oracle**: 0 failed, 0 not run.
    - 3x labels learned nothing about logg_A, logg_B, teff_A, teff_B (posterior width >= 80% of the prior)
    - 3x labels learned nothing about logg_A, teff_A (posterior width >= 80% of the prior)
    - 1x orbit from the table
    - 1x smoothness of 'B' did not move from its start
    - 1x labels learned nothing about logg_A, logg_B, mh, teff_A, teff_B (posterior width >= 80% of the prior)
    - 1x B is weakly detected (delta chi2 < 25) in 2 epoch(s)
    - 1x 1 of 25 epochs had their components re-assigned by the disentangling's orbit
    - 1x 2 of 25 epochs unusable in the velocity table (2 blended, 0 at the search edge)
    - 1x residual z-score rms 45.047
    - 1x labels
    - 1x 15 of 15 epochs unusable in the velocity table (7 blended, 15 at the search edge)
    - 1x orbit from the table skipped
- **eclipsing**: 0 failed, 0 not run.
    - 1x smoothness of 'B' did not move from its start
    - 1x the semi-amplitude scan moved K_A from 152.1 to 88.9 km/s (+205 nats)
    - 1x labels learned nothing about logg_A, mh, teff_A (posterior width >= 80% of the prior)
    - 1x B is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 1x labels learned nothing about logg_A (posterior width >= 80% of the prior)
    - 1x the semi-amplitude scan moved K_A from 99.5 to 19.2 km/s, K_B from 102.2 to 41.3 km/s (+293 nats)
    - 1x labels learned nothing about logg_B (posterior width >= 80% of the prior)
- **orbit**: 0 failed, 0 not run.
    - 14x light fractions measured by correlation against library templates rather than declared
    - 3x orbit from the table
    - 2x labels learned nothing about v_B, vsini_B (posterior width >= 80% of the prior)
    - 1x labels learned nothing about logg_A, logg_B, mh, teff_B, v_A, vsini_A (posterior width >= 80% of the prior)
    - 1x labels
    - 1x the semi-amplitude scan moved K_B from 167.3 to 22.8 km/s (+43 nats)
    - 1x the label fit measures a light fraction of 0.80 for 'A' against the declared 0.60
    - 1x the semi-amplitude scan moved K_A from 27.8 to 62.9 km/s, K_B from 27.7 to 74.8 km/s (+1069 nats)
    - 1x labels learned nothing about logg_A, logg_B, mh, teff_A, teff_B (posterior width >= 80% of the prior)
    - 1x smoothness of 'A' did not move from its start
    - 1x smoothness of 'B' did not move from its start
    - 1x 2 of 25 epochs unusable in the velocity table (2 blended, 0 at the search edge)
- **blind**: 0 failed, 0 not run.
    - 14x light fractions measured by correlation against library templates rather than declared
    - 10x orbit from the table
    - 9x bootstrap
    - 2x components recovered in the other order
    - 1x the label fit measures a light fraction of 0.96 for 'A' against the declared 0.60
    - 1x 1 of 11 epochs unusable in the velocity table (0 blended, 1 at the search edge)
    - 1x bootstrap semi-amplitudes [524.7, 507.2] km/s fall outside the declared range 2-250 for ['A', 'B']
    - 1x the semi-amplitude scan moved K_A from 84.7 to 7.0 km/s (+955 nats)
    - 1x labels learned nothing about mh, v_B, vsini_B (posterior width >= 80% of the prior)
    - 1x the label fit measures a light fraction of 0.92 for 'A' against the declared 0.54
    - 1x 2 of 25 epochs unusable in the velocity table (2 blended, 0 at the search edge)
    - 1x 1 of 13 epochs had their components re-assigned by the disentangling's orbit

## Files

- `rows.csv`: one line per star run with the truth and every metric.
- `velocities.csv`: every measured epoch velocity with its error, flags, the injected velocity and the pull.
- `summary.json`: the per-tier statistics tabulated above.
- `population.json`, `manifest.json`: the systems and the settings.
- one directory per star with the pipeline's `summary.txt`, `result.json`, tables, spectra and figures.
