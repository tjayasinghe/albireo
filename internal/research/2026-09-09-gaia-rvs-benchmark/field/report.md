# albireo on Gaia RVS double-lined binaries: the field population (D62)

Generated 2026-09-10T20:37:26+00:00 by albireo 0.1.0.dev0 on Windows 11, AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, 32 threads. Product `dr4-epoch` (DR4 epoch-spectrum sampling: 0.025 nm, 961 samples from 846.0 to 870.0 nm); library `bosz2024-fgk-rvs`; cadence `scanning-law`; resolving power `nominal`; model grid 3.0 km/s; K prior 2.0 to 250.0 km/s; eccentricity to 0.9; 100 disentangling and 80 label steps; noise model AR(1) with the delivered grid's lag-one correlation; seed 0.

Every system was simulated once with albireo's Gaia RVS model (photon noise per detector pixel at the S/N that follows from G_RVS, one transit per epoch, delivered on the archive grid with its correlated noise) and run through the pipeline under each tier; the numbers below come from the pipeline's own comparison against the injected truth. A pull is a difference over the quoted error; a calibrated error gives a pull rms near one.

## The population

```
20 systems (2 eclipsing), sources parametric
  P [d]        quartiles 4.76 / 34.5 / 267
  e            quartiles 0 / 0.434 / 0.649
  K1 + K2 [km/s] quartiles 57.4 / 69.8 / 122
  q = M2/M1    quartiles 0.872 / 0.942 / 0.976
  F2/F1 (RVS)  quartiles 0.618 / 0.805 / 0.887
  Teff1 [K]    quartiles 6.11e+03 / 6.4e+03 / 6.71e+03
  G_RVS        quartiles 8.46 / 9.24 / 10.1
  transits     quartiles 27.8 / 40.5 / 53
```

20 systems drawn, 59 star runs planned over the tiers oracle, eclipsing, orbit, blind; 1 system left out for having fewer than 10 transits.

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
| completed / planned | 19 / 19 | 2 / 2 | 19 / 19 | 19 / 19 |
| abs(dK1 / K1) [%], disentangling | 0.767 (0.138 to 20.1) | 1.99 (1.06 to 2.91) | 1.85 (0.354 to 65.9) | 3.44 (0.483 to 84.4) |
| abs(dK2 / K2) [%], disentangling | 0.787 (0.222 to 4.13) | 1.23 (0.706 to 1.76) | 1.82 (0.762 to 112) | 5.37 (0.598 to 49.2) |
| K1 within 1%, disentangling | 53% | 50% | 42% | 37% |
| K2 within 1%, disentangling | 53% | 50% | 32% | 32% |
| K1 within 5%, disentangling | 58% | 100% | 63% | 53% |
| K2 within 5%, disentangling | 84% | 100% | 58% | 47% |
| abs(dK1 / K1) [%], velocity table | 2.13 (0.27 to 46.5) | 1.55 (1.03 to 2.08) | 1.21 (0.223 to 11.8) | 3.59 (0.408 to 74.5) |
| abs(dK2 / K2) [%], velocity table | 1.01 (0.171 to 42.1) | 0.89 (0.369 to 1.41) | 1.77 (0.472 to 62.3) | 14.4 (0.514 to 80.9) |
| K1 within 1%, velocity table | 37% | 50% | 47% | 37% |
| K1 within 5%, velocity table | 53% | 100% | 63% | 53% |
| K2 within 5%, velocity table | 63% | 100% | 53% | 47% |
| K1 pull rms, velocity table | 6.79 | 4.69 | 2.5 | 6.43 |
| K2 pull rms, velocity table | 5.28 | 3.22 | 6.06 | 6.69 |
| K1 pull within 1, velocity table | 26% | 0% | 42% | 42% |
| K2 pull within 2, velocity table | 63% | 50% | 47% | 47% |
| components recovered in the other order | 0% | 0% | 26% | 16% |
| abs(dP / P) from the table | 2.67e-05 (4.2e-06 to 0.00124) | 4.66e-06 (2.51e-06 to 6.81e-06) | 2.65e-05 (3.65e-06 to 0.00965) | 1.86e-05 (2.35e-06 to 0.999) |
| abs(de) | 0.0076 (0.0015 to 0.0604) | 0.0103 (0.00971 to 0.0109) | 0.00554 (0.00179 to 0.064) | 0.0154 (0.0019 to 0.266) |
| abs(d omega) [deg] | 3.32 (0.782 to 15.3) | n/a | 71.4 (0.858 to 176) | 77.7 (4.29 to 175) |
| abs(d t_conj) [phase] | 0.00149 (0.000918 to 0.0394) | 0.000686 (0.000238 to 0.00113) | 0.018 (0.000657 to 0.438) | 0.024 (0.000787 to 0.263) |
| abs(d gamma) A [km/s] | 0.25 (0.0773 to 0.704) | 0.103 (0.0826 to 0.124) | 0.156 (0.0626 to 2.06) | 0.185 (0.0414 to 1.47) |
| epoch velocity rms, A [km/s] | 2.17 (1.03 to 7.32) | 2.6 (2 to 3.21) | 1.91 (0.996 to 10.7) | 4.02 (1.08 to 10.9) |
| epoch velocity rms, B [km/s] | 2.72 (1.1 to 13.5) | 2.61 (2.06 to 3.15) | 2.73 (1.08 to 51.4) | 4.5 (1.15 to 33.7) |
| epoch velocity pull rms, A | 1.37 (0.977 to 2.79) | 1.29 (1.24 to 1.34) | 1.4 (1.06 to 3.58) | 1.34 (1.07 to 5.13) |
| epoch velocity pull rms, B | 1.1 (0.949 to 2.96) | 1.2 (1.14 to 1.25) | 1.35 (0.927 to 6.54) | 1.63 (0.98 to 34.1) |
| secondary detected per epoch (dchi2 > 25) | 1 (0.744 to 1) | 1 (1 to 1) | 1 (0.868 to 1) | 1 (0.91 to 1) |
| spectrum correlation, A | 0.939 (0.701 to 0.989) | 0.909 (0.886 to 0.932) | 0.895 (0.694 to 0.972) | 0.89 (0.616 to 0.972) |
| spectrum correlation, B | 0.782 (0.502 to 0.961) | 0.905 (0.88 to 0.931) | 0.781 (0.312 to 0.944) | 0.716 (0.286 to 0.957) |
| spectrum pull rms, A | 2.3 (1.86 to 4.27) | 0.62 (0.562 to 0.678) | 2.13 (0.785 to 4.04) | 2.53 (1.77 to 4.41) |
| EW ratio, metal windows, A | 1.5 (1.15 to 1.79) | 1.1 (1.04 to 1.15) | 1.18 (1.05 to 1.34) | 0.843 (-0.00231 to 1.33) |
| EW ratio, Ca II triplet, A | 1.17 (1.05 to 1.38) | 1.03 (1.02 to 1.04) | 1.02 (0.886 to 1.27) | 1.01 (0.777 to 1.25) |
| EW ratio, metal windows, B | 0.0472 (-0.156 to 0.453) | 0.874 (0.822 to 0.926) | 0.507 (-0.00587 to 0.914) | 1.21 (-0.0182 to 1.79) |
| abs(dTeff) A [K] | 123 (72.2 to 167) | 114 (86.9 to 141) | 128 (49.1 to 310) | 191 (72.7 to 573) |
| abs(dTeff) B [K] | 186 (59.3 to 236) | 104 (44.3 to 164) | 424 (224 to 1.04e+03) | 488 (179 to 1.54e+03) |
| abs(dlog g) A | 0.182 (0.105 to 0.243) | 0.178 (0.176 to 0.181) | 0.193 (0.074 to 0.377) | 0.282 (0.103 to 0.825) |
| abs(dvsini) A [km/s] | 7.43 (3.29 to 20.3) | 2.87 (2.46 to 3.27) | 4.66 (1.11 to 21.2) | 7.94 (2.18 to 17.9) |
| abs(dlight) from the label fit, A | 0.0249 (0.00388 to 0.133) | 0.00477 (0.00399 to 0.00556) | 0.0341 (0.00982 to 0.165) | 0.0391 (0.0114 to 0.225) |
| abs(dlight) as declared, A | 0 (0 to 0) | 0 (0 to 0) | 0.0409 (0.0243 to 0.158) | 0.0391 (0.0188 to 0.158) |
| residual z rms | 0.993 (0.989 to 0.995) | 0.979 (0.976 to 0.983) | 0.991 (0.982 to 0.994) | 0.995 (0.991 to 1.01) |
| period search recovered within 2% | n/a | n/a | n/a | 63% of 19 |
| wall per star [s] | 421 (303 to 604) | 477 (405 to 549) | 614 (421 to 916) | 580 (444 to 736) |

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
| systems, epochs measured | 19, 859 | 2, 81 | 19, 859 | 19, 859 |
| epochs usable | 98% | 100% | 99% | 98% |
| rms residual, A [km/s] | 4.27 | 3 | 5.52 | 22.3 |
| rms residual, B [km/s] | 9.67 | 2.95 | 38.9 | 40.1 |
| median abs residual, A [km/s] | 0.912 | 2.57 | 0.93 | 1.16 |
| median abs residual, B [km/s] | 1.32 | 1.85 | 1.38 | 1.93 |
| median quoted error, A [km/s] | 1.03 | 2.47 | 0.968 | 1.1 |
| median quoted error, B [km/s] | 1.22 | 2.58 | 1.76 | 2.02 |
| pull rms, A | 1.89 | 1.31 | 2.39 | 14.1 |
| pull rms, B | 36.1 | 1.22 | 40.4 | 41.8 |
| within 3 sigma, A | 90% | 99% | 86% | 80% |
| within 3 sigma, B | 91% | 100% | 84% | 80% |

![rv_curves_oracle](figures/rv_curves_oracle.png)

![rv_curves_eclipsing](figures/rv_curves_eclipsing.png)

![rv_curves_orbit](figures/rv_curves_orbit.png)

![rv_curves_blind](figures/rv_curves_blind.png)

Phase-folded measured velocities (circles the primary, squares the secondary; filled where the table calls the epoch usable, open where it flags it) against the injected orbit, one panel per system, per tier. A pair recovered in the other order is drawn against the exchanged truth and marked.

## Failures and flags

- **oracle**: 0 failed, 0 not run.
    - 4x labels learned nothing about logg_A, logg_B, teff_A, teff_B (posterior width >= 80% of the prior)
    - 4x orbit from the table
    - 3x labels learned nothing about logg_A, teff_A (posterior width >= 80% of the prior)
    - 3x A is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 2x labels learned nothing about logg_B, teff_B (posterior width >= 80% of the prior)
    - 2x labels learned nothing about logg_A, logg_B, mh, teff_A, teff_B (posterior width >= 80% of the prior)
    - 2x 1 of 38 epochs had their components re-assigned by the disentangling's orbit
    - 1x 2 of 47 epochs unusable in the velocity table (2 blended, 0 at the search edge)
    - 1x labels learned nothing about logg_B, teff_A, teff_B (posterior width >= 80% of the prior)
    - 1x 3 of 53 epochs unusable in the velocity table (3 blended, 0 at the search edge)
    - 1x A is weakly detected (delta chi2 < 25) in 3 epoch(s)
    - 1x B is weakly detected (delta chi2 < 25) in 1 epoch(s)
- **eclipsing**: 0 failed, 0 not run.
    - 1x labels learned nothing about logg_A, logg_B, mh, teff_A, teff_B (posterior width >= 80% of the prior)
    - 1x labels
    - 1x the semi-amplitude scan moved K_A from 36.5 to 135.4 km/s, K_B from 36.9 to 135.4 km/s (+3807 nats)
- **orbit**: 0 failed, 0 not run.
    - 19x light fractions measured by correlation against library templates rather than declared
    - 5x orbit from the table
    - 5x components recovered in the other order
    - 2x labels
    - 2x 1 of 38 epochs had their components re-assigned by the disentangling's orbit
    - 2x A is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 2x B is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 2x smoothness of 'B' did not move from its start
    - 1x 2 of 47 epochs unusable in the velocity table (2 blended, 0 at the search edge)
    - 1x 1 of 53 epochs had their components re-assigned by the disentangling's orbit
    - 1x 19 of 80 epochs had their components re-assigned by the disentangling's orbit
    - 1x A is weakly detected (delta chi2 < 25) in 9 epoch(s)
- **blind**: 0 failed, 0 not run.
    - 19x light fractions measured by correlation against library templates rather than declared
    - 16x bootstrap
    - 7x orbit from the table
    - 3x B is weakly detected (delta chi2 < 25) in 2 epoch(s)
    - 3x labels learned nothing about v_B, vsini_B (posterior width >= 80% of the prior)
    - 3x components recovered in the other order
    - 2x A is weakly detected (delta chi2 < 25) in 3 epoch(s)
    - 2x B is weakly detected (delta chi2 < 25) in 28 epoch(s)
    - 2x A is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 1x labels learned nothing about logg_B (posterior width >= 80% of the prior)
    - 1x 2 of 47 epochs unusable in the velocity table (2 blended, 0 at the search edge)
    - 1x 3 of 53 epochs unusable in the velocity table (3 blended, 0 at the search edge)

## Files

- `rows.csv`: one line per star run with the truth and every metric.
- `velocities.csv`: every measured epoch velocity with its error, flags, the injected velocity and the pull.
- `summary.json`: the per-tier statistics tabulated above.
- `population.json`, `manifest.json`: the systems and the settings.
- one directory per star with the pipeline's `summary.txt`, `result.json`, tables, spectra and figures.
