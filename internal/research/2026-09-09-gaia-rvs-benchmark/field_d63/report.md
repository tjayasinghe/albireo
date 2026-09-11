# albireo on Gaia RVS double-lined binaries: field population (D63)

Generated 2026-09-11T13:53:52+00:00 by albireo 0.1.0.dev0 on Windows 11, AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, 32 threads. Product `dr4-epoch` (DR4 epoch-spectrum sampling: 0.025 nm, 961 samples from 846.0 to 870.0 nm); library `bosz2024-fgk-rvs`; cadence `scanning-law`; resolving power `nominal`; model grid 3.0 km/s; K prior 2.0 to 250.0 km/s; eccentricity to 0.9; 300 disentangling and 80 label steps; noise model AR(1) with the delivered grid's lag-one correlation; seed 0.

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
| abs(dK1 / K1) [%], disentangling | 1.47 (0.119 to 20.1) | 0.924 (0.8 to 1.05) | 1.48 (0.288 to 73.4) | 4.87 (0.298 to 51.3) |
| abs(dK2 / K2) [%], disentangling | 0.977 (0.255 to 6) | 0.204 (0.175 to 0.232) | 1.75 (0.209 to 36.1) | 9.92 (0.425 to 78.7) |
| K1 within 1%, disentangling | 47% | 50% | 42% | 32% |
| K2 within 1%, disentangling | 53% | 100% | 37% | 26% |
| K1 within 5%, disentangling | 58% | 100% | 58% | 53% |
| K2 within 5%, disentangling | 74% | 100% | 58% | 42% |
| abs(dK1 / K1) [%], velocity table | 2.73 (0.261 to 56.1) | 0.853 (0.512 to 1.19) | 4.58 (0.385 to 29) | 5.83 (0.358 to 74.2) |
| abs(dK2 / K2) [%], velocity table | 1.05 (0.185 to 19.3) | 0.0927 (0.0624 to 0.123) | 6.93 (0.203 to 78.6) | 9.23 (0.227 to 72.2) |
| K1 within 1%, velocity table | 37% | 50% | 42% | 37% |
| K1 within 5%, velocity table | 58% | 100% | 53% | 47% |
| K2 within 5%, velocity table | 63% | 100% | 47% | 47% |
| K1 pull rms, velocity table | 6.96 | 1.77 | 7.68 | 6.54 |
| K2 pull rms, velocity table | 4.52 | 0.18 | 5.5 | 5.45 |
| K1 pull within 1, velocity table | 26% | 50% | 32% | 26% |
| K2 pull within 2, velocity table | 63% | 100% | 47% | 58% |
| components recovered in the other order | 0% | 0% | 16% | 16% |
| abs(dP / P) from the table | 2.41e-05 (3.81e-06 to 0.00207) | 5.6e-06 (2.43e-06 to 8.77e-06) | 1.71e-05 (3.14e-06 to 0.00456) | 0.000747 (6.12e-06 to 0.999) |
| abs(de) | 0.00336 (0.0018 to 0.0781) | 0.0123 (0.0113 to 0.0134) | 0.00413 (0.00154 to 0.0534) | 0.0438 (0.0018 to 0.212) |
| abs(d omega) [deg] | 3.32 (1.54 to 16.7) | n/a | 24.4 (2.2 to 176) | 77.5 (5.07 to 179) |
| abs(d t_conj) [phase] | 0.00347 (0.000979 to 0.0384) | 0.000854 (0.000438 to 0.00127) | 0.00923 (0.000515 to 0.252) | 0.0951 (0.00124 to 0.356) |
| abs(d gamma) A [km/s] | 0.241 (0.0646 to 0.483) | 0.139 (0.0482 to 0.231) | 0.116 (0.0268 to 0.707) | 0.272 (0.101 to 1.09) |
| epoch velocity rms, A [km/s] | 2.21 (1.03 to 7.84) | 2.58 (2.35 to 2.82) | 2.98 (0.899 to 8.62) | 4.62 (1.16 to 12.9) |
| epoch velocity rms, B [km/s] | 2.69 (1.11 to 13.4) | 2.74 (2.42 to 3.05) | 2.74 (0.853 to 16.4) | 6.2 (1.29 to 32.8) |
| epoch velocity pull rms, A | 1.37 (0.973 to 2.81) | 1.01 (0.916 to 1.1) | 1.23 (0.984 to 3.12) | 1.9 (1.1 to 8.47) |
| epoch velocity pull rms, B | 1.05 (0.915 to 2.91) | 0.976 (0.956 to 0.996) | 1.32 (0.935 to 7.32) | 1.77 (0.926 to 11.9) |
| secondary detected per epoch (dchi2 > 25) | 1 (0.72 to 1) | 0.991 (0.984 to 0.997) | 1 (0.813 to 1) | 1 (0.772 to 1) |
| spectrum correlation, A | 0.939 (0.701 to 0.989) | 0.954 (0.942 to 0.966) | 0.919 (0.686 to 0.988) | 0.907 (0.691 to 0.972) |
| spectrum correlation, B | 0.786 (0.502 to 0.961) | 0.94 (0.923 to 0.958) | 0.851 (0.477 to 0.957) | 0.785 (0.495 to 0.949) |
| spectrum pull rms, A | 2.23 (1.72 to 4.28) | 0.887 (0.681 to 1.09) | 2.37 (1.18 to 4.78) | 2.87 (1.88 to 4.99) |
| EW ratio, metal windows, A | 1.38 (1.15 to 1.74) | 1.2 (1.14 to 1.27) | 0.885 (0.266 to 1.33) | 1.16 (0.0146 to 1.33) |
| EW ratio, Ca II triplet, A | 1.11 (1.04 to 1.38) | 1.03 (1.02 to 1.05) | 0.938 (0.763 to 1.24) | 0.981 (0.745 to 1.24) |
| EW ratio, metal windows, B | 0.0493 (-0.214 to 0.645) | 0.743 (0.677 to 0.809) | 1.11 (-0.00865 to 1.88) | 0.651 (-0.0951 to 1.93) |
| abs(dTeff) A [K] | 124 (79.2 to 170) | 117 (91.7 to 142) | 150 (64.4 to 363) | 256 (137 to 592) |
| abs(dTeff) B [K] | 187 (59 to 236) | 215 (200 to 230) | 412 (170 to 865) | 430 (117 to 1.01e+03) |
| abs(dlog g) A | 0.182 (0.104 to 0.238) | 0.178 (0.176 to 0.181) | 0.193 (0.0731 to 0.286) | 0.276 (0.0985 to 0.732) |
| abs(dvsini) A [km/s] | 6.7 (3.2 to 20.1) | 4.75 (3.74 to 5.75) | 6.08 (2.01 to 21) | 8.33 (2.98 to 23.1) |
| abs(dlight) from the label fit, A | 0.0236 (0.00382 to 0.134) | 0.0179 (0.0154 to 0.0205) | 0.0331 (0.00948 to 0.175) | 0.0378 (0.0101 to 0.231) |
| abs(dlight) as declared, A | 0 (0 to 0) | 0 (0 to 0) | 0.0391 (0.0188 to 0.158) | 0.0391 (0.0188 to 0.158) |
| residual z rms | 0.993 (0.99 to 0.995) | 0.986 (0.986 to 0.987) | 0.992 (0.988 to 0.995) | 0.994 (0.989 to 1.03) |
| period search recovered within 2% | n/a | n/a | n/a | 58% of 19 |
| wall per star [s] | 506 (197 to 945) | 779 (707 to 850) | 908 (573 to 1.58e+03) | 757 (579 to 2.73e+03) |

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
| epochs usable | 97% | 100% | 98% | 96% |
| rms residual, A [km/s] | 4.84 | 2.71 | 8 | 36 |
| rms residual, B [km/s] | 9.72 | 2.91 | 13.2 | 39 |
| median abs residual, A [km/s] | 0.91 | 1.8 | 0.947 | 1.18 |
| median abs residual, B [km/s] | 1.21 | 1.98 | 1.33 | 1.97 |
| median quoted error, A [km/s] | 1.04 | 3.22 | 1.04 | 1.12 |
| median quoted error, B [km/s] | 1.21 | 3.05 | 1.9 | 1.94 |
| pull rms, A | 1.78 | 0.974 | 2.85 | 12.2 |
| pull rms, B | 36.2 | 0.986 | 36.3 | 38.5 |
| within 3 sigma, A | 91% | 100% | 85% | 76% |
| within 3 sigma, B | 92% | 100% | 86% | 77% |

![rv_curves_oracle](figures/rv_curves_oracle.png)

![rv_curves_eclipsing](figures/rv_curves_eclipsing.png)

![rv_curves_orbit](figures/rv_curves_orbit.png)

![rv_curves_blind](figures/rv_curves_blind.png)

Phase-folded measured velocities (circles the primary, squares the secondary; filled where the table calls the epoch usable, open where it flags it) against the injected orbit, one panel per system, per tier. A pair recovered in the other order is drawn against the exchanged truth and marked.

## Failures and flags

- **oracle**: 0 failed, 0 not run.
    - 5x labels learned nothing about logg_A, logg_B, teff_A, teff_B (posterior width >= 80% of the prior)
    - 4x orbit from the table
    - 2x labels learned nothing about logg_B, teff_B (posterior width >= 80% of the prior)
    - 2x labels learned nothing about logg_A, teff_A (posterior width >= 80% of the prior)
    - 2x B is weakly detected (delta chi2 < 25) in 26 epoch(s)
    - 2x A is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 1x 2 of 47 epochs unusable in the velocity table (2 blended, 0 at the search edge)
    - 1x labels learned nothing about logg_B, teff_A, teff_B (posterior width >= 80% of the prior)
    - 1x 3 of 53 epochs unusable in the velocity table (3 blended, 0 at the search edge)
    - 1x A is weakly detected (delta chi2 < 25) in 3 epoch(s)
    - 1x B is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 1x 19 of 80 epochs had their components re-assigned by the disentangling's orbit
- **eclipsing**: 0 failed, 0 not run.
    - 1x labels learned nothing about logg_B, teff_A, teff_B (posterior width >= 80% of the prior)
    - 1x labels
    - 1x labels learned nothing about logg_A, logg_B, mh, teff_A, teff_B (posterior width >= 80% of the prior)
    - 1x A is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 1x B is weakly detected (delta chi2 < 25) in 1 epoch(s)
- **orbit**: 0 failed, 0 not run.
    - 19x light fractions measured by correlation against library templates rather than declared
    - 5x orbit from the table
    - 3x A is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 3x components recovered in the other order
    - 3x labels learned nothing about logg_A, mh, teff_A (posterior width >= 80% of the prior)
    - 2x labels learned nothing about v_B, vsini_B (posterior width >= 80% of the prior)
    - 1x 2 of 47 epochs unusable in the velocity table (2 blended, 0 at the search edge)
    - 1x B is weakly detected (delta chi2 < 25) in 1 epoch(s)
    - 1x 19 of 80 epochs had their components re-assigned by the disentangling's orbit
    - 1x 10 of 80 epochs unusable in the velocity table (0 blended, 10 at the search edge)
    - 1x A is weakly detected (delta chi2 < 25) in 7 epoch(s)
    - 1x K_A from the velocity table (10.18 km/s) disagrees with the disentangling (22.59) by 12.41
- **blind**: 0 failed, 0 not run.
    - 20x light fractions measured by correlation against library templates rather than declared
    - 18x bootstrap
    - 8x orbit from the table
    - 3x components recovered in the other order
    - 2x A is weakly detected (delta chi2 < 25) in 7 epoch(s)
    - 2x B is weakly detected (delta chi2 < 25) in 27 epoch(s)
    - 2x period ambiguous
    - 1x labels learned nothing about logg_B, teff_B (posterior width >= 80% of the prior)
    - 1x 2 of 47 epochs unusable in the velocity table (2 blended, 0 at the search edge)
    - 1x 9 of 53 epochs had their components re-assigned by the disentangling's orbit
    - 1x 9 of 53 epochs unusable in the velocity table (9 blended, 0 at the search edge)
    - 1x B is weakly detected (delta chi2 < 25) in 7 epoch(s)

## Files

- `rows.csv`: one line per star run with the truth and every metric.
- `velocities.csv`: every measured epoch velocity with its error, flags, the injected velocity and the pull.
- `summary.json`: the per-tier statistics tabulated above.
- `population.json`, `manifest.json`: the systems and the settings.
- one directory per star with the pipeline's `summary.txt`, `result.json`, tables, spectra and figures.
