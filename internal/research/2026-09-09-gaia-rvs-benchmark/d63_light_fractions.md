# Measured light fractions in the D62 orbit and blind tiers

Every number below is read from the archived per-star `result.json` files under the two D62 run directories, with no re-running of the pipeline. Only the `orbit` and `blind` tiers are tabulated, because those are the two that ran with `light="measure"`; the `oracle` and `eclipsing` tiers declared the injected fractions and have nothing to compare.

## How each column is derived

The truth block of a result records differences, result minus injected, so the injected fraction of the secondary is recovered as `l_B(true) = l_B(declared) - truth["light_declared"]["B"]`. That value agrees with the population file, where `light_ratio = F2/F1` converts to `l_2 = light_ratio / (1 + light_ratio)`, to 1.39e-17 in the worst case over all 66 stars (with the two components swapped where `components_exchanged` is set). The label fit's fraction is `labels.flux_ratio.B`, which reproduces `l_B(true) + truth["light_label_fit"]["B"]` to 2.78e-17. The declared fraction is in every case exactly the one the light stage measured, so `declared` and `measured` are the same column.

The light stage keeps no per-epoch detection statistics in `result.json`: its `light` block (orbit tier) and `bootstrap` block (blind tier) record only the fractions, the usable epoch count and the templates. The columns `sig_B`, `rc2` and `R2` are therefore scraped from the light stage's own TODCOR report in `summary.txt`, and the `dchi2_B` and `blend` columns come from `result["velocities"]`, which is the later TODCOR pass that runs after the disentangling with the label-fit templates rather than the light stage itself.

One convention needs stating. In 11 of the runs the disentangling recovered the pair in the other order, and the truth block then scores the fitted B against the injected primary. The `l_B true` column follows that convention, so for those runs it is the injected primary's fraction; the `l_2 inj` column carries the injected secondary's fraction in every case, and the two differ by exactly the swap. The exchange only matters for the systems that are far from equal light, since l_1 and l_2 sum to one.

## Table

`l_B` values are fractions of the total flux. `r_dec` is declared over true, `r_lab` is label-fit over true. `K_A` and `K_B` are the disentangling's semi-amplitudes in km/s and `dK_A`, `dK_B` its errors against the injected values, also in km/s. `sig_B` is the light stage's median velocity error for B in km/s, `rc2` and `R2` its median reduced chi-square and median R-squared. `dchi2_B` is the median per-epoch detection statistic for B in the post-disentangling velocity stage and `blend` the number of epochs it flagged as blended. `exch` marks the runs in which the components came out exchanged, and `flag` the runs in which the pipeline itself raised the flag that the label fit's light fraction disagrees with the declared one.

| system | tier | l_B true | l_2 inj | l_B declared | l_B label | r_dec | r_lab | sig_B | rc2 | R2 | dchi2_B | blend | K_A | K_B | dK_A | dK_B | exch | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed-0004 | blind | 0.0520 | 0.0520 | 0.0517 | 0.0426 | 0.99 | 0.82 | 4.807 | 4.32 | 0.977 | 167 | 0/80 | 22.61 | 42.08 | -0.04 | -0.26 |  |  |
| mixed-0015 | blind | 0.0731 | 0.0731 | 0.1242 | 0.0180 | 1.70 | 0.25 | 2.963 | 1.48 | 0.925 | 38 | 0/27 | 4.26 | 12.13 | -7.76 | -9.74 |  |  |
| mixed-0008 | blind | 0.1119 | 0.1119 | 0.1425 | 0.0728 | 1.27 | 0.65 | 3.685 | 1.03 | 0.851 | 39 | 0/38 | 25.92 | 58.43 | -1.62 | +13.98 |  |  |
| mixed-0007 | blind | 0.1434 | 0.1434 | 0.1843 | 0.0939 | 1.28 | 0.65 | 6.308 | 1.01 | 0.614 | 16 | 0/38 | 25.99 | 38.56 | -0.88 | -2.19 |  |  |
| mixed-0010 | blind | 0.4194 | 0.4194 | 0.2354 | 0.1719 | 0.56 | 0.41 | 3.827 | 1.02 | 0.688 | 50 | 0/62 | 3.52 | 17.59 | -17.15 | -4.44 |  |  |
| mixed-0016 | blind | 0.4203 | 0.4203 | 0.3751 | 0.3916 | 0.89 | 0.93 | 0.804 | 1.08 | 0.941 | 1852 | 1/50 | 59.95 | 64.95 | +1.99 | +2.61 |  |  |
| mixed-0011 | blind | 0.4204 | 0.4204 | 0.3868 | 0.3630 | 0.92 | 0.86 | 1.243 | 1.24 | 0.873 | 466 | 4/24 | 38.35 | 49.03 | +8.27 | +15.77 |  |  |
| mixed-0019 | blind | 0.4242 | 0.4242 | 0.2678 | 0.4284 | 0.63 | 1.01 | 3.155 | 1.32 | 0.566 | 1192 | 0/11 | 122.72 | 184.43 | +101.15 | +160.72 |  | yes |
| mixed-0005 | blind | 0.4443 | 0.4443 | 0.2740 | 0.0064 | 0.62 | 0.01 | 4.718 | 1.02 | 0.481 | 6 | 0/29 | 14.64 | 45.47 | -4.77 | +24.55 |  | yes |
| mixed-0002 | blind | 0.4477 | 0.4477 | 0.4288 | 0.4403 | 0.96 | 0.98 | 1.370 | 1.16 | 0.753 | 932 | 0/28 | 94.72 | 100.79 | +0.85 | +1.49 |  |  |
| mixed-0000 | blind | 0.4586 | 0.4586 | 0.4253 | 0.4469 | 0.93 | 0.97 | 0.573 | 1.18 | 0.952 | 5594 | 2/47 | 67.68 | 68.88 | +0.04 | -0.54 |  |  |
| mixed-0018 | blind | 0.4616 | 0.4616 | 0.4247 | 0.4303 | 0.92 | 0.93 | 0.560 | 1.64 | 0.950 | 7435 | 1/71 | 52.13 | 52.74 | +1.03 | -0.28 |  |  |
| mixed-0009 | blind | 0.4682 | 0.4682 | 0.4291 | 0.4490 | 0.92 | 0.96 | 0.969 | 1.16 | 0.890 | 2115 | 1/35 | 65.53 | 67.93 | -0.33 | -0.66 |  |  |
| mixed-0003 | blind | 0.4755 | 0.4755 | 0.4505 | 0.4583 | 0.95 | 0.96 | 1.784 | 1.22 | 0.494 | 476 | 3/53 | 134.77 | 138.48 | -0.47 | +0.38 |  |  |
| mixed-0014 | blind | 0.4846 | 0.4846 | 0.4761 | 0.1219 | 0.98 | 0.25 | 0.672 | 1.44 | 0.943 | 2397 | 0/48 | 2.93 | 62.79 | -61.03 | -2.28 |  | yes |
| mixed-0012 | blind | 0.4860 | 0.4860 | 0.4434 | 0.4627 | 0.91 | 0.95 | 0.958 | 1.24 | 0.888 | 2446 | 2/100 | 59.77 | 59.90 | +0.50 | -0.29 |  |  |
| mixed-0006 | blind | 0.5113 | 0.4887 | 0.3757 | 0.4550 | 0.73 | 0.89 | 1.212 | 1.76 | 0.778 | 1589 | 0/53 | 48.03 | 51.32 | -0.40 | +3.50 | yes |  |
| mixed-0013 | blind | 0.5354 | 0.4646 | 0.3019 | 0.4838 | 0.56 | 0.90 | 1.095 | 2.16 | 0.926 | 2000 | 0/43 | 57.98 | 48.06 | +27.54 | +18.93 | yes | yes |
| mixed-0017 | blind | 0.7668 | 0.2332 | 0.7490 | 0.9883 | 0.98 | 1.29 | 0.826 | 1.15 | 0.883 | 6315 | 0/22 | 43.16 | 29.82 | +14.49 | +7.59 | yes | yes |
| mixed-0004 | orbit | 0.0520 | 0.0520 | 0.0517 | 0.0402 | 0.99 | 0.77 | 5.469 | 4.32 | 0.977 | 166 | 0/80 | 22.53 | 42.84 | -0.12 | +0.51 |  |  |
| mixed-0015 | orbit | 0.0731 | 0.0731 | 0.1242 | 0.0215 | 1.70 | 0.29 | 2.963 | 1.48 | 0.925 | 23 | 0/27 | 10.13 | 18.48 | -1.89 | -3.38 |  |  |
| mixed-0008 | orbit | 0.1119 | 0.1119 | 0.1425 | 0.1112 | 1.27 | 0.99 | 3.694 | 1.03 | 0.851 | 50 | 0/38 | 26.84 | 46.83 | -0.70 | +2.38 |  |  |
| mixed-0007 | orbit | 0.1434 | 0.1434 | 0.1843 | 0.0225 | 1.28 | 0.16 | 6.308 | 1.01 | 0.614 | 11 | 0/38 | 24.29 | 107.48 | -2.59 | +66.73 |  | yes |
| mixed-0010 | orbit | 0.4194 | 0.4194 | 0.2354 | 0.0211 | 0.56 | 0.05 | 3.920 | 1.02 | 0.688 | 47 | 0/62 | 2.04 | 29.84 | -18.63 | +7.80 |  | yes |
| mixed-0016 | orbit | 0.4203 | 0.4203 | 0.3751 | 0.3959 | 0.89 | 0.94 | 0.804 | 1.08 | 0.941 | 1904 | 1/50 | 58.50 | 62.83 | +0.54 | +0.50 |  |  |
| mixed-0011 | orbit | 0.4204 | 0.4204 | 0.3868 | 0.3839 | 0.92 | 0.91 | 1.243 | 1.24 | 0.873 | 563 | 0/24 | 26.99 | 35.49 | -3.09 | +2.23 |  |  |
| mixed-0019 | orbit | 0.4242 | 0.4242 | 0.2678 | 0.0296 | 0.63 | 0.07 | 3.155 | 1.32 | 0.566 | 72 | 0/11 | 7.75 | 69.84 | -13.82 | +46.13 |  | yes |
| mixed-0005 | orbit | 0.4443 | 0.4443 | 0.2740 | 0.0006 | 0.62 | 0.00 | 4.774 | 1.02 | 0.481 | 9 | 0/29 | 3.79 | 57.41 | -15.62 | +36.49 |  | yes |
| mixed-0002 | orbit | 0.4477 | 0.4477 | 0.4288 | 0.4493 | 0.96 | 1.00 | 1.362 | 1.16 | 0.753 | 924 | 0/28 | 94.54 | 100.78 | +0.67 | +1.47 |  |  |
| mixed-0000 | orbit | 0.4586 | 0.4586 | 0.4253 | 0.4437 | 0.93 | 0.97 | 0.573 | 1.18 | 0.952 | 5551 | 2/47 | 67.89 | 69.59 | +0.25 | +0.17 |  |  |
| mixed-0009 | orbit | 0.4682 | 0.4682 | 0.4291 | 0.4400 | 0.92 | 0.94 | 0.971 | 1.16 | 0.890 | 2064 | 0/35 | 65.58 | 67.96 | -0.27 | -0.62 |  |  |
| mixed-0003 | orbit | 0.4755 | 0.4755 | 0.4505 | 0.4554 | 0.95 | 0.96 | 1.714 | 1.22 | 0.494 | 471 | 0/53 | 132.73 | 135.84 | -2.51 | -2.25 |  |  |
| mixed-0012 | orbit | 0.4860 | 0.4860 | 0.4434 | 0.4756 | 0.91 | 0.98 | 0.954 | 1.24 | 0.888 | 2385 | 2/100 | 59.41 | 59.88 | +0.14 | -0.31 |  |  |
| mixed-0006 | orbit | 0.5113 | 0.4887 | 0.3757 | 0.4715 | 0.73 | 0.92 | 1.212 | 1.76 | 0.778 | 1702 | 0/53 | 48.90 | 46.93 | +0.48 | -0.88 | yes |  |
| mixed-0014 | orbit | 0.5154 | 0.4846 | 0.4761 | 0.5096 | 0.92 | 0.99 | 0.666 | 1.44 | 0.943 | 6305 | 2/48 | 65.23 | 63.93 | +0.16 | -0.03 | yes |  |
| mixed-0013 | orbit | 0.5354 | 0.4646 | 0.3019 | 0.4020 | 0.56 | 0.75 | 1.100 | 2.16 | 0.926 | 924 | 0/43 | 53.52 | 61.07 | +23.08 | +31.94 | yes |  |
| mixed-0018 | orbit | 0.5384 | 0.4616 | 0.4247 | 0.5043 | 0.79 | 0.94 | 0.560 | 1.64 | 0.950 | 9970 | 1/71 | 53.06 | 51.53 | +0.03 | +0.43 | yes |  |
| mixed-0017 | orbit | 0.7668 | 0.2332 | 0.7490 | 0.8240 | 0.98 | 1.07 | 0.820 | 1.15 | 0.883 | 3109 | 0/22 | 29.23 | 20.91 | +0.56 | -1.32 | yes |  |
| gaia-17170287112790656 | blind | 0.0673 | 0.0673 | 0.3992 | 0.0374 | 5.93 | 0.56 | 1.628 | 1.53 | 0.590 | 360 | 0/11 | 3.70 | 77.04 | -99.33 | -115.66 |  | yes |
| gaia-45788547559850496 | blind | 0.1442 | 0.1442 | 0.1529 | 0.1368 | 1.06 | 0.95 | 1.375 | 1.66 | 0.973 | 1242 | 0/15 | 60.51 | 94.22 | -0.04 | +2.39 |  |  |
| gaia-51574864941234048 | blind | 0.3960 | 0.3960 | 0.1838 | 0.1554 | 0.46 | 0.39 | 2.006 | 1.70 | 0.915 | 123 | 0/14 | 39.23 | 179.97 | +25.82 | +164.89 |  |  |
| gaia-53290511099783296 | blind | 0.4445 | 0.4445 | 0.2116 | 0.0446 | 0.48 | 0.10 | 1.918 | 1.87 | 0.881 | 68 | 0/16 | 13.62 | 92.41 | -19.53 | +56.93 |  | yes |
| gaia-47620265212420096 | blind | 0.4589 | 0.4589 | 0.4279 | 0.4523 | 0.93 | 0.99 | 0.431 | 2.54 | 0.983 | 16897 | 0/13 | 23.29 | 24.43 | -2.69 | -2.17 |  |  |
| gaia-51884824140205824 | blind | 0.4617 | 0.4617 | 0.4350 | 0.4614 | 0.94 | 1.00 | 0.520 | 1.54 | 0.971 | 9896 | 0/24 | 61.30 | 62.41 | +10.17 | +10.16 |  |  |
| gaia-53680017390963072 | blind | 0.4669 | 0.4669 | 0.3502 | 0.3651 | 0.75 | 0.78 | 1.692 | 1.37 | 0.698 | 733 | 0/17 | 116.42 | 120.94 | +19.16 | +19.02 |  |  |
| gaia-7289006178185856 | blind | 0.4843 | 0.4843 | 0.4605 | 0.4744 | 0.95 | 0.98 | 1.695 | 1.48 | 0.366 | 538 | 0/10 | 147.32 | 144.89 | +18.99 | +12.71 |  |  |
| gaia-40041022325608704 | blind | 0.4905 | 0.4905 | 0.4517 | 0.5093 | 0.92 | 1.04 | 1.885 | 1.04 | 0.682 | 623 | 0/12 | 73.21 | 73.88 | -28.84 | -29.42 |  |  |
| gaia-45354171749336960 | blind | 0.4911 | 0.4911 | 0.4542 | 0.4703 | 0.92 | 0.96 | 0.465 | 1.63 | 0.968 | 13557 | 2/25 | 75.41 | 77.02 | +1.27 | +1.19 |  |  |
| gaia-56716765427964800 | blind | 0.4947 | 0.4947 | 0.4547 | 0.4755 | 0.92 | 0.96 | 0.477 | 1.61 | 0.968 | 13869 | 1/12 | 157.23 | 163.81 | +93.53 | +99.25 |  |  |
| gaia-37608387208382848 | blind | 0.4951 | 0.4951 | 0.4629 | 0.0836 | 0.93 | 0.17 | 1.655 | 1.03 | 0.690 | 243 | 0/15 | 3.36 | 117.74 | -65.17 | +49.04 |  | yes |
| gaia-50868531799454720 | blind | 0.5561 | 0.4439 | 0.5078 | 0.5832 | 0.91 | 1.05 | 0.732 | 1.21 | 0.917 | 4599 | 0/16 | 114.07 | 110.96 | +58.77 | +58.00 | yes |  |
| gaia-56861694804053120 | blind | 0.5732 | 0.4268 | 0.5432 | 0.5782 | 0.95 | 1.01 | 0.462 | 3.05 | 0.965 | 28375 | 0/15 | 72.21 | 67.06 | -2.24 | -2.27 | yes |  |
| gaia-17170287112790656 | orbit | 0.0673 | 0.0673 | 0.3992 | 0.1979 | 5.93 | 2.94 | 1.704 | 1.53 | 0.590 | 90 | 0/11 | 116.33 | 19.35 | +13.30 | -173.35 |  | yes |
| gaia-45788547559850496 | orbit | 0.1442 | 0.1442 | 0.1529 | 0.1358 | 1.06 | 0.94 | 1.375 | 1.66 | 0.973 | 1288 | 0/15 | 60.47 | 97.47 | -0.07 | +5.64 |  |  |
| gaia-51574864941234048 | orbit | 0.3960 | 0.3960 | 0.1838 | 0.0062 | 0.46 | 0.02 | 2.006 | 1.70 | 0.915 | 42 | 0/14 | 3.06 | 41.74 | -10.35 | +26.67 |  | yes |
| gaia-50868531799454720 | orbit | 0.4439 | 0.4439 | 0.5078 | 0.4307 | 1.14 | 0.97 | 0.732 | 1.21 | 0.917 | 2896 | 0/16 | 61.75 | 63.71 | +8.79 | +8.41 |  |  |
| gaia-53290511099783296 | orbit | 0.4445 | 0.4445 | 0.2116 | 0.0455 | 0.48 | 0.10 | 1.924 | 1.87 | 0.881 | 57 | 0/16 | 21.25 | 110.65 | -11.91 | +75.17 |  | yes |
| gaia-47620265212420096 | orbit | 0.4589 | 0.4589 | 0.4279 | 0.4535 | 0.93 | 0.99 | 0.431 | 2.54 | 0.983 | 16080 | 0/13 | 25.52 | 26.07 | -0.46 | -0.53 |  |  |
| gaia-51884824140205824 | orbit | 0.4617 | 0.4617 | 0.4350 | 0.4567 | 0.94 | 0.99 | 0.522 | 1.54 | 0.971 | 7834 | 2/24 | 50.98 | 52.24 | -0.15 | -0.01 |  |  |
| gaia-53680017390963072 | orbit | 0.4669 | 0.4669 | 0.3502 | 0.3787 | 0.75 | 0.81 | 1.692 | 1.37 | 0.698 | 671 | 0/17 | 99.36 | 101.40 | +2.11 | -0.53 |  |  |
| gaia-7289006178185856 | orbit | 0.4843 | 0.4843 | 0.4605 | 0.5228 | 0.95 | 1.08 | 1.695 | 1.48 | 0.366 | 241 | 0/10 | 62.84 | 42.53 | -65.49 | -89.65 |  |  |
| gaia-40041022325608704 | orbit | 0.4905 | 0.4905 | 0.4517 | 0.4503 | 0.92 | 0.92 | 1.825 | 1.04 | 0.682 | 507 | 0/12 | 97.18 | 106.03 | -4.87 | +2.74 |  |  |
| gaia-45354171749336960 | orbit | 0.4911 | 0.4911 | 0.4542 | 0.4731 | 0.92 | 0.96 | 0.461 | 1.63 | 0.968 | 13510 | 2/25 | 73.77 | 76.02 | -0.37 | +0.19 |  |  |
| gaia-56716765427964800 | orbit | 0.4947 | 0.4947 | 0.4547 | 0.4805 | 0.92 | 0.97 | 0.468 | 1.61 | 0.968 | 13587 | 0/12 | 66.83 | 68.32 | +3.13 | +3.75 |  |  |
| gaia-37608387208382848 | orbit | 0.4951 | 0.4951 | 0.4629 | 0.4947 | 0.93 | 1.00 | 1.543 | 1.03 | 0.690 | 800 | 0/15 | 68.33 | 67.73 | -0.21 | -0.97 |  |  |
| gaia-56861694804053120 | orbit | 0.5732 | 0.4268 | 0.5432 | 0.5764 | 0.95 | 1.01 | 0.462 | 3.05 | 0.965 | 28206 | 0/15 | 75.64 | 70.18 | +1.19 | +0.85 | yes |  |

Absent from the archive: mixed-0001 (orbit, field3), mixed-0001 (blind, field3).

## Summary

66 runs are tabulated, 33 on the orbit tier and 33 on the blind tier, drawn from 33 distinct systems across the two populations.

### How often the measured fraction is wrong by more than a factor 1.5

16 of 66 runs (24 per cent) have a measured l_B off by more than a factor 1.5: 4 too bright and 12 too faint. By tier that is 8 of 33 on orbit, 8 of 33 on blind; by population it is 6 of 28 in gaia4, 10 of 38 in field3.

The two directions are cleanly separated by the true fraction. Every run in which the light stage measured B too bright has a true l_B below 0.078 (the 4 of them span 0.0673 to 0.0731, and they are 2 systems seen on both tiers). Every run in which it measured B too faint has a true l_B above 0.391 (the 12 of them span 0.3960 to 0.5354). Nothing in between fails in either direction. The single worst case is the one already known: gaia-17170287112790656 at a true l_B of 0.0673, measured as 0.3992, a factor 5.93 too bright, on both tiers.

The failures belong to the system, not to the tier. Over every system present on both tiers the orbit and blind light stages returned the same fraction to the last recorded digit, so the 16 runs are 8 systems counted twice, and each of those 8 fails on both tiers. This also means the blind tier's period search costs the light stage nothing: the measurement happens before any of that.

The magnitudes are as lopsided as the directions. The too-bright group reaches a factor 5.93, while no too-faint run is worse than a factor 2.15.

The exchange convention moves the count by one system. Scoring the declared fraction against the injected secondary in every run, exchange ignored, gives 18 bad runs rather than 16; the only system that changes is mixed-0017. Every system in the list above is a failure under either convention.

| system | tier | l_B true | l_B declared | r_dec | l_B label | r_lab | dK_A | dK_B |
|---|---|---|---|---|---|---|---|---|
| gaia-17170287112790656 | orbit | 0.0673 | 0.3992 | 5.93 | 0.1979 | 2.94 | +13.30 | -173.35 |
| gaia-17170287112790656 | blind | 0.0673 | 0.3992 | 5.93 | 0.0374 | 0.56 | -99.33 | -115.66 |
| mixed-0015 | orbit | 0.0731 | 0.1242 | 1.70 | 0.0215 | 0.29 | -1.89 | -3.38 |
| mixed-0015 | blind | 0.0731 | 0.1242 | 1.70 | 0.0180 | 0.25 | -7.76 | -9.74 |
| gaia-51574864941234048 | orbit | 0.3960 | 0.1838 | 0.46 | 0.0062 | 0.02 | -10.35 | +26.67 |
| gaia-51574864941234048 | blind | 0.3960 | 0.1838 | 0.46 | 0.1554 | 0.39 | +25.82 | +164.89 |
| mixed-0010 | orbit | 0.4194 | 0.2354 | 0.56 | 0.0211 | 0.05 | -18.63 | +7.80 |
| mixed-0010 | blind | 0.4194 | 0.2354 | 0.56 | 0.1719 | 0.41 | -17.15 | -4.44 |
| mixed-0019 | orbit | 0.4242 | 0.2678 | 0.63 | 0.0296 | 0.07 | -13.82 | +46.13 |
| mixed-0019 | blind | 0.4242 | 0.2678 | 0.63 | 0.4284 | 1.01 | +101.15 | +160.72 |
| mixed-0005 | orbit | 0.4443 | 0.2740 | 0.62 | 0.0006 | 0.00 | -15.62 | +36.49 |
| mixed-0005 | blind | 0.4443 | 0.2740 | 0.62 | 0.0064 | 0.01 | -4.77 | +24.55 |
| gaia-53290511099783296 | orbit | 0.4445 | 0.2116 | 0.48 | 0.0455 | 0.10 | -11.91 | +75.17 |
| gaia-53290511099783296 | blind | 0.4445 | 0.2116 | 0.48 | 0.0446 | 0.10 | -19.53 | +56.93 |
| mixed-0013 | orbit | 0.5354 | 0.3019 | 0.56 | 0.4020 | 0.75 | +23.08 | +31.94 |
| mixed-0013 | blind | 0.5354 | 0.3019 | 0.56 | 0.4838 | 0.90 | +27.54 | +18.93 |

### Where the failures sit in true l_B

The short answer is that it is not simply the faint secondaries. True l_B of the badly measured runs: min 0.0673, median 0.4218, max 0.5354. True l_B of the rest: min 0.0520, median 0.4682, max 0.7668. The faint end has by far the highest failure rate, but the faint end is thinly populated, and three quarters of the failures by count sit among the near-equal pairs.

| true l_B bin | runs | off by >1.5x | fraction |
|---|---|---|---|
| 0.00 to 0.05 | 0 | 0 | n/a |
| 0.05 to 0.10 | 6 | 4 | 67% |
| 0.10 to 0.20 | 6 | 0 | 0% |
| 0.20 to 0.30 | 0 | 0 | n/a |
| 0.30 to 0.40 | 2 | 2 | 100% |
| 0.40 to 0.50 | 41 | 8 | 20% |
| >= 0.50 | 11 | 2 | 18% |

### What the light stage itself reported

The light stage's own diagnostics shift in the expected direction but do not separate the failures. Median velocity error for B: 2.48 km/s on the badly measured runs against 0.97 km/s on the rest. Median reduced chi-square: 1.51 against 1.24. Median R-squared: 0.784 against 0.890.

Read as classifiers they range from weakly informative to blind. The rank-based area under the curve for calling a run bad, where 0.5 is a coin toss and 1.0 is perfect separation, is 0.83 for sigma_B, 0.56 for the reduced chi-square and 0.70 for R-squared (the last two taken in the direction that helps them). Only sigma_B carries real signal, and it is not enough to gate on: the largest sigma_B in the whole table, 6.31 km/s, belongs to a run whose fraction came out right, while the worst failure of all had a sigma_B of 1.70 km/s.

The later velocity stage does separate them, but only because it runs after the disentangling and so inherits the damage: median dchi2 for B is 63 on the badly measured runs against 1878 on the rest. The pipeline's own light-fraction flag, raised when the label fit disagrees with the declared fraction, fired on 11 of the 16 badly measured runs and on 4 of the 50 others, so it is a sensitive but not a specific warning.

### Is the label fit closer

The label fit's l_B is closer to the truth in 41 of 66 runs (62 per cent). Median absolute log ratio: measured 0.086, label fit 0.070 (a factor of 1.09 and 1.07 respectively). 18 runs are still off by more than a factor 1.5 after the label fit, against 16 before.

Restricted to the 16 badly measured runs, the label fit is closer in 5 of them, with median absolute log ratio 1.312 against 0.575 for the measured fraction (factors 3.71 and 1.78); 13 of them are still off by more than a factor 1.5 after the label fit.

Splitting the two populations of runs makes the pattern plain. Among the 50 runs whose measured fraction was already within a factor 1.5, the label fit is closer in 36 and it pushes 5 of them past the factor 1.5 line that they had been inside. Among the 16 badly measured runs the label fit collapses l_B toward zero rather than correcting it: 5 of them end below a tenth of the true fraction, the extreme being mixed-0005 on the orbit tier at l_B = 0.0006 against a true 0.4443. The label fit therefore inherits the disentangling's failure rather than diagnosing it, which is what one expects when the spectrum it is fitting is the one the disentangling produced.

### Do the bad fractions coincide with the large K errors

Absolute K_A error: badly measured runs median 16.39 km/s (range 1.89 to 101.15); the rest median 0.96 km/s (range 0.03 to 93.53).
Absolute K_B error: badly measured runs median 34.22 km/s (range 3.38 to 173.35); the rest median 2.18 km/s (range 0.01 to 99.25).

|dK_B| above 10 km/s: 23 of 66 runs, of which 12 also have a measured l_B off by more than a factor 1.5 (52 per cent, against 9 per cent among the 43 runs below that threshold).
|dK_B| above 25 km/s: 16 of 66 runs, of which 10 also have a measured l_B off by more than a factor 1.5 (62 per cent, against 12 per cent among the 50 runs below that threshold).
|dK_B| above 50 km/s: 10 of 66 runs, of which 6 also have a measured l_B off by more than a factor 1.5 (60 per cent, against 18 per cent among the 56 runs below that threshold).

Rank correlation (Spearman) of |log(declared/true)| against |dK_B| is 0.49 and against |dK_A| is 0.37. The signed log ratio against the true l_B correlates at -0.25, which is the too-bright-when-faint and too-faint-when-equal split restated as a monotone trend. The rank-based area under the curve for using |dK_B| to call a run badly measured is 0.88.

The association is real but it runs in both directions and neither implies the other. Of the 16 badly measured runs, 4 have |dK_B| under 10 km/s, so a wrong light fraction does not always wreck the semi-amplitude; and 4 runs with |dK_B| over 50 km/s had their light measured to within a factor 1.5, so a wrecked semi-amplitude does not always follow from the light. What the numbers support is the weaker statement that a badly measured light fraction raises the median |dK_B| from 2.2 to 34.2 km/s and makes a gross semi-amplitude failure roughly 5 times more likely.

The case that motivated this tabulation is the extreme of the trend. gaia-17170287112790656 has a true l_B of 0.0673; the light stage measured 0.3992, and on the orbit tier the disentangling then returned K_B = 19.4 km/s against an injected 192.7 km/s, an almost static secondary. The same system on the blind tier moved the failure into the primary instead, K_A = 3.7 km/s against an injected 103.0 km/s.

### Exchanged components

11 of 66 runs came out with the components exchanged; 2 of those 11 also have a measured l_B off by more than a factor 1.5, against 14 of the 55 runs that were not exchanged. Exchange is therefore not a marker of a badly measured light fraction. The exchanged runs are the near-equal pairs, as expected: their injected secondary fractions run from 0.233 to 0.489, and 9 of the 11 sit above 0.42, so the disentangling had little in the light to tell the two stars apart. The most lopsided of them is mixed-0017 at 0.233. A further 3 of them (gaia-50868531799454720, mixed-0014, mixed-0018) exchanged on one tier and not on the other, which is the same ambiguity resolving differently rather than a difference in the data.

---

Produced by `tabulate_light.py` in this directory, from the archived `result.json`, `summary.txt` and `population.json` files of the D62 gaia4 and field3 runs.
