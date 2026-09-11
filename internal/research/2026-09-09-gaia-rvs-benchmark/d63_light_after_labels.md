# The light fraction carried by the post-label-fit TODCOR pass

This is a follow-up to `../wp-g/light_fractions.md`, which tabulated three quantities for the 66 orbit- and blind-tier runs of the D62 benchmark: the injected secondary light fraction, the fraction the light stage measured before the disentangling, and the fraction the label fit reported afterwards. It found 16 runs (8 systems) whose measured fraction is off by more than a factor 1.5, and that the label fit does not correct them. The question here is what the pipeline's *second* TODCOR pass says about the same fraction. That pass runs after the disentangling and the label fit, against the label fit's own templates, and its per-epoch table is `result["velocities"]` and `velocities.csv`.

## What the second pass actually records

The premise of the question has to be corrected before the table can be read. The pass with a `light_mode` of "global median" is the *first* TODCOR pass, the light stage itself; the second pass holds the light fixed. Over all 66 runs the `light_mode` recorded in `result["velocities"]` is 'fixed', and the mode printed in the second `TODCOR velocities:` block of `summary.txt` is 'fixed' while the first block reports 'global median'. Every star writes 2 such blocks.

The reason is in `albireo/pipeline.py`, stage 6, which measures the epoch velocities as

```python
templates = _templates(ctx, fit, match)
table = fit.measure_velocities(templates=templates, light=[s.light for s in dis.stars])
```

The label fit reaches this call through `templates` only, and `_templates` uses it for one thing: it replaces each template's velocity zero point with the label fit's `v_kms`, which is what makes the epoch velocities absolute rather than differential. The light amplitudes come from `dis.stars`, the components as declared to the disentangler, which on the orbit and blind tiers are exactly the fractions the light stage measured. Passing a list of floats as `light` selects `mode="fixed"` in `albireo.todcor`, so the second pass does not measure a light fraction at all. It carries the first pass's.

The numbers bear that out. Taking the median of the per-epoch `light_B` column over the usable epochs, the second pass's secondary fraction equals the light stage's exactly, bit for bit, in all 66 runs, and taking the median over all epochs instead changes nothing. The medians are needed because the column is not constant in 18 of the 66 runs: `reassign_by_orbit` exchanges the two components at epochs where the disentangling's orbit says the per-epoch correlation landed in the mirror minimum, and the light column travels with the exchange, so `light_B` at those epochs holds l_A. The swapped epochs are always a minority (at most 19 of 80 in the worst run, mixed-0004 orbit) and they are never flagged usable-and-blended, so the median over usable epochs recovers the declared value in every case.

So the column added below, `l_B vel`, is the median over usable epochs, and it is the light stage's fraction to machine precision. Everything that follows is the consequence of that identity, stated in the terms the earlier pass used.

## Table

`l_B true` is the injected fraction under the exchange convention of the earlier pass. `l_B ls` is the light stage's fraction (the declared one), `l_B lab` the label fit's, `l_B vel` the second TODCOR pass's median over usable epochs. `r_ls`, `r_lab`, `r_vel` are those over the truth. `swap` is the number of epochs `reassign_by_orbit` exchanged, out of the epochs of the run. The remaining columns are the second pass's own diagnostics for B: `dchi2_B` the median per-epoch detection statistic over usable epochs, `blend` the blended epochs as a fraction of all epochs, `sig_B` the median velocity error in km/s over usable epochs, `rc2` the pass's median reduced chi-square. `K_B` and `K_B inj` are the disentangling's and the injected semi-amplitude in km/s and `K_B/inj` their ratio.

| system | tier | l_B true | l_B ls | l_B lab | l_B vel | r_ls | r_lab | r_vel | swap | dchi2_B | blend | sig_B | rc2 | K_B | K_B inj | K_B/inj | dK_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixed-0004 | blind | 0.0520 | 0.0517 | 0.0426 | 0.0517 | 0.99 | 0.82 | 0.99 | 17/80 | 167 | 0/80 (0%) | 2.264 | 1.01 | 42.08 | 42.34 | 0.99 | -0.26 |
| mixed-0015 | blind | 0.0731 | 0.1242 | 0.0180 | 0.1242 | 1.70 | 0.25 | 1.70 | 0/27 | 38 | 0/27 (0%) | 3.286 | 0.98 | 12.13 | 21.86 | 0.55 | -9.74 |
| mixed-0008 | blind | 0.1119 | 0.1425 | 0.0728 | 0.1425 | 1.27 | 0.65 | 1.27 | 3/38 | 39 | 0/38 (0%) | 5.893 | 0.96 | 58.43 | 44.45 | 1.31 | +13.98 |
| mixed-0007 | blind | 0.1434 | 0.1843 | 0.0939 | 0.1843 | 1.28 | 0.65 | 1.28 | 1/38 | 16 | 0/38 (0%) | 14.191 | 0.99 | 38.56 | 40.75 | 0.95 | -2.19 |
| mixed-0010 | blind | 0.4194 | 0.2354 | 0.1719 | 0.2354 | 0.56 | 0.41 | 0.56 | 0/62 | 50 | 0/62 (0%) | 6.655 | 0.99 | 17.59 | 22.04 | 0.80 | -4.44 |
| mixed-0016 | blind | 0.4203 | 0.3751 | 0.3916 | 0.3751 | 0.89 | 0.93 | 0.89 | 1/50 | 1867 | 1/50 (2%) | 0.937 | 0.99 | 64.95 | 62.34 | 1.04 | +2.61 |
| mixed-0011 | blind | 0.4204 | 0.3868 | 0.3630 | 0.3868 | 0.92 | 0.86 | 0.92 | 0/24 | 562 | 4/24 (17%) | 2.310 | 0.98 | 49.03 | 33.26 | 1.47 | +15.77 |
| mixed-0019 | blind | 0.4242 | 0.2678 | 0.4284 | 0.2678 | 0.63 | 1.01 | 0.63 | 0/11 | 1192 | 0/11 (0%) | 1.885 | 1.04 | 184.43 | 23.72 | 7.78 | +160.72 |
| mixed-0005 | blind | 0.4443 | 0.2740 | 0.0064 | 0.2740 | 0.62 | 0.01 | 0.62 | 0/29 | 5 | 0/29 (0%) | 95.294 | 0.98 | 45.47 | 20.92 | 2.17 | +24.55 |
| mixed-0002 | blind | 0.4477 | 0.4288 | 0.4403 | 0.4288 | 0.96 | 0.98 | 0.96 | 0/28 | 932 | 0/28 (0%) | 2.159 | 0.96 | 100.79 | 99.30 | 1.02 | +1.49 |
| mixed-0000 | blind | 0.4586 | 0.4253 | 0.4469 | 0.4253 | 0.93 | 0.97 | 0.93 | 0/47 | 5598 | 2/47 (4%) | 0.670 | 0.98 | 68.88 | 69.42 | 0.99 | -0.54 |
| mixed-0018 | blind | 0.4616 | 0.4247 | 0.4303 | 0.4247 | 0.92 | 0.93 | 0.92 | 0/71 | 7448 | 1/71 (1%) | 0.575 | 0.98 | 52.74 | 53.03 | 0.99 | -0.28 |
| mixed-0009 | blind | 0.4682 | 0.4291 | 0.4490 | 0.4291 | 0.92 | 0.96 | 0.92 | 0/35 | 2168 | 1/35 (3%) | 1.166 | 0.98 | 67.93 | 68.58 | 0.99 | -0.66 |
| mixed-0003 | blind | 0.4755 | 0.4505 | 0.4583 | 0.4505 | 0.95 | 0.96 | 0.95 | 0/53 | 486 | 3/53 (6%) | 4.523 | 0.97 | 138.48 | 138.10 | 1.00 | +0.38 |
| mixed-0014 | blind | 0.4846 | 0.4761 | 0.1219 | 0.4761 | 0.98 | 0.25 | 0.98 | 11/48 | 2397 | 0/48 (0%) | 1.868 | 2.34 | 62.79 | 65.07 | 0.96 | -2.28 |
| mixed-0012 | blind | 0.4860 | 0.4434 | 0.4627 | 0.4434 | 0.91 | 0.95 | 0.91 | 5/100 | 2459 | 2/100 (2%) | 1.077 | 0.99 | 59.90 | 60.19 | 1.00 | -0.29 |
| mixed-0006 | blind | 0.5113 | 0.3757 | 0.4550 | 0.3757 | 0.73 | 0.89 | 0.73 | 0/53 | 1589 | 0/53 (0%) | 1.984 | 0.98 | 51.32 | 47.81 | 1.07 | +3.50 |
| mixed-0013 | blind | 0.5354 | 0.3019 | 0.4838 | 0.3019 | 0.56 | 0.90 | 0.56 | 0/43 | 2000 | 0/43 (0%) | 2.158 | 0.99 | 48.06 | 29.13 | 1.65 | +18.93 |
| mixed-0017 | blind | 0.7668 | 0.7490 | 0.9883 | 0.7490 | 0.98 | 1.29 | 0.98 | 0/22 | 6315 | 0/22 (0%) | 0.917 | 1.00 | 29.82 | 22.23 | 1.34 | +7.59 |
| mixed-0004 | orbit | 0.0520 | 0.0517 | 0.0402 | 0.0517 | 0.99 | 0.77 | 0.99 | 19/80 | 166 | 0/80 (0%) | 2.218 | 1.01 | 42.84 | 42.34 | 1.01 | +0.51 |
| mixed-0015 | orbit | 0.0731 | 0.1242 | 0.0215 | 0.1242 | 1.70 | 0.29 | 1.70 | 0/27 | 24 | 0/27 (0%) | 8.014 | 0.97 | 18.48 | 21.86 | 0.85 | -3.38 |
| mixed-0008 | orbit | 0.1119 | 0.1425 | 0.1112 | 0.1425 | 1.27 | 0.99 | 1.27 | 1/38 | 50 | 0/38 (0%) | 4.463 | 0.95 | 46.83 | 44.45 | 1.05 | +2.38 |
| mixed-0007 | orbit | 0.1434 | 0.1843 | 0.0225 | 0.1843 | 1.28 | 0.16 | 1.28 | 1/38 | 12 | 0/38 (0%) | 25.886 | 1.00 | 107.48 | 40.75 | 2.64 | +66.73 |
| mixed-0010 | orbit | 0.4194 | 0.2354 | 0.0211 | 0.2354 | 0.56 | 0.05 | 0.56 | 0/62 | 47 | 0/62 (0%) | 3.553 | 0.99 | 29.84 | 22.04 | 1.35 | +7.80 |
| mixed-0016 | orbit | 0.4203 | 0.3751 | 0.3959 | 0.3751 | 0.89 | 0.94 | 0.89 | 2/50 | 1922 | 1/50 (2%) | 0.918 | 0.98 | 62.83 | 62.34 | 1.01 | +0.50 |
| mixed-0011 | orbit | 0.4204 | 0.3868 | 0.3839 | 0.3868 | 0.92 | 0.91 | 0.92 | 0/24 | 563 | 0/24 (0%) | 1.770 | 0.97 | 35.49 | 33.26 | 1.07 | +2.23 |
| mixed-0019 | orbit | 0.4242 | 0.2678 | 0.0296 | 0.2678 | 0.63 | 0.07 | 0.63 | 0/11 | 72 | 0/11 (0%) | 4.431 | 0.98 | 69.84 | 23.72 | 2.95 | +46.13 |
| mixed-0005 | orbit | 0.4443 | 0.2740 | 0.0006 | 0.2740 | 0.62 | 0.00 | 0.62 | 0/29 | 9 | 0/29 (0%) | 217.698 | 0.99 | 57.41 | 20.92 | 2.74 | +36.49 |
| mixed-0002 | orbit | 0.4477 | 0.4288 | 0.4493 | 0.4288 | 0.96 | 1.00 | 0.96 | 0/28 | 924 | 0/28 (0%) | 1.616 | 0.93 | 100.78 | 99.30 | 1.01 | +1.47 |
| mixed-0000 | orbit | 0.4586 | 0.4253 | 0.4437 | 0.4253 | 0.93 | 0.97 | 0.93 | 0/47 | 5577 | 2/47 (4%) | 0.637 | 0.96 | 69.59 | 69.42 | 1.00 | +0.17 |
| mixed-0009 | orbit | 0.4682 | 0.4291 | 0.4400 | 0.4291 | 0.92 | 0.94 | 0.92 | 0/35 | 2064 | 0/35 (0%) | 1.019 | 0.96 | 67.96 | 68.58 | 0.99 | -0.62 |
| mixed-0003 | orbit | 0.4755 | 0.4505 | 0.4554 | 0.4505 | 0.95 | 0.96 | 0.95 | 1/53 | 471 | 0/53 (0%) | 2.717 | 0.95 | 135.84 | 138.10 | 0.98 | -2.25 |
| mixed-0012 | orbit | 0.4860 | 0.4434 | 0.4756 | 0.4434 | 0.91 | 0.98 | 0.91 | 3/100 | 2403 | 2/100 (2%) | 1.015 | 0.98 | 59.88 | 60.19 | 0.99 | -0.31 |
| mixed-0006 | orbit | 0.5113 | 0.3757 | 0.4715 | 0.3757 | 0.73 | 0.92 | 0.73 | 0/53 | 1702 | 0/53 (0%) | 1.912 | 0.98 | 46.93 | 47.81 | 0.98 | -0.88 |
| mixed-0014 | orbit | 0.5154 | 0.4761 | 0.5096 | 0.4761 | 0.92 | 0.99 | 0.92 | 0/48 | 6445 | 2/48 (4%) | 0.572 | 0.97 | 63.93 | 63.97 | 1.00 | -0.03 |
| mixed-0013 | orbit | 0.5354 | 0.3019 | 0.4020 | 0.3019 | 0.56 | 0.75 | 0.56 | 0/43 | 924 | 0/43 (0%) | 1.963 | 0.97 | 61.07 | 29.13 | 2.10 | +31.94 |
| mixed-0018 | orbit | 0.5384 | 0.4247 | 0.5043 | 0.4247 | 0.79 | 0.94 | 0.79 | 1/71 | 9980 | 1/71 (1%) | 0.484 | 0.98 | 51.53 | 51.10 | 1.01 | +0.43 |
| mixed-0017 | orbit | 0.7668 | 0.7490 | 0.8240 | 0.7490 | 0.98 | 1.07 | 0.98 | 0/22 | 3109 | 0/22 (0%) | 0.800 | 0.93 | 20.91 | 22.23 | 0.94 | -1.32 |
| gaia-17170287112790656 | blind | 0.0673 | 0.3992 | 0.0374 | 0.3992 | 5.93 | 0.56 | 5.93 | 0/11 | 358 | 0/11 (0%) | 2.267 | 0.95 | 77.04 | 192.70 | 0.40 | -115.66 |
| gaia-45788547559850496 | blind | 0.1442 | 0.1529 | 0.1368 | 0.1529 | 1.06 | 0.95 | 1.06 | 0/15 | 1242 | 0/15 (0%) | 1.111 | 0.91 | 94.22 | 91.83 | 1.03 | +2.39 |
| gaia-51574864941234048 | blind | 0.3960 | 0.1838 | 0.1554 | 0.1838 | 0.46 | 0.39 | 0.46 | 0/14 | 123 | 0/14 (0%) | 1.856 | 0.92 | 179.97 | 15.08 | 11.94 | +164.89 |
| gaia-53290511099783296 | blind | 0.4445 | 0.2116 | 0.0446 | 0.2116 | 0.48 | 0.10 | 0.48 | 0/16 | 68 | 0/16 (0%) | 4.026 | 0.96 | 92.41 | 35.48 | 2.60 | +56.93 |
| gaia-47620265212420096 | blind | 0.4589 | 0.4279 | 0.4523 | 0.4279 | 0.93 | 0.99 | 0.93 | 1/13 | 16897 | 0/13 (0%) | 0.272 | 0.96 | 24.43 | 26.60 | 0.92 | -2.17 |
| gaia-51884824140205824 | blind | 0.4617 | 0.4350 | 0.4614 | 0.4350 | 0.94 | 1.00 | 0.94 | 5/24 | 9896 | 0/24 (0%) | 0.688 | 1.38 | 62.41 | 52.25 | 1.19 | +10.16 |
| gaia-53680017390963072 | blind | 0.4669 | 0.3502 | 0.3651 | 0.3502 | 0.75 | 0.78 | 0.75 | 0/17 | 733 | 0/17 (0%) | 3.801 | 1.02 | 120.94 | 101.92 | 1.19 | +19.02 |
| gaia-7289006178185856 | blind | 0.4843 | 0.4605 | 0.4744 | 0.4605 | 0.95 | 0.98 | 0.95 | 0/10 | 538 | 0/10 (0%) | 2.034 | 0.87 | 144.89 | 132.18 | 1.10 | +12.71 |
| gaia-40041022325608704 | blind | 0.4905 | 0.4517 | 0.5093 | 0.4517 | 0.92 | 1.04 | 0.92 | 0/12 | 623 | 0/12 (0%) | 3.109 | 0.96 | 73.88 | 103.30 | 0.72 | -29.42 |
| gaia-45354171749336960 | blind | 0.4911 | 0.4542 | 0.4703 | 0.4542 | 0.92 | 0.96 | 0.92 | 0/25 | 13577 | 2/25 (8%) | 0.451 | 0.95 | 77.02 | 75.83 | 1.02 | +1.19 |
| gaia-56716765427964800 | blind | 0.4947 | 0.4547 | 0.4755 | 0.4547 | 0.92 | 0.96 | 0.92 | 0/12 | 14126 | 1/12 (8%) | 0.444 | 0.94 | 163.81 | 64.56 | 2.54 | +99.25 |
| gaia-37608387208382848 | blind | 0.4951 | 0.4629 | 0.0836 | 0.4629 | 0.93 | 0.17 | 0.93 | 0/15 | 243 | 0/15 (0%) | 2.518 | 1.10 | 117.74 | 68.70 | 1.71 | +49.04 |
| gaia-50868531799454720 | blind | 0.5561 | 0.5078 | 0.5832 | 0.5078 | 0.91 | 1.05 | 0.91 | 0/16 | 4599 | 0/16 (0%) | 0.821 | 0.97 | 110.96 | 52.96 | 2.10 | +58.00 |
| gaia-56861694804053120 | blind | 0.5732 | 0.5432 | 0.5782 | 0.5432 | 0.95 | 1.01 | 0.95 | 0/15 | 28375 | 0/15 (0%) | 0.263 | 0.93 | 67.06 | 69.33 | 0.97 | -2.27 |
| gaia-17170287112790656 | orbit | 0.0673 | 0.3992 | 0.1979 | 0.3992 | 5.93 | 2.94 | 5.93 | 0/11 | 90 | 0/11 (0%) | 3.030 | 0.89 | 19.35 | 192.70 | 0.10 | -173.35 |
| gaia-45788547559850496 | orbit | 0.1442 | 0.1529 | 0.1358 | 0.1529 | 1.06 | 0.94 | 1.06 | 1/15 | 1288 | 0/15 (0%) | 1.067 | 0.90 | 97.47 | 91.83 | 1.06 | +5.64 |
| gaia-51574864941234048 | orbit | 0.3960 | 0.1838 | 0.0062 | 0.1838 | 0.46 | 0.02 | 0.46 | 1/14 | 42 | 0/14 (0%) | 3.698 | 0.96 | 41.74 | 15.08 | 2.77 | +26.67 |
| gaia-50868531799454720 | orbit | 0.4439 | 0.5078 | 0.4307 | 0.5078 | 1.14 | 0.97 | 1.14 | 0/16 | 2896 | 0/16 (0%) | 0.832 | 0.92 | 63.71 | 55.30 | 1.15 | +8.41 |
| gaia-53290511099783296 | orbit | 0.4445 | 0.2116 | 0.0455 | 0.2116 | 0.48 | 0.10 | 0.48 | 0/16 | 57 | 0/16 (0%) | 5.662 | 0.96 | 110.65 | 35.48 | 3.12 | +75.17 |
| gaia-47620265212420096 | orbit | 0.4589 | 0.4279 | 0.4535 | 0.4279 | 0.93 | 0.99 | 0.93 | 0/13 | 16080 | 0/13 (0%) | 0.266 | 0.91 | 26.07 | 26.60 | 0.98 | -0.53 |
| gaia-51884824140205824 | orbit | 0.4617 | 0.4350 | 0.4567 | 0.4350 | 0.94 | 0.99 | 0.94 | 2/24 | 8057 | 2/24 (8%) | 0.455 | 0.96 | 52.24 | 52.25 | 1.00 | -0.01 |
| gaia-53680017390963072 | orbit | 0.4669 | 0.3502 | 0.3787 | 0.3502 | 0.75 | 0.81 | 0.75 | 0/17 | 671 | 0/17 (0%) | 3.759 | 0.96 | 101.40 | 101.92 | 0.99 | -0.53 |
| gaia-7289006178185856 | orbit | 0.4843 | 0.4605 | 0.5228 | 0.4605 | 0.95 | 1.08 | 0.95 | 0/10 | 241 | 0/10 (0%) | 2.095 | 0.87 | 42.53 | 132.18 | 0.32 | -89.65 |
| gaia-40041022325608704 | orbit | 0.4905 | 0.4517 | 0.4503 | 0.4517 | 0.92 | 0.92 | 0.92 | 0/12 | 507 | 0/12 (0%) | 1.774 | 0.91 | 106.03 | 103.30 | 1.03 | +2.74 |
| gaia-45354171749336960 | orbit | 0.4911 | 0.4542 | 0.4731 | 0.4542 | 0.92 | 0.96 | 0.92 | 0/25 | 13516 | 2/25 (8%) | 0.436 | 0.94 | 76.02 | 75.83 | 1.00 | +0.19 |
| gaia-56716765427964800 | orbit | 0.4947 | 0.4547 | 0.4805 | 0.4547 | 0.92 | 0.97 | 0.92 | 0/12 | 13587 | 0/12 (0%) | 0.436 | 0.92 | 68.32 | 64.56 | 1.06 | +3.75 |
| gaia-37608387208382848 | orbit | 0.4951 | 0.4629 | 0.4947 | 0.4629 | 0.93 | 1.00 | 0.93 | 0/15 | 800 | 0/15 (0%) | 2.380 | 0.93 | 67.73 | 68.70 | 0.99 | -0.97 |
| gaia-56861694804053120 | orbit | 0.5732 | 0.5432 | 0.5764 | 0.5432 | 0.95 | 1.01 | 0.95 | 0/15 | 28206 | 0/15 (0%) | 0.251 | 0.92 | 70.18 | 69.33 | 1.01 | +0.85 |

Absent from the archive: mixed-0001 (orbit, field3), mixed-0001 (blind, field3).

## The second pass's fraction, judged as the earlier pass judged the others

**Ratio to the truth.** The second pass's r_vel is the light stage's r_ls in every run: median absolute log ratio 0.0858 against 0.0858, the two columns agreeing exactly. Its range is 0.46 to 5.93, the same span as the light stage's.

**Outside a factor 1.5.** 16 of 66 runs, the same 16 runs and the same 8 systems the light stage failed on, against 18 runs for the label fit.

**Closer to the truth than the light stage's fraction?** It is closer in 0 of 66 runs, further in 0, and identical in 66. Median absolute log ratio: light stage 0.086, second pass 0.086, label fit 0.070 (factors 1.09, 1.09 and 1.07). There is nothing to choose between the first two because they are the same number.

**On the 16 badly measured runs.** The second pass corrects 0 of them. Its median absolute log ratio there is 0.575 (factor 1.78), the light stage's 0.575 (factor 1.78), the label fit's 1.312 (factor 3.71); 16 of the 16 are still outside a factor 1.5 after the second pass, against 13 after the label fit.

| system | tier | l_B true | l_B ls | l_B lab | l_B vel | r_vel | dchi2_B | blend | sig_B | K_B/inj | dK_B |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gaia-17170287112790656 | orbit | 0.0673 | 0.3992 | 0.1979 | 0.3992 | 5.93 | 90 | 0/11 | 3.030 | 0.10 | -173.35 |
| gaia-17170287112790656 | blind | 0.0673 | 0.3992 | 0.0374 | 0.3992 | 5.93 | 358 | 0/11 | 2.267 | 0.40 | -115.66 |
| mixed-0015 | orbit | 0.0731 | 0.1242 | 0.0215 | 0.1242 | 1.70 | 24 | 0/27 | 8.014 | 0.85 | -3.38 |
| mixed-0015 | blind | 0.0731 | 0.1242 | 0.0180 | 0.1242 | 1.70 | 38 | 0/27 | 3.286 | 0.55 | -9.74 |
| gaia-51574864941234048 | orbit | 0.3960 | 0.1838 | 0.0062 | 0.1838 | 0.46 | 42 | 0/14 | 3.698 | 2.77 | +26.67 |
| gaia-51574864941234048 | blind | 0.3960 | 0.1838 | 0.1554 | 0.1838 | 0.46 | 123 | 0/14 | 1.856 | 11.94 | +164.89 |
| mixed-0010 | orbit | 0.4194 | 0.2354 | 0.0211 | 0.2354 | 0.56 | 47 | 0/62 | 3.553 | 1.35 | +7.80 |
| mixed-0010 | blind | 0.4194 | 0.2354 | 0.1719 | 0.2354 | 0.56 | 50 | 0/62 | 6.655 | 0.80 | -4.44 |
| mixed-0019 | orbit | 0.4242 | 0.2678 | 0.0296 | 0.2678 | 0.63 | 72 | 0/11 | 4.431 | 2.95 | +46.13 |
| mixed-0019 | blind | 0.4242 | 0.2678 | 0.4284 | 0.2678 | 0.63 | 1192 | 0/11 | 1.885 | 7.78 | +160.72 |
| mixed-0005 | orbit | 0.4443 | 0.2740 | 0.0006 | 0.2740 | 0.62 | 9 | 0/29 | 217.698 | 2.74 | +36.49 |
| mixed-0005 | blind | 0.4443 | 0.2740 | 0.0064 | 0.2740 | 0.62 | 5 | 0/29 | 95.294 | 2.17 | +24.55 |
| gaia-53290511099783296 | orbit | 0.4445 | 0.2116 | 0.0455 | 0.2116 | 0.48 | 57 | 0/16 | 5.662 | 3.12 | +75.17 |
| gaia-53290511099783296 | blind | 0.4445 | 0.2116 | 0.0446 | 0.2116 | 0.48 | 68 | 0/16 | 4.026 | 2.60 | +56.93 |
| mixed-0013 | orbit | 0.5354 | 0.3019 | 0.4020 | 0.3019 | 0.56 | 924 | 0/43 | 1.963 | 2.10 | +31.94 |
| mixed-0013 | blind | 0.5354 | 0.3019 | 0.4838 | 0.3019 | 0.56 | 2000 | 0/43 | 2.158 | 1.65 | +18.93 |

## How the second pass's diagnostics relate to the disentangling's K_B error

The pass carries no new light fraction, but it does carry evidence about the template it was given, and that evidence tracks the disentangling's failure. Median detection statistic for B over usable epochs: 63 on the 16 badly measured runs against 1895 on the other 50. Median velocity error for B: 3.63 km/s against 1.09 km/s. Median reduced chi-square: 0.98 against 0.96. Blended epochs: 0.0 per cent of all epochs against 1.6 per cent.

The blended fraction runs the wrong way, and it is worth saying why. All 29 blended epochs in the whole table belong to the 50 runs whose light fraction was measured well; not one of the 16 badly measured runs has a single blended epoch. Blending is flagged when the two velocities were measured along a ridge rather than at a peak, which requires both components to be detected at once. Where the secondary's template is wrong, B is not detected well enough to be confused with A, so the diagnostic stays silent. Its absence is not reassurance.

Read as classifiers for a badly measured light fraction, the rank-based areas under the curve are 0.87 for a low median dchi2_B, 0.84 for a high median sigma_B, 0.66 for a low blended fraction and 0.57 for a high reduced chi-square. Only the first two separate them, and they do so because they run after the disentangling: a wrong light fraction produces a wrong secondary spectrum, the label fit turns that into a wrong template, and the template then fails to detect B. Both statistics are downstream of the failure, not independent of it.

The K_B error does not merely grow with a bad light fraction; its sign is set by the sign of the light error, and that is the cleanest structure in the table. The 4 runs whose secondary was measured too bright return a K_B that collapses: median K_B over injected 0.48, range 0.10 to 0.85. The 12 runs whose secondary was measured too faint return a K_B that inflates: median 2.67, range 0.80 to 11.94. The 50 well measured runs sit at a median of 1.01. The reading is the obvious one: the disentangling is told how much of the flux belongs to B, and if it is given too much it spreads a faint star's signal over an amplitude it cannot support, while if it is given too little it must move a bright star further to fit the same line displacement.

The outright collapse the question anticipated is therefore the minority case. 3 of the 66 runs returned a K_B below half the injected value and 2 of those 3 also have a badly measured light fraction, while 10 of the 16 badly measured runs inflated K_B past 1.5 times the injected value. Where K_B did collapse, the median dchi2_B is 241 against 1192 for the rest, so the template built from that secondary's spectrum barely detects it, which is the mechanism the question described. Rank correlation of the median dchi2_B against |dK_B| is -0.47, against |log(K_B/injected)| -0.50, and against |log r_ls| -0.47; the median sigma_B against |log r_ls| is 0.47.

| system | tier | K_B | K_B inj | K_B/inj | r_ls | r_lab | dchi2_B | sig_B | bad light |
|---|---|---|---|---|---|---|---|---|---|
| gaia-17170287112790656 | orbit | 19.35 | 192.70 | 0.10 | 5.93 | 2.94 | 90 | 3.030 | yes |
| gaia-7289006178185856 | orbit | 42.53 | 132.18 | 0.32 | 0.95 | 1.08 | 241 | 2.095 |  |
| gaia-17170287112790656 | blind | 77.04 | 192.70 | 0.40 | 5.93 | 0.56 | 358 | 2.267 | yes |

## Would a second disentangling started from this pass's fractions do better

No, and not for any of the 8 systems. The fractions this pass carries are the fractions the first disentangling was given, bit for bit. A second disentangling started from them would be started from its own input, so it would reproduce the first disentangling and with it the first light fractions: 0 of the 16 badly measured runs would improve, 0 of the 8 badly measured systems. The same holds for the 50 runs whose fraction was already good. There is no sense in which this pass is a second opinion about the light; it is the first opinion written down a second time.

Nor does any recorded statistic separate the runs where such a restart would help from those where it would not, because the improvement is identically zero in all 66 runs. A statistic can only separate a distribution that has two sides, and this one does not.

The question is worth restating in the form the archive can answer: of the fractions the pipeline does record after the disentangling, is any of them a better restart point than the light stage's? The only candidate is the label fit's. Restarting a second disentangling from it would bring 3 of the 16 badly measured runs inside a factor 1.5 (mixed-0013 orbit, mixed-0013 blind, mixed-0019 blind), which is 2 of the 8 badly measured systems, and only 1 of them on both tiers (mixed-0013). It would leave the other 13 outside, and on the badly measured runs taken together it would move the median error from a factor 1.78 to a factor 3.71, because on the runs it does not fix it collapses the fraction rather than correcting it. It would also push 5 of the 50 currently good runs outside the same threshold (gaia-37608387208382848 blind, mixed-0007 orbit, mixed-0007 blind, mixed-0008 blind, mixed-0014 blind). On this evidence a label-fit restart is not worth making: it trades 3 failures for 5, and the failures it creates are on systems the pipeline currently handles.

What the second pass does contribute is a warning, not a correction. Its median detection statistic for B reaches an area under the curve of 0.87 for calling a run badly measured, and its median sigma_B 0.84. Neither can choose a restart point, since both are computed after the disentangling they would be diagnosing, and neither gives a clean cut: a cut at dchi2_B < 100 catches 11 of the 16 bad runs and flags 4 of the 50 good ones; a cut at dchi2_B < 500 catches 13 of the 16 bad runs and flags 10 of the 50 good ones; a cut at dchi2_B < 1000 catches 14 of the 16 bad runs and flags 20 of the 50 good ones. The bad runs span dchi2_B 5 to 2000 and the good ones 12 to 28375, so the two overlap over most of the bad range. Used as a gate on whether a run deserves a second look rather than as a measurement, the tightest cut is the useful one: 11 of the 15 runs it flags are genuinely badly measured.

---

Produced by `tabulate_light_after_labels.py` in this directory, from the archived `result.json`, `summary.txt` and `population.json` files of the D62 gaia4 and field3 runs. json and numpy only; nothing under `src/` or `tests/` was read for data and nothing was re-run.
