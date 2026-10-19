# Comparison with other disentangling codes

The disentangling codes in use share one model: each composite spectrum is the sum of the
component spectra, each shifted by its star's radial velocity. They differ in where the problem
is solved (Fourier or wavelength space), in which unknowns they fit, in how they treat the
broad features that the data do not assign to either star, and in the uncertainties they
report. The tables below follow Table 1 of Seeburger et al. (2024), extended by PSOAP, by the
method of Seeburger et al. itself and by albireo, and by the treatment of the regularization
and of the uncertainties. The [sources](#sources) of every entry are listed below the tables,
and an entry is left out where no accessible source settles it. The
[method overview](method.md) describes albireo's own steps.

## 1. What each code solves for

| Code | Domain | Solves for | Input required | Data grid | Weights and masks | Components |
|---|---|---|---|---|---|---|
| KOREL | Fourier | spectra and orbital elements, or spectra and per-epoch velocities; line-strength factors | starting orbital elements, each of which can be held | common grid, equidistant in $`\ln\lambda`$, $`2^n`$ bins per spectral region | one weight per spectrum; no per-pixel weights or masks | up to five stars on four nested orbits; telluric lines |
| FDBinary, fd3 | Fourier | spectra and any subset of the orbital elements | starting values and ranges of the elements; light factors per spectrum | one common grid, equidistant in $`\ln\lambda`$ | one noise level per spectrum; no per-pixel weights or masks | one to three (hierarchical triple) |
| CRES | wavelength, singular value decomposition | spectra only | velocities and light factors of every spectrum | each spectrum on its own pixels | inverse-error weight per pixel; points can be left out | any number |
| Spectangular | wavelength, singular value decomposition | spectra and orbital elements, or spectra and per-epoch velocities; a flux ratio per spectrum (2019) | starting orbit or velocities; light ratio | common logarithmic grid | none (unweighted; equal S/N required) | two, plus a static component for telluric or nebular lines |
| Shift-and-add | wavelength, iterative | spectra, and $`K_2`$ or $`(K_1, K_2)`$ from a $`\chi^2`$ grid; per-epoch velocities in the scheme of González & Levato (2006) | period, time of periastron, $`e`$ and $`\omega`$ from the primary's orbit; light ratio | logarithmic grid, shifts by interpolation | weights or pixel rejection allowed by the method; one noise level per epoch in the $`\chi^2`$ | two; triples and a static nebular component in later work |
| PSOAP | wavelength, Gaussian processes | orbital elements and prior hyperparameters, sampled; spectra integrated out | starting orbit and hyperparameters | each epoch on its own pixels | per-pixel noise; masks | two (three in the code) |
| Seeburger et al. (2024) | wavelength, sparse least squares | spectra, the primary's velocity at each epoch, the mass ratio | starting velocities, from templates or TIRAVEL | common grid, uniform in $`\ln\lambda`$ | unweighted linear step; bad pixels removed before resampling | two |
| albireo | wavelength, banded least squares | orbital elements, sampled, with the spectra integrated out; or per-epoch velocities | ranges of the orbital elements; light fractions | each epoch on its own pixels | per-pixel inverse variance; masks | two or more stars, hierarchical triples, telluric and nebular components |

## 2. Regularization and uncertainties

| Code | Regularization of the spectra | Light ratio | Uncertainty of the orbit | Uncertainty of the spectra | Implementation |
|---|---|---|---|---|---|
| KOREL | none; the lowest Fourier modes can be left out of the fit | line-strength factors per spectrum, held or fitted; the continuum ratio only from eclipses | $`\chi^2`$ maps over pairs of parameters (later versions) | none | Fortran 77; web service with registration (VO-KOREL); no licence stated; KOREL11b, 2011 |
| FDBinary, fd3 | none; singular modes take the minimum-norm solution | light factors per spectrum, given by the user | none reported by fd3; a jackknife in FDBinary V.2 | none | C with GSL; source download; no licence stated for fd3 (FDBinary V.2: GPL); fd3 v3.1, 2014 |
| CRES | truncated singular value decomposition, chosen by the user | light factors per spectrum, given by the user | not applicable (velocities given) | formal error per pixel | C with GSL; GPL; v2.2, 2006; not maintained |
| Spectangular | none | set by the user; a flux ratio per spectrum fitted with the orbit held (2019) | curvature of $`\chi^2`$ at the optimum | none | C++ with a Qt interface; Apache-2.0; last change 2023 |
| Shift-and-add | none explicit and no stopping rule; spectra can be held below the continuum | set by the user; it rescales the output only | contour of the $`\chi^2`$ map over $`K_2`$ or $`(K_1, K_2)`$, other elements held | none | Python; no licence stated; v2.0, 2023 |
| PSOAP | Gaussian-process prior, hyperparameters sampled | fixed afterwards by matching templates | posterior (Metropolis-Hastings) | posterior draws | Python and Cython; MIT; v0.1-beta, 2017; not maintained |
| Seeburger et al. (2024) | second-difference (Tikhonov) penalty, weight set by trial and error | fitted afterwards against templates | spread of refits with one epoch left out | none | Python; not public |
| albireo | second-difference penalty plus a continuum term, weights fitted by maximizing the marginal likelihood (ML-II) | declared; per-epoch fractions can be inferred | posterior (NUTS) | conditional draws at each posterior draw | Python with JAX; BSD-3-Clause |

## 3. The families of codes

**Fourier space.** On a grid uniform in $`\ln\lambda`$ a Doppler shift is a translation, and
each Fourier mode of the composite spectra decouples into a small linear system (Hadrava 1995).
KOREL and FDBinary/fd3 solve these systems and optimize the orbital elements around them. The
method is fast and assumes no template, but the spectra must be resampled onto a common
equidistant grid, per-pixel weights and masks cannot be used, and the periodicity of the
Fourier transform requires continuum at both ends of the fitted region. The lowest Fourier
modes are poorly determined, and small errors of the continuum normalization are amplified in
them into low-frequency undulations (Hadrava 2004; Ilijić et al. 2004; Hensberge, Ilijić &
Torres 2008).

**Wavelength space.** For given velocities the composite spectra are linear in the component
spectra (Simon & Sturm 1994), and the spectra follow from a sparse linear least-squares
problem. CRES and Spectangular solve it by singular value decomposition, Seeburger et al.
(2024) by an iterative sparse solver with a curvature penalty. The formulation admits per-pixel
weights, masks and spectra on different grids; CRES uses them, while Spectangular and
Seeburger et al. resample the spectra onto a common grid and do not weight the linear step.

**Iterative shift-and-add.** The current estimate of one component is subtracted from every
epoch, the residuals are shifted to the frame of the other and averaged, and the step is
repeated (Marchenko, Moffat & Eenens 1998). González & Levato (2006) added the measurement of
the velocities at every epoch. In its current use the semi-amplitudes are found by repeating
the iteration on a grid of $`K_2`$ or of $`(K_1, K_2)`$, with the other orbital elements taken
from the primary's orbit (Shenar et al. 2020, 2022). The method is simple and widely used,
in particular in work on massive binaries and on candidate compact companions. It has no
stopping rule, and stopping early leaves spurious line wings and wrong line strengths
(Quintero, Eenens & Rauw 2020).

**Probabilistic.** PSOAP (Czekala et al. 2017) places Gaussian-process priors on the component
spectra, integrates them out analytically, and samples the orbit together with the
hyperparameters of the priors. Its dense covariance matrices over epochs and pixels limited it
to spectral segments of about 10 Å. albireo uses the same structure with banded matrices, so
that the cost is linear in the number of pixels, and with a smoothness prior equivalent to the
curvature penalty of Seeburger et al. (2024). Unlike PSOAP, it fixes the prior's weights at
their ML-II values when it samples the orbit, so their uncertainty is not propagated.

## 4. Accuracy and speed on identical data

The [benchmark record](benchmarks.md#the-three-code-comparison-at-a-common-resolution-2026-10-15)
compares albireo with fd3 and with a reimplementation of shift-and-add written for the
benchmark from the published algorithm, on one simulated SB2 (20 epochs, signal-to-noise
ratio 100, a Gaussian line-spread function of 6 km/s, a common $`\ln\lambda`$ grid so that no
code resamples) with the orbit held at its injected values for every code. fd3 and
shift-and-add take no line-spread function and return the spectra at the instrumental
resolution, while albireo is given the function and returns the spectra deconvolved. All
three are therefore scored at the instrumental resolution, against the injected spectra
convolved with the line-spread function, and albireo's spectra are convolved with the same
function before scoring.

| | RMS error, primary | RMS error, secondary | Wall-clock time | Uncertainty on the spectra |
|---|---|---|---|---|
| albireo | 0.0033 | 0.0052 | 0.059 s | yes |
| fd3 | 0.0055 | 0.0090 | 0.064 s | no |
| shift-and-add, 7 iterations | 0.0145 | 0.0226 | 0.026 s | no |
| shift-and-add, 20,000 iterations | 0.0041 | 0.0064 | 103 s | no |

The errors are measured in the line cores after a constant offset is removed from each
spectrum. The data do not determine that offset: albireo's smoothness prior sets it, and the
other two codes leave it to the user. Before the removal fd3's errors are 21 times albireo's.
Over twenty noise realizations albireo's errors are 0.6 times fd3's. The two codes have the
same error at low frequencies and in the lines, and differ above the cutoff of the
line-spread function, where albireo, which is given that function, removes the noise. Given
no line-spread function, as the other codes are, albireo's errors are within 10% of fd3's at
the fixed regularization weights of the benchmark and 0.66 times fd3's with the weights
fitted. The errors of shift-and-add depend on the number of iterations. At the seven
iterations of the published method they are 4.4 times albireo's, and almost all of the
error lies at low frequencies, where the iteration converges slowly; the iteration reaches
its limit after about 20,000 iterations. The times are those of the [re-run on one
machine](benchmarks.md#re-run-of-the-three-code-comparison-on-one-machine-2026-08-16), with
fd3's linear algebra restricted to one thread, and the 20,000 iterations were timed in the
same way. Shift-and-add is the fastest of the three at seven iterations.

## 5. Disentangling and template fitting

Seeburger et al. (2024, Section 2.2) distinguish disentangling, which reconstructs each
component's spectrum without a template, from spectral model fitting, which describes the
composite with templates from a library. The disentangling in albireo uses no template. Two
optional stages of albireo do: the label fit compares a disentangled component with a
synthetic grid, and TODCOR measures epoch velocities against templates. Both are separate from
the disentangling and are described in the tutorials on [labels](tutorials/labels.md) and
[TODCOR](tutorials/todcor.md).

## 6. Limitations of albireo

- **Speed.** At the seven iterations of the published method, shift-and-add is faster on
  every machine measured (Section 4), and the NUTS stage takes minutes per system on a CPU.
- **The light ratio.** As for every disentangling code, the continuum light ratio is not
  measurable from the spectra when the light fractions are constant. albireo requires it to be
  declared and does not infer it.
- **The regularization weights** are fitted and then held fixed when the orbit is sampled, so
  their uncertainty is not propagated (a plug-in approximation).
- **Time-invariant spectra.** The component spectra are assumed constant in time, apart from
  per-epoch light fractions and a nebular component with a per-epoch amplitude. Line-profile
  variability is not modelled.
- **Software.** albireo requires JAX in double precision; a speed-up on a GPU needs one with
  fast double-precision arithmetic.

## Sources

The entries were checked against these sources in October 2026.

- **KOREL.** The Fourier solution and the joint fit of the spectra and the orbital elements:
  Hadrava (1995, Sections 2 and 3). Line-strength factors, and the continuum ratio from
  eclipses: Hadrava (1997). Per-epoch velocities, the input files, the grid of $`2^n`$ bins per
  spectral region, the weight per spectrum, up to five stars on four orbits, telluric lines,
  the exclusion of the lowest Fourier modes, and the $`\chi^2`$ maps over pairs of parameters:
  the updated user guide (Hadrava 2009). Error ellipses from those maps: Hadrava (2025).
  Distribution as a web service with registration: Škoda & Hadrava (2010) and the
  [VO-KOREL page](http://stel.asu.cas.cz/vo-korel/). Language and licence: Hadrava (2009) and
  the [KOREL page](http://space.asu.cas.cz/~had/korel.html).
- **FDBinary and fd3.** The method: Ilijić et al. (2004). The fit of any subset of the orbital
  elements by a downhill simplex with restarts, the noise level and light factors per
  spectrum, the single common grid, the treatment of singular modes, and the absence of
  reported uncertainties: the source code and examples of fd3 v3.1 on the
  [fd3 page](http://sail.zpf.fer.hr/fdbinary/). The jackknife and the GPL of FDBinary V.2: the
  archived FDBinary pages. Catalogue entries: ascl:1705.011 (FDBinary) and ascl:1705.012 (fd3).
- **CRES.** The [CRES page](http://sail.zpf.fer.hr/cres/), the source file `cres.c` (v2.2), and
  Ilijić (2004).
- **Spectangular.** Sablowski & Weber (2017, Sections 2, 3 and 5), Sablowski, Järvinen & Weber
  (2019), and the manual and source code in the
  [repository](https://github.com/DPSablowski/Spectangular).
- **Shift-and-add.** The iteration: Marchenko, Moffat & Eenens (1998) and González & Levato
  (2006, Section 2), whose averages may include weights or a rejection of pixels. The
  $`\chi^2`$ grid over $`K_2`$ for LB-1 and HR 6819: Shenar et al. (2020, Section 4) and
  Bodensteiner et al. (2020, Section 4); over $`(K_1, K_2)`$, with the period, time of
  periastron, $`e`$ and $`\omega`$ held at the primary's orbit: Shenar et al. (2022, Section
  3.1). The light ratio supplied by the user, the constraint that holds the spectra below the
  continuum, triples, and the absence of a licence: the README of the public implementation
  ([Disentangling_Shift_And_Add](https://github.com/TomerShenar/Disentangling_Shift_And_Add),
  v2.0). Convergence and the effect of stopping early: Quintero, Eenens & Rauw (2020). The
  comparison of Section 4 uses a reimplementation of the published algorithm, not this code.
- **PSOAP.** Czekala et al. (2017, Sections II and III), and the documentation and source code
  in the [repository](https://github.com/iancze/PSOAP).
- **Seeburger et al. (2024).** Sections 2.3 to 2.6, 3 and 4 of the paper. No code has been
  released.
- **albireo.** The [method overview](method.md) and the
  [mathematical foundations](math.md).

## References

- Bodensteiner, J., Shenar, T., Mahy, L., et al. 2020, A&A, 641, A43. [ADS](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A..43B)
- Czekala, I., Mandel, K. S., Andrews, S. M., et al. 2017, ApJ, 840, 49. [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...840...49C)
- González, J. F. & Levato, H. 2006, A&A, 448, 283. [ADS](https://ui.adsabs.harvard.edu/abs/2006A%26A...448..283G)
- Hadrava, P. 1995, A&AS, 114, 393. [ADS](https://ui.adsabs.harvard.edu/abs/1995A%26AS..114..393H)
- Hadrava, P. 1997, A&AS, 122, 581. [ADS](https://ui.adsabs.harvard.edu/abs/1997A%26AS..122..581H)
- Hadrava, P. 2004, Publications of the Astronomical Institute of the Czechoslovak Academy of Sciences, 92, 15. [ADS](https://ui.adsabs.harvard.edu/abs/2004PAICz..92...15H)
- Hadrava, P. 2009, arXiv:0909.0172. [arXiv](https://arxiv.org/abs/0909.0172)
- Hadrava, P. 2025, Contributions of the Astronomical Observatory Skalnaté Pleso, 55, 325. [DOI](https://doi.org/10.31577/caosp.2025.55.3.325)
- Hensberge, H., Ilijić, S. & Torres, K. B. V. 2008, A&A, 482, 1031. [ADS](https://ui.adsabs.harvard.edu/abs/2008A%26A...482.1031H)
- Ilijić, S. 2004, in ASP Conf. Ser. 318, Spectroscopically and Spatially Resolving the Components of the Close Binary Stars, 107. [ADS](https://ui.adsabs.harvard.edu/abs/2004ASPC..318..107I)
- Ilijić, S., Hensberge, H., Pavlovski, K. & Freyhammer, L. M. 2004, in ASP Conf. Ser. 318, Spectroscopically and Spatially Resolving the Components of the Close Binary Stars, 111. [ADS](https://ui.adsabs.harvard.edu/abs/2004ASPC..318..111I)
- Marchenko, S. V., Moffat, A. F. J. & Eenens, P. R. J. 1998, PASP, 110, 1416. [ADS](https://ui.adsabs.harvard.edu/abs/1998PASP..110.1416M)
- Quintero, E. A., Eenens, P. & Rauw, G. 2020, Astronomische Nachrichten, 341, 628. [ADS](https://ui.adsabs.harvard.edu/abs/2020AN....341..628Q)
- Sablowski, D. P. & Weber, M. 2017, A&A, 597, A125. [ADS](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A.125S)
- Sablowski, D. P., Järvinen, S. & Weber, M. 2019, A&A, 623, A31. [ADS](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..31S)
- Seeburger, R., Rix, H.-W., El-Badry, K., Xiang, M. & Fouesneau, M. 2024, MNRAS, 530, 1935. [ADS](https://ui.adsabs.harvard.edu/abs/2024MNRAS.530.1935S)
- Shenar, T., Bodensteiner, J., Abdul-Masih, M., et al. 2020, A&A, 639, L6. [ADS](https://ui.adsabs.harvard.edu/abs/2020A%26A...639L...6S)
- Shenar, T., Sana, H., Mahy, L., et al. 2022, A&A, 665, A148. [ADS](https://ui.adsabs.harvard.edu/abs/2022A%26A...665A.148S)
- Simon, K. P. & Sturm, E. 1994, A&A, 281, 286. [ADS](https://ui.adsabs.harvard.edu/abs/1994A%26A...281..286S)
- Škoda, P. & Hadrava, P. 2010, in ASP Conf. Ser. 435, Binaries - Key to Comprehension of the Universe, 71. [ADS](https://ui.adsabs.harvard.edu/abs/2010ASPC..435...71S)
