# Scientific background

This page summarises the science that albireo implements and places each part of the
package in the context of the published literature. It is written for astronomers who
work with spectroscopic binaries and is intended to be read before the
[mathematical foundations](math.md), which give the equations, and the
[design document](design.md), which records the decisions. Every method with a literature
source is cited in the text, and the [reference list](#references) at the end carries an
ADS link for each entry.

## 1. Spectroscopic binaries and the disentangling problem

Most binary stars are not spatially resolved. A spectrograph records a single composite
spectrum in which the absorption lines of both stars are superposed, each Doppler-shifted
by its own orbital motion and blended differently at every epoch. A system in which only
one set of lines is visible is a single-lined spectroscopic binary (SB1); one in which
both are visible is double-lined (SB2), and triple-lined systems (SB3) also occur.

Two quantities are wanted from such a time series. The first is the orbit: the period
$P$, eccentricity $e$, argument of periastron $\omega$, a reference epoch, and the
velocity semi-amplitudes $K_1$ and $K_2$. Together with an inclination from eclipses or
astrometry these give the dynamical masses and the semi-major axis (Hilditch 2001;
Torres, Andersen & Giménez 2010). The second is the spectrum of each component on its own,
from which effective temperatures, surface gravities, rotation rates and abundances follow.

*Spectral disentangling* is the inverse problem that recovers both from the composite
spectra without a template for either star. Its scientific reach is wide: the multiplicity
of massive stars (Sana et al. 2012), where most systems will interact and where the
components must be characterised separately; benchmark eclipsing binaries, where a
disentangled spectrum feeds a model-atmosphere analysis of each star; and searches for
dark companions, where the presence or absence of a faint second set of lines decides
between a compact object and a stripped star (Section 5). Pavlovski & Hensberge (2010)
review the method and its applications.

## 2. Methods of spectral disentangling

### 2.1 Tomographic separation

Bagnuolo & Gies (1991) introduced the first practical decomposition: with the radial
velocities of both components known at every epoch, the component spectra are recovered
by an iterative tomographic algorithm. The method requires the velocities as input and
therefore depends on a prior cross-correlation analysis.

### 2.2 Wavelength-space linear inversion

Simon & Sturm (1994) recognised that, for given velocities, the composite spectra are
linear in the unknown component spectra. Disentangling then becomes a large sparse linear
least-squares problem, solved by singular value decomposition, with the orbital elements
optimised in an outer loop. The formulation accepts arbitrary per-epoch weights and masks
and arbitrary velocity laws. Its limitations were the cost of the decomposition for large
systems and the absence of regularisation for the ill-conditioned low-frequency modes
discussed in Section 3. albireo follows this formulation.

### 2.3 Fourier-space disentangling

Hadrava (1995) showed that on a grid uniform in $\ln\lambda$ a Doppler shift is a
translation, so that each Fourier mode of the composite spectra decouples into a small
linear system. The resulting code, KOREL, solves for the component spectra and the orbital
elements simultaneously and was later extended to variable line strengths (Hadrava 1997)
and provided as a Virtual Observatory service (Škoda & Hadrava 2010; Hadrava 2004 is the
user guide). FDBinary and its successor fd3 (Ilijić et al. 2004) implement the same
Fourier approach as an open command-line program. The Fourier methods are fast and
template-free but require the data to be resampled onto a common equidistant
log-wavelength grid, and the lowest Fourier modes are poorly determined by the data, which
produces the low-frequency undulations analysed by Ilijić, Hensberge & Pavlovski (2001)
and Hensberge, Ilijić & Torres (2008).

### 2.4 Iterative shift-and-add

González & Levato (2006) proposed an iterative scheme: subtract the current estimate of
one component, shift the residuals to the rest frame of the other, average, and repeat,
with the semi-amplitudes chosen by a $\chi^2$ grid over $(K_1, K_2)$. Shenar et al. (2020)
applied it to LB-1, and the same approach identified the companion of HR 6819
(Bodensteiner et al. 2020). It is simple and robust at low signal-to-noise, and it is the
most widely used method in the massive-star and compact-companion literature. Its costs are
a grid search whose size grows exponentially with the number of parameters, an implicit
regularisation set by when the iteration is stopped, light ratios that must be fixed by
hand, and no estimate of uncertainty on the recovered spectra. Quintero, Eenens & Rauw
(2020) analyse the artefacts of the algorithm and propose corrections.

### 2.5 Singular value decomposition with global optimisation

Spectangular (Sablowski & Weber 2017, 2019) returns to the wavelength-space formulation
with a global optimiser over the orbital elements or the per-epoch velocities and a
graphical interface. It supports weights and masks and variable line strengths but returns
point estimates only.

### 2.6 Probabilistic and survey-scale approaches

Czekala et al. (2017) formulated disentangling as a joint Bayesian inference in wavelength
space with Gaussian-process priors on the component spectra, and marginalised the spectra
analytically. That work (PSOAP) is the closest methodological predecessor of albireo. Its
dense covariance matrices over epochs and pixels scale cubically in time and quadratically
in memory, which restricted it to narrow spectral chunks. Seeburger et al. (2024) target
survey-scale data with a second-derivative Tikhonov penalty and sparse iterative solves,
without a posterior. Sairam et al. (2024) model an SB2 as a sum of two Doppler-shifted
Gaussian processes with the aim of precise radial velocities for circumbinary-planet
searches. For the SB1 regime, González, Martínez & Alejo (2024) search a grid in mass
ratio, subtract the primary and cross-correlate the residuals against templates.

### 2.7 The approach taken by albireo

albireo combines the wavelength-space linear model of Simon & Sturm (1994) with the
analytic marginalisation of Czekala et al. (2017), and replaces dense covariances with
banded precision matrices so that the cost is linear in the number of pixels. The
component spectra are given Gaussian smoothness priors whose precision is a scaled
second-difference operator plus a weak ridge; the deterministic limit of that prior is the
Tikhonov penalty of Seeburger et al. (2024). Conditional on the orbit and the other
nonlinear parameters the model is linear-Gaussian in the spectra, so the spectra are
integrated out in closed form and only the low-dimensional nonlinear parameters are
sampled, by Hamiltonian Monte Carlo. The spectra and their full covariance are recovered
afterwards as a conditional Gaussian at each posterior draw. The model is evaluated on
each epoch's native wavelength grid, so the data are never resampled and masks, chip gaps,
per-pixel weights and mixed instruments are handled without special cases. The equations
are in [Mathematical foundations](math.md), Sections 1 to 4.

## 3. Degeneracies intrinsic to the problem

Several quantities cannot be determined from the composite spectra alone, whatever the
method. albireo makes each of them proper with an explicit prior scale, reports it in the
posterior, and requires a user decision where only external information can resolve it
([math.md, Section 5](math.md#5-degeneracies-and-identifiability)).

**Low-frequency separation.** The data constrain the sum of the component spectra at every
spatial frequency, but the difference between them only through the epoch-to-epoch
variation of the relative Doppler shift. Features broader than the root-mean-square
differential shift cannot be attributed to one star or the other, and the constant term is
exactly unconstrained. This is the origin of the undulations in Fourier-disentangled spectra
(Ilijić et al. 2001; Hensberge et al. 2008). In albireo the smoothness prior sets the scale
of these modes explicitly, the posterior covariance reports their inflation, and the
uncertainty band widens where the data give no leverage.

**Light ratio and line depth.** With constant light fractions the composite spectrum
depends only on the product of each component's light fraction and its line depths, so the
continuum light ratio is not measurable from the spectra alone (Pavlovski & Hensberge
2010; Tamajo, Pavlovski & Southworth 2011). It is broken by per-epoch light variation
during eclipses, by external photometry, or by assumption. Because an assumed light ratio
propagates into every line depth and every atmospheric parameter derived from it, albireo
has no default light fraction: the treatment must be declared, and per-epoch light
fractions can be inferred where eclipses exist. The interpretation of LB-1 and HR 6819
turned on exactly this choice (Shenar et al. 2020; Bodensteiner et al. 2020; El-Badry &
Quataert 2021).

**Systemic velocity.** Translating every component spectrum by the same amount and
shifting the systemic velocity by the opposite amount leaves the composite unchanged, so
$\gamma$ is not identified by disentangling and must be measured afterwards from the
disentangled spectra against a template. When per-epoch velocities are fitted instead of
a Keplerian, the same argument applies once per component: each star's velocities carry
an arbitrary zero point, while the semi-amplitudes and the mass ratio, which is the slope
of one velocity against the other (Wilson 1941), survive.

**Line-spread function and intrinsic line width.** A broader instrumental profile
combined with intrinsically narrower lines is observationally almost identical to the
reverse, so a template-free model cannot measure an absolute LSF width. Instruments that
share the same component spectra identify the differences between their widths; the
absolute scale must come from one instrument whose profile is known.

## 4. The albireo model

### 4.1 Forward model

The model grid is uniform in $x = \ln\lambda$ so that a Doppler shift is a translation
(Hadrava 1995). The mapping from velocity to log-shift uses the relativistic radial
Doppler formula, $\xi(v) = \operatorname{artanh}(v/c)$, which is exactly antisymmetric so
that shifts compose and invert exactly; the classical form differs by about
0.6 km s$^{-1}$ at 600 km s$^{-1}$. Each component's deviation from the continuum is
shifted, convolved with the instrument's line-spread function, weighted by its light
fraction, summed, projected onto the epoch's native pixels by a flux-conserving rebinning
operator, and multiplied by a low-order response polynomial. Telluric absorption is an
additional component at rest in the observatory frame, treated additively to first order.
Nebular emission from an H II region (Osterbrock & Ferland 2006) is a component at rest in
the barycentric frame with a free amplitude per exposure and a prior that confines it to
its lines. Light fractions may be constant or vary per epoch, the latter for eclipsing
systems. The barycentric correction is composed inside the model rather than applied to
the data.

### 4.2 Priors and analytic marginalisation

Each deviation spectrum receives an independent Gaussian prior whose precision is
$\tau\,\mathbf{D}_2^\top\mathbf{D}_2 + \eta\,\mathbf{I}$, a curvature penalty plus a weak
ridge. The ridge makes the affine null space of the curvature penalty proper; these are
exactly the low-frequency directions of Section 3. Because the model is linear-Gaussian in
the spectra conditional on the nonlinear parameters, the marginal likelihood has the
standard closed form for a linear-Gaussian model (Rasmussen & Williams 2006, Chapter 2).
The posterior precision of the stacked spectra is block-tridiagonal, is assembled
analytically epoch by epoch, and is factorised by a block Cholesky decomposition; the
log-determinant, the conditional mean, and the marginal variances of the spectra follow
from the same factor, the last through the selected-inverse recursion of Takahashi, Fagan
& Chin (1973) and Erisman & Tinney (1975). The exploitation of linear structure to
marginalise a high-dimensional latent function at linear cost is the same strategy that
celerite applies to Gaussian-process time series (Foreman-Mackey et al. 2017). Gradients
of the marginal likelihood are obtained in closed form through the same recursion rather
than by reverse-mode differentiation of the factorisation.

### 4.3 Inference

Inference over the nonlinear parameters proceeds in three stages. A maximum a posteriori
fit by L-BFGS also maximises the marginal likelihood over the prior hyperparameters
$(\tau, \eta)$, which is the type-II maximum likelihood or empirical Bayes estimate
(MacKay 1992). A Laplace approximation at the optimum supplies the inverse mass matrix.
The posterior is then sampled with the No-U-Turn Sampler (Hoffman & Gelman 2014;
Betancourt 2017) as implemented in numpyro (Phan, Pradhan & Jankowiak 2019; Bingham et al.
2019). Posterior calibration is checked by injection and recovery on simulated data, in the
manner of simulation-based calibration (Talts et al. 2018); the results are recorded in
the [benchmark record](benchmarks.md).

### 4.4 Orbits and conventions

Radial velocities follow the standard Keplerian law (Hilditch 2001) with Kepler's equation
solved by Newton iteration; the gradient with respect to the elements is supplied by the
implicit-function theorem so that it is exact at the converged solution. The eccentricity
and argument of periastron are sampled as $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$,
which is smooth through $e = 0$ and maps a uniform prior on the unit disk to a uniform
prior on $e$ (Ford 2006; Eastman, Gaudi & Agol 2013). The reference epoch is the time of
conjunction, which remains defined for circular orbits. Times are barycentric Julian dates
in the TDB time scale (Eastman, Siverd & Gaudi 2010), and the barycentric correction to
the velocities follows Wright & Eastman (2014). Minimum masses and projected semi-major
axes follow the relations collected by Hilditch (2001).

### 4.5 Instrumental effects and preprocessing

The line-spread function is Gaussian in velocity, constant or varying with wavelength,
with an optional asymmetric term parameterised by the Gauss-Hermite coefficient $h_3$
(van der Marel & Franx 1993). Rotational broadening of synthetic templates uses the
limb-darkened profile of Gray (2005). Conversion between air and vacuum wavelengths uses
the Edlén (1966) dispersion formula as revised by Birch & Downs (1994), the form tabulated
by Morton (2000); the two scales differ by about 83 km s$^{-1}$ in the optical, and albireo
requires the medium to be declared wherever absolute line positions matter. Where reduced
spectra carry no usable error array, the noise is estimated from the spectrum itself with
the DER_SNR estimator (Stoehr et al. 2008). Correlated noise introduced by pipeline
resampling is modelled as a first-order autoregressive process per epoch, with the
tridiagonal precision and the determinant in closed form. Archival spectra are identified
by their IVOA Spectrum Data Model utypes, because column names, units and quality
conventions differ between ESO instrument pipelines.

## 5. Faint companions: the SB1 scan and detection limits

Candidate dormant black holes such as LB-1 (Liu et al. 2019) and HR 6819 (Rivinius et al.
2020) were reinterpreted when disentangling revealed a luminous but faint companion whose
lines had not been recognised in the composite spectra (Shenar et al. 2020; Bodensteiner
et al. 2020; El-Badry & Quataert 2021), a conclusion later confirmed for HR 6819 by
interferometry (Frost et al. 2022; Klement et al. 2025). Genuine dormant compact objects
have since been identified astrometrically (El-Badry et al. 2023). Whether a second set of
lines is present, at what light fraction it would have been seen, and how often noise
alone produces a comparable signal are therefore the questions a companion search must
answer.

albireo's SB1 mode scans the companion semi-amplitude $K_2$ with the companion spectrum
marginalised at every trial, which is a matched filter that assumes no template; the
detection statistic is twice the log ratio of the marginal likelihoods with and without
the companion ([math.md, Section 6](math.md#6-sb1-faint-companion-mode-k_2-scan)). The
primary's semi-amplitude may be integrated out over a Gaussian prior instead of held at a
literature value, because an error in $K_1$ leaves coherent primary signal that the free
companion spectrum absorbs, which both distorts the recovered companion and increases the
detection statistic. The statistic has no closed-form null distribution, so the false-alarm
probability and the completeness are measured by injection and recovery through the
observed data's own operators, and the result is a completeness limit at a stated
confidence. The light fraction of the putative companion must be chosen explicitly, for the
reason given in Section 3.

## 6. Observing-strategy forecasts

The posterior covariance of the component spectra depends on the epoch times, the
weights, the masks, the line-spread functions, the light fractions and the prior, but not
on the fluxes. It can therefore be evaluated for observations that have not yet been
taken, given an assumed orbit. albireo reports the pointwise uncertainty band, the
worst-determined spectral modes, the number of constrained degrees of freedom, and the
expected information gain, which for a linear-Gaussian model is the Bayesian
D-optimality criterion (Chaloner & Verdinelli 1995). The variance of the differential
shift over the epochs, which the small-frequency expansion of Section 3 suggests as a
figure of merit, is shown to rank cadences incorrectly when the cadence is aliased to the
orbital period ([math.md, Section 5.5](math.md#55-forecasting-a-design-d47)).

## 7. Atmospheric parameters from disentangled spectra

Disentangled spectra are routinely analysed with model atmospheres to obtain effective
temperatures, gravities and abundances of each component (Pavlovski & Hensberge 2005;
Pavlovski, Southworth & Tamajo 2018; Mahy et al. 2020). The uncertainty introduced by the
disentangling itself is usually not propagated: the residuals of a disentangled spectrum
are correlated across wavelength, and formal errors from a fit that treats them as white
are optimistic by factors of several (Czekala et al. 2015; Gebruers et al. 2022). Kıran et
al. (2016) estimated the effect by adding noise to a disentangled profile and refitting
repeatedly. albireo supplies the disentangling posterior directly: draws from the joint
posterior of both components, correlated across wavelength and between the stars, can be
exported for analysis with GSSP (Tkachenko 2015), iSpec (Blanco-Cuaresma et al. 2014),
Korg (Wheeler et al. 2023) or SME (Piskunov & Valenti 2017), and the spread of the fitted
parameters across draws is the disentangling contribution to their uncertainty. The light
ratio, which is assumed rather than inferred under constant light fractions, is not
included in that spread.

For the purpose of measuring epoch radial velocities, albireo also fits $T_{\rm eff}$,
$\log g$, [M/H] and $v\sin i$ to the disentangled components against published synthetic
grids, so that each component can be rendered as a template. The grids are read, not
computed: BOSZ (Bohlin et al. 2017; Mészáros et al. 2024), POLLUX (Palacios et al. 2010),
PHOENIX (Husser et al. 2013) and TLUSTY (Lanz & Hubeny 2003, 2007) are the public
libraries of this kind. Interpolation is performed in flux rather than in the model
atmospheres, which Mészáros & Allende Prieto (2013) found to be more accurate on the grid
spacing BOSZ uses, and a cubic interpolant on a well-sampled grid reaches an accuracy
comparable to neural-network emulators such as The Payne (Ting et al. 2019). The dilution
of each component is fitted jointly through a shared radius ratio with wavelength-dependent
light fractions that sum to one, following the binary mode of GSSP (Tkachenko 2015), and
the constant offset that disentangling leaves unconstrained is absorbed by an additive
nuisance term and reported. The accuracy that a radial-velocity template requires is
modest: template temperature errors of several hundred kelvin bias solar-type velocities by
of order 0.2 km s$^{-1}$ (Posbic et al. 2012), least-squares-deconvolution profiles are
insensitive to comparable label changes (Tkachenko et al. 2022), and the dominant effect of
a wrong template is a constant velocity offset per component, which was the origin of the
hot-star radial-velocity offsets in Gaia DR3 before correction (Blomme et al. 2023).

## 8. Epoch radial velocities

Cross-correlation against a template (Tonry & Davis 1979) is the standard estimator of a
single star's radial velocity. For double-lined systems the two-dimensional correlation of
Zucker & Mazeh (1994), TODCOR, correlates each observed spectrum against a combination of
two templates with independent shifts and reads both velocities from the location of the
maximum, which removes the bias that blended peaks introduce in a one-dimensional
correlation near conjunction. Extensions to three and four components were given by Zucker,
Torres & Mazeh (1995) and Torres, Latham & Stefanik (2007), and Zucker (2003) showed that
the correlation peak is a maximum-likelihood estimate, which provides its uncertainty and a
rule for combining spectral orders. Broadening functions (Rucinski 2002) are an alternative
that recovers the velocity profile itself.

albireo evaluates the same estimator as the weighted least-squares fit of $N$ shifted,
LSF-convolved templates projected onto each epoch's pixels
([math.md, Section 10](math.md#10-epoch-velocities-by-n-dimensional-correlation-d56d57)).
On a uniform grid with uniform weights this reproduces the TODCOR expressions exactly; the
least-squares form additionally admits masks, chip gaps, per-pixel weights, mixed
instruments, exact fractional shifts and the maximum-likelihood errors of Zucker (2003).
Templates may come from a synthetic library, from the label fit of Section 7, or from the
disentangled components themselves. Velocities measured against a disentangled component
are differential, with one unknown zero point per component, for the reason given in
Section 3. A Keplerian is fitted to the resulting table by weighted nonlinear least
squares, with a period search by the Lomb-Scargle periodogram (Lomb 1976; Scargle 1982;
VanderPlas 2018) and with one systemic velocity per component whenever a component is
differential.

## 9. Systems used for validation

**AI Phoenicis** is a detached eclipsing binary with a K-type subgiant and an F-type
main-sequence star on a 24.6 d orbit, with masses known to 0.3 per cent (Andersen et al.
1988; Kirkby-Kent et al. 2016) and independently measured effective temperatures (Maxted
et al. 2020; Miller, Maxted & Smalley 2020). Archival HARPS spectra of this system are the
real-data test of the label fit ([tutorial](tutorials/aiphe-labels.ipynb)).

**HR 6819** was proposed as a triple system containing a dormant black hole (Rivinius et
al. 2020) and reinterpreted as a binary of a stripped star and a rapidly rotating Be star
(Bodensteiner et al. 2020; El-Badry & Quataert 2021), which interferometry confirmed
(Frost et al. 2022; Klement et al. 2025). Its public FEROS spectra are the real-data test
of the disentangling and of the instrumental model
([tutorial](tutorials/real-data.md)).

**BLOeM**, the Binarity at LOw Metallicity survey (Shenar et al. 2024), obtained about 25
epochs of VLT/FLAMES spectroscopy for 929 massive stars in the Small Magellanic Cloud.
Villaseñor et al. (2025) classify the early B-type dwarfs and giants and identify 59
double-lined systems, and Bodensteiner et al. (2025) the B-type supergiants. albireo can
resolve a BLOeM identifier to its archival spectra ([tutorial](tutorials/bloem-sb2.md)).

**V453 Cygni** is the published test case distributed with fd3 (Pavlovski & Southworth
2009); its output is used to validate the fd3 build against which albireo is benchmarked.

**LB-1** (Liu et al. 2019; Shenar et al. 2020) is the reference case for the SB1 companion
scan, which is validated on simulated data with injected companions.

## 10. Software foundations

albireo is written in JAX (Bradbury et al. 2018) with NumPy (Harris et al. 2020) and SciPy
(Virtanen et al. 2020). Sampling uses numpyro (Phan et al. 2019; Bingham et al. 2019) and
optimisation uses optax. Optional dependencies are astropy (Astropy Collaboration 2013,
2018, 2022) for FITS input and output, matplotlib (Hunter 2007) for figures, and ArviZ
(Kumar et al. 2019) for posterior diagnostics. The [citing](citing.md) page lists what to
cite for each.

## References

Each entry links to its record in the NASA Astrophysics Data System where one exists.

- Andersen, J., Clausen, J. V., Gustafsson, B., Nordström, B. & VandenBerg, D. A. 1988, A&A, 196, 128. [ADS](https://ui.adsabs.harvard.edu/abs/1988A%26A...196..128A)
- Astropy Collaboration 2013, A&A, 558, A33. [ADS](https://ui.adsabs.harvard.edu/abs/2013A%26A...558A..33A)
- Astropy Collaboration 2018, AJ, 156, 123. [ADS](https://ui.adsabs.harvard.edu/abs/2018AJ....156..123A)
- Astropy Collaboration 2022, ApJ, 935, 167. [ADS](https://ui.adsabs.harvard.edu/abs/2022ApJ...935..167A)
- Bagnuolo, W. G. & Gies, D. R. 1991, ApJ, 376, 266. [ADS](https://ui.adsabs.harvard.edu/abs/1991ApJ...376..266B)
- Betancourt, M. 2017, arXiv:1701.02434. [ADS](https://ui.adsabs.harvard.edu/abs/2017arXiv170102434B)
- Bingham, E., Chen, J. P., Jankowiak, M., et al. 2019, Journal of Machine Learning Research, 20, 28. [ADS](https://ui.adsabs.harvard.edu/abs/2018arXiv181009538B)
- Birch, K. P. & Downs, M. J. 1994, Metrologia, 31, 315. [ADS](https://ui.adsabs.harvard.edu/abs/1994Metro..31..315B)
- Blanco-Cuaresma, S., Soubiran, C., Heiter, U. & Jofré, P. 2014, A&A, 569, A111. [ADS](https://ui.adsabs.harvard.edu/abs/2014A%26A...569A.111B)
- Blomme, R., Frémat, Y., Sartoretti, P., et al. 2023, A&A, 674, A7. [ADS](https://ui.adsabs.harvard.edu/abs/2023A%26A...674A...7B)
- Bodensteiner, J., Shenar, T., Mahy, L., et al. 2020, A&A, 641, A43. [ADS](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A..43B)
- Bodensteiner, J., Sana, H., Dufton, P. L., et al. 2025, A&A, 698, A40. [ADS](https://ui.adsabs.harvard.edu/abs/2025A%26A...698A..40B)
- Bohlin, R. C., Mészáros, Sz., Fleming, S. W., et al. 2017, AJ, 153, 234. [ADS](https://ui.adsabs.harvard.edu/abs/2017AJ....153..234B)
- Bradbury, J., Frostig, R., Hawkins, P., et al. 2018, JAX: composable transformations of Python+NumPy programs. [GitHub](https://github.com/jax-ml/jax)
- Chaloner, K. & Verdinelli, I. 1995, Statistical Science, 10, 273. [DOI](https://doi.org/10.1214/ss/1177009939)
- Czekala, I., Andrews, S. M., Mandel, K. S., Hogg, D. W. & Green, G. M. 2015, ApJ, 812, 128. [ADS](https://ui.adsabs.harvard.edu/abs/2015ApJ...812..128C)
- Czekala, I., Mandel, K. S., Andrews, S. M., et al. 2017, ApJ, 840, 49. [ADS](https://ui.adsabs.harvard.edu/abs/2017ApJ...840...49C)
- Eastman, J., Siverd, R. & Gaudi, B. S. 2010, PASP, 122, 935. [ADS](https://ui.adsabs.harvard.edu/abs/2010PASP..122..935E)
- Eastman, J., Gaudi, B. S. & Agol, E. 2013, PASP, 125, 83. [ADS](https://ui.adsabs.harvard.edu/abs/2013PASP..125...83E)
- Edlén, B. 1966, Metrologia, 2, 71. [ADS](https://ui.adsabs.harvard.edu/abs/1966Metro...2...71E)
- El-Badry, K. & Quataert, E. 2021, MNRAS, 502, 3436. [ADS](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.3436E)
- El-Badry, K., Rix, H.-W., Quataert, E., et al. 2023, MNRAS, 518, 1057. [ADS](https://ui.adsabs.harvard.edu/abs/2023MNRAS.518.1057E)
- Erisman, A. M. & Tinney, W. F. 1975, Communications of the ACM, 18, 177. [DOI](https://doi.org/10.1145/360680.360704)
- Ford, E. B. 2006, ApJ, 642, 505. [ADS](https://ui.adsabs.harvard.edu/abs/2006ApJ...642..505F)
- Foreman-Mackey, D., Agol, E., Ambikasaran, S. & Angus, R. 2017, AJ, 154, 220. [ADS](https://ui.adsabs.harvard.edu/abs/2017AJ....154..220F)
- Frost, A. J., Bodensteiner, J., Rivinius, Th., et al. 2022, A&A, 659, L3. [ADS](https://ui.adsabs.harvard.edu/abs/2022A%26A...659L...3F)
- Gebruers, S., Tkachenko, A., Bowman, D. M., et al. 2022, A&A, 665, A36. [ADS](https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..36G)
- González, J. F. & Levato, H. 2006, A&A, 448, 283. [ADS](https://ui.adsabs.harvard.edu/abs/2006A%26A...448..283G)
- González, J. F., Martínez, M. J. & Alejo, A. D. 2024, A&A, 690, A124. [ADS](https://ui.adsabs.harvard.edu/abs/2024A%26A...690A.124G)
- Gray, D. F. 2005, The Observation and Analysis of Stellar Photospheres, 3rd ed. (Cambridge: Cambridge University Press). [ADS](https://ui.adsabs.harvard.edu/abs/2005oasp.book.....G)
- Hadrava, P. 1995, A&AS, 114, 393. [ADS](https://ui.adsabs.harvard.edu/abs/1995A%26AS..114..393H)
- Hadrava, P. 1997, A&AS, 122, 581. [ADS](https://ui.adsabs.harvard.edu/abs/1997A%26AS..122..581H)
- Hadrava, P. 2004, Publications of the Astronomical Institute of the Czechoslovak Academy of Sciences, 92, 15. [ADS](https://ui.adsabs.harvard.edu/abs/2004PAICz..92...15H)
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 2020, Nature, 585, 357. [ADS](https://ui.adsabs.harvard.edu/abs/2020Natur.585..357H)
- Hensberge, H., Ilijić, S. & Torres, K. B. V. 2008, A&A, 482, 1031. [ADS](https://ui.adsabs.harvard.edu/abs/2008A%26A...482.1031H)
- Hilditch, R. W. 2001, An Introduction to Close Binary Stars (Cambridge: Cambridge University Press). [ADS](https://ui.adsabs.harvard.edu/abs/2001icbs.book.....H)
- Hoffman, M. D. & Gelman, A. 2014, Journal of Machine Learning Research, 15, 1593. [ADS](https://ui.adsabs.harvard.edu/abs/2011arXiv1111.4246H)
- Hunter, J. D. 2007, Computing in Science & Engineering, 9, 90. [ADS](https://ui.adsabs.harvard.edu/abs/2007CSE.....9...90H)
- Husser, T.-O., Wende-von Berg, S., Dreizler, S., et al. 2013, A&A, 553, A6. [ADS](https://ui.adsabs.harvard.edu/abs/2013A%26A...553A...6H)
- Ilijić, S., Hensberge, H. & Pavlovski, K. 2001, in Lecture Notes in Physics, 573, Astrotomography, 269. [ADS](https://ui.adsabs.harvard.edu/abs/2001LNP...573..269I)
- Ilijić, S., Hensberge, H., Pavlovski, K. & Freyhammer, L. M. 2004, in ASP Conf. Ser. 318, Spectroscopically and Spatially Resolving the Components of the Close Binary Stars, 111. [ADS](https://ui.adsabs.harvard.edu/abs/2004ASPC..318..111I)
- Kıran, E., Bakış, V., Hensberge, H. & Harmanec, P. 2016, A&A, 587, A127. [ADS](https://ui.adsabs.harvard.edu/abs/2016A%26A...587A.127K)
- Kirkby-Kent, J. A., Maxted, P. F. L., Serenelli, A. M., et al. 2016, A&A, 591, A124. [ADS](https://ui.adsabs.harvard.edu/abs/2016A%26A...591A.124K)
- Klement, R., Rivinius, Th., Baade, D., et al. 2025, A&A, 694, A208. [ADS](https://ui.adsabs.harvard.edu/abs/2025A%26A...694A.208K)
- Kumar, R., Carroll, C., Hartikainen, A. & Martin, O. 2019, Journal of Open Source Software, 4, 1143. [ADS](https://ui.adsabs.harvard.edu/abs/2019JOSS....4.1143K)
- Lanz, T. & Hubeny, I. 2003, ApJS, 146, 417. [ADS](https://ui.adsabs.harvard.edu/abs/2003ApJS..146..417L)
- Lanz, T. & Hubeny, I. 2007, ApJS, 169, 83. [ADS](https://ui.adsabs.harvard.edu/abs/2007ApJS..169...83L)
- Liu, J., Zhang, H., Howard, A. W., et al. 2019, Nature, 575, 618. [ADS](https://ui.adsabs.harvard.edu/abs/2019Natur.575..618L)
- Lomb, N. R. 1976, Ap&SS, 39, 447. [ADS](https://ui.adsabs.harvard.edu/abs/1976Ap%26SS..39..447L)
- MacKay, D. J. C. 1992, Neural Computation, 4, 415. [DOI](https://doi.org/10.1162/neco.1992.4.3.415)
- Mahy, L., Sana, H., Abdul-Masih, M., et al. 2020, A&A, 634, A118. [ADS](https://ui.adsabs.harvard.edu/abs/2020A%26A...634A.118M)
- Maxted, P. F. L., Gaulme, P., Graczyk, D., et al. 2020, MNRAS, 498, 332. [ADS](https://ui.adsabs.harvard.edu/abs/2020MNRAS.498..332M)
- Mészáros, Sz. & Allende Prieto, C. 2013, MNRAS, 430, 3285. [ADS](https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.3285M)
- Mészáros, Sz., Bohlin, R., Allende Prieto, C., et al. 2024, A&A, 688, A171. [ADS](https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.171M)
- Miller, N. J., Maxted, P. F. L. & Smalley, B. 2020, MNRAS, 497, 2899. [ADS](https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.2899M)
- Morton, D. C. 2000, ApJS, 130, 403. [ADS](https://ui.adsabs.harvard.edu/abs/2000ApJS..130..403M)
- Osterbrock, D. E. & Ferland, G. J. 2006, Astrophysics of Gaseous Nebulae and Active Galactic Nuclei, 2nd ed. (Sausalito: University Science Books). [ADS](https://ui.adsabs.harvard.edu/abs/2006agna.book.....O)
- Palacios, A., Gebran, M., Josselin, E., et al. 2010, A&A, 516, A13. [ADS](https://ui.adsabs.harvard.edu/abs/2010A%26A...516A..13P)
- Pavlovski, K. & Hensberge, H. 2005, A&A, 439, 309. [ADS](https://ui.adsabs.harvard.edu/abs/2005A%26A...439..309P)
- Pavlovski, K. & Hensberge, H. 2010, in ASP Conf. Ser. 435, Binaries - Key to Comprehension of the Universe, 207. [ADS](https://ui.adsabs.harvard.edu/abs/2010ASPC..435..207P)
- Pavlovski, K. & Southworth, J. 2009, MNRAS, 394, 1519. [ADS](https://ui.adsabs.harvard.edu/abs/2009MNRAS.394.1519P)
- Pavlovski, K., Southworth, J. & Tamajo, E. 2018, MNRAS, 481, 3129. [ADS](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.3129P)
- Phan, D., Pradhan, N. & Jankowiak, M. 2019, arXiv:1912.11554. [ADS](https://ui.adsabs.harvard.edu/abs/2019arXiv191211554P)
- Piskunov, N. & Valenti, J. A. 2017, A&A, 597, A16. [ADS](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A..16P)
- Posbic, H., Katz, D., Caffau, E., et al. 2012, A&A, 544, A154. [ADS](https://ui.adsabs.harvard.edu/abs/2012A%26A...544A.154P)
- Quintero, E. A., Eenens, P. & Rauw, G. 2020, Astronomische Nachrichten, 341, 628. [ADS](https://ui.adsabs.harvard.edu/abs/2020AN....341..628Q)
- Rasmussen, C. E. & Williams, C. K. I. 2006, Gaussian Processes for Machine Learning (Cambridge, MA: MIT Press). [ADS](https://ui.adsabs.harvard.edu/abs/2006gpml.book.....R)
- Rivinius, Th., Baade, D., Hadrava, P., Heida, M. & Klement, R. 2020, A&A, 637, L3. [ADS](https://ui.adsabs.harvard.edu/abs/2020A%26A...637L...3R)
- Rucinski, S. M. 2002, AJ, 124, 1746. [ADS](https://ui.adsabs.harvard.edu/abs/2002AJ....124.1746R)
- Sablowski, D. P. & Weber, M. 2017, A&A, 597, A125. [ADS](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A.125S)
- Sablowski, D. P. & Weber, M. 2019, A&A, 623, A31. [ADS](https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..31S)
- Sairam, L., Triaud, A. H. M. J., Baycroft, T. A., et al. 2024, MNRAS, 534, 3999. [ADS](https://ui.adsabs.harvard.edu/abs/2024MNRAS.534.3999S)
- Sana, H., de Mink, S. E., de Koter, A., et al. 2012, Science, 337, 444. [ADS](https://ui.adsabs.harvard.edu/abs/2012Sci...337..444S)
- Scargle, J. D. 1982, ApJ, 263, 835. [ADS](https://ui.adsabs.harvard.edu/abs/1982ApJ...263..835S)
- Seeburger, R., Rix, H.-W., El-Badry, K., Xiang, M. & Fouesneau, M. 2024, MNRAS, 530, 1935. [ADS](https://ui.adsabs.harvard.edu/abs/2024MNRAS.530.1935S)
- Shenar, T., Bodensteiner, J., Abdul-Masih, M., et al. 2020, A&A, 639, L6. [ADS](https://ui.adsabs.harvard.edu/abs/2020A%26A...639L...6S)
- Shenar, T., Bodensteiner, J., Sana, H., et al. 2024, A&A, 690, A289. [ADS](https://ui.adsabs.harvard.edu/abs/2024A%26A...690A.289S)
- Simon, K. P. & Sturm, E. 1994, A&A, 281, 286. [ADS](https://ui.adsabs.harvard.edu/abs/1994A%26A...281..286S)
- Škoda, P. & Hadrava, P. 2010, in ASP Conf. Ser. 435, Binaries - Key to Comprehension of the Universe, 71. [ADS](https://ui.adsabs.harvard.edu/abs/2010ASPC..435...71S)
- Stoehr, F., White, R., Smith, M., et al. 2008, in ASP Conf. Ser. 394, Astronomical Data Analysis Software and Systems XVII, 505. [ADS](https://ui.adsabs.harvard.edu/abs/2008ASPC..394..505S)
- Takahashi, K., Fagan, J. & Chin, M.-S. 1973, in Proceedings of the 8th PICA Conference (Minneapolis), 63.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A. & Gelman, A. 2018, arXiv:1804.06788. [ADS](https://ui.adsabs.harvard.edu/abs/2018arXiv180406788T)
- Tamajo, E., Pavlovski, K. & Southworth, J. 2011, A&A, 526, A76. [ADS](https://ui.adsabs.harvard.edu/abs/2011A%26A...526A..76T)
- Ting, Y.-S., Conroy, C., Rix, H.-W. & Cargile, P. 2019, ApJ, 879, 69. [ADS](https://ui.adsabs.harvard.edu/abs/2019ApJ...879...69T)
- Tkachenko, A. 2015, A&A, 581, A129. [ADS](https://ui.adsabs.harvard.edu/abs/2015A%26A...581A.129T)
- Tkachenko, A., Tsymbal, V., Zvyagintsev, S., et al. 2022, A&A, 666, A180. [ADS](https://ui.adsabs.harvard.edu/abs/2022A%26A...666A.180T)
- Tonry, J. & Davis, M. 1979, AJ, 84, 1511. [ADS](https://ui.adsabs.harvard.edu/abs/1979AJ.....84.1511T)
- Torres, G., Andersen, J. & Giménez, A. 2010, A&ARv, 18, 67. [ADS](https://ui.adsabs.harvard.edu/abs/2010A%26ARv..18...67T)
- Torres, G., Latham, D. W. & Stefanik, R. P. 2007, ApJ, 662, 602. [ADS](https://ui.adsabs.harvard.edu/abs/2007ApJ...662..602T)
- van der Marel, R. P. & Franx, M. 1993, ApJ, 407, 525. [ADS](https://ui.adsabs.harvard.edu/abs/1993ApJ...407..525V)
- VanderPlas, J. T. 2018, ApJS, 236, 16. [ADS](https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V)
- Villaseñor, J. I., Sana, H., Mahy, L., et al. 2025, A&A, 698, A41. [ADS](https://ui.adsabs.harvard.edu/abs/2025A%26A...698A..41V)
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. 2020, Nature Methods, 17, 261. [ADS](https://ui.adsabs.harvard.edu/abs/2020NatMe..17..261V)
- Wheeler, A. J., Abruzzo, M. W., Casey, A. R. & Ness, M. K. 2023, AJ, 165, 68. [ADS](https://ui.adsabs.harvard.edu/abs/2023AJ....165...68W)
- Wilson, O. C. 1941, ApJ, 93, 29. [ADS](https://ui.adsabs.harvard.edu/abs/1941ApJ....93...29W)
- Wright, J. T. & Eastman, J. D. 2014, PASP, 126, 838. [ADS](https://ui.adsabs.harvard.edu/abs/2014PASP..126..838W)
- Zucker, S. 2003, MNRAS, 342, 1291. [ADS](https://ui.adsabs.harvard.edu/abs/2003MNRAS.342.1291Z)
- Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806. [ADS](https://ui.adsabs.harvard.edu/abs/1994ApJ...420..806Z)
- Zucker, S., Torres, G. & Mazeh, T. 1995, ApJ, 452, 863. [ADS](https://ui.adsabs.harvard.edu/abs/1995ApJ...452..863Z)
