# Candidate model extensions: physics and literature

Research note for the albireo lead. Compiled 2026-09-03. No repository file was modified.

## 0. Scope, notation, and how to read this

Each section covers one research item and ends with two fixed subsections, **What the
incumbents do** and **What a linear-Gaussian implementation would have to represent**. The
second subsection classifies every new quantity as *linear in the spectra* (so the analytic
marginalisation of §3 of `math.md` survives untouched), *nonlinear* (so it joins $`\theta`$ and
is sampled), or *degenerate* with something albireo already carries.

Throughout, the reference model is `math.md` §1.4,

```math
m_j(\theta, d) \;=\; \mathrm{diag}(r_j)\,\mathbf{R}_j
\Big[\mathbf{1} + \sum_{i} \ell_{ij}\, \mathbf{B}_j\, \mathbf{T}(\delta_{ij})\, d_i \Big],
```

with $`\delta_{ij} = \xi(v_{ij})/\Delta`$ the log-shift in pixels, $`\mathbf{B}_j`$ the LSF,
$`\mathbf{R}_j`$ the rebin onto epoch $`j`$'s native grid, and $`r_j`$ the response polynomial.

**Sourcing.** Equations were read from arXiv full texts or from the code distributions
themselves, not from abstracts. A&A's own servers and ADS abstract pages both refuse
automated fetches from this environment (HTTP 403 and 405 respectively), so bibcodes are
either quoted from ADS URLs returned by search, constructed from a journal reference read on
an arXiv abstract page, or verified through Crossref (volume and page). Section 8 is a ledger
with the verification state of every bibcode. Four claims are explicitly flagged as
unverified; nothing was invented.

---

## 1. Time-variable line strengths and profiles

### 1.1 Hadrava's line-strength factors

Hadrava (1997, `1997A&AS..122..581H`) generalised Fourier disentangling by attaching a free
multiplicative factor $`s_j(t)`$ to each component at each exposure. The clearest statement of
the algebra is in his lecture notes (`2009arXiv0909.0172H`, §1.6.3), where the per-component
broadening kernel becomes

```math
\Delta_j(x,t,p) = s_j(t)\,\delta\big(x - v_j(t,p)\big),
\qquad
\tilde\Delta_j(y,t,p) = s_j(t)\,\exp\!\big(\mathrm{i}\,y\,v_j(t,p)\big).
```

Three structural facts follow, all stated by Hadrava:

1. The objective is **bilinear** in the coefficients $`s_{jl} \equiv s_j(t_l)`$ and the component
   spectra. He exploits this: for fixed spectra, $`\partial S/\partial s_{ml} = 0`$ gives a small
   linear system (his Eq. 1.44) solved directly, "before optimizing $`S`$ with respect to either
   $`v_j(t_l)`$ or $`p`$, in which it is non-linear."
2. There is an **exact scale degeneracy**. "It is obvious ... that for each component its
   spectrum $`\tilde I_j(y)`$ and strengths $`s_{jl}`$ are defined by the observations up to a
   reciprocal multiplicator. This must be fixed by a normalization condition."
   (`2009arXiv0909.0172H`, §1.6.3.)
3. Convergence is fragile. "There is no guaranty that this scheme will converge from every
   arbitrarily chosen initial condition. Instead, it can achieve some false local minimum by
   suppressing lines in exposures for which the true radial velocities differ from the
   instantaneous approximation." His recommended practice is to fit the orbit and spectra with
   the strengths fixed at unity first and release them only for final tuning.

The motivation Hadrava gives is not cosmetic: "the experience that in some binaries errors of
radial velocities increased significantly close to conjunctions where an eclipse could be
expected" (`2009arXiv0909.0172H`, §1.6.3). He lists as further targets the Struve-Sahade
effect, circumstellar lines, telluric lines, and "some instrumental and data-processing
imperfections like incorrect flatfielding or rectification."

### 1.2 Line photometry: what the strength factors actually measure

The strength factors are not the eclipse light curve. Hadrava's §1.6.4 derives, for a system
with continuum fractions normalised to $`I_1 + I_2 = 1`$ out of eclipse and component 1 dimmed
by a factor $`z`$,

```math
s_1 L_1 = \frac{z_{\rm lin} L_1}{z_{\rm cont} I_1 + I_2},
\qquad
s_2 L_2 = \frac{L_2}{z_{\rm cont} I_1 + I_2},
```

so that

```math
z_{\rm lin} = \frac{s_1}{s_2},
\qquad\text{and, if } z_{\rm cont} \equiv z_{\rm lin},\qquad
\frac{I_1}{I_2} = \frac{1 - s_2}{s_1 - 1}.
```

Only the **ratio** $`s_1/s_2`$ is used, and the continuum light ratio follows only under the
assumption $`z_{\rm lin} = z_{\rm cont}`$, which Hadrava then argues is false. In the
Milne-Eddington approximation the specific intensity across the disc is
$`I(x,\mu) = S_0 + S_1\mu`$, so continuum limb darkening is $`u = S_1/(S_0+S_1) \simeq 0.3`$ in
broad bands; but for a weak line whose opacity scales with the continuum opacity,
$`\tau(x,r) = [1+\phi(x)]\tau_{\rm cont}(r)`$, the line contribution to the emergent flux is
$`-S_{1,\rm cont}\,\phi/(1+\phi)\,\mu`$, which has $`u_{\rm lin} = 1`$
(`2009arXiv0909.0172H`, Eqs. 1.56 to 1.61). Consequence, in his words: "at initial phases of an
eclipse when only a part of disk edge is hidden, the light missing in line represents larger
portion of the overall flux in that frequency than the light missing in the continuum
($`z_{\rm lin} < z_{\rm cont}`$), so that in some cases line-strengths of both components can be
enhanced." He notes that fitting $`s_{1,2}(x)`$ with a proper eclipse model per wavelength "can
reveal the limb-darkening variations within the line and thus yield information about the
structure of atmosphere of the eclipsed component."

### 1.3 The general line-profile-variation formulation, and why it stays linear

Hadrava's §1.6.5 is the load-bearing result for any linear-Gaussian extension. Writing the
observed flux as a surface integral of Doppler-shifted local intensities,

```math
I(x,t) = \sum_j \int_s \mu\, I_j(x,s,\mu,t) * \delta\big(x - v_j(s,t)\big)\, \mathrm{d}^2 s,
```

he expands the local intensity in a small basis of spectral functions with coefficients that
carry all the geometry and time dependence,

```math
I_j(x,s,\mu,t) = \sum_k f_j^k(s,\mu,t)\, I_j^k(x)
\;\;\Longrightarrow\;\;
I(x,t) = \sum_{j,k} I_j^k(x) * \Delta_j^k(x,t,p),
\qquad
\Delta_j^k = \int_s \mu\, f_j^k\, \delta(x - v_j)\, \mathrm{d}^2 s .
```

He remarks that this "is formally identical with (1.27)", the ordinary disentangling equation,
"apart of the fact that each component $`j`$ can be now characterized by several spectra
$`I_j^k(x)`$ ... with different spectral broadenings." That is the whole trick: a component with
variable line profiles is **several ordinary components with different, known, time-dependent
kernels**. The problem stays linear in the unknown spectra. The same device reappears as
multiprofile LSD in Kochukhov, Makaganiuk & Piskunov (2010, `2010A&A...524A...5K`, Eq. 21),
where $`N`$ mean profiles are recovered simultaneously from a concatenated design matrix.

Two worked cases:

*Radial pulsation.* For an atmosphere moving radially with instantaneous speed $`v_p(t)`$ and
local intensity $`\propto \mu^k`$, the broadening kernel and its transform are (Hadrava,
Šlechta & Škoda 2009, `2009A&A...507..397H`, Eqs. 5, 8, 9)

```math
\Delta^k(x,t) = \frac{2\pi R^2}{v_p^{k+2}}\Big[x^{k+1}\Big]_0^{v_p},
\qquad
\tilde\Delta^1(y,t) = \frac{2\pi \mathrm{i} R^2}{y^3 v_p^3}
\Big[e^{\mathrm{i} y v_p}\big(2 - 2\mathrm{i}yv_p - y^2v_p^2\big) - 2\Big],
```

with the measured first-moment shift $`v_r = \frac{k+2}{k+3}v_p`$, i.e. the Baade-Wesselink
projection factor $`p = 3/2`$ for $`k=0`$, $`4/3`$ for $`k=1`$, tending to 1 for high $`k`$.
Standard disentangling is the $`k \to \infty`$ limit, $`\tilde\Delta^\infty \simeq
e^{\mathrm{i}yv_p}`$. They absorb the radius variation into the strength factor,
$`2\pi R^2 \equiv (k+2)s(t)`$.

*Rotational (Schlesinger-Rossiter-McLaughlin) effect.* For a rigidly rotating star partly
eclipsed by a companion at projected position $`(u_i, v_i)`$, the kernels are integrals over
the visible disc (Hadrava `2009arXiv0909.0172H`, Eqs. 1.73 to 1.75). He objects explicitly to
the usual practice of fixing the in-line limb darkening to the continuum value, "it is
incorrect to fix the limb darkening in lines to the values obtained for continua either from
light-curve solution or from model atmospheres", and notes that "centers of different
broadening profiles $`\Delta_j^k`$ are generally different", so the RM amplitude differs from
line to line.

### 1.4 What the codes actually implement

**KOREL** solves for $`s_{jl}`$ per exposure per component, optionally with some components'
strengths held fixed, and offers template-constrained disentangling
(`2009arXiv0909.0172H`, Eq. 1.77) so that a known component (typically telluric) can be
subtracted rather than fitted. KOREL09 additionally fits the free parameters of a prescribed
pulsational broadening kernel (`2009A&A...507..397H`). Applied to δ Cep, the line-strength
semi-amplitudes were $`s_1 = 0.577`$ for the low-excitation component and $`-0.152`$ (that is,
in antiphase) for the high-excitation one, and the amplitude of the strength variation
anticorrelates with the excitation potential of the lower level (`2009A&A...507..397H`,
Table 2 and §3).

**fd3** does not fit strengths. Its control file carries a per-observation table whose columns
its own output labels "SECTION 4: DESCRIPTORS TO OBSERVED SPECTRA AND LIGHT-FACTORS ASSIGNED
TO COMPONENTS", with columns `t_obs`, `rv_corr`, `noise_rms`, `lf_A`, `lf_B` (and `lf_C`), and
no uncertainty or free-parameter flag (fd3 v3.1 distribution, `V453_Cyg.in` and
`V453_Cyg.out`). Ilijić et al. (2004, `2004ASPC..318..111I`) state the restriction outright:
"Light-factors (LFs) can be used only as fixed, possibly time-dependent parameters." The
shipped V453 Cyg example uses $`(\ell_A,\ell_B) = (0.7273, 0.2727)`$ out of eclipse and
$`(1.0, 0.0)`$ at one epoch in total eclipse; the shipped triple example uses
$`(0.375, 0.000, 0.625)`$ and $`(0.125, 0.250, 0.625)`$ at two eclipse epochs. There is no
"lfvar" keyword; the mechanism is the per-epoch table.

**Spectangular** (Sablowski & Weber 2017, `2017A&A...597A.125S`; Sablowski, Järvinen & Weber
2019, `2019A&A...623A..31S`) added the ability to *optimise* the per-spectrum flux ratios:
"it is now possible to run an optimization on the flux ratios of each observed spectrum. This
now provides a method for obtaining information about the flux variation ... the disentangling
approach is also usable to extract light curve information from the spectroscopic data"
(`2019A&A...623A..31S`, §4). It also records the limitation that matters for §2 below: "the
code cannot account for wavelength dependent flux ratios." Its treatment of *shape* variability
is entirely post-hoc: pulsations, flares and spots are recovered from the residuals
(observation minus shifted disentangled spectra), and the disentangled spectrum is described as
"an extracted mean" whose least-squares nature means "random variability will be strongly
suppressed" (`2019A&A...623A..31S`, §§2.2, 3.1, 3.2, 3.3, 3.5). A flare in one exposure was
shown to tilt the continuum of the *output* spectra, with the sign of the tilt depending on the
orbital phase at which the flare occurred (§3.3).

**UNWIND** could not be documented from a primary source in this session; searches surfaced
only secondary mentions. Treat any claim about its variable-line-strength capability as
unverified.

### 1.5 Be-star disc emission: what was actually done

The uniform answer in the compact-companion literature is *exclusion*, not modelling.

- HR 6819 (Bodensteiner et al. 2020, `2020A&A...641A..43B`, §4): disentangling was run
  separately on He I (photospheric), Fe II and O I (disc emission), and the Balmer lines.
  "Given the large variability observed in the Hα line ... we do not consider Hα in our
  analysis." Because of "strong variability between the 1999 and 2004 epochs" they also refused
  to combine the two data sets. The recovered semi-amplitudes were $`K_2 = 3.9\pm3.2`$ (Fe II +
  O I), $`3.6\pm1.5`$ (Balmer) and $`7.5\pm12.5`$ km s$`^{-1}`$ (He I), combining to
  $`4.0\pm0.8`$, with the caveat "The true error may be larger (e.g., due to pulsational
  variability), but is difficult to quantify."
- MWC 656 (Janssens et al. 2023, `2023A&A...677L...9J`, §3.3): "Due to strong variability in
  the emission lines, we avoid regions such as the Balmer and Paschen lines, and the
  He II λ4686 line." The cost is stated: "While excluding Balmer lines from the disentangling,
  we cannot detect a WD companion."
- ALS 8814 (El-Badry, Fabry, Sana, Shenar & Seeburger 2025, arXiv:2509.01545): "While the MRS
  spectra also contain Hα, obvious changes in the strong emission line profile across epochs
  make it unsuitable for disentangling." The quantitative consequence is the most useful
  published number in this whole area: the three usable He I lines give $`K_2 = 37, 22`$ and
  $`24`$ km s$`^{-1}`$ against a formal emcee error of $`\pm1.6`$, and "All three lines have a
  best-fit $`\chi^2_{\rm red}`$ that is larger than 1, meaning that the two disentangled
  components cannot fully describe all the variance in the observed spectra. This is probably a
  result of time-variability in the emission lines."
- El-Badry & Quataert (2021, `2021MNRAS.502.3436E`, Appendix B) diagnose the same effect for
  LB-1: the line-to-line scatter in $`K_{\rm Be}`$ "is larger than would be expected if the
  formal uncertainties ... were reliable. This is most likely a consequence of small errors in
  continuum normalization as well as time-variability in the flux ratio, which violate the
  underlying assumptions of the spectral disentangling algorithm."

The Struve-Sahade effect is the same phenomenon in O-type binaries and was quantified by
Linder et al. (2007, `2007A&A...474..193L`; erratum `2012A&A...541C...2L`).

### 1.6 Pulsating components in binaries

- KIC 9850387 (Sekaran et al. 2020, `2020A&A...643A.162S`, §§3.2, 3.3): FDBinary with the
  orbit **fixed** to the LSD radial-velocity solution, because releasing it produced "wildly
  varying values for $`K_2`$ depending on the wavelength region that was being disentangled".
  The pulsational signal survives into the output: "a significant amount of pulsational
  distortion (from the high-amplitude g modes) is present in the disentangled primary component
  HIRES spectrum. This pulsational distortion manifests as asymmetric line-profile variations".
  They declined to fit macroturbulence because it mimics pulsational broadening (Aerts et al.
  2014, `2014A&A...569A.118A`; see also Aerts et al. 2009, `2009A&A...508..409A`).
- Spectangular's artificial tests recover a pulsation period from the summed residuals to
  $`6.009 \pm 0.011`$ against a true 6.0 cycles per orbit at a 5 km s$`^{-1}`$ amplitude
  (25 per cent of $`v\sin i`$), degrading to $`6.138\pm0.048`$ at 0.5 km s$`^{-1}`$
  (`2019A&A...623A..31S`, Table 1).

### What the incumbents do

| Code | Per-epoch strength | Per-epoch shape | How variability is handled |
|---|---|---|---|
| KOREL | free scalar $`s_{jl}`$ per component per exposure, solved by linear least squares inside the iteration; scale fixed by convention | prescribed kernels: radial pulsation, rigid rotation, eclipse geometry, limb-darkening moments | modelled |
| fd3 | fixed per-epoch light factors read from the control file; never free | none | user must supply the light curve; variable lines are excluded by hand |
| Spectangular | per-spectrum flux ratios, optionally optimised | none | recovered post hoc from residuals; wavelength-independent only |
| shift-and-add (Shenar/Bodensteiner lineage) | fixed light ratio, applied only as an output scaling | none | affected lines and epochs are excluded |
| LSDBinary (Tkachenko et al. 2022, `2022A&A...666A.180T`) | wavelength-dependent flux ratio and radii ratio in the composite model | analytic RM correction in the initial-guess module | modelled at profile level, not spectrum level |

No incumbent propagates line-strength uncertainty into the recovered spectra, and none reports
a covariance for the strengths.

### What a linear-Gaussian implementation would have to represent

Three distinct designs, in increasing order of what they buy and cost.

**(a) A free per-epoch scalar per component**, $`\ell_{ij} \to \ell_{ij} s_{ij}`$.

- *Linear in the spectra:* yes, unchanged. $`\mathbf{A}(\theta)`$ merely acquires a different
  scalar in each block, so the analytic marginalisation, the band structure, and the
  Takahashi recursion are all untouched.
- *Nonlinear:* $`s_{ij}`$, adding $`N_c J`$ parameters to $`\theta`$. Note that Hadrava's trick
  of profiling them out by linear least squares is *not* available here, because $`d`$ has
  already been integrated out; the marginal is not quadratic in $`s`$.
- *Degenerate:*
  1. With $`\ell_{ij}`$ **exactly and pointwise**. A per-epoch light fraction and a per-epoch
     line-strength factor are the same parameter; albireo's existing `PerEpoch` light-fraction
     mode already *is* Hadrava's line-strength mode, differing only by the simplex constraint
     $`\sum_i \ell_{ij} = 1`$. The physically meaningful distinction, from §1.2, is that the
     simplex enforces $`z_{\rm lin} = z_{\rm cont}`$, which is precisely the assumption Hadrava
     shows to be wrong. Releasing the simplex is the extension; the constraint that replaces
     it must come from a light curve or from a limb-darkening model.
  2. With the norm of $`d_i`$ **exactly**: $`(c\,s_{ij},\ d_i/c)`$ is the same fit. This is
     `math.md` §5.2 wearing a per-epoch hat, and it needs the same fix albireo already applies
     to the nebular amplitude, a centred log site pinning $`\prod_j s_{ij} = 1`$.
  3. A per-epoch factor **common to all components**, $`s_j`$, is nearly degenerate with the
     constant term of the response polynomial $`r_j`$: the two differ only in whether the
     continuum is rescaled along with the lines, so the separation is carried by the continuum
     signal-to-noise alone. Only the *differential* strength $`s_{1j}/s_{2j}`$ is well
     identified, which is exactly the combination Hadrava's line photometry uses.

**(b) A low-rank basis with known coefficients** (the linear route, and the one that costs
nothing structurally). Give component $`i`$ a set of $`R+1`$ spectra with fixed, known epoch
coefficients $`c_{ijr}`$,

```math
m_j = \mathrm{diag}(r_j)\mathbf{R}_j\Big[\mathbf{1}
+ \sum_i \sum_{r=0}^{R} c_{ijr}\, \mathbf{B}_j \mathbf{T}(\delta_{ij})\, d_i^{(r)}\Big].
```

- *Linear:* all of $`d_i^{(r)}`$. They are extra columns of $`\mathbf{A}`$ and extra diagonal
  blocks of $`\boldsymbol\Lambda`$; the marginal likelihood, its gradient, and the selected
  inverse all carry through verbatim. Cost rises from $`N_cP`$ to $`N_c(1+R)P`$ unknowns, with
  the block size of the block-tridiagonal factor growing by the same factor.
- *Nonlinear:* nothing new, provided the coefficients come from outside (a pulsation
  ephemeris, an eclipse light curve, a limb-darkening moment expansion). If they are
  parameterised, only those parameters are nonlinear.
- *Identifiability:* $`d_i^{(r)}`$ is separated from $`d_i^{(0)}`$ only through the epoch-to-epoch
  variation of $`c_{ijr}`$, by the same argument as `math.md` §5.1. If the modulation is
  phase-locked to the orbit, $`c_{ij1}`$ correlates with $`\delta_{ij}`$ and the separation
  degrades; the exact statement is available from the forecast machinery of §5.5, since
  $`\tilde{\boldsymbol\Lambda}`$ is flux-free and this is just a wider $`\mathbf{A}`$.
- albireo's nebular component is the $`R=0`$ special case of this family with a *free*
  amplitude, and `math.md` §1.3 already says so ("rank-one time variation, a fixed shape with a
  free per-epoch scale, which is the same structure Tier 3's variable-disc component
  generalizes").

**(c) Per-epoch kernels.** Replace $`\mathbf{B}_j \mathbf{T}(\delta_{ij})`$ by a physical
kernel $`\mathbf{K}_{ij}(\psi)`$ (pulsational, rotational, eclipse-distorted).

- *Linear:* the spectra, still. A kernel is a linear operator; it changes $`\mathbf{A}`$, not
  the model class. albireo's `operators.convolve_varying` already accepts arbitrary profile
  banks; the extension is per-epoch rather than per-wavelength banks.
- *Nonlinear:* $`\psi`$ (pulsation velocity curve, $`v\sin i`$, eclipse geometry).
- *Degenerate:* a kernel change that is identical at every epoch is absorbed exactly by the
  free spectrum, $`d \to \mathbf{K}'\mathbf{K}^{-1}d`$, which is the same commutation argument
  albireo already makes for the LSF in `math.md` §1.3. Hadrava reports the observational
  consequence: with pulsational kernels, "a systematic shift of all pulsational velocities can
  be relatively well compensated by a shift of the disentangled spectrum, [so] the convergence
  of the solution to the true radial velocity is very slow and the value of the velocity is
  poorly defined" (`2009A&A...507..397H`, §3).

**Be-star discs specifically.** A disc whose emission changes *shape*, not just scale, is not
rank one. Three honest options, in order of ambition: mask the affected windows (what everyone
does); give the disc a rank-$`R`$ basis with free per-epoch coefficients, which is bilinear and
therefore case (a) plus case (b) combined; or admit the mismatch through a windowed per-epoch
noise inflation, which is `math.md` §3.2a's jitter restricted to a wavelength mask. The third
is the cheapest and the only one that will report honestly rather than absorb the variability
into a distorted spectrum. Note that §3.2a already warns that jitter "widens the intervals
around an unchanged, still-biased point estimate", which is exactly the ALS 8814 situation.

---

## 2. Wavelength-dependent light ratios across broad windows

### 2.1 The renormalisation problem

Disentangling returns each component's contribution to the *composite* continuum. Converting to
a spectrum normalised to that star's *own* continuum is an affine transformation involving both
a multiplicative and an additive term, and the additive term is exactly the unconstrained
$`k=0`$ mode.

Pavlovski & Hensberge (2005, `2005A&A...439..309P`, §3) write the composite as

```math
O(\lambda) = s_A(\lambda) + s_B(\lambda) + c_{\rm obs},
\qquad
O(\lambda) = \ell_A(t)\,S_A(\lambda) + \ell_B(t)\,S_B(\lambda),
```

where $`\langle s_{A,B}\rangle = 0`$, $`b_{\rm obs} = 1 - c_{\rm obs}`$ is the observed line
blocking, and $`S_{A,B}`$ are the spectra normalised to their own continua. They note in
passing that the light fractions carry "a slow dependence on wavelength". The renormalisation
is then

```math
S_{A,B} \;=\; 1 - b_{\rm obs} + \frac{s_{A,B} + C_{A,B}}{\ell_{A,B}},
\qquad
C_A = -C_B = \frac{b_{\rm obs}\,\ell_A\,(1 - b_A/b_B)}{1 + (b_A/b_B)(\ell_A/\ell_B)},
```

with the intrinsic line-blocking coefficients $`b_{A,B}`$ coupled only by

```math
\ell_A b_A + \ell_B b_B = b_{\rm obs},
```

so that one of them, or their ratio, "can be chosen freely in the case of time-independent
light fractions."

Ilijić et al. (2004, `2004ASPC..318..111I`) give the operational recipe used by FDBinary and
fd3. Separation is run with *generic* light factors $`\ell_1 = \ell_2 = 0.5`$, giving generic
spectra $`z_{1,2}`$; the physical spectra follow from a chosen continuum flux ratio $`L`$ and
line-blocking ratio $`B = b_1/b_2 = (1-\bar x_1)/(1-\bar x_2)`$ as

```math
x_{1i} = \frac{L+1}{2L}\,(z_{1i} + c),
\qquad
x_{2i} = \frac{L+1}{2}\,(z_{2i} - c),
\qquad
c = \frac{L-1}{L+1} - \frac{BL-1}{BL+1}\,b ,
```

with $`b = 1 - \bar y`$ the mean observed line blocking. They also note that the rank-deficient
$`n=0`$ system has the one-parameter solution

```math
\begin{pmatrix}\bar x_1\\ \bar x_2\end{pmatrix}
= \frac{\bar y}{\ell_1^2+\ell_2^2}\begin{pmatrix}\ell_1\\ \ell_2\end{pmatrix}
+ a\begin{pmatrix}\ell_2\\ -\ell_1\end{pmatrix},
```

the first term being the minimum-norm SVD solution and $`a`$ arbitrary. When the output shows
undulations they replace the constant $`c`$ by a slowly varying, zero-mean function
$`c \to c + f_i`$ fitted with a few parameters, which they describe candidly as a step that
"a posteriori applies physical constraints onto the model component spectra that were not
embedded in the mathematical formulation of the separation problem."

KOREL's version (Hadrava `2009arXiv0909.0172H`, §1.6.10) solves for all components' continuum
offsets $`\Delta_j = C_j - \langle I_j\rangle`$ jointly under the closure
$`Q \equiv 1 - \langle I\rangle - \sum_j \Delta_j = 0`$, by minimising
$`S = \sum_j w_j \int [I'_j - 1 - \Delta_j]^2`$ with a Lagrange multiplier, then rectifies
$`I_j/C_j = 1 + (I'_j - 1 - \Delta_j)/C_j`$.

### 2.2 Making the dilution wavelength dependent

The explicit formulation is GSSP's binary mode (Tkachenko 2015, `2015A&A...581A.129T`,
Eqs. 3 and 4). Rather than fitting two wavelength-independent light factors, it fits the
**radius ratio** and lets the continuum-intensity ratio supply the wavelength dependence:

```math
R_1^{\rm dis} = \frac{\alpha R_1^{\rm th} + 1}{1+\alpha},
\qquad
R_2^{\rm dis} = \frac{\alpha + R_2^{\rm th}}{1+\alpha},
\qquad
\alpha = \frac{I^c_{\nu,1}}{I^c_{\nu,2}}\left(\frac{\Re_1}{\Re_2}\right)^{2},
```

or, in terms of line depths, $`r_1^{\rm dis} = r_1^{\rm th}\,\alpha/(1+\alpha)`$ and
$`r_2^{\rm dis} = r_2^{\rm th}/(1+\alpha)`$. The two dilution factors sum to unity by
construction, as in the constrained mode of Tamajo et al. (2011, `2011A&A...526A..76T`), but
"instead of assuming wavelength-independent light dilution for each of the components, we
optimize the ratio of the radii of the binary components which is obviously the same for both
stars."

The magnitude of the effect, measured:

- KIC 6352430 (`2015A&A...581A.129T`, §2.3, Fig. 5): the primary is twice as hot as the
  secondary, and their continuum-intensity ratio "changes from $`\sim 4.9`$ at
  $`\lambda\lambda\,4750`$ Å to $`\sim 3.9`$ at $`\lambda\lambda\,5750`$ Å", a 20 per cent drift
  across 1000 Å, and the unconstrained fit "is likely to deliver not entirely correct
  parameters" as a result.
- KIC 11285625, two similar stars: the same quantity "does not exceed 0.2% of the value of
  0.955 measured at 5250 Å" (`2015A&A...581A.129T`).
- Algol (Kolbas et al. 2015, `2015MNRAS.451.4150K`, §5.1): "Optimisation was performed
  separately for each line to find the wavelength-dependent fractional light contribution",
  giving for the primary $`\ell(4471{+}4713) = 0.943\pm0.002`$ and
  $`\ell(5015) = 0.915\pm0.002`$, and for the secondary
  $`\ell(4500) = 0.008\pm0.001`$ and $`\ell(5500) = 0.018\pm0.001`$. The primary's dilution
  moves by 14 formal standard deviations across 550 Å; the secondary's more than doubles across
  1000 Å.

Sekaran et al. (2020, `2020A&A...643A.162S`, §3.2) state the requirement in one sentence: each
composite spectrum is a sum of components "scaled by the wavelength-dependent light
contribution of the individual components to the total flux, which depends on the spectral
energy distribution of the individual components."

### 2.3 Where $`\ell(\lambda)`$ comes from, and third light

Four published routes, in decreasing order of directness:

1. **Multi-band eclipse light ratios.** The standard for detached eclipsing binaries. Pavlovski
   & Southworth (2009, `2009MNRAS.394.1519P`, §5) put it plainly for V453 Cyg: "the solution of
   the Cohen (1974) light curves ... allows us to assign a light factor to the two stars for
   each observed spectrum. We have renormalised the disentangled spectra using the method and
   formulae presented by PH05." One light ratio per passband gives $`\ell(\lambda)`$ sampled at
   the effective wavelengths of the filters.
2. **Model-atmosphere continuum flux ratios**, i.e. GSSP's $`\alpha(\lambda)`$ above, tied to a
   single geometric parameter $`\Re_1/\Re_2`$.
3. **Per-line spectroscopic light factors**, as in Kolbas et al. (2015): fit the light
   dilution independently for each diagnostic line and read the wavelength dependence off the
   result. Tamajo et al. (2011) established that this works, recovering the light factors "to
   well within the errorbars for all S/N ratios considered" in synthetic tests where the
   temperature and gravity themselves were degenerate ($`\rho(T_{\rm eff}, \log g) = 0.98`$,
   with "the correlation with the LFs ... much weaker"). Pavlovski & Hensberge (2010,
   `2010ASPC..435..207P`) report agreement with light-curve values "within 1.5%" for V615 Per.
4. **Physicality bracketing.** Frémat et al.'s use of the deep Ca II K core in DG Leo, requiring
   that no component spectrum crosses zero (reported in `2010ASPC..435..207P`), and
   Bodensteiner et al.'s HR 6819 bound: the light contributions "cannot differ from these values
   by more than $`\approx 20\%`$, because the scaled disentangled spectra would otherwise become
   unphysical", specifically "assuming a light contribution less than 30% for the primary would
   cause its Balmer absorption lines in the scaled spectrum to reach negative flux values"
   (`2020A&A...641A..43B`, §4).

**Third light** enters as $`\sum_i \ell_i(\lambda) + \ell_3(\lambda) = 1`$. Tamajo et al. (2011)
note that "Contaminating light from a third star can in principle be found, in cases when the
light contributions of the two stars in the binary sum to less than unity", and their
unconstrained mode is the test. Kolbas et al. (2015, §5.1) show what happens when the third
light is real and poorly constrained: for Algol, the light-curve solutions disagree among
themselves precisely because the third-light contribution had to be assumed, and they
abandoned the photometry in favour of spectroscopic light factors.

### What the incumbents do

Nobody carries $`\ell(\lambda)`$ inside the disentangling. It is applied afterwards, at the
renormalisation step, and it is applied as a constant per spectral window:

- **fd3 / FDBinary**: constant light factors per epoch, applied as scalars, with the
  renormalisation of `2004ASPC..318..111I` done offline per window.
- **KOREL**: constant per region; the $`\Delta_j`$ closure of §1.6.10 is solved per region, and
  the ratios $`C_j`$ "must be estimated from some additional information".
- **Spectangular**: explicitly cannot, and says so, offering the mitigation that "since the
  wavelength ranges are rather narrow (in high-resolution spectra) the slope in such a small
  region may have a negligible effect" (`2019A&A...623A..31S`, §4).
- **shift-and-add**: the light ratio "only impacts the final scaling of the disentangled
  spectra"; it is a per-window constant chosen after the fact.
- **GSSP binary** (`2015A&A...581A.129T`) and **STARFIT** (`2014MNRAS.444.3118K`) are the two
  codes that make the dilution wavelength dependent, and both do so *downstream* of
  disentangling, on the already-recovered spectra, with GSSP tying it to a radius ratio and
  STARFIT fitting one light factor per line.

### What a linear-Gaussian implementation would have to represent

Replace the scalar $`\ell_{ij}`$ by a diagonal operator on the model grid, applied **after** the
shift and the LSF:

```math
m_j = \mathrm{diag}(r_j)\mathbf{R}_j\Big[\mathbf{1}
+ \sum_i \mathrm{diag}\big(\ell_i(\lambda)\big)\,\mathbf{B}_j\,\mathbf{T}(\delta_{ij})\,d_i\Big].
```

- *Linear:* the spectra, unchanged. $`\mathrm{diag}(\ell_i)`$ is diagonal on the model grid, so
  the half-bandwidth of $`\mathbf{A}^\top\mathbf{W}\mathbf{A}`$ does not change and the cost of
  the marginal likelihood is identical. This is the cheapest of all the extensions considered
  here.
- *Placement is physics, not convention.* The dilution is a ratio of continuum fluxes at the
  *observed* wavelength and does not co-move with the star, so the diagonal must sit outside
  $`\mathbf{T}`$. Placing it inside would make the dilution ride along with the Doppler shift,
  which is wrong at exactly the order of the effect being modelled.
- *Nonlinear:* whatever parameterises $`\ell_i(\lambda)`$. Under the GSSP form, that is
  $`\Re_1/\Re_2`$ plus whatever sets $`I^c_{\nu,i}(\lambda)`$, which requires a continuum-flux
  model from a synthetic library. albireo already reads BOSZ, POLLUX, PHOENIX and TLUSTY for
  the label fit (`science.md` §7), so the ingredient exists.
- *Degenerate, and this is the decisive point.* Under **time-constant** light fractions the
  *shape* of $`\ell_i(\lambda)`$ is not identified by the disentangling data. The would-be
  invariance $`\ell_i \to \ell_i(1+\epsilon f)`$, $`d_i \to d_i/(1+\epsilon f)`$ is broken only
  because $`f`$ acts in the observed frame while the compensating change acts in the rest frame,
  leaving a residual of order

  ```math
  \epsilon\,\xi_{ij}\,\frac{\mathrm{d}\ln \ell_i}{\mathrm{d}\ln\lambda}\;\ell_i d_i .
  ```

  Using Kolbas et al.'s Algol numbers, $`\mathrm{d}\ln\ell/\mathrm{d}\ln\lambda \approx -0.3`$,
  and a generous $`|v| = 100`$ km s$`^{-1}`$ giving $`\xi = 3.3\times10^{-4}`$, the residual is
  $`\sim 10^{-4}`$ of the line depth, that is $`\sim10^{-5}`$ in normalised flux. It is below
  the noise of any real data set. **Conclusion for the design: with constant light,
  $`\ell_i(\lambda)`$ is an external-information channel only, exactly like the constant light
  ratio of `math.md` §5.2, and it should be declared, not fitted.** Its value is that it makes
  the renormalisation and the downstream label fit wavelength-correct across a broad window, not
  that it adds information.
- *Not degenerate under eclipses.* Once $`\ell_{ij}(\lambda)`$ varies with epoch, the shape is
  identified, because the epoch variation cannot be absorbed by a single rest-frame spectrum.
  This is the same mechanism as §5.2's breaker 1, now resolved in wavelength as well as time,
  and it is exactly Hadrava's proposal to "reveal the limb-darkening variations within the line"
  (§1.2 above).
- *Third light* $`\ell_3(\lambda)`$ with $`d_3 \equiv 0`$ only rescales $`\sum_i \ell_i`$, so
  under constant light it is exactly degenerate with a common rescaling of the stellar light
  fractions. It becomes separable only when the stellar fractions vary and it does not. The
  API consequence is that a third-light declaration belongs on the same footing as the light
  ratio: an explicit user choice, not a default.
- One row should be added to the §5.4 ledger: *shape of $`\ell_i(\lambda)`$ vs low-frequency
  content of $`d_i`$; approximately exact under constant light, leverage
  $`O(\xi\,\mathrm{d}\ln\ell/\mathrm{d}\ln\lambda) \sim 10^{-4}`$; broken by eclipses or by
  external photometry.*

---

## 3. Per-epoch light fractions during eclipses from a light-curve model

### 3.1 How the codes accept per-epoch light factors

**fd3** reads them from a fixed table, one row per observation, columns `lf_A lf_B lf_C`, with
no free-parameter flag (fd3 v3.1 distribution; `2004ASPC..318..111I`). The shipped V453 Cyg
example demonstrates the intended use: a total-eclipse epoch is entered as
$`(\ell_A,\ell_B) = (1.0, 0.0)`$, which reveals one component's spectrum outright and thereby
fixes the zero-points that are otherwise free.

**KOREL** does the opposite: it *solves* for the strengths (§1.1) and then infers the light
ratio from them by line photometry (§1.2). The two conventions are the two ends of the same
axis.

**Spectangular** takes them as user input per spectrum and, since 2019, can optimise them
(`2019A&A...623A..31S`, §3.4), for which it carries an internal spherical-geometry light-curve
model with circular-segment eclipse areas (their Eqs. 5 to 12), producing
$`k(\phi) = f_A(\phi)/[1-f_A(\phi)]`$.

**Practice in the Pavlovski and Southworth line of papers.** The light ratio comes from a
Wilson-Devinney or JKTEBOP solution of published light curves and is assigned per spectrum
(`2009MNRAS.394.1519P`, §5). Where the photometry is degenerate the direction is reversed: for
V478 Cyg, "The Sezer et al. (1983) BV light curves do not yield a determinate ratio of the
radii because the eclipses are partial and quite shallow. We therefore obtained a light ratio
between the two stars from the spectroscopic analysis and used this to break the solution
degeneracy. The uncertainty in this light ratio was propagated into the final results by
running solutions with the light ratio perturbed by its errorbar" (Pavlovski, Southworth &
Tamajo 2018, `2018MNRAS.481.3129P`, §6). For AH Cep they "opted for the spectroscopically
determined light ratio" in the light-curve analysis (same paper, §5.3).

### 3.2 In-eclipse spectra and the Rossiter-McLaughlin distortion

The problem is that an eclipse changes each component's line *shape*, not only its scale, so
the "time-invariant component spectrum" ansatz fails at exactly the epochs that break the
light-ratio degeneracy. Three published responses:

**Exclude, then subtract.** Albrecht et al. (2007, `2007A&A...474..565A`, §3) build the
component spectra by Bagnuolo & Gies tomography using "all observations taken outside the
eclipses", then, for the in-eclipse spectra, "the spectrum of the foreground star is
subtracted. For an observation out of eclipse this is straightforward ... During eclipses, one
has to incorporate the change in the light ratio of the two stars due to the eclipses. For this
we assumed a linear limb-darkening law with a limb-darkening coefficient ($`u`$) of 0.6 for
both stars." They then measure the RM effect from the residual broadening function. Their own
verdict on the method is a warning: "for systems with low eccentricity, blending of the
spectral lines during eclipses is stronger than in V1143 Cyg. Therefore, every systematic error
in the tomography or in the subtraction of the foreground spectrum due to the parameters used
in the subtraction, will have a substantial effect on the value of $`\beta`$ derived."

**Model the kernel.** Their second method "avoids the problems of the first method by taking
the influence of the eclipsing star into account explicitly", fitting the whole broadening
function shape, which "makes the fitting of the parameters $`\beta`$ and $`v\sin i`$ more
precise". This is the same object as Hadrava's $`\Delta_j^k`$ of §1.3, and Hadrava's version is
the only one embedded inside a disentangling code.

**Analytic correction at profile level.** LSDBinary (`2022A&A...666A.180T`) computes the RM
distortion analytically in its initial-guess module, so the returned velocities can be RM-
corrected or RM-contaminated by choice. Lehmann et al. (2018, `2018A&A...615A.131L`) report
LSD velocities of both components of R CMa *through* primary eclipse.

### 3.3 Tools that output per-component flux versus phase programmatically

| Code | Reference | Per-component flux vs phase | Line profiles in eclipse |
|---|---|---|---|
| JKTEBOP | Southworth et al. 2004; `2023arXiv230102531S` for the current limb-darkening laws | yes, biaxial-ellipsoid model, light ratio per passband | no |
| Wilson-Devinney | Wilson & Devinney 1971 | yes, Roche geometry | no (RM as velocities only) |
| ellc | Maxted 2016, `2016A&A...591A.111M` | yes, and it computes an RM radial-velocity curve | RM velocities, not profiles |
| eb | Irwin et al. 2011 (`arXiv:1109.2055`) | yes | no |
| PHOEBE 2 | Prša et al. 2016 `2016ApJS..227...29P`; Horvat et al. 2018 `2018ApJS..237...26H`; Conroy et al. 2020 `2020ApJS..250...34C` | yes, per-component sums over mesh triangles | **yes**, via the spectroscopic module below |

The PHOEBE 2 spectroscopic module (Brož, Prša, Conroy & Abdul-Masih 2025, arXiv:2506.20868) is
the one that matters here. Its "complex" model generates a synthetic spectrum per surface
triangle, Doppler-shifts each by its local radial velocity, and sums,

```math
\Phi_\lambda = \frac{1}{L_{\rm tot}}\sum_i I_{{\rm pass},i}\,S_i\,\cos\theta_i\,f_i\,I''_{\lambda,i},
```

with $`f_i`$ the visible fraction of triangle $`i`$. Because $`f_i`$ carries the eclipse, the
RM distortion is self-consistent: "only the complex model shows asymmetries of line profiles
due to the partially eclipsed surfaces." And "Summing over triangles belonging to one component
allows to compute also per-component properties." Caveats they state themselves: the module is
in a development branch, not the official PHOEBE repository, and limb darkening is imposed by
an analytic law rather than interpolated in $`\mu`$.

### What the incumbents do

Per-epoch light factors are a *fixed input table* everywhere except Spectangular (which can
optimise them, free, per spectrum, with no wavelength dependence) and KOREL (which solves for
strengths and infers the light ratio afterwards). No disentangling code takes a light-curve
model as a parameterisation of the light factors; the light curve is solved separately, the
numbers are transcribed into the control file, and the two solutions are iterated by hand.
In-eclipse spectra are either excluded (the tomography and shift-and-add tradition), used only
for the mid-eclipse zero-point (fd3's V453 Cyg example), or given an explicit eclipse-distorted
kernel (KOREL alone, and PHOEBE's forward model outside disentangling).

### What a linear-Gaussian implementation would have to represent

**Light fractions from a light-curve model.** Replace the $`J(N_c-1)`$ free numbers of the
`PerEpoch` mode by $`\ell_{ij} = L_i(t_j; \psi)`$ with $`\psi`$ the geometric parameters
$`(r_1, r_2, i, e, \omega, \text{limb darkening}, \text{surface-brightness ratio}, \ell_3)`$.

- *Linear:* the spectra, unchanged.
- *Nonlinear:* $`\psi`$, roughly six to eight parameters replacing up to $`J`$ of them. That is
  a large reduction in the sampled dimension and, more importantly, it couples the spectroscopy
  to the photometry through a shared geometry rather than through a transcribed table.
- *Degenerate:* the scale still is. $`(c\,\ell_{ij},\ d_i/c)`$ remains an exact invariance of the
  likelihood; what an eclipse gives is the *shape* of $`\ell_{ij}(t)`$, and the scale is pinned
  only by the closure $`\sum_i \ell_{ij} + \ell_3 = 1`$. So a light-curve model breaks §5.2
  only in combination with an explicit third-light declaration. This should be stated in the API
  the same way the light ratio already is.
- *The physical caveat that must not be swallowed:* the closure enforces
  $`z_{\rm lin} = z_{\rm cont}`$, which §1.2 shows is wrong at the level of the difference
  between $`u_{\rm cont} \approx 0.3`$ and $`u_{\rm lin} \to 1`$. Either the residual is
  reported as a diagnostic (Hadrava's proposal), or the light factors are allowed to differ
  from the photometric ones by a fitted line-to-continuum limb-darkening contrast, which is one
  extra nonlinear parameter per component.

**Rossiter-McLaughlin.** Replace $`\mathbf{B}_j\mathbf{T}(\delta_{ij})`$ by an eclipse-distorted
kernel $`\mathbf{K}_{ij}(\psi)`$, precomputed from the same geometry.

- *Linear:* the spectra, again. This is the single most important structural observation of
  this section: **RM disentangling is a change of the linear operator, not of the model class.**
  The kernel is banded on the model grid exactly as $`\mathbf{B}_j`$ is, so the band assembly,
  the block-Cholesky and the selected inverse are untouched; only the kernel bank is per-epoch
  instead of shared.
- *Nonlinear:* $`v\sin i`$ per component, the projected spin-orbit angle $`\beta`$, and the same
  $`\psi`$ that drives the light fractions. These are shared with the light-curve model, which
  is the point.
- *Degenerate:* an epoch-independent kernel change is absorbed exactly by the free spectrum
  (`math.md` §1.3). Only the *in-eclipse* deviation of the kernel from the out-of-eclipse one
  carries information, so the identifiable quantity is the eclipse-induced kernel asymmetry, not
  the absolute rotational profile. This is the same structure as albireo's existing statement
  that only the *wavelength variation* of the LSF is data-identified.
- *A cheaper intermediate:* the two-term limb-darkening expansion, $`d_i^{(0)}`$ and
  $`d_i^{(1)}`$ with kernels $`\Delta^0`$ and $`\Delta^1`$ (Hadrava Eqs. 1.69, 1.74, 1.75), is
  case (b) of §1: fully linear, no new nonlinear parameters beyond the geometry, at the cost of
  doubling the number of latent spectra per component.
- *Import path:* PHOEBE 2's spectroscopic module already computes both the per-component light
  fractions and the mesh-integrated in-eclipse kernels. Reading them as tabulated inputs is a
  strictly smaller piece of work than reimplementing the geometry, and it keeps the
  responsibility for the photometry where the community already puts it.

---

## 4. Propagating light-ratio uncertainty

### 4.1 The exact first-order relations

Because the renormalisation of §2.1 is affine in $`1/\ell_i`$, the sensitivities are simple and
exact at fixed disentangled contribution $`s_i`$. Writing the renormalised line depth
$`D_i(\lambda) = 1 - S_i(\lambda)`$,

```math
D_i \;=\; b_{\rm obs} - \frac{s_i + C_i}{\ell_i}
\qquad\Longrightarrow\qquad
\frac{\partial \ln (D_i - b_{\rm obs})}{\partial \ln \ell_i} \;=\; -1 ,
```

so, for the part of the depth that comes from the multiplicative term,

```math
\frac{\delta W_i}{W_i} \;=\; -\frac{\delta \ell_i}{\ell_i},
```

an equivalent width in inverse proportion to the assumed light fraction. This is the
$`\mathrm{d}W/\mathrm{d}\ell`$ the brief asks for, and it is a property of the transformation
rather than of any particular star. Pavlovski & Hensberge state the same fact qualitatively:
"The multiplicative operation amplifies random as well as systematic errors by a factor
inversely proportional to $`\ell_{A,B}`$, i.e. a factor 1.45 for V578 Mon A and 3.2 for
V578 Mon B" (`2005A&A...439..309P`, §3), and again in `2018MNRAS.481.3129P`, §5.1: "in
renormalisation random and systematic errors are multiplied by a factor inversely proportional
to the light dilution factor."

The additive term is the other half. Pavlovski & Hensberge: "The difference in the additive
terms reflects the difference in the line blocking for the two components. Hence, from the
viewpoint of an abundance analysis, an error in these coupled additive terms would bias the
abundances of one component in the opposite sense than for the other"
(`2005A&A...439..309P`, §3). Note that $`C_A = -C_B`$ is exactly albireo's $`k=0`$ nuisance,
already identified in the D52/D53 label work as needing to be *additive* rather than
multiplicative; the literature agrees, and gives the coupling.

Signal-to-noise is the compensating effect, and two independent statements of it exist:

```math
(S/N)_i = (S/N)_{\rm obs}\,\sqrt{n_{\rm obs}}\;\ell_i
```

(Hensberge & Pavlovski 2007, `2007IAUS..240..136H`, Eq. 2, described there as "useful, but
somewhat optimistic"), and, identically, $`{\rm SNR}_{A,B} = {\rm SNR}\,f_{A,B}\sqrt{N}`$ with
the corollary that "to conserve the S/N of the observations, at least
$`N = 1/\min[f_{A,B}^2]`$ spectra are required" (`2019A&A...623A..31S`, Eqs. 1 and 2). Measured
example: V578 Mon, 12 input spectra at S/N 125 to 250, giving 360 for the primary
($`\ell = 0.69`$) and 160 for the secondary ($`\ell = 0.31`$) (`2005A&A...439..309P`).

### 4.2 What is published for $`\mathrm{d}T_{\rm eff}/\mathrm{d}\ell`$ and
$`\mathrm{d}\log g/\mathrm{d}\ell`$

**No paper found in this survey quotes a partial derivative of $`T_{\rm eff}`$ or $`\log g`$
with respect to the light ratio.** What exists instead is a set of indirect anchors, and the
brief should be answered honestly on that point:

- Tamajo et al. (2011, `2011A&A...526A..76T`, §2, Tables 1 and 2): in synthetic tests the light
  factors are recovered "to well within the errorbars for all S/N ratios considered", while
  $`T_{\rm eff}`$ and $`\log g`$ are both underestimated at low S/N. The reason given is a
  correlation structure: $`\rho(T_{\rm eff}, \log g) = 0.98`$ for both components, while "The
  correlation with the LFs is much weaker, which is why the LF values are reliable even for
  low-S/N spectra." That is the closest thing to a published statement of the sensitivity, and
  its direction is reassuring: the light factor is *not* strongly covariant with the
  atmospheric parameters when the fit is constrained to $`\sum_i \ell_i = 1`$.
- Pavlovski & Hensberge (2010, `2010ASPC..435..207P`): light factors from the light curve and
  from line-profile fitting of the disentangled spectra agree "within 1.5%" for V615 Per.
- Tkachenko (2015, `2015A&A...581A.129T`): ignoring the *wavelength dependence* of the dilution
  over 1000 Å for a pair with a factor-two temperature contrast produced "certain deviations in
  the metallicity and effective temperature of the secondary", enough that "unconstrained
  fitting is likely to deliver not entirely correct parameters."
- Pavlovski et al. (2018, `2018MNRAS.481.3129P`, §5.4.2): the AH Cep components differ in
  $`[\rm Mg/H]`$ by $`0.36\pm0.10`$ dex, which for a coeval pair is unphysical, and the authors'
  first two candidate explanations are "An incorrect light ratio, or dilution by some third
  light", with the observation that "any corrections for these effects would also affect
  determination of the atmospheric parameters, and thus the CNO abundances." That is the
  practical size of the problem: a 0.36 dex abundance discrepancy in a benchmark system, with
  the light ratio as prime suspect.
- Kolbas et al. (2015, `2015MNRAS.451.4150K`, §5.1): the wavelength dependence itself is
  measured at 14σ formal significance (§2.2 above), and the secondary's light fraction found
  spectroscopically, 0.8 to 1.8 per cent, is far below every published light-curve value.

### 4.3 Who marginalises

Nobody marginalises the *disentangling* posterior over the light ratio. The three codes that
come closest all condition on a single point-estimate disentangled spectrum and treat the light
factor as one more parameter of the downstream atmospheric fit:

- **genfitt** (Tamajo et al. 2011, `2011A&A...526A..76T`): genetic-algorithm optimisation of
  $`(T_{\rm eff}, \log g, v\sin i, \text{Doppler shift}, {\rm LF})`$ per component with the
  option $`{\rm LF}_1 + {\rm LF}_2 = 1`$; errors from a Levenberg-Marquardt covariance matrix.
- **STARFIT** (Kolbas et al. 2014, `2014MNRAS.444.3118K`; used in `2015MNRAS.451.4150K` and
  `2018MNRAS.481.3129P`): the same parameter set plus a continuum-correction term, with
  uncertainties from MCMC. This is the only published MCMC over a light dilution factor in this
  literature.
- **GSSP binary** (`2015A&A...581A.129T`): optimises $`\Re_1/\Re_2`$ instead, so the light
  dilution is a derived, wavelength-dependent quantity rather than a free parameter.

None of the three carries the covariance of the disentangled spectra into the fit. Kıran et al.
(2016, `2016A&A...587A.127K`) approximate that missing piece by adding noise to a disentangled
profile and refitting repeatedly, which albireo's docs already cite.

### What the incumbents do

The light ratio is fixed by external photometry, or fitted downstream against synthetic
spectra, and its uncertainty is propagated either not at all or by rerunning the analysis at
perturbed values (`2018MNRAS.481.3129P`, §6). The generic amplification statement, that
renormalisation multiplies both random and systematic errors by $`1/\ell_i`$, is universally
acknowledged and never turned into a covariance. The one place the propagation is done
properly is in the *reverse* direction, where a spectroscopic light ratio with an error bar is
fed into the light-curve solution to break a degeneracy in the radii.

### What a linear-Gaussian implementation would have to represent

- *Linear:* nothing new. The light ratio never was linear in the spectra; it multiplies them.
- *Nonlinear:* $`\ell_i`$ if freed. But under constant light this is pointless in isolation,
  because §5.2 makes it exactly degenerate with $`\|d_i\|`$: sampling it with a flat prior
  samples a ridge, and sampling it with an informative photometric prior returns the prior.
  The honest statement is that freeing $`\ell`$ inside the sampler buys nothing unless one of
  §5.2's four breakers is active, and albireo's existing API, which forces the user to declare
  `Fixed`, `Free(prior=...)` or `PerEpoch`, already encodes that.
- *Where the marginalisation actually belongs:* downstream, and it is cheap because the map is
  affine. Draw $`\ell_i^{(t)}`$ from its external (photometric or SED) prior alongside each
  posterior draw $`d^{(t)}`$ from the conditional Gaussian of §3.3, and form

  ```math
  S_i^{(t)} = 1 - b_{\rm obs} + \frac{d_i^{(t)} + C_i(\ell^{(t)})}{\ell_i^{(t)}} .
  ```

  The induced posterior on the renormalised spectra is a Gaussian scale mixture, exact to the
  extent that the photometric prior is, and it is the object the label fit of `science.md` §7
  should consume. This closes the gap that section currently names explicitly: "The light
  ratio, which is assumed rather than inferred under constant light fractions, is not included
  in that spread."
- *The additive nuisance already exists.* $`C_A = -C_B`$ of `2005A&A...439..309P` Eq. 4 is the
  same object as albireo's additive $`k=0`$ nuisance from D52/D53. The literature supplies the
  coupling $`\ell_A b_A + \ell_B b_B = b_{\rm obs}`$ that ties the two components' offsets
  together, which is stronger than treating them as independent, and is worth adopting.
- *Reportable derived quantities:* $`\delta W/W = -\delta\ell/\ell`$ exactly, and the error
  amplification $`1/\ell_i`$, both of which follow from the affine map and can be quoted
  alongside every renormalised spectrum without any extra computation.

---

## 5. Broadening functions and least-squares deconvolution

### 5.1 The broadening function

Rucinski's method (1992, `1992AJ....104.1968R`; the SVD formulation in 1999,
`1999ASPC..185...82R`; the radial-velocity pipeline in 2002, `2002AJ....124.1746R`) solves the
convolution equation

```math
P(\lambda') = \int B(\lambda' - \lambda)\, T(\lambda)\, \mathrm{d}\lambda,
\qquad
\vec P = \mathbf{D}\,\vec B,
\qquad
P_i = \sum_{j=0}^{m-1} T_{i+m-j}\,B_j ,
```

where $`\mathbf{D}`$ is built "from the vector $`\vec T`$ by placing it as columns of
$`\mathbf{D}`$ after shifting it downward by one index for each successive column"
(`2002AJ....124.1746R`, §5). The design matrix is exactly a bank of shifted templates. His
argument for not using Fourier division is that "the division operation usually ... does not
work: the high frequency noise becomes amplified and some sort of frequency filtering is
needed. The result may then actually depend on the applied filter" (`1999ASPC..185...82R`, §3);
the Fourier-quotient reference he gives is Anderson, Stanford & Leininger (1983), not Sargent
et al. or Tonry & Davis. The BF differs from the CCF because the CCF "inherits the common
broadening components (such as thermal, micro-turbulence, instrumental) from both spectra"
(`1999ASPC..185...82R`, §2).

Two regularisations exist in Rucinski's own work, and they are not the same one:

- For *shape* recovery, truncated SVD, $`\vec B = \mathbf{V}\mathbf{W}^{-1}(\mathbf{U}^\top
  \vec P)`$ with the small singular values discarded, the cut chosen where "the error of the
  fit stops decreasing" (`1999ASPC..185...82R`, §§5 to 7).
- For *velocities*, no truncation at all: "with the radial velocities in mind, we do not in
  fact eliminate any singular values ... if some basis functions are eliminated, there exists a
  possibility that the spectral features may acquire asymmetries through an unwanted conspiracy
  of the basis functions which remain in the definition of the BF" (`2002AJ....124.1746R`, §5).
  The noise is instead removed by convolving the recovered BF with a Gaussian of
  $`\sigma = 1.5`$ pixels, chosen because that "operation is strictly local whereas removal of
  some singular values may introduce non-local effects."

Velocities are read by fitting Gaussians, or in later work rotational profiles, to the BF peaks
(`2002AJ....124.1746R`, Eq. 4; Rucinski 2015, `2015AJ....149...49R`). Multi-component fits are
"numerically unstable forcing us to fix or manually adjust the width parameters."

The property that matters for a linear code is linearity in the spectrum, and Rucinski makes it
his central argument against TODCOR: "the non-linear nature of the cross-correlation
complicates derivation of relative luminosities of components and requires a complex
calibration while our linear approach gives directly the relative luminosities through
integration of the individual features in the broadening functions"
(`2002AJ....124.1746R`, §4). Hensberge & Pavlovski (`2007IAUS..240..136H`) restate the
condition: "The integrated intensity of the different components in the bf is directly
proportional to the light ratios, in the case of identical line blocking coefficients and on
condition that the continuum is determined correctly."

Uncertainties are the weak point, and Rucinski says so. "The error analysis for the truncated
case has not yet been done. In this situation, it may be advantageous to utilize techniques of
the external estimates, such as the bootstrap or Monte Carlo" (`1999ASPC..185...82R`, §7); and
in practice "we do not determine the radial-velocity errors from the individual BF's, but
evaluate them externally later from the orbital velocity solutions"
(`2002AJ....124.1746R`, §6). The systematic he does quantify is instructive: for GM Dra,
varying the *fixed* Gaussian widths by $`\pm20`$ km s$`^{-1}`$ moves $`V_0`$ by 0.1 to 0.2 and
$`K_1`$ by 0.15 to 0.34 km s$`^{-1}`$, but $`K_2`$ by $`-4.30`$ to $`+5.97`$ km s$`^{-1}`$,
against a bootstrap $`\sigma(K_2) = 2.50`$ (`2002AJ....124.1746R`, §9).

### 5.2 Least-squares deconvolution

Donati et al. (1997, `1997MNRAS.291..658D`) introduced LSD; the rigorous linear-algebra
statement is Kochukhov, Makaganiuk & Piskunov (2010, `2010A&A...524A...5K`). Writing
$`Y = 1 - I/I_c`$ and $`M`$ the line-pattern matrix built by linear interpolation of each mask
line onto the velocity grid,

```math
M_{i,j} = w_l\frac{v_{j+1}-v_i}{v_{j+1}-v_j},
\qquad
M_{i,j+1} = w_l\frac{v_i - v_j}{v_{j+1}-v_j},
```

the weighted least-squares solution is

```math
\chi^2 = (Y_o - MZ)^\top S^2 (Y_o - MZ) \to \min,
\qquad
Z = \big(M^\top S^2 M\big)^{-1} M^\top S^2 Y_o .
```

Their own reading of that expression is the sentence a linear-Gaussian code should quote:
"The $`M^\top S^2 Y_o`$ term ... represents a weighted cross-correlation between the line mask
and the observed spectrum. The left-hand part is the inverse of autocorrelation matrix
$`M^\top S^2 M`$, which effectively deconvolves the cross-correlation vector and provides
formal uncertainty estimates of the LSD profile through the diagonal elements of
$`(M^\top S^2 M)^{-1}`$."

They give a regularised variant, first-order Tikhonov,

```math
Z = \big(M^\top S^2 M + \Lambda R\big)^{-1} M^\top S^2 Y_o,
\qquad
R = \mathrm{tridiag}(-1, 2, -1),
```

motivated because "the second step systematically increases the noise in the mean profile
compared to CCF independently of the numerical technique used to invert the autocorrelation
matrix", and demonstrated to gain about 50 per cent in signal-to-noise on a 500-line Stokes V
simulation.

The multiprofile extension is the SB2-capable one:

```math
Y = \sum_{k=1}^{N} M_k Z_k = M' Z',
```

with $`M'`$ an $`n \times (mN)`$ concatenation. Kochukhov et al. name the binary case explicitly
("a composite spectrum of a binary star") but do not apply it to one. They also record the
amplitude degeneracy that recurs throughout this note: "Observations in each Stokes parameter
constrain the product of the corresponding line weight and the LSD profile."

Their validity limits, which are quantified and rarely quoted: self-similarity holds only for
lines "weaker than $`\approx 40\%`$ central depth", and linear addition of blends is accurate to
$`\le 10^{-2}`$ only for blend components with residual intensity $`\lesssim 20\%`$.

Tkachenko et al. (2013, `2013A&A...560A..37T`) generalise to $`N`$ components $`\times`$ $`n`$
profiles per component (with $`n=2`$ optimal above $`v\sin i = 30`$ km s$`^{-1}`$) and add a
line-strength correction, and they applied it to the SB2 eclipsing binary KIC 11285625 with
$`2\times3 = 6`$ simultaneous profiles. But they record two failures: "the LSD method suffers
from degeneracy between spectral contributions of the two stars when the average profiles are
computed on (largely) overlapping RV grids", and it is expected "to fail completely in eclipse
phases"; and their attempt to disentangle from LSD profiles was abandoned because the line
optimisation "suffers from strong degeneracy between contributions of individual stellar
components and delivers unreliable disentangled spectra."

The paper that actually delivers SB2 velocities is Tkachenko et al. (2022,
`2022A&A...666A.180T`, code `LSDBinary`). Their diagnosis of why plain LSD does not: the
composite LSD profile "will contain signatures of both binary components ... Measuring RVs
still proves nearly as difficult as from the original observed composite spectra because,
despite the significantly enhanced S/N in the LSD profile, line blending remains the dominant
source of uncertainty." Their model is a linear sum of two components' LSD-based model spectra
weighted by a wavelength-dependent flux ratio (fitted as a second-degree polynomial) and the
radii ratio, with local fractional intensity corrections, and RM handled analytically. Their
velocities are explicitly differential: "RVs delivered by the LSDBinary algorithm are on the
scale relative to the initial guess LSD profiles." They also report the multiplex gain,
"typically anywhere between a factor of 5 to 50", and that "the orbital inclination angle and
components' radii ratio are found to have the largest effect on the shapes of the LSD profiles
and RV curves extracted from them."

A Gaussian-process prior on an LSD profile already exists: Asensio Ramos & Petit (2015,
`2015A&A...583A..51A`) replace the Tikhonov term by a GP covariance,

```math
\Sigma_Z = \Big[K(\alpha_Z)^{-1} + W^\top \mathrm{diag}(1/\sigma^2) W\Big]^{-1},
\qquad
\mu_Z = \Sigma_Z W^\top \mathrm{diag}(1/\sigma^2) V,
```

with squared-exponential or Matérn kernels whose length scale and amplitude are learned by
type-II maximum likelihood, so that "this prior allows the result to automatically adapt to the
presence of signal", and with the posterior covariance giving "the uncertainty at each velocity
bin". It is developed for Stokes V on single stars, not for SB2 velocities.

### 5.3 How widely they are used

BF: the DDO contact-binary survey ran to fifteen papers, 60 orbits by Paper VI
(`2002AJ....124.1746R`); AW UMa gave $`v\sin i = 181.4\pm2.5`$ km s$`^{-1}`$ measured directly
from the BF (`2015AJ....149...49R`); Rawls et al. (2016, `2016ApJ...818..108R`) used it for a
Kepler double red giant, noting the normalisation $`\int B(v)\,\mathrm{d}v = 1`$ for an exact
template match; Smith et al. (2021, MNRAS, DOI 10.1093/mnras/stab2374, bibcode unverified)
used it in NGTS J0002-29 to obtain primary, secondary *and tertiary* velocities at every epoch,
spectroscopic light ratios from the BF peak areas in three separate windows, and per-epoch
uncertainties from an MCMC fit of three Gaussians plus a Gaussian-process noise model. That
last paper is the strongest modern exemplar.

LSD: Sekaran et al. (2020, `2020A&A...643A.162S`, §3) is a clean case of LSD as the SB2
velocity product, with a 3000-line mask, a synthetic-LSD-profile grid fitted to the observed
profiles, "$`v\sin i`$ as a proxy of rotational broadening ... as a free parameter to account
for the effects of line-profile variations", and "a scaling factor for the depth of each
profile to account for the light contribution of each component to the total flux"; velocity
precision 0.4 km s$`^{-1}`$ at S/N 20. Lehmann et al. (2018, `2018A&A...615A.131L`) obtained
velocities of both components of R CMa through primary eclipse.

One negative worth recording: Matson et al. (2017, `2017AJ....154..216M`), often cited as BF
work, in fact uses a two-dimensional double cross-correlation validated against TODCOR. The
Kepler eclipsing-binary follow-up community is genuinely split between the two families.

### What the incumbents do

BF and LSD are separate tools, run before or beside disentangling, not inside it. Their
regularisations are ad hoc by their authors' own admission (truncated SVD for shape, post-hoc
Gaussian smoothing for velocities, a hand-set Tikhonov $`\Lambda`$ for iLSD), and their
uncertainties are either not propagated at all (Rucinski's pipeline bootstraps the orbit
instead) or computed as $`\mathrm{diag}[(M^\top S^2 M)^{-1}]`$ rescaled so that
$`\chi^2_\nu = 1`$. No published broadening function is solved with a Tikhonov or
Gaussian-process prior in place of SVD truncation, and no published Gaussian-process LSD has
been applied to SB2 velocities.

### What a linear-Gaussian implementation would have to represent

The observation that makes this cheap: **BF and LSD are the same normal equations albireo
already assembles, with the latent moved from the spectrum to the profile.** Hold the component
spectra fixed at templates $`t_i`$ and let the unknown be a per-epoch velocity profile
$`z_{ij}`$ on a velocity grid,

```math
m_j = \mathrm{diag}(r_j)\,\mathbf{R}_j\Big[\mathbf{1} + \sum_i \mathbf{Z}_{ij}\,t_i\Big],
\qquad
\mathbf{Z}_{ij} = \sum_k z_{ijk}\,\mathbf{T}(\delta_k),
```

which is linear in $`z_{ij}`$ because $`\mathbf{T}(\delta_k)t_i`$ is a fixed vector for each
grid node $`k`$.

- *Linear:* $`z_{ij}`$, the profile itself. Everything in `math.md` §3 applies verbatim: the
  marginal likelihood, the conditional mean, the exact posterior covariance from the same
  factor.
- *Structurally easier than disentangling.* Each epoch's profile appears only in that epoch's
  data, so $`\tilde{\boldsymbol\Lambda}`$ is **block diagonal by epoch** rather than coupled
  across epochs. The block-Cholesky reduces to $`J`$ independent small solves of size
  $`N_c m`$, with $`m \sim 10^2`$ to $`10^3`$.
- *The prior is already written.* albireo's $`\boldsymbol\Lambda = \tau \mathbf{D}_2^\top
  \mathbf{D}_2 + \eta\mathbf{I}`$ applied to $`z_{ij}`$ **is** Kochukhov et al.'s Eq. 24 with a
  second-difference instead of a first-difference operator, and the ML-II choice of
  $`(\tau,\eta)`$ **is** Asensio Ramos & Petit's empirical-Bayes length scale. This is the
  single clearest place where albireo's existing machinery answers an acknowledged open problem
  in the incumbent literature, and where the argument against SVD truncation is Rucinski's own:
  truncation is non-local in velocity and can manufacture profile asymmetry, whereas a
  curvature penalty is local by construction.
- *Nonlinear:* the template labels (or, equivalently, which disentangled component is used as
  the template), the LSF, the response polynomial. Velocity ceases to be a parameter at all and
  becomes a coordinate, read off the recovered profile afterwards, which removes the
  two-dimensional grid search that TODCOR requires.
- *Degenerate:*
  1. The overall scale of $`z_{ij}`$ against the template's line depths, exactly Kochukhov
     et al.'s amplitude degeneracy and exactly `math.md` §5.2. The integral
     $`\int z_{ij}\,\mathrm{d}v`$ is the light fraction only up to that scale and only if the
     line blocking of the two components matches (`2007IAUS..240..136H`).
  2. A common velocity shift of all profiles against $`\gamma`$, exactly `math.md` §5.3. The
     velocities are differential unless an absolute template anchors them, which albireo's
     pipeline already handles through the label fit's `v_kms`.
  3. Profile shape against the LSF, exactly `math.md` §5.4's LSF row, since a stationary kernel
     is absorbed by the free profile.
- *Relation to the existing TODCOR mode:* albireo's §10 already fits $`N`$ shifted,
  LSF-convolved templates by weighted least squares. The BF/LSD extension is the same design
  matrix with the two-column restriction lifted: TODCOR is the profile problem under the hard
  prior $`z(v) = \alpha_1\delta(v-v_1) + \alpha_2\delta(v-v_2)`$. Everything TODCOR gains in
  conditioning it pays for in that prior; everything the profile form gains in shape freedom it
  pays for in needing a regulariser, which albireo has. Stating the pair this way is the
  cleanest framing for the documentation.

---

## 6. The $`K_1`$-$`K_2`$ chi-square map convention

### 6.1 What is plotted

The only explicit definition in this literature appears in Shenar et al. (2022,
`2022A&A...665A.148S`, Eq. 1) and Shenar et al. (2022, `2022NatAs...6.1085S`, Eq. 1):

```math
\chi^2_{\rm reduced}(K_1, K_2) =
\frac{1}{N_\lambda\,(N_{\rm epochs}-2)}
\sum_{i=1}^{N_{\rm epochs}}\sum_{k=1}^{N_\lambda}
\frac{\big(A_{i,k} + B_{i,k} - O_{i,k}\big)^2}{\sigma_i^2}.
```

Details that matter for reproducing it:

- It is already reduced, with $`\nu = N_\lambda(N_{\rm epochs}-2)`$, justified because "each
  pixel in each of the two disentangled spectra is considered a free variable".
- $`\sigma_i`$ is a **single scalar per epoch**, "the S/N in the continuum region in the
  vicinity of the spectral line". There is no per-pixel error array.
- The sum runs over one selected line or line group, not the whole spectrum; lines are
  disentangled individually and combined afterwards.
- $`A_{i,k}, B_{i,k}`$ are the *converged* shift-and-add spectra at that trial $`(K_1,K_2)`$, so
  every grid node requires a full convergence.
- All other orbital elements are held fixed. Mahy et al. (2022, `2022A&A...664A.159M`): "we
  fixed the orbital period, eccentricity, longitude of the periastron passage and the time of
  reference, and only let the RV semi-amplitudes ... vary". Shenar et al.
  (`2022A&A...665A.148S`) give the reason: extending to $`e`$ or $`P`$ "becomes computationally
  expensive for the shift-and-add algorithm."

### 6.2 How the minimum and the errors are read

**No paper in this lineage states a $`\Delta\chi^2`$ threshold, an F-distribution criterion, or
a rescaling of $`\chi^2_\nu`$ to unity for the $`K_1`$-$`K_2`$ map.** The contours are only
labelled, for example "showing the 1σ contour as well (green dashed line)"
(`2022A&A...665A.148S`, Fig. 2 caption). The hypothesis that $`\chi^2_\nu`$ is rescaled to unity
before reading errors is refuted by the published colour bars, whose minima are neither unity
nor consistent between papers: 1.42 (VFTS 350, `2022A&A...665A.148S`), 1.375 (VFTS 243,
`2022NatAs...6.1085S`), $`\approx 28.25`$ and $`\approx 4.7`$ (R 144,
`2021A&A...650A.147S`), $`\approx 1.8`$ and $`\approx 2.55`$ (LB-1, `2020A&A...639L...6S`), and
1.025, 1.235, 1.605 (ALS 8814, arXiv:2509.01545). LB-1's Fig. A.3 does show curves "normalised
to unity", but the paper says this is "to allow for an easy comparison between the different
curves" and gives the absolute values separately in Fig. A.4.

What is actually done:

- **LB-1** (`2020A&A...639L...6S`): "A parabola fit to the minimum region in the combined
  reduced $`\chi^2`$ of all Balmer lines yields $`K_2 = 11.3\pm1.4`$ km/s, where the 1σ error is
  calculated from the corresponding $`\chi^2(K_2)`$ contour." The contour level is not
  specified. Their FDBinary cross-check used a genuine Monte Carlo, "3000 Monte Carlo samples of
  the minimisation process, while altering the spectra and the orbit by adding white noise",
  giving $`11.2\pm1.0`$. Grid: $`K_2`$ from 0 to 100 km s$`^{-1}`$ in steps of 0.5.
- **HR 6819** (`2020A&A...641A..43B`): per-line-group $`\chi^2_\nu(K_2)`$ curves, 0 to 20
  km s$`^{-1}`$ in steps of 0.5, with a marked 1σ interval, then a weighted mean over line
  groups, and the caveat quoted in §1.5.
- **TMBM VI** (`2022A&A...665A.148S`): the de facto error estimator is the line-to-line spread,
  "A weighted mean of the measurements obtained for the He I λ4026, λ4144, and λ4388 lines
  yields $`K_2 = 238\pm96`$ km/s".
- No bootstrap over epochs was found in any of these papers.

The characteristic structure reported is not a diagonal ridge but a **flat valley along
$`K_2`$**: "We find that $`\chi^2`$ is flat with respect to $`K_2`$"
(`2022NatAs...6.1085S`), and the general rule "The fainter the secondary, the less of an impact
$`K_2`$ has on the disentangled spectra. In contrast, even small deviations in $`K_1`$ can
result in spurious features in the disentangled spectrum of the secondary"
(`2022A&A...665A.148S`). Diagonal ridges are reported only in the older critique.

### 6.3 The critiques

The sharpest is the oldest. Hynes & Maxted (1998, `1998A&A...331..167H`) is titled *A critique
of disentangling as a method of deriving spectroscopic orbits*, and is the only source that
states the contour recipe and then tests it:

- The recipe: "construct a contour of constant $`\chi^2`$, defined by
  $`\chi^2 = \chi^2_{\rm min} + \Delta\chi^2`$ and project this onto the $`K_1`$ and $`K_2`$
  axes. For the 1σ, two parameter case $`\Delta\chi^2 = 2.3`$."
- The result: "While the deduced uncertainties for $`K_1`$ are satisfactory, those for $`K_2`$
  are somewhat low. This is not unique to this data set, it is a common problem."
- The diagnosis, which is precisely the nuisance-parameter question: "In applying curvature
  analysis as above we are assuming that the separated spectra are simply 'nuisance parameters'
  and can be ignored in the error analysis ... the fact that curvature analysis consistently
  gets the relative errors on $`K_1`$ and $`K_2`$ wrong indicates that the rôle of the separated
  spectra may be more subtle."
- On the surfaces: "One grid shows a 'ridge' near the minimum, a feature that was commonly seen
  in these grids. The uncertainties derived from the curvature of such a surface are clearly
  unreliable."
- On correlated noise: "the noise distribution is unlikely to be the idealized uncorrelated
  Poisson noise assumed in this work, e.g. interpolating spectra onto a logarithmic grid is
  known to introduce a short scale autocorrelation into the noise, which can lead to a
  'rippling' on the residual surfaces." albireo's AR(1) model of `math.md` §1.4a is the direct
  answer to that sentence.

El-Badry & Quataert (`2021MNRAS.502.3436E`) and El-Badry et al. (arXiv:2509.01545) confirm the
underestimate empirically, the latter with the factor-of-five discrepancy quoted in §1.5.
Quintero, Eenens & Rauw (`2020AN....341..628Q`) is a critique of the reconstructed *spectra*,
not of the map: the shift-and-add "line fluxes are poorly reproduced and spurious wings appear",
severe enough that "the classification of massive stars in binaries can be off by several
subtypes". Its influence on the convention is indirect but important: `2022A&A...665A.148S`
adopts the mitigation, "A workaround is to enforce the disentangled spectra to lie below the
continuum for absorption-line spectra ... This assumption is adopted throughout the paper." That
is an unmodelled prior sitting inside the very $`\chi^2`$ whose contours are then read as 1σ
errors.

Three corrections to the brief, established by the background search: Mahy et al. 2020,
A&A 634, A118 (`2020A&A...634A.118M`) is *TMBM IV, double-lined photometric binaries*, contains
no $`K_1`$-$`K_2`$ map and no shift-and-add disentangling; the intended paper is almost
certainly Mahy et al. 2022 (`2022A&A...664A.159M`). Shenar et al. 2022, A&A 665, A148
(`2022A&A...665A.148S`) is *TMBM VI*, the methods paper, not the VFTS 243 paper, which is
`2022NatAs...6.1085S`. Shenar et al. 2021, A&A 650, A147 (`2021A&A...650A.147S`, R 144) is real
and does carry two-dimensional maps.

### What the incumbents do

They publish an uncalibrated profile-likelihood-shaped object: a reduced $`\chi^2`$ built with a
scalar per-epoch $`\sigma`$, a degrees-of-freedom count that treats all $`2N_\lambda`$
component-spectrum pixels as free parameters, a positivity constraint injected into the fit,
and a "1σ contour" whose numerical threshold is stated in no paper found. The practical error
bar is the weighted mean and scatter across independently disentangled lines, and every author
who has compared that scatter to the formal contour reports the formal error is too small, with
$`K_2`$ the worse offender.

### What a linear-Gaussian implementation would have to represent

Nothing new is needed. The figure is a reparameterisation of what albireo already computes.

- *Linear:* the spectra, marginalised. That is precisely the step the incumbent statistic gets
  wrong by counting them as free parameters.
- *The direct analogue* of the published map is
  $`\Delta(K_1,K_2) = -2\big[\log p(y\,|\,K_1,K_2,\hat\theta_{\rm rest}) - \max\log p\big]`$,
  the profile of the *marginal* likelihood over the remaining nonlinear parameters. Because the
  marginal already contains the Occam term
  $`\tfrac12(\log\det\boldsymbol\Lambda - \log\det\tilde{\boldsymbol\Lambda})`$, the spectra are
  integrated rather than counted, and the $`\Delta = 2.30`$ two-parameter contour is the correct
  1σ level to the extent that the marginal is locally Gaussian. That single change is the whole
  methodological claim: albireo can draw the community's figure with a defensible contour on it.
- *The degrees-of-freedom question* the community leaves open has an answer in albireo's own
  notation: `math.md` §3.2a's $`p_{\rm eff} = \mathrm{tr}[\tilde{\boldsymbol\Lambda}^{-1}
  \mathbf{A}^\top\mathbf{W}\mathbf{A}]`$ is the number of data-determined spectral modes,
  measured at $`\approx 2900`$ against $`N_cP = 19{,}876`$ on HR 6819, that is the number of
  resolution elements rather than of pixels. Hynes & Maxted's complaint that "each pixel of each
  separated spectrum is a parameter to be fitted" is exactly the quantity $`p_{\rm eff}`$
  replaces.
- *The scalar $`\sigma_i`$* is replaced by the per-pixel $`w`$ plus the per-epoch jitter
  $`\alpha_j`$ of §1.4, whose profiled estimate $`\hat\alpha^2 = \chi^2_0/(N-p_{\rm eff})`$ is
  the degrees-of-freedom-corrected version of what the incumbents do by hand.
- *The positivity constraint* is `math.md` §5.2's breaker 3, the physicality floor
  $`\ell_i d_i \ge -\ell_i`$. Making it an explicit optional constraint rather than an
  undocumented step inside the iteration is the difference between a prior and a fudge.
- *Nothing becomes degenerate that was not already.* The light ratio does not enter the
  statistic under constant light (which is why the community says "the $`\chi^2`$ statistic is
  independent of the light ratio"), and that statement is the shift-and-add form of §5.2.
- *Practical note for reproducing the figure:* the published maps hold $`P, e, \omega, T_0`$
  fixed and show a flat $`K_2`$ valley. Producing the same plot from the marginal likelihood at
  fixed nuisance parameters is the like-for-like comparison; producing it with those parameters
  marginalised is the improvement, and the difference between the two contours is itself worth
  publishing.

---

## 7. The Gaia RVS window (846 to 870 nm, $`R \approx 11\,500`$)

### 7.1 Has anyone disentangled Ca II triplet spectra?

Essentially once. Nazé et al. (2023, `2023MNRAS.525.1641N`) applied shift-and-add disentangling
to 49 X-Shooter epochs of HD 25631 over 8400 to 8800 Å, chosen because the range "contains the
Paschen lines which are ubiquitous in many stars but, in low-mass main-sequence stars, this
spectral range is also dominated by the strong absorptions of the near-IR Ca II triplet". The
orbit was **not** solved from those data: primary velocities were fixed to a pre-determined
circular solution and the mass ratio scanned only over 0.12 to 0.15. The light ratio over that
band is 0.003 to 0.01. The recovered secondary shows narrow Ca II *emission*, not absorption.

That is the whole precedent. RAVE (Matijević et al. 2010, `2010AJ....140..184M`) classifies CCF
shapes and fits two-component synthetic spectra, never recovering component spectra. Gaia-ESO
HR21 (Merle et al. 2017, `2017A&A...608A..95M`; Van der Swaelmen et al. 2025,
`2025A&A...693A.289V`) detects multiplicity from CCF structure only. Gaia DR3 RVS mean spectra
are structurally unusable: one spectrum per source, already shifted to the rest frame using the
epoch or combined radial velocity and averaged onto a common grid, which destroys exactly the
differential Doppler information disentangling consumes, and for a faint SB2 the shift applied
is the combined velocity the pipeline itself discards as meaningless.

Two indirect confirmations that the gap is real. Binnenfeld et al. (2025, arXiv:2507.12363)
built deep networks for SB2 parameters from *single-exposure* simulated RVS spectra precisely
because "the number of available spectra per target is often not enough for a proper spectral
disentangling". And Seeburger et al. (2024, `2024MNRAS.530.1935S`), the survey-scale
disentangler, cites Gaia DR4 as motivation but never tests the RVS window.

### 7.2 Why this window is hard

- **Few lines carrying velocity.** The band was designed for cool stars. Blomme et al. (2023,
  `2023A&A...674A...7B`) on hot stars: "The hotter star ... is dominated by the hydrogen Paschen
  lines, with no other substantial spectral lines that can be used for radial velocity
  determination", and "The proximity to and systematic blueward offset of the calcium infrared
  triplet to the hydrogen Paschen lines in hot stars can result in a systematic offset in radial
  velocity." DR3 hot-star velocities stop at 14 500 K; hotter stars were deferred to DR4.
- **The strongest lines are the least useful.** Merle et al. (`2017A&A...608A..95M`) on HR21:
  "the rate of SBn detection in this setup is very low because it is dominated by the presence
  of the Ca II triplet, which is a very strong feature in late-type stars, thus resulting in a
  broad CCF that can mask possible multiple peaks ... only two firm detections among the 31 970
  stars observed with this setup only." Van der Swaelmen et al. (`2025A&A...693A.289V`) rescued
  the window by *deleting* the triplet: they fit the Ca II and Mg I lines with Lorentzian
  profiles and subtract them, "which resulted in an effective masking of these strong Ca II and
  Mg I lines", after which HR21 matches the 5300 to 5600 Å window, 340 SB2 detections in 34 053
  observations versus 94 in 10 323.
- **Velocity floors set by line width, not resolution.** `2025A&A...693A.289V` state it
  explicitly: "The difference between the HR10 and the HR21 resolutions does not translate into
  a difference between the smallest detectable $`\Delta v_{\rm rad}`$", which is
  $`\sim 25`$ km s$`^{-1}`$ with the triplet removed. RAVE's floor at $`R \sim 7500`$ is 50
  km s$`^{-1}`$ (`2010AJ....140..184M`); Gaia-ESO's is 20 to 60 across GIRAFFE setups
  (`2017A&A...608A..95M`).
- **Chromospheric emission manufactures false SB2 signatures.** "emission in the line cores of
  this triplet induces fake double-peak CCFs because in the templates the lines are always in
  absorption" (`2017A&A...608A..95M`), with RAVE hitting the same thing in seven RS CVn spectra
  (`2010AJ....140..184M`), and the one successful CaT disentangling recovering a secondary whose
  Ca II is pure emission (`2023MNRAS.525.1641N`).
- **DR3 SB2 processing is unpublished.** Katz et al. (`2023A&A...674A...5K`) state that
  double-line spectra "are analysed by a dedicated method (Damerdji et al. 2022) ... but not
  published in Gaia DR3", and "Approximately 40 000 sources with 10% or more transits flagged as
  double-line were considered SB2 candidates and their combined radial velocities were
  discarded." The promised SB2 companion paper has not appeared; El-Badry (2024,
  arXiv:2403.12146) confirms that "the processing of the epoch RV data and fitting of the orbits
  has not yet been described in a publication." No claim about whether DR3 used a TODCOR-like
  two-dimensional CCF can be supported from the published record.

### What the incumbents do

Nothing, in this window. Every published effort detects multiplicity from CCF structure, and
the two efforts that work best in the band do so by *removing* the Ca II triplet before
correlating. The only disentangling in the band held the orbit fixed and used the window as an
extreme-contrast detection channel.

### What a linear-Gaussian implementation would have to represent

Nothing new is linear. The window is short and the physics is adversarial, and the value albireo
can add is to say so quantitatively rather than to delete the offending lines.

- **The window's own numbers.** $`\ln(870/846) = 0.0280`$, so the entire band spans
  $`8.4\times10^{3}`$ km s$`^{-1}`$ of log-wavelength. A differential shift of, say, 300
  km s$`^{-1}`$ peak to peak is 3.6 per cent of the window, and the Ca II lines themselves have
  a velocity width of order $`10^2`$ km s$`^{-1}`$. `math.md` §5.1's criterion, that features
  broader than the RMS differential shift cannot be attributed to either star, therefore fails
  for the dominant lines in exactly the regime the surveys operate in. The $`\lambda_-(k)`$
  forecast should be reported for the triplet's own spatial frequency, not just for the window.
- **The per-pixel prior profile is the right tool.** `math.md` §2's
  $`\boldsymbol\Lambda_i = \mathbf{D}_2^\top\mathrm{diag}(\tau_i p^\tau_i)\mathbf{D}_2 +
  \mathrm{diag}(\eta_i p^\eta_i)`$ lets the triplet be *represented*, because it is real signal
  in both components, while the posterior covariance reports honestly that the *separation*
  there is prior-dominated. That is strictly better than the community practice of Lorentzian
  subtraction, which discards the signal and cannot report what was lost.
- **Chromospheric emission is a variable component, not noise.** The false-SB2 mechanism of
  `2017A&A...608A..95M` is a component with $`K=0`$ relative to one star and a per-epoch
  amplitude, which is structurally identical to albireo's nebular component of §1.3: a fixed
  shape at a fixed shift with a free per-epoch scale, confined by a per-pixel prior profile to
  the three triplet cores. The §1.3 nebular test, where omitting such a component moved
  $`K_2`$ by $`-59`$ per cent, is the warning that applies here.
- **Not linear, and not available:** anything requiring epochs. DR3 publishes one mean spectrum
  per source in the rest frame. The window becomes an albireo target only with DR4's
  `rvs_epoch_spectrum`, released 2 December 2026, and the memory note
  [[gaia-rvs-facts]] already records the schema.

---

## 8. Bibcode ledger

Verification states: **[A]** bibcode seen in a live ADS URL returned by search, with a matching
title; **[C]** constructed from a journal reference read on an arXiv abstract page or confirmed
through Crossref (volume and page); **[U]** unverified, no refereed volume and page established.

### Disentangling theory and codes

| Bibcode | State | Note |
|---|---|---|
| `1994A&A...281..286S` | [A] | Simon & Sturm, wavelength-space disentangling |
| `1995A&AS..114..393H` | [A] | Hadrava, Fourier disentangling (KOREL) |
| `1997A&AS..122..581H` | [A] | Hadrava, **variable line strengths**; the origin of $`s_j(t)`$ |
| `2009arXiv0909.0172H` | [C] | Hadrava, *Disentangling of spectra: theory and practice*. The complete derivations: §1.6.3 strengths, §1.6.4 line photometry, §1.6.5 LPVs, §1.6.6 pulsation, §1.6.7 RM, §1.6.8 constraints, §1.6.10 renormalisation |
| `2009A&A...507..397H` | [A] | Hadrava, Šlechta & Škoda, *Notes on disentangling II*, Cepheid pulsation kernels; δ Cep |
| `2001LNP...573..269I` | [A] | Ilijić, Hensberge & Pavlovski, undulations |
| `2004ASPC..318..111I` | [A] | Ilijić et al., **normalisation of FDBinary output**; "Light-factors can be used only as fixed, possibly time-dependent parameters" |
| `2008A&A...482.1031H` | [C] | Hensberge, Ilijić & Torres, A&A 482, 1031, bias progression and spurious patterns (Crossref-confirmed; not on arXiv, not read) |
| `1998A&A...331..167H` | [A] | Hynes & Maxted, *A critique of disentangling*; $`\Delta\chi^2 = 2.3`$, ridges, $`K_2`$ errors too small |
| `2006A&A...448..283G` | [A] | González & Levato, shift-and-add |
| `2017A&A...597A.125S` | [C] | Sablowski & Weber, Spectangular I |
| `2019A&A...623A..31S` | [C] | Sablowski, Järvinen & Weber, **Spectangular on variable spectra**; per-spectrum flux ratios; "cannot account for wavelength dependent flux ratios" |
| `2020AN....341..628Q` | [A] | Quintero, Eenens & Rauw, shift-and-add spectral artefacts |
| `2024MNRAS.530.1935S` | [A] | Seeburger et al., survey-scale Tikhonov disentangling |
| fd3 v3.1 distribution | n/a | `sail.zpf.fer.hr/fd3`, retrieved via the Internet Archive; the `.in`/`.out` pair documents the per-epoch light-factor table |

### Light ratios, renormalisation and atmospheric parameters

| Bibcode | State | Note |
|---|---|---|
| `2005A&A...439..309P` | [A] | Pavlovski & Hensberge, V578 Mon; **the renormalisation equations** and the $`1/\ell`$ amplification (1.45 and 3.2) |
| `2009MNRAS.394.1519P` | [C] | Pavlovski & Southworth, V453 Cyg; light factor per spectrum from the light curve |
| `2010ASPC..435..207P` | [A] | Pavlovski & Hensberge, review; the four renormalisation routes; 1.5 per cent agreement for V615 Per |
| `2011A&A...526A..76T` | [C] | Tamajo, Pavlovski & Southworth, **genfitt**; constrained LF fitting; $`\rho(T_{\rm eff},\log g) = 0.98`$ |
| `2014MNRAS.444.3118K` | [C] | Kolbas et al., **STARFIT**; MCMC over $`(T_{\rm eff}, \log g, {\rm lf}, v\sin i, v_0, cc)`$ |
| `2015A&A...581A.129T` | [C] | Tkachenko, **GSSP**; the wavelength-dependent dilution $`\alpha = (I^c_1/I^c_2)(\Re_1/\Re_2)^2`$; KIC 6352430 |
| `2015MNRAS.451.4150K` | [C] | Kolbas et al., Algol; **per-line light factors**, 0.943 vs 0.915 and 0.008 vs 0.018 |
| `2018MNRAS.481.3129P` | [C] | Pavlovski, Southworth & Tamajo; renormalisation practice; AH Cep Mg discrepancy |
| `2007IAUS..240..136H` | [C] | Hensberge & Pavlovski, *Modern analysis techniques for spectroscopic binaries*; $`(S/N)_i = (S/N)_{\rm obs}\sqrt{n_{\rm obs}}\,\ell_i`$; BF light-ratio statement. Journal reference from the arXiv listing (astro-ph/0611422); IAU Symposium volume and page **not independently confirmed** |
| `2016A&A...587A.127K` | [A] | Kıran et al., noise-injection estimate of disentangling uncertainty |

### Eclipses, RM and light-curve codes

| Bibcode | State | Note |
|---|---|---|
| `2007A&A...474..565A` | [C] | Albrecht et al., V1143 Cyg; out-of-eclipse tomography, in-eclipse foreground subtraction with $`u=0.6`$, BF modelling |
| `2014arXiv1403.0583A` | [U] | Albrecht et al., BANANA V, CV Vel; arXiv only in this session |
| `2016A&A...591A.111M` | [C] | Maxted, **ellc** |
| `2016ApJS..227...29P` | [A] | Prša et al., PHOEBE 2.0 |
| `2018ApJS..237...26H` | [C] | Horvat et al., PHOEBE 2, spin-orbit misalignment |
| `2020ApJS..250...34C` | [A] | Conroy et al., PHOEBE 2.3 |
| arXiv:2506.20868 | [U] | Brož, Prša, Conroy & Abdul-Masih, **PHOEBE IX, spectroscopic module**; per-component mesh sums and self-consistent in-eclipse line profiles. Development branch, not the official repository |
| arXiv:1109.2055 | [U] | Irwin et al., LSPM J1112+7626; the paper that introduces the `eb` light-curve code |
| `2023arXiv230102531S` | [C] | Southworth, JKTEBOP limb-darkening laws |

### Variability, Be discs and pulsators

| Bibcode | State | Note |
|---|---|---|
| `2020A&A...641A..43B` | [C] | Bodensteiner et al., HR 6819; Hα excluded, epochs not combined, $`K_2 = 4.0\pm0.8`$, light ratio bracketed by physicality |
| `2020A&A...639L...6S` | [C] | Shenar et al., LB-1 |
| `2021MNRAS.502.3436E` | [A] | El-Badry & Quataert; line-to-line scatter exceeds formal errors |
| `2023A&A...677L...9J` | [C] | **Janssens** et al., MWC 656 (not Shenar); Balmer, Paschen and He II λ4686 excluded for variability |
| arXiv:2509.01545 | [U] | El-Badry, Fabry, Sana, Shenar & Seeburger, ALS 8814; $`K_2 = 37, 22, 24`$ km s$`^{-1}`$ from three lines against a formal $`\pm1.6`$ |
| `2020A&A...643A.162S` | [C] | **Sekaran** et al. (not Tkachenko), *Tango of celestial dancers*; orbit fixed during disentangling; pulsational distortion in the output |
| `2014A&A...569A.118A` | [C] | Aerts et al., pulsational broadening mimics macroturbulence |
| `2009A&A...508..409A` | [C] | Aerts et al., collective pulsational velocity broadening |
| `2007A&A...474..193L` | [C] | Linder et al., the Struve-Sahade effect in O-type binaries (erratum `2012A&A...541C...2L`) |

### Broadening functions and LSD

| Bibcode | State | Note |
|---|---|---|
| `1992AJ....104.1968R` | [A] | Rucinski, original BF paper. **Correction to the brief:** it is AJ 104, 1968, not ASP Conf. Ser. 38 |
| `1999ASPC..185...82R` | [C] | Rucinski, SVD formulation and truncation; why not Fourier division; "the error analysis for the truncated case has not yet been done" |
| `2002AJ....124.1746R` | [A] | Rucinski, the BF as an RV product; **no truncation** in the RV pipeline, $`\sigma = 1.5`$ px smoothing instead; bootstrap errors; the $`K_2`$ systematic |
| `1997MNRAS.291..658D` | [A] | Donati et al., LSD. Not on arXiv; formulation sourced from Kochukhov et al. |
| `2010A&A...524A...5K` | [A] | Kochukhov, Makaganiuk & Piskunov, **rigorous LSD**; Eq. 20 as CCF times inverse Gram; multiprofile Eq. 21; Tikhonov Eqs. 24 to 25; 40 per cent and 20 per cent validity limits |
| `2013A&A...560A..37T` | [C] | Tkachenko et al., improved LSD; SB2 denoising; the overlapping-grid degeneracy |
| `2022A&A...666A.180T` | [C] | Tkachenko et al., **LSDBinary**; wavelength-dependent flux ratio, radii ratio, analytic RM; differential velocities |
| `2015A&A...583A..51A` | [A] | Asensio Ramos & Petit, **Bayesian LSD**; GP prior, empirical-Bayes length scale, posterior covariance per velocity bin |
| `2016ApJ...818..108R` | [C] | Rawls et al., BF for a Kepler double red giant; $`\int B\,\mathrm{d}v = 1`$ |
| DOI 10.1093/mnras/stab2374 | [U] | Smith et al. 2021, NGTS J0002-29; three-component BF velocities, BF-area light ratios, GP + MCMC |
| `2015AJ....149...49R` | [C] | Rucinski, AW UMa BF time series; $`v\sin i = 181.4\pm2.5`$ km s$`^{-1}`$ |
| `2018A&A...615A.131L` | [C] | Lehmann et al., R CMa; LSD velocities through primary eclipse |
| `2017AJ....154..216M` | [A] | Matson et al.; a *counter*-example, double CCF plus TODCOR, not BF |
| arXiv:2401.17225 | [U] | Yi 2024, block-matrix two-template BF; read only through a fetch summariser, verify before quoting |

### The $`K_1`$-$`K_2`$ map

| Bibcode | State | Note |
|---|---|---|
| `2022A&A...665A.148S` | [A] | Shenar et al., **TMBM VI**; the definitive Eq. 1, $`\nu = N_\lambda(N_{\rm ep}-2)`$, positivity constraint, undefined 1σ contour |
| `2022NatAs...6.1085S` | [A] | Shenar et al., VFTS 243; same equation; flat $`\chi^2`$ in $`K_2`$ |
| `2021A&A...650A.147S` | [A] | Shenar et al., TMBM V, R 144; two-dimensional maps with $`\chi^2_\nu`$ minima of 28 and 4.7 |
| `2022A&A...664A.159M` | [C] | Mahy et al.; Fourier-space grid, same convention; $`\Delta\chi^2 = 2.30`$ used only for $`T_{\rm eff}`$ and $`\log g`$, **not** for the map |
| `2020A&A...634A.118M` | [C] | Mahy et al., TMBM IV. **Correction to the brief:** contains no $`K_1`$-$`K_2`$ map |

### The RVS window

| Bibcode | State | Note |
|---|---|---|
| `2018A&A...616A...5C` | [A] | Cropper et al., the RVS instrument |
| `2023A&A...674A...5K` | [A] | Katz et al., DR3 radial velocities; SB2 detection in STA; 40 000 candidates discarded |
| `2023A&A...674A...7B` | [A] | Blomme et al., hot-star RVS velocities; Paschen-only spectra above 14 500 K |
| `2023A&A...674A..34G` | [A] | Gaia Collaboration / Arenou et al., DR3 non-single stars |
| `2023MNRAS.525.1641N` | [A] | **Nazé et al., the only CaT-window disentangling found**; HD 25631, orbit fixed |
| `2010AJ....140..184M` | [A] | Matijević et al., RAVE SB2s from CCF shape; 50 km s$`^{-1}`$ floor |
| `2017A&A...608A..95M` | [A] | Merle et al., Gaia-ESO DOE; the HR21 verdict, two detections in 31 970 stars |
| `2025A&A...693A.289V` | [A] | Van der Swaelmen et al.; Lorentzian subtraction of the Ca II triplet rescues HR21 |
| arXiv:2507.12363 | [U] | Binnenfeld et al. 2025; deep networks for RVS SB2s, built because disentangling is not available |
| arXiv:2403.12146 | [U] | El-Badry 2024, Gaia binary review; DR3 SB2 processing unpublished |

### Explicitly not established

- **UNWIND (2026):** no primary source located. Any claim about its variable-line-strength
  capability is unverified.
- **The numerical definition of the "1σ contour"** in the Shenar shift-and-add maps appears to
  be unpublished.
- **$`\mathrm{d}T_{\rm eff}/\mathrm{d}\ell`$ and $`\mathrm{d}\log g/\mathrm{d}\ell`$:** no paper
  found quotes either. §4.2 lists what does exist instead.
- **N I and He I lines in the RVS band:** not supported by any source read; Blomme et al.
  describe hot RVS spectra as Paschen-dominated.
- **A Damerdji et al. SB2 paper** is cited by the Gaia DR3 papers but could not be confirmed to
  exist.
