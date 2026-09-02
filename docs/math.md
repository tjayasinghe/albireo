# Mathematical foundations

This document defines the albireo forward model, derives the analytically marginalized
likelihood, analyzes its computational structure, and works out the degeneracy theory behind
the API design. Every method implemented in the code traces back to an equation here, and
every equation here is intended to be covered by a test.

**Status:** v1 design (M0). Frozen only after review.

---

## 0. Notation and conventions

| Symbol | Meaning |
|---|---|
| $c$ | speed of light, $299\,792.458\ \mathrm{km\,s^{-1}}$ |
| $x = \ln\lambda$ | log-wavelength coordinate |
| $P$ | number of pixels of the common model grid |
| $N_c$ | number of components (2–3 stars, optionally + 1 telluric) |
| $J$ | number of epochs; epoch index $j$; component index $i$ |
| $N_j$ | number of native data pixels at epoch $j$; $N = \sum_j N_j$ |
| $s_i \in \mathbb{R}^P$ | component $i$ spectrum on the model grid (continuum-normalized) |
| $d_i = s_i - \mathbf{1}$ | *deviation* spectrum (0 in the continuum; lines are negative dips) |
| $\theta$ | all nonlinear parameters (orbit, light fractions, LSF, response, noise, prior hypers) |
| $y_j \in \mathbb{R}^{N_j}$ | observed flux at epoch $j$, on its **native** wavelength grid |
| $w_j$ | inverse variances; $w = 0$ encodes a masked pixel |

Conventions:

- **Radial velocity sign:** $v > 0$ means the source is receding; observed wavelengths are
  redshifted.
- **Frames:** data may be supplied in the topocentric (observed) or barycentric frame; the model
  handles either (§1.2). Internally all shifts compose additively in $x$.
- All flux vectors are continuum-normalized. The linear model is formulated in deviation space
  $d_i$, so that "no signal" is exactly the zero vector and edge padding of shift operators is
  zeros, not continuum.

---

## 1. The forward model

### 1.1 Log-wavelength grid and Doppler shift as translation

The model grid is uniform in $x=\ln\lambda$:

$$
x_p = x_0 + p\,\Delta,\qquad p = 0,\dots,P-1 ,
$$

with $\Delta = \ln(1 + \delta v/c)$ for a chosen pixel velocity width $\delta v$ (default: half
the finest instrument pixel).

A Doppler shift with radial velocity $v$ maps emitted to observed wavelength as
$\lambda_{\rm obs} = \lambda_{\rm em}(1+z)$, i.e. a **translation** in $x$ by

$$
\xi(v) \;=\; \ln(1+z) \;=\;
\begin{cases}
\operatorname{artanh}(v/c) & \text{relativistic (default)},\\[2pt]
\ln(1 + v/c) & \text{classical}.
\end{cases}
$$

The relativistic form follows from $1+z = \sqrt{(1+\beta)/(1-\beta)}$ for purely radial motion,
whose logarithm is $\operatorname{artanh}\beta$. It is the default for two reasons:
(i) at $|v| \sim 600\ \mathrm{km\,s^{-1}}$ the classical form is wrong by
$\sim 0.6\ \mathrm{km\,s^{-1}}$, far above the RV error budget; (ii)
$\operatorname{artanh}$ is exactly antisymmetric, so shifts compose and invert exactly:
$\xi(-v) = -\xi(v)$, and a barycentric correction is exact additive composition in $x$
(velocity composition is not additive, log-shifts are).

**The shift operator.** For a shift of $\delta = \xi(v)/\Delta$ pixels, the observed deviation
spectrum is $d(x - \xi)$, discretized by sparse linear interpolation:

$$
\left[\mathbf{T}(\delta)\, d\right]_p \;=\; (1-f)\, d_{i_p} + f\, d_{i_p+1},
\qquad i_p = \lfloor p - \delta \rfloor,\quad f = (p-\delta) - i_p ,
$$

with zero fill outside $[0, P-1]$ (correct because $d\to 0$ in the continuum). Each row has
exactly two nonzeros; $\mathbf{T}$ is a banded linear operator, and its adjoint is the
corresponding scatter-add. $\mathbf{T}(\delta)d$ is differentiable in $\delta$:

$$
\frac{\partial\,[\mathbf{T}(\delta)d]_p}{\partial \delta} = -\left(d_{i_p+1} - d_{i_p}\right),
$$

which is piecewise constant in $\delta$ with kinks at integer crossings. The *summed* gradient
used by HMC is a sum of $\mathcal{O}(P)$ such terms whose breakpoints are all distinct, so the
total log-density gradient is effectively smooth at realistic $P$; a cubic (4-tap) interpolant
with $C^1$ gradients is a planned option behind a flag if this ever limits sampler performance.

**Resampling operators.** Data are *never* interpolated onto the model grid (that would
correlate their noise and invalidate the diagonal noise model). Instead the model is projected
onto each epoch's native grid by a static sparse operator $\mathbf{R}_j$. Two flavors:

- *Point interpolation* $\mathbf{R}^{\rm interp}$: linear interpolation weights (2 nonzeros/row),
  for quick tests.
- *Pixel-integral rebinning* $\mathbf{R}^{\rm rebin}$ (default): instrument pixels integrate flux
  density, so the value in output pixel $k$ with edges $[a_k, b_k]$ is the bin average

$$
\left[\mathbf{R}\, f\right]_k = \frac{1}{b_k - a_k} \sum_l \left|\,[a_k,b_k] \cap [e_l, e_{l+1}]\,\right|\; f_l ,
$$

which conserves integrated flux exactly over fully-covered ranges:
$\sum_k (b_k - a_k)\,[\mathbf{R}f]_k = \sum_l (e_{l+1}-e_l)\, f_l$. Both are precomputed sparse
matrices (built once in NumPy, applied as gather/segment-sum in JAX).

### 1.2 Velocities

**Keplerian mode (default).** For component $i$ at time $t_j$ (BJD), the radial velocity is

$$
v_{ij} = \gamma + K_i\left[\cos(\nu_j + \omega_i) + e\cos\omega_i\right],
\qquad \omega_2 = \omega_1 + \pi ,
$$

with true anomaly $\nu_j$ from Kepler's equation for mean anomaly
$M_j = 2\pi (t_j - T_{\rm p})/P_{\rm orb}$. Both the $T_{\rm p}$ (periastron) and the
$T_0$ (conjunction) parameterization are supported, and sampling in
$(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$ is used to behave well at low eccentricity. For SB3, a
hierarchical outer orbit adds its contribution to the inner pair. The Kepler solver is a
JAX-differentiable Newton iteration with a fixed iteration count (gradients via implicit
differentiation).

**Free-velocity mode (diagnostic).** $v_{ij}$ free parameters with optional priors, used to
detect non-Keplerian residuals and to validate the orbit model.

**Frames and tellurics.** Let $u_j$ be the barycentric correction velocity at epoch $j$
(defined so that $v^{\rm bary} = v^{\rm topo} \oplus u_j$). All shifts are composed in log-shift
space, where composition is exact addition:

| | stellar component $i$ | telluric component |
|---|---|---|
| data in topocentric frame | $\xi(v_{ij}) - \xi(u_j)$ | $0$ |
| data barycentric-corrected | $\xi(v_{ij})$ | $+\xi(u_j)$ |

The telluric spectrum is therefore one more linear component, whose velocity law is the known
topocentric one; no additional machinery is required.

**Telluric linearity approximation.** Telluric transmission is physically multiplicative:
$(1 + d_\star)\,(1 + d_{\rm tell})$. albireo models it additively,
$1 + d_\star + d_{\rm tell}$, which is first-order accurate with error
$d_\star d_{\rm tell}$. That error reaches order $10^{-2}$ only where a deep stellar line
overlaps a deep telluric line, and deep tellurics are normally masked in any case. The
approximation keeps the model linear in all component spectra. Exact multiplicative tellurics
(alternating solves) are a v2 candidate; the approximation is testable in the simulator.

### 1.3 LSF, light fractions, response

**LSF.** Per-instrument Gaussian in velocity space with width $\sigma_v$: on the uniform
log-$\lambda$ grid this is a stationary discrete convolution $\mathbf{B}_j$ (Toeplitz, banded,
kernel truncated at $\pm 4\sigma$), consistent with a constant-resolving-power spectrograph.
Because $\mathbf{B}_j$ is stationary on the same uniform grid, it commutes with $\mathbf{T}$
(up to edges); it is applied after shifting, matching the physical picture in which the
instrument acts in the observed frame. The v2 seam is open (D37): a wavelength-dependent LSF
is the banded non-stationary matrix $K[m, c] = P[m,\, m - c + r]$, in which each model pixel
applies its own kernel row, realized from per-anchor kernels by linear interpolation in
log-$\lambda$ (`operators.convolve_varying`; per-anchor Gaussian widths are the built-in
parameterization, but the operator accepts arbitrary profile banks, including asymmetric
ones). Two consequences follow. *Identifiability:* the commutation argument runs in both
directions, since a stationary kernel change is absorbed exactly by the free component
spectra (reparameterize $d \to \mathbf{B}'\mathbf{B}^{-1} d$), so the marginal identifies only
the *wavelength variation* of the LSF, and only through the epoch-dependent shifts that
sample $\sigma(\lambda)$ at different observed wavelengths. Measured at gate scale, the
absolute width level is prior-dominated: the flat profile was preferred over an injected ramp
by ~3 nats under ML-II-style freedom, so fitted anchor widths are diagnostics rather than
measurements. *Orbit relevance:* by the same argument a stationary LSF error of any shape
cannot bias the orbit; only wavelength dependence can, and symmetric width variation only at
second order, since a symmetric kernel moves no centroid. The first-order channel is
wavelength-dependent *asymmetry*, whose epoch-coupled part enters as an apparent velocity
perturbation $\propto \lambda\, c'(\lambda)\, v(t)/c$.

The asymmetry channel is parameterized (D38): per-anchor Gauss–Hermite $h_3$
(`operators.gauss_hermite_kernel_traced`; van der Marel & Franx (1993) truncated at the
first asymmetric term, $|h_3| \le 0.2$), imprinting a centroid-warp field
$c(\lambda) \approx \sqrt{3}\, h_3(\lambda)\, \sigma$. Its identifiability is worse
than the width's, and measurably so. A free spectrum can represent any static warp
outright: the closed loop injected an $h_3$ ramp of $\mp 0.12$ and the joint fit
returned it flat (fitted $|h_3| \lesssim 0.03$) with the orbit unbiased, so the
data-identified remainder is only the epoch-coupled sampling of the warp's gradient,
$\Delta c \approx c'(\lambda)\,\lambda\,(v(t) - v_\mathrm{bary}(t))/c$. Magnitude at
the HR 6819 configuration: $|h_3| \sim 0.1$ varying on the echelle-order scale
(~40 Å) gives $c' \sim 0.025$ km/s/Å and shifts of $\pm 1.3$ Å, i.e. ~30 m/s of
epoch-coupled apparent-velocity modulation, two orders below the ~4 km/s of
accumulated RV signature a 0.04 d period offset represents over that baseline. This
estimate is why the *instrument-frame per-epoch* kernel realization (the pipeline's
barycentric correction makes the true kernel epoch-dependent in the analysis frame,
which a shared bank cannot express) is recorded but not built: the channel it would
add is bounded at the tens-of-m/s level. Fitted $h_3$ profiles are diagnostics; the
orbit's response is the readout.

**Light fractions.** $\ell_{ij} \ge 0$ with $\sum_i \ell_{ij} = 1$ over the stellar components
(continuum-normalized data), telluric fixed at $\ell = 1$. Constant per component by default;
a per-epoch (eclipse) mode is also supported (§5.2 gives the reason).

**Nebular emission (D40).** Massive stars are born in H II regions, so their spectra carry
emission lines that belong to neither star. Three properties set the structure. The lines do
not move with either component: they are at rest in the *barycentric* frame, the mirror of
the telluric convention. Their strength varies from exposure to exposure with seeing, slit
losses and sky subtraction, while their *shape* does not. And nebular flux is added on top
of the total stellar continuum rather than taken out of it, so its amplitude does not
belong on the light-fraction simplex at all: with $F_\star$ the (normalized) stellar
composite and $n(\lambda)$ the nebular line profile, the observed normalized flux is
$F_\star + a_j n$, not a convex combination.

All three are expressible without leaving the affine family: the nebular component is one
more column of $\mathbf{A}$ with $\delta_{\mathrm{neb},j} = \xi(v_\mathrm{neb})$ (minus
$\xi(v_{\mathrm{bary},j})$ for topocentric data) and $\ell_{\mathrm{neb},j} = a_j$ free. It
is therefore rank-one time variation, a fixed shape with a free per-epoch scale, which
is the same structure Tier 3's variable-disc component generalizes, and the reason that
generalization stays inside the linear-Gaussian family.

Two exact degeneracies come with it, and both are resolved by convention rather than by data.

1. **Scale.** Only the products $a_j d_\mathrm{neb}$ are observable, so $(c\,a_j,\,
   d_\mathrm{neb}/c)$ is the same fit for every $c > 0$. The spectral prior breaks it only
   weakly, leaving a nearly flat, unbounded direction that is worse for sampling than an
   unbroken one. albireo pins the geometric mean, $\prod_j a_j = 1$, by centering the
   log-amplitude site (`inference.nebular_amplitudes`). What the data then constrain is the
   epoch-to-epoch variation; the level lives in $d_\mathrm{neb}$.
2. **Velocity.** The nebular shift is the same at every epoch (barycentric data) or differs
   only by $\xi(v_\mathrm{bary})$ (topocentric), and a constant shift of a *free* spectrum is
   the reparameterization $d \to \mathbf{T}(\delta)d$, so $v_\mathrm{neb}$ is not identified,
   for the same reason $\gamma$ is not (§5.3). It survives as a parameter only because it
   sets *where on the model grid* the component's lines land, which matters as soon as the
   prior confines the component to windows (§2): the windows and the shift must agree.

The cost of omitting the component has been measured. In the closed loop of
`tests/test_nebular.py`, an SB2 with $K = (58, 41)$ km/s whose H$\beta$ absorption carries a
static nebular emission line of peak 0.45 varying $\pm 30$% per epoch, disentangling without
a nebular component raises the mean flux in the line core by $+0.154$ against a true depth of
$-0.506$ (the recovered core bottoms out at $-0.375$, 26% shallower) and understates the
H$\beta$ equivalent width by 11.5%. With the component the same
numbers are $+0.0015$, $-0.508$, and 0.14%, and the marginal likelihood prefers it by
$8.1\times10^4$ nats. An 11.5% error in a Balmer
equivalent width is a large error in $\log g$, and it is systematic rather than random, so no
reported uncertainty covers it.

The orbit is affected more strongly. A static line is a component with $K = 0$, so a model
with nowhere else to put it represents it with whichever star can be made to move least:
fitting the same data jointly from a cold start without the component returns $K_2 = 16.8$
against an injected 41.0 ($-59$%), a period long by 0.171 d, and a circular orbit reported at
$e = 0.95$, which is the solver's clip rather than a fit, while $K_1$ survives at $-1$%
because 70% of the light pins it. With the component:
$K_2 - 0.29$%, $\Delta P = -1.4\times10^{-4}$ d, $e = 0.0022$. Nebular contamination therefore
propagates into the dynamical answer (the masses) as well as the atmospheric one.

**Response.** Per-epoch multiplicative Chebyshev polynomial on the native grid,
$r_j(\lambda) = \sum_{m=0}^{M} c_{jm}\,\phi_m(\lambda)$ (default $M=2$), absorbing
continuum-normalization errors. Coefficients live in $\theta$.

### 1.4 The stacked linear model

The linear system is posed in wavelength space, in the manner of Simon & Sturm (1994), rather
than in the Fourier domain, so that each epoch keeps its native grid, its mask and its
per-pixel weights. Collecting §1.1 to §1.3, the model for epoch $j$ is

$$
m_j(\theta, d) \;=\; \operatorname{diag}\!\big(r_j\big)\,
\mathbf{R}_j \Big[\mathbf{1} + \sum_{i=1}^{N_c} \ell_{ij}\, \mathbf{B}_j\, \mathbf{T}(\delta_{ij})\, d_i \Big],
$$

and the property the rest of the document rests on is that, conditional on $\theta$, $m_j$ is
affine in the stacked deviation vector
$d = (d_1^\top,\dots,d_{N_c}^\top)^\top \in \mathbb{R}^{N_c P}$:

$$
y = a_0(\theta) + \mathbf{A}(\theta)\, d + n, \qquad n \sim \mathcal{N}\big(0, \mathbf{C}_n\big),
\quad \mathbf{C}_n = \operatorname{diag}(w)^{-1},
$$

where row-block $j$ of $\mathbf{A}$ is
$\big[\ \operatorname{diag}(r_j)\mathbf{R}_j \mathbf{B}_j \mathbf{T}(\delta_{1j})\ \big|\ \cdots\ \big|\
\operatorname{diag}(r_j)\mathbf{R}_j \mathbf{B}_j \mathbf{T}(\delta_{N_c j})\ \big]$ scaled by
$\ell_{ij}$, and $a_0 = \operatorname{diag}(r_j)\mathbf{R}_j \mathbf{1}$ collects the continuum.
Masked pixels have $w=0$ and drop out of every inner product below. An optional per-epoch
noise-inflation ("jitter") factor $\alpha_j$ rescales $w_j \to w_j/\alpha_j^2$ and joins
$\theta$ (`forward.with_jitter`, θ site `log_jitter`); §3.2a derives what profiling it
estimates, and why it is not the same as rescaling by the residual scatter.

$\mathbf{A}$ is never formed densely: it is a composition of gathers, stationary convolutions,
and segment-sums, each with an exact adjoint (tested against `jax.linear_transpose`).

### 1.4a Correlated noise: the AR(1) chain (D34)

A pipeline that resamples spectra onto a common wavelength step correlates adjacent
pixels; this is the mechanism left standing by D31, where a rescaled diagonal model whitened
the residual scale while relocating the orbit. The v1 correlated model keeps every term
closed form: per epoch, the noise covariance of the standardized residual is AR(1)
in native-pixel index,

$$
\mathbf{C}_j \;=\; \alpha_j^2\, \mathbf{D}_j^{-1/2}\, \mathbf{R}_{\phi_j}\, \mathbf{D}_j^{-1/2},
\qquad \mathbf{D}_j = \operatorname{diag}(w_j),\quad
(\mathbf{R}_\phi)_{pq} = \phi^{|p-q|},
$$

so $w$ keeps the per-pixel scale, $\alpha$ (D31) keeps the overall scale, and $\phi$
correlates; $\phi = 0$ is exactly the diagonal model. Two facts make this exact and
cheap:

1. **A subset of a Markov chain is Markov.** The observed (unmasked) pixels form a
   chain whose consecutive pairs, an index distance $g$ apart, carry correlation
   $\rho = \phi^{g}$, so masking introduces no approximation. Links are capped at
   `ar1_max_gap` (short-range noise does not span a chip gap; the cap also bounds the
   bandwidth cost below), beyond which the chain restarts.
2. **A chain's precision is tridiagonal in closed form.** With per-link
   $a_k = \rho_k^2/(1-\rho_k^2)$ and $c_k = \rho_k/(1-\rho_k^2)$, each link adds $a_k$
   to *both* endpoints' diagonal and $-c_k$ to their off-diagonal pair, on top of the
   identity; and
   $\log\det \mathbf{R}^{-1} = -\sum_{\rm links} \log(1-\rho_k^2)$, so the marginal's
   noise-normalization term is
   $\log\det \mathbf{W}_j = \sum_{\rm good}\log w - 2 n_j \log\alpha_j - \sum_{\rm links}\log(1-\rho_k^2)$.
   The determinant term is what makes $\phi$ identifiable rather than an unconstrained
   scale factor, as it does for $\alpha$ (§3.2a).

The whitener is the Markov factorization
$\tilde\varepsilon_k = (\varepsilon_k - \rho_k\,\varepsilon_{k-1})/\sqrt{1-\rho_k^2}$
(`data_residual_zscores` uses it when the problem is correlated). The discriminating
diagnostic is the lag-1 autocorrelation rather than the residual scale: AR(1) noise has unit
marginal variance, so a diagonal whitener still reports sd 1 while the residuals carry
autocorrelation $\approx \phi$.

Two structural consequences follow, both static. $\mathbf{A}^\top \mathbf{W} \mathbf{A}$
couples rebin rows up to `ar1_max_gap` apart, widening the half-bandwidth by the
largest model-pixel offset between a stored link's row supports
(`Problem.ar_bandwidth_extra`; `MarginalOrbitModel` reserves it behind its ``ar1``
flag, D21-style). And the inner sandwich of the D28 band assembly gains the chain's
cross-row terms: each link contributes the symmetrized outer product
$-c_k \sqrt{w_n w_p}\, r_n r_p (\mathbf{R}_n^\top \mathbf{R}_p + \mathbf{R}_p^\top \mathbf{R}_n)$
of two *different* rebin rows, carried by static link pair tables (§4.5a, D35), so the
correlated marginal stays on the band path; comb probing remains the reference
implementation.

---

## 2. Priors on the component spectra

The deviation spectra carry independent Gaussian priors,
$d_i \sim \mathcal{N}(0, \boldsymbol\Lambda_i^{-1})$, specified by **banded precision**
matrices rather than by dense kernels. A dense covariance over $P$ model pixels does not scale
to survey-sized grids, which is the limiting cost of the dense Gaussian-process formulation
used by PSOAP (Czekala et al. 2017):

$$
\boldsymbol\Lambda_i \;=\; \tau_i\, \mathbf{D}_2^\top \mathbf{D}_2 \;+\; \eta_i\, \mathbf{I},
$$

where $\mathbf{D}_2$ is the second-difference operator. Interpretation:

- $\tau_i$ penalizes curvature. The continuum limit is an integrated Wiener process, i.e. a
  smoothness prior with correlation length set by $(\tau_i/\eta_i)^{1/4}$ pixels. Matérn-class
  priors via their SPDE/state-space banded precision are a drop-in v1.x extension.
- $\mathbf{D}_2^\top\mathbf{D}_2$ has an affine nullspace (constant + slope per component).
  These are the directions of the low-frequency separation degeneracy (§5.1), so they
  must be proper: the weak ridge $\eta_i$ anchors them to the continuum ($d_i = 0$) with a
  large but finite variance. The choice sets the scale of the low-frequency uncertainty
  explicitly rather than leaving it implicit.

Hyperparameters $\tau_i, \eta_i$ are part of $\theta$: they can be fixed, optimized (ML-II, at
no extra cost, since the marginal likelihood is already computed), or sampled. The prior mean
is $0$ in deviation space; emission-line components need no special treatment.

$\log\det\boldsymbol\Lambda_i$ is cheap (banded Cholesky, bandwidth 2).

**Per-pixel strengths (D40).** Both weights may carry a static per-pixel profile:

$$
\boldsymbol\Lambda_i \;=\; \mathbf{D}_2^\top
  \operatorname{diag}\!\big(\tau_i\, p^\tau_i\big)\, \mathbf{D}_2
\;+\; \operatorname{diag}\!\big(\eta_i\, p^\eta_i\big),
$$

with row $k$ of $\mathbf{D}_2$ (which spans pixels $k, k{+}1, k{+}2$) taking the profile
value of its *center* pixel, so a profile is indexed like the spectrum it regularizes. The
scalars stay separate from the profiles so that ML-II is unchanged: a profile says
*where* a component may deviate from the continuum, the scalar says *how much*. One
consequence of that split is that an inferred $(\tau_i, \eta_i)$ replaces only the scalars and
keeps the profiles the model was constructed with. The
pentadiagonal entries generalize without new structure. Writing $t_k$ for the row
weights and reading $t$ as zero outside $[0, P{-}3]$,

$$
\Lambda_{aa} = t_a + 4t_{a-1} + t_{a-2} + e_a, \qquad
\Lambda_{a,a+1} = -2\,(t_a + t_{a-1}), \qquad
\Lambda_{a,a+2} = t_a,
$$

which reduces to the Toeplitz form (6, $-4$, 1 with the usual boundary corrections) at
uniform weight, and feeds the same $O(P)$ determinant recursion (§4.5) and the same band
assembly.

The motivating use is confinement. A nebular component (§1.3) has structure only at a
handful of known lines, and $p^\eta = 10^6$ away from them states that: the prior standard
deviation there is $10^{-3}$ of the in-window value, which is negligible against any line
and leaves the precision better conditioned than before, not worse. The constraint is soft:
a hard zero would require different linear algebra, and it would remove the model's ability
to report a disagreement with the assumed windows. The mechanism is not
nebular-specific; interstellar bands, or any component known a priori to be line-poor, take
the same treatment.

---

## 3. Marginalizing the spectra analytically

Conditional on $\theta$, the model is linear-Gaussian in $d$,
so $d$ integrates out in closed form and the sampler sees only the low-dimensional
$p(y\,|\,\theta)$.

### 3.1 Derivation

Write $\tilde y = y - a_0(\theta)$, $\mathbf{W} = \operatorname{diag}(w)$,
$\boldsymbol\Lambda = \operatorname{blkdiag}(\boldsymbol\Lambda_1,\dots,\boldsymbol\Lambda_{N_c})$.
The joint log-density is

$$
\log p(y, d \,|\, \theta) = -\tfrac12 (\tilde y - \mathbf{A}d)^\top \mathbf{W} (\tilde y - \mathbf{A}d)
-\tfrac12 d^\top \boldsymbol\Lambda d
+ \tfrac12\log\big|\tfrac{\mathbf{W}}{2\pi}\big|_{+}
+ \tfrac12\log\big|\tfrac{\boldsymbol\Lambda}{2\pi}\big| ,
$$

where $|\cdot|_+$ runs over unmasked pixels. Collect the terms quadratic and linear in $d$:

$$
-\tfrac12\, d^\top \underbrace{\left(\boldsymbol\Lambda + \mathbf{A}^\top \mathbf{W} \mathbf{A}\right)}_{\displaystyle \tilde{\boldsymbol\Lambda}(\theta)}\, d
\;+\; d^\top \underbrace{\mathbf{A}^\top \mathbf{W} \tilde y}_{\displaystyle b(\theta)} .
$$

Completing the square, $d^\top\tilde{\boldsymbol\Lambda}d - 2d^\top b =
(d-\hat d)^\top\tilde{\boldsymbol\Lambda}(d-\hat d) - b^\top\tilde{\boldsymbol\Lambda}^{-1}b$
with $\hat d = \tilde{\boldsymbol\Lambda}^{-1} b$, and integrating the Gaussian in $d$ gives the
**marginal log-likelihood**

$$
\boxed{\;
\log p(y\,|\,\theta) \;=\;
-\tfrac12\Big[\tilde y^\top \mathbf{W} \tilde y - b^\top \tilde{\boldsymbol\Lambda}^{-1} b\Big]
\;-\;\tfrac12\log\det\tilde{\boldsymbol\Lambda}
\;+\;\tfrac12\log\det\boldsymbol\Lambda
\;+\;\tfrac12\textstyle\sum_{w_u>0}\log\frac{w_u}{2\pi}
\;}
$$

Every piece depends on $\theta$: $\mathbf{A}$, $a_0$, and possibly $\mathbf{W}$ (jitter) and
$\boldsymbol\Lambda$ (hyperparameters).

A numerically preferable equivalent form uses the residual at the conditional mean: with
$\hat r = \tilde y - \mathbf{A}\hat d$,

$$
\tilde y^\top\mathbf{W}\tilde y - b^\top\tilde{\boldsymbol\Lambda}^{-1}b
\;=\; \hat r^\top \mathbf{W} \hat r + \hat d^\top \boldsymbol\Lambda \hat d ,
$$

i.e. *(weighted misfit at the regularized least-squares solution)* + *(prior penalty of that
solution)*. In code, with the Cholesky factorization
$\tilde{\boldsymbol\Lambda} = \mathbf{L}\mathbf{L}^\top$:
$b^\top\tilde{\boldsymbol\Lambda}^{-1}b = \|\mathbf{L}^{-1}b\|^2$ and
$\log\det\tilde{\boldsymbol\Lambda} = 2\sum_k \log L_{kk}$.

### 3.2 Relation to profile likelihood + Laplace

$\hat d(\theta)$ is exactly the (regularized) profile solution, and the bracketed quadratic is
the profile objective. The marginal differs from the profile likelihood only by
$\tfrac12(\log\det\boldsymbol\Lambda - \log\det\tilde{\boldsymbol\Lambda})$, the Laplace
correction, which is exact here because the model is linear-Gaussian in $d$. This
identifies the three computational strategies of §4 as the same estimator with different
$\log\det$ treatments: exact (A), stochastic (B), or frozen/dropped (C). Dropping the
$\theta$-dependence of the $\log\det$ biases the parameters that
change the information geometry (light ratios, LSF widths, prior hyperparameters), which is why
strategy C is used for quick looks only.

### 3.2a What profiling the jitter estimates

The marginal gives the noise-inflation factor of §1.4 a different denominator from the one a
direct residual calculation gives. Take one shared $\alpha$ for clarity, so
$\mathbf{W} = \mathbf{W}_0/\alpha^2$. Only two terms of the boxed marginal depend on $\alpha$
once the data-dominated directions are separated out: the weight term contributes
$-N\log\alpha$ ($N$ = unmasked pixels), and

$$
-\tfrac12\log\det\big(\boldsymbol\Lambda + \mathbf{A}^\top\mathbf{W}_0\mathbf{A}/\alpha^2\big)
\;\longrightarrow\; +\,p_{\text{eff}}\log\alpha + \text{const},
\qquad
p_{\text{eff}} = \operatorname{tr}\!\big[\tilde{\boldsymbol\Lambda}^{-1}\mathbf{A}^\top\mathbf{W}\mathbf{A}\big],
$$

$p_{\text{eff}}$ being the usual effective number of parameters (prior-dominated directions
carry no $\alpha$ dependence and drop out). With $\chi^2_0 = \hat r^\top\mathbf{W}_0\hat r$,
setting $\partial_{\log\alpha} = 0$ gives

$$
\hat\alpha^2 \;=\; \frac{\chi^2_0}{N - p_{\text{eff}}},
$$

the degrees-of-freedom-corrected variance estimate. Whitening the residuals and reading off
their standard deviation instead estimates $\chi^2_0/N$, low by $\sqrt{1 - p_{\text{eff}}/N}$.
The Occam term is doing the same work here that it does for $(\tau,\eta)$ in §5.1.

How large the correction is depends on the run, and the quantity that sets it is
$p_{\text{eff}}$, *not* the nominal parameter count $N_c P$. An oversampled model grid with a
fitted smoothness prior has far fewer data-determined modes than pixels: on HR 6819,
$p_{\text{eff}} \approx 2900$ against $N_c P = 19{,}876$, roughly the number of resolution
elements rather than of pixels, so the correction was 0.4%. Inverting the two estimators gives
$p_{\text{eff}}$ at no extra cost, $p_{\text{eff}} = N\,[1 - (\text{residual sd}/\hat\alpha)^2]$,
a diagnostic of how much of the spectrum the data constrain that is otherwise awkward to
obtain.

What the jitter does not provide: $\mathbf{W}$ stays diagonal, so a jitter can only rescale a
residual, never decorrelate one. Against systematics such as continuum errors, LSF mismatch or
intrinsically variable line profiles, $\hat\alpha$ widens the intervals around an unchanged,
still-biased point estimate, and it removes the residual-scale diagnostic while doing so.

### 3.3 Recovering the spectra and their uncertainties

Conditional on $\theta$, the posterior of the spectra is Gaussian:

$$
d \,|\, y, \theta \;\sim\; \mathcal{N}\!\left(\hat d(\theta),\; \tilde{\boldsymbol\Lambda}(\theta)^{-1}\right).
$$

The full posterior of the spectra marginalizes over the $\theta$ posterior. In practice, one
spectrum realization is drawn for each NUTS draw $\theta^{(t)}$,

$$
d^{(t)} = \hat d(\theta^{(t)}) + \mathbf{L}^{-\top} z, \qquad z \sim \mathcal{N}(0,\mathbf{I}),
$$

giving samples from $p(d\,|\,y)$ that include both the linear-Gaussian pixel noise and the
orbit/calibration uncertainty. The product is a set of disentangled spectra whose
uncertainties carry both contributions. Pointwise error bars come from the sample
variance and/or the diagonal of $\tilde{\boldsymbol\Lambda}^{-1}$ (computable without dense
inversion via the selected-inversion recursions of Takahashi et al. (1973) on the
banded/block factor). The posterior covariance between pixels, including the inflated
low-frequency modes of §5.1, is available from the same factor and is part of the standard
output.

### 3.4 Gradients

NUTS needs $\nabla_\theta \log p(y|\theta)$. All terms are compositions of JAX primitives, so
reverse-mode AD applies end-to-end, including through the Cholesky factorization (the
$\log\det$ gradient $\tfrac12\operatorname{tr}(\tilde{\boldsymbol\Lambda}^{-1}\partial\tilde{\boldsymbol\Lambda})$
emerges automatically). Two engineering notes:

- The banded/block factorization is a `lax.scan`; reverse-mode memory is controlled with
  checkpointing (`jax.checkpoint` per scan block).
- If AD through the factorization ever dominates, a custom VJP using the Jacobi formula with
  the Takahashi selected inverse computes the same gradient with one extra banded sweep.

---

## 4. Computational structure and strategy

### 4.1 The structure of $\tilde{\boldsymbol\Lambda}$

$\tilde{\boldsymbol\Lambda} = \boldsymbol\Lambda + \mathbf{A}^\top\mathbf{W}\mathbf{A}$ is an
$N_c \times N_c$ grid of $P\times P$ blocks:

$$
\big[\mathbf{A}^\top\mathbf{W}\mathbf{A}\big]_{ii'} \;=\;
\sum_j \ell_{ij}\ell_{i'j}\; \mathbf{T}(\delta_{ij})^\top \mathbf{B}_j^\top \mathbf{R}_j^\top
\operatorname{diag}(r_j^2 w_j) \mathbf{R}_j \mathbf{B}_j \mathbf{T}(\delta_{i'j}) .
$$

Let $m$ be the half-bandwidth of $\mathbf{B}^\top(\cdots)\mathbf{B}$ (LSF + rebin support,
$m \sim 10$–$30$ px). The structural fact that matters is that
$\mathbf{T}(\delta)^\top \mathbf{M}\, \mathbf{T}(\delta')$
for banded $\mathbf{M}$ is banded around the offset diagonal $\delta' - \delta$. Therefore:

- **Diagonal blocks** ($i = i'$): offset $0$ for every epoch → half-bandwidth $\approx m$.
- **Off-diagonal blocks**: offsets range over the epoch-by-epoch relative shifts, so the union
  is a band of half-width $b_{ii'} = \max_j |\delta_{ij} - \delta_{i'j}| + m$, set by the
  **relative RV excursion**: for an SB2, $b \approx (K_1+K_2)(1+e)/\delta v + m$ pixels.

Interleaving the component index gives a single banded matrix of dimension $N_c P$ and
half-bandwidth $p \approx N_c\,(\max_{ii'} b_{ii'} + 1)$. It is near-block-Toeplitz: it would
be exactly Toeplitz-structured for stationary weights, but masks and response break that
structure, which is why the implementation factorizes rather than using an FFT.

Design-target numbers ($P = 2\times10^5$, $J=50$, $N_c=2$, $\delta v = 1\ \mathrm{km\,s^{-1}}$,
$K_1+K_2 = 400\ \mathrm{km\,s^{-1}}$, $m=20$): $b \approx 420$, matrix dimension
$4\times10^5$, half-bandwidth $p \approx 850$.

### 4.2 Strategy A (primary): block-tridiagonal Cholesky

Partition the interleaved banded matrix into $K = N_cP/p$ dense blocks of size $p$; the matrix
is block-tridiagonal, and the Cholesky factor follows from the recursion

$$
\mathbf{L}_{kk}\mathbf{L}_{kk}^\top = \mathbf{D}_k - \mathbf{L}_{k,k-1}\mathbf{L}_{k,k-1}^\top,
\qquad
\mathbf{L}_{k+1,k} = \mathbf{E}_k \mathbf{L}_{kk}^{-\top},
$$

implemented as a `lax.scan` over $K$ steps of dense $p\times p$ operations (GPU-friendly), with
solves and $\log\det$ from the same sweep. Cost $\approx K \cdot \mathcal{O}(p^3) =
\mathcal{O}(N_c P\, p^2)$:

$$
4\times10^5 \times (850)^2 \approx 3\times10^{11}\ \text{flops}
\;\Rightarrow\; \mathcal{O}(0.1\ \mathrm{s})\ \text{per likelihood+gradient on a modern GPU (fp64)},
$$

and much less for typical SB2s (at $K_1+K_2 \lesssim 150\ \mathrm{km\,s^{-1}}$ or
$\delta v = 2\ \mathrm{km\,s^{-1}}$, cost drops by $\sim 10\times$). Assembling the band of
$\mathbf{A}^\top\mathbf{W}\mathbf{A}$ is $\mathcal{O}(J N_c^2 P (m + \text{taps}))$ via
shifted-product accumulation, which is subdominant. Two orthogonal levers batch this further:
`vmap` over independent wavelength chunks (echelle orders and natural mask gaps make chunks
exactly independent; otherwise chunking is an explicit, benchmarked approximation) and
`vmap` over systems (survey mode). NUTS with $\sim 10^3$–$10^4$ gradient evaluations then
falls within the minutes-on-one-GPU budget of the definition of done.

### 4.3 Strategy B: matrix-free CG + stochastic log-det

Apply $\tilde{\boldsymbol\Lambda}$ as operators (shift–conv–rebin chains,
$\mathcal{O}(J N_c P)$ per matvec), solve $\tilde{\boldsymbol\Lambda}\hat d = b$ by
preconditioned CG (circulant/Toeplitz preconditioner from the epoch-averaged stationary
operator, applied by FFT), and estimate $\log\det$ by stochastic Lanczos quadrature.
Assessment: the matvec is cheap, but hundreds of CG iterations times the number of probes
generally lose to Strategy A at the bandwidths of interest, and SLQ log-det estimates are
biased and stochastic. They are usable inside MAP optimization, but they violate the
exactness NUTS requires (a noisy log-density is not a valid target; pseudo-marginal MCMC needs
unbiased *likelihood*, not log-likelihood, estimates). Role: fallback for pathological
bandwidths (extreme $K_1+K_2$, very fine grids), and cross-validation of Strategy A.

### 4.4 Strategy C: profile likelihood with frozen log-det

Compute the profile term only (CG solve, no factorization), freezing
$\log\det\tilde{\boldsymbol\Lambda}$ at a reference $\theta_0$ (or dropping it). By §3.2 this
biases light ratios, LSF and hyperparameter inference. Role: fast MAP quick-look and
initialization only, never final inference.

**Decision:** implement A as the default engine; B behind the same interface for benchmarks;
C powers `fit_map(quick=True)`. The A-vs-B crossover is measured at M2/M3 on the design-target
benchmark and recorded in `docs/benchmarks.md`.

### 4.5 Direct band assembly and the closed-form gradient (D28)

Through M5 the band of $\tilde{\boldsymbol\Lambda}$ was assembled by comb probing: $2p+1$
applications of the matrix-free operator, paying for the union of all epochs' band offsets.
The shipped engine assembles the band directly from its analytic per-epoch structure
(the shifted-product accumulation of §4.2), and keeps probing as the reference implementation
and validation oracle.

**Per-epoch band structure.** The data term is
$\mathbf{A}^\top\mathbf{W}\mathbf{A} = \sum_j \mathbf{S}_j^\top \mathbf{W}'_j \mathbf{S}_j$
with $\mathbf{S}_j = \mathbf{R}\,\mathbf{K}\sum_i \ell_{ij}\mathbf{T}(\delta_{ij})$ and
$\mathbf{W}'_j = \mathrm{diag}(w_j r_j^2)$. Its $(i,i')$ component block for epoch $j$ is

$$
\ell_{ij}\,\ell_{i'j}\;\mathbf{T}(\delta_{ij})^\top\, \mathbf{G}_j\, \mathbf{T}(\delta_{i'j}),
\qquad
\mathbf{G}_j = \mathbf{K}^\top \big(\mathbf{R}^\top \mathbf{W}'_j \mathbf{R}\big) \mathbf{K},
$$

and $\mathbf{G}_j$ is a band of half-width $(s-1) + 2r$ (rebin row support $s$, kernel
radius $r$) *independent of the velocities*: the T-sandwich only translates the band to
the offset $\lfloor\delta_{ij}\rfloor - \lfloor\delta_{i'j}\rfloor$ and mixes adjacent
entries with the interpolation tent weights. Column $q$ of $\mathbf{T}(\delta)$
has exactly two entries, at rows $q + \lfloor\delta\rfloor$ (weight $1-\mathrm{frac}\,\delta$)
and $q + \lfloor\delta\rfloor + 1$ (weight $\mathrm{frac}\,\delta$), so each block band is
a four-term tent-weighted combination of row-translated copies of $\mathbf{G}_j$. The
computation is: (i) $\mathbf{R}^\top\mathbf{W}'\mathbf{R}$ by one `segment_sum` over
static *pair tables* precomputed from the rebin sparsity; (ii) the kernel sandwich as two
unrolled diagonal-shifted accumulations on the band image; (iii) translation + tent
mixing + accumulation into a global band tensor. A wavelength-dependent kernel (D37)
keeps stage (ii)'s structure with the scalar taps replaced by row-shifted profile
columns, $K[c+d,\,c] = P[c+d,\, r-d]$. Only *left* applications $\mathbf{K}^\top
\mathbf{M}$ broadcast on a row-major band image, since a right application's taps vary along
the columns, so the second application runs against the band-transpose of the first,
using the symmetry of $\mathbf{G}$:
$\mathbf{G} = \mathbf{K}^\top (\mathbf{K}^\top \mathbf{H})^\top$. That costs one band
transpose (a static column-slice shuffle) per epoch, at the same width and flop count. Cost
per epoch is $\mathcal{O}(P \cdot w)$ with $w = 2(s + 2r) + \mathcal{O}(1)$, against probing's
$\mathcal{O}(p)$ operator applications, an order of magnitude at survey bandwidths
($w \sim 50$, $2p+1 \sim 10^3$), with identical results up to floating-point summation
order (regression-tested; the `validate` path checks the assembled matrix against the
matrix-free operator directly).

Because $\mathbf{G}_j$ does not depend on the velocities, stage (ii) is a pre-pass over
epochs rather than part of the accumulation body. How many epochs it covers at once
(`epoch_chunk`) is purely a memory choice: `vmap` batches every intermediate of the
chain, so covering all of them costs ~9 GB at the design target, while any partition
costs exactly one extra $\mathbf{G}$ pass in the rematerialized backward regardless of
its granularity. The default keeps the whole pre-pass live below a threshold and
batches above it (D29).

The prior determinant needs no factorization at all: $\boldsymbol\Lambda_p$ is block
diagonal over components and pentadiagonal within each, so
$\log\det\boldsymbol\Lambda_p = 2\sum_i \log c_i$ from the three-term banded Cholesky
recursion $a_i = \alpha_i/c_{i-2}$, $b_i = (\beta_i - a_i b_{i-1})/c_{i-1}$,
$c_i = \sqrt{\gamma_i - a_i^2 - b_i^2}$ on the analytic diagonals of
$\tau\mathbf{D}_2^\top\mathbf{D}_2 + \eta\mathbf{I}$ (`assembly.prior_logdet`).

**Closed-form gradient.** With the band cheap, reverse mode through the Cholesky and
solve scans becomes the bottleneck (it stores or recomputes every scan step). The solve
stage instead defines a custom VJP from the standard identities: for
$\ell d = \log\det\tilde{\boldsymbol\Lambda}$, $Q = b^\top\tilde{\boldsymbol\Lambda}^{-1}b$,
$\hat d = \tilde{\boldsymbol\Lambda}^{-1} b$,

$$
\frac{\partial\,\ell d}{\partial \tilde{\boldsymbol\Lambda}} = \tilde{\boldsymbol\Sigma},
\qquad
\frac{\partial Q}{\partial \tilde{\boldsymbol\Lambda}} = -\hat d\,\hat d^\top,
\qquad
\bar b \mathrel{+}= \tilde{\boldsymbol\Sigma}\,\bar g_{\hat d},
\quad
\bar{\tilde{\boldsymbol\Lambda}} \mathrel{+}= -\big(u\,\hat d^\top\big),\;
u = \tilde{\boldsymbol\Sigma}\,\bar g_{\hat d}.
$$

Because every perturbation of $\tilde{\boldsymbol\Lambda}$ is block-tridiagonal, only the
banded part of $\tilde{\boldsymbol\Sigma}$ is ever contracted, which is what the block
Takahashi recursion delivers ($\Sigma_{k+1,k} = -\Sigma_{k+1,k+1}W_k$,
$\Sigma_{kk} = L_{kk}^{-\top}L_{kk}^{-1} + W_k^\top\Sigma_{k+1,k+1}W_k$,
$W_k = L_{k+1,k}L_{kk}^{-1}$). Cross-block cotangents carry a factor 2 (the stored lower
block represents both triangles); the within-block cotangent is left *unsymmetrized*,
which is exact end-to-end because `diag[k]` stores both triangles of a symmetric block
and the band packing reads each entry exactly once, so mirrored entries receive the two
halves of $u\hat d^\top + \hat d u^\top$ separately and sum to the same parameter
gradient. Each $\Sigma$ block is contracted at the step of the recursion that produces
it (`solver.selected_inverse_cotangent`), so the selected inverse is never materialized
($2K-1$ blocks, 3.1 GB at the design target, D29); `selected_inverse_blocks` remains as
the reference form and test oracle. Gradient contract: gradients flow through the
log-likelihood, the quadratic form, and $\hat d$. The Cholesky factor is not an output of the
custom-VJP stage, because a cotangent on it cannot be honoured by this
rule: propagating one is the reverse pass through the factorization that the
rule exists to avoid. `MarginalResult` rebuilds it outside the boundary, where plain
autodiff applies. Verified against plain autodiff to $10^{-13}$ relative and by finite
differences.

**Grid boundaries.** $\mathbf{G}$ is a matrix on the model grid, so entry $(x,y)$ exists
only for $y \in [0, n)$. $\mathbf{H} = \mathbf{R}^\top\mathbf{W}'\mathbf{R}$ is exactly
zero outside the grid (no rebin pairs there), but the $\mathbf{K}$ convolutions smear
in-grid mass *outward*, populating band-image entries at absolute columns that do not
correspond to grid pixels. The T-sandwich reads column $c + \lfloor\delta_j\rfloor + b$,
which leaves the grid whenever an epoch's shift places a component's support against an
edge; $T(\delta_j)$ has no row there, so the contribution is zero. Those columns
are therefore masked once per group. (Left unmasked this cost $6\times10^{-4}$ relative
in the assembled matrix and $2\times10^{-7}$ in $\log p$ for data spanning the grid;
with any margin between data and grid edge the weights vanish there and the effect is
absent, D29.)

**Second derivatives.** Hessians are taken reverse-over-reverse
(`jacrev(jacrev(...))`), which is exact here *because* the forward rule recomputes its
primal inline: the second reverse pass then walks plain graphs instead of re-entering
the custom boundary, where the un-propagated Cholesky cotangent would silently lose the
factor-mediated second-order terms (measured $8\times10^{-3}$ relative before that fix;
equal to plain autodiff at $10^{-15}$ after). Forward mode applied directly to the
marginal is not available, since JAX rejects `jvp` of a `custom_vjp` function, whereas
`jax.hessian` is forward-over-*reverse* and does run, since the inner `jacrev` resolves
the custom boundary first. It nonetheless produces an appreciably asymmetric Hessian
on this stack, and does so on the plain-autodiff path too, so the cause is the solver
scans rather than the custom rule; reverse-over-reverse matches central finite
differences of the gradient to 8 digits where forward-over-reverse does not.
`laplace_inverse_mass` uses reverse-over-reverse, fixing a defect present since M3.

Since D49 there are two custom boundaries rather than one: the band accumulate of §4.5
carries its own `custom_vjp` (`assembly._band_accumulate`), because reverse mode
otherwise rebuilds the entire band tensor to reproduce its own input. Everything above
still holds, since the forward rule recomputes its primal inline for the same reason and
`jax.hessian` still runs because the inner `jacrev` resolves both boundaries, but the
consequence for forward mode is now package-wide rather than confined to the solve
stage. `forecast._effective_parameters` was the last `jax.jvp` in albireo and is now a
`jax.grad` of the same scalar-to-scalar function, returning the identical `p_eff`.

### 4.5a The tridiagonal noise sandwich: link pair tables (D35)

The stage-1 sandwich above assumed $\mathbf{W}'$ diagonal. With the AR(1) chain
(§1.4a) the noise precision adds one symmetric off-diagonal term per link,
so per epoch

$$
\mathbf{H} \;=\; \mathbf{R}^\top \mathbf{W}' \mathbf{R}
\;=\; \underbrace{\mathbf{R}^\top \mathrm{diag}\!\big(w\, r^2 d^{\rm chain}/\alpha^2\big) \mathbf{R}}_{\text{equal-row pair tables}}
\;-\; \sum_{\rm links} \frac{c_k \sqrt{w_n w_p}\, r_n r_p}{\alpha^2}
\underbrace{\big(\mathbf{R}_n^\top \mathbf{R}_p + \mathbf{R}_p^\top \mathbf{R}_n\big)}_{\text{cross-row pair tables}},
$$

where $d^{\rm chain} = 1 + \sum_{\text{links at pixel}} a_k$ is the chain diagonal and
$(n, p)$ are the link's endpoint rows. The diagonal part reuses the D28 pair tables
with a re-weighted per-pixel vector; the cross-row part gets its own static tables
(`operators.rebin_link_pair_tables`). For every realized link (the union over
epochs of `ar_gap[e, n] == g`, since masks differ by epoch) and every ordered entry
pair of the two rows, the product $v_1 v_2$ lands on the upper band entry
$(\min(c_1, c_2), |c_1 - c_2|)$; the two orderings supply the two transposes, and
coincide on the diagonal, where the value is doubled instead. Per epoch the increment
is one `segment_sum` whose traced weights carry $\phi$, $\alpha$, and $r$, and the gap
test `ar_gap[e, link_row] == link_gap` selects each epoch's own links against the
shared union tables. $\mathbf{H}$ widens by the group's static `ar_step` (the largest
model-pixel offset between a realized link's row supports, the same quantity
`Problem.ar_bandwidth_extra` reserves), and every downstream stage is untouched: the
$\mathbf{K}$ convolutions, the T-sandwich, the band accumulation, and the custom-VJP
solve see only a slightly wider velocity-independent band image. This restores the full
D28/D29 cost profile for correlated problems, measured at HR-window scale: eval
$7.2\times$, gradient $19\times$ faster than probing, gradient peak memory
$23.7 \to 1.9$ GiB (benchmarks.md D35). Comb probing remains the reference path and
the `validate` oracle; band $=$ probe $=$ dense LAPACK is pinned in
`tests/test_ar1.py`.

---

## 5. Degeneracies and identifiability

The degeneracies below are properties of the problem rather than defects of a particular
method. albireo derives them, regularizes them explicitly, reports them in the posterior, and
requires an explicit choice from the user where only external information can break them.

### 5.1 The low-frequency degeneracy (the "undulations" theorem)

Take two components, constant light fractions absorbed into $u_i = \ell_i d_i$, no LSF/response,
unit weights, common grid, epochs $j=1..J$. In Fourier space (continuous transform, mode
$e^{\mathrm{i}kx}$), a shift is a phase: the model at epoch $j$ is
$\hat u_1(k)\,e^{-\mathrm{i}k\xi_{1j}} + \hat u_2(k)\,e^{-\mathrm{i}k\xi_{2j}}$. The per-mode
normal ("information") matrix is

$$
\mathbf{G}(k) = \begin{pmatrix} J & g(k) \\ g^*(k) & J\end{pmatrix},
\qquad g(k) = \sum_j e^{\mathrm{i}k\Delta_j},\quad \Delta_j = \xi_{1j} - \xi_{2j},
$$

with eigenvalues $J \pm |g(k)|$ for the sum and difference modes
$\hat u_1 \pm e^{\mathrm{i}\phi}\hat u_2$. The sum mode is always well constrained. For the
difference mode, expanding at small $k$:

$$
|g(k)| = J\left|\left\langle e^{\mathrm{i}k\Delta}\right\rangle\right|
\approx J\left(1 - \tfrac{k^2}{2}\operatorname{Var}_j(\Delta)\right)
\;\;\Rightarrow\;\;
\lambda_-(k) \approx \tfrac{J k^2}{2} \operatorname{Var}_j(\Delta).
$$

So the noise amplification of the separation is

$$
\sigma_-(k) \;\propto\; \frac{1}{k\,\sqrt{J\operatorname{Var}_j(\Delta)}} ,
$$

diverging as $k\to0$, with $k=0$ exactly singular. **Interpretation:** spectral features
narrower than the RMS *differential* orbital shift separate cleanly; features broader than it
cannot be attributed to either star from the data alone. This is the low-frequency
"undulation" artifact familiar from the output of KOREL (Hadrava 1995) and fd3: not a
numerical artifact of those codes, but the nullspace of the problem, excited by noise.
Consequences for the design:

1. The prior (§2) makes these directions proper and *sets their scale explicitly* ($\eta_i$).
2. The posterior covariance of §3.3 *reports* the inflation instead of hiding it.
3. $\operatorname{Var}_j(\Delta)$ is an **observing-strategy diagnostic**: albireo exposes
   $\lambda_-(k)$ forecasts from planned epochs (`sensitivity_forecast`, §5.5), which show
   which phase sampling improves separation quality. §5.5 also records
   where $\operatorname{Var}_j(\Delta)$ on its own ranks designs incorrectly, which is why
   the exact covariance is computed alongside it rather than in place of it.
4. Per-epoch response polynomials absorb the per-epoch near-constant modes;
   their order is kept low (default 2) so that they cannot absorb real broad features, and the
   response–low-$k$ covariance is visible in the posterior.

### 5.2 Light ratio ↔ line depth

With constant light fractions the composite is $1 + \sum_i \ell_i\,\mathbf{T}_{ij} d_i$: the
likelihood depends on the *products* $\ell_i d_i$ only. Therefore $(\ell_i, d_i)$ are exactly
degenerate along $\ell_i \to \ell_i/\alpha$, $d_i \to \alpha d_i$: the data cannot measure the
continuum light ratio, only each star's *contribution* to the composite lines. The degeneracy
is broken only by:

1. **Per-epoch light variation** (eclipses): $\ell_{ij}$ varying with known or parameterized
   geometry makes the products epoch-dependent while $d_i$ is shared, which is why per-epoch
   light fractions are supported directly.
2. **External photometric priors** on $\ell_i$ (from light-curve solutions or SED fits).
3. **Physicality floor**: $s_i \ge 0 \Rightarrow \ell_i d_i \ge -\ell_i$; a saturated line in
   the composite bounds $\ell_i$ from below. The bound is weak but real, and is available as
   an optional constraint.
4. **Fixing $\ell$** by assumption.

The API requires an explicit choice: `light_ratio=` must be given as `Fixed(values)`,
`Free(prior=...)` (requires 1–3 to be informative, and the docs say so), or
`PerEpoch(...)`.

### 5.3 Systemic velocity / zero-point

A change $\gamma \to \gamma + \epsilon$ composed with translating every $d_i$ by
$-\xi(\epsilon)$ leaves the likelihood invariant (up to grid edges), and the stationary priors
of §2 are translation-invariant too, so $\gamma$ is unidentified by disentangling itself
(a known property, inherited from the physics, of all disentangling methods). Default:
$\gamma \equiv 0$; recovered spectra live in the systemic frame, and $\gamma$ is measured
afterwards by template cross-correlation *of the disentangled spectra*, outside the sampler.
Rest-frame information from another source allows $\gamma$ to be freed with an informative
prior. $K_i$, $e$, $\omega$, $P_{\rm orb}$, $T_{\rm p}$ are unaffected.

### 5.4 Degeneracy ledger

| Degeneracy | Exact/approx | Broken by | albireo policy |
|---|---|---|---|
| low-$k$ mode exchange between components | exact at $k=0$, $\propto 1/k$ | phase coverage ($\operatorname{Var}\Delta$), priors | proper priors; covariance reported; forecast tool (§5.5) returns it as the leading eigenvector, at ~1× the prior for *every* design |
| $\ell_i$ vs. line depth | exact (constant $\ell$) | eclipses, photometry, saturation floor, assumption | explicit `light_ratio=` choice required |
| $\gamma$ vs. common shift | exact up to edges | external rest-frame info | $\gamma \equiv 0$ default, post-hoc measurement |
| per-epoch constants vs. response | approx | low poly order | order $\le 2$ default, covariance reported |
| telluric constant vs. common stellar constant | exact up to edges | ridge anchors ($\eta$) on both | measured in the telluric closed loop: the two offsets cancel in the sum to $\lesssim 10^{-3}$; report both |
| LSF width vs. intrinsic line widths | near-exact per instrument | cross-instrument spectrum sharing | absolute widths need a *reference instrument* anchor (tight prior); only relative widths are data-identified (M4, benchmarks.md) |
| nebular amplitude scale vs. nebular spectrum | exact ($a_j \to c\,a_j$, $d \to d/c$) | nothing; it is a convention | geometric mean pinned to 1 by centering `log_nebular_amp` (§1.3) |
| nebular velocity vs. common shift of *its* spectrum | exact up to edges (the §5.3 argument, one component at a time) | nothing on barycentric data | `nebular_v_kms` is a *placement* choice; it must agree with the prior's line windows and is not a measurement |

Two of these rows need expanding. **Telluric constant exchange:** with $\sum_i \ell_i = 1$
and a telluric component of light fraction 1, adding a constant $a$ to the telluric
deviation while subtracting $a$ from *every* stellar deviation changes no epoch's
prediction (constants are shift-invariant away from the grid edges), giving a second exact
$k = 0$ mode, split only by the $\eta$ ridges. **LSF ↔ intrinsic widths:** for one
instrument a wider Gaussian kernel composed with intrinsically narrower lines is
observationally near-identical (Gaussian widths add in quadrature), so a template-free
model cannot measure an absolute LSF width; empirically, ML-II with all widths free
inflates them by tens of percent while leaving the orbit untouched. Multiple
instruments *sharing the same spectra* identify the width differences; the absolute
scale must come from one instrument whose LSF is known.

### 5.5 Forecasting a design (D47)

§5.1 is a statement about the *design*, not about the data, and that generalizes exactly.
The posterior precision of the stacked spectra,

$$
\tilde{\boldsymbol\Lambda} \;=\; \boldsymbol\Lambda_p \;+\; \mathbf{A}^\top \mathbf{W} \mathbf{A},
$$

is built from the epoch times (through the velocities, hence the shifts $\delta_{ij}$), the
per-pixel weights, the masks, the LSF kernels, the light fractions, the response and the
prior. No flux appears in it. The fluxes enter the marginal likelihood only through
$b = \mathbf{A}^\top\mathbf{W}z$ and $z^\top\mathbf{W}z$, which move the posterior *mean*
and the evidence, not the covariance $\boldsymbol\Sigma = \tilde{\boldsymbol\Lambda}^{-1}$.
So $\boldsymbol\Sigma$ is computable for observations that have not happened, given only
their times, their instrument, and an assumed orbit. `albireo.forecast.sensitivity_forecast`
is that computation; `plan_epochs` builds the epochs to hand it.

Three summaries follow, each an exact quantity rather than an estimate.

**Pointwise band.** $\sqrt{\operatorname{diag}\boldsymbol\Sigma}$, by the same Takahashi
selected-inversion sweep as §3.3, quoted against
$\sqrt{\operatorname{diag}\boldsymbol\Lambda_p^{-1}}$. The second is reported because
a band that has relaxed back onto the prior is otherwise indistinguishable from one the data
constrained.

**Worst-determined modes.** The largest eigenpairs of $\boldsymbol\Sigma$, by subspace
iteration on the banded factor, which is the end of the spectrum a factorization does not
provide directly. The leading eigenvector is §5.1's degeneracy in its exact, non-asymptotic
form, and gives the shape of the error the disentangled spectra will carry. Two
coordinate sets are projected out first. The solver's pad block is the
identity, so its coordinates are eigenvectors with eigenvalue exactly 1. And the model grid
is made wider than the data (§1.1, the shift-plus-kernel margin), so its margin
pixels are prior-only and carry the largest eigenvalue on essentially every real problem;
left in, the worst-determined mode would report how much margin the grid was given.

**Constrained degrees of freedom.** $p_{\rm eff} = \operatorname{tr}[\boldsymbol\Sigma\,
\mathbf{A}^\top\mathbf{W}\mathbf{A}]$, the same quantity §3.2a profiles the jitter against.
It comes from one directional derivative rather than a stochastic trace estimator: scaling
every epoch's noise by $\alpha \to \alpha e^{t}$ sends $\mathbf{A}^\top\mathbf{W}\mathbf{A}
\to e^{-2t}\mathbf{A}^\top\mathbf{W}\mathbf{A}$ and leaves $\boldsymbol\Lambda_p$ alone, so

$$
\left.\frac{\mathrm{d}}{\mathrm{d}t}\log\det\!\left(\boldsymbol\Lambda_p + e^{-2t}\mathbf{A}^\top\mathbf{W}\mathbf{A}\right)\right|_{t=0}
\;=\; -2\,p_{\rm eff}.
$$

For ranking whole designs the scalar is the expected information gain. For a linear-Gaussian
model the prior-predictive expectation of $\mathrm{KL}(p(d|y)\,\|\,p(d))$ is exactly

$$
\mathbb{E}\,\mathrm{KL} \;=\; \tfrac12\left(\log\det\tilde{\boldsymbol\Lambda} - \log\det\boldsymbol\Lambda_p\right),
$$

the data-free half of §3.1's marginal likelihood, and the Bayesian D-optimality criterion
(Chaloner & Verdinelli 1995).

**The idealized diagnostic, and how it misleads.** §5.1's closed form costs no linear algebra
and is therefore what screens a hundred candidate cadences: with $J$ epochs and differential
log-shifts $\Delta_j$, separating the pair is noisier than measuring their sum by
$\sqrt{(J + |g(k)|)/(J - |g(k)|)}$ with $g(k) = \sum_j e^{\mathrm{i}k\Delta_j}$. But
$\operatorname{Var}_j(\Delta)$ is only the small-$k$ expansion of that, and maximizing it is
not the same as optimizing the design. A cadence aliased to the orbital period visits the two
*extreme* values of $\Delta$ repeatedly: it maximizes the variance, and it leaves
$|g(k)| = J|\cos(k\,\Delta_{\rm sep}/2)|$, which returns to $J$ at a whole comb of scales.
Measured in `examples/08_forecast.py` on a 13.7 d circular SB2 with eight epochs in hand and
twelve to plan:

| twelve planned nights | RMS $\Delta v$ | blind fraction | 2nd mode $\sigma$ | information gain |
|---|---|---|---|---|
| at $P/2$ (aliased) | 117.8 km/s | 58% | 0.518 | 243 nats |
| continuing the existing cadence | 115.7 km/s | 56% | 0.106 | 295 nats |
| spread over phase | **99.3 km/s** | **33%** | **0.071** | **375 nats** |

The aliased plan is best in the column the §5.1 expansion would maximize and worst in every
other one. The exact covariance gives the ranking; the closed form explains it and screens
candidates.

**What is not forecastable.** The covariance of the orbit is not. Its Fisher information runs
through $\partial(\text{model})/\partial v \propto \ell_i d_i'$, the derivative of the
component spectrum, so an error bar on $K_2$ requires the line depths, which are what has not
been measured. albireo therefore forecasts the spectra and not the orbit, rather than
forecasting the orbit against an assumed template.

---

## 6. SB1 + faint companion mode ($K_2$ scan)

This is the dormant-compact-object workflow. Given an SB1 solution (fixed $P_{\rm orb}, e, \omega,
T_{\rm p}, K_1$; primary spectrum either fixed from a single-component fit or left free), scan
a grid of trial $K_2$ (optionally × light fraction $\ell_2$): for each trial, the secondary
deviation spectrum $d_2$ is a linear component and marginalizes analytically, so the detection
statistic

$$
D(K_2) = 2\left[\log p(y \,|\, K_2) - \log p(y \,|\, \text{no companion})\right]
$$

costs one linear solve per grid point. This is the matched
filter *marginalized over the unknown companion spectrum*, more sensitive than a CCF
grid search with an assumed template, and it returns the recovered companion spectrum
$\hat d_2$ with its covariance at the peak. Because $d_2$'s prior scale enters, $D$ is calibrated
empirically by injection–recovery (same simulator as M1) rather than by an asymptotic $\chi^2$
claim; the null distribution is estimated, not assumed (§6.2).

### 6.1 Marginalizing $K_1$

Holding $K_1$ at the SB1 value conditions the whole scan on a number that has an error bar,
and the resulting bias does not stay in $K_1$: unremoved primary signal is *coherent* across
epochs, and the companion's free spectrum is the only component that can absorb it. The
symptom reported throughout the literature is spurious structure in the recovered secondary;
the less often reported symptom is that $D$ increases while this happens, so the artifact
reads as a stronger detection (measured in benchmarks.md D41: $K_1$ 10% high tripled $D$ and
took the companion's recovered line pattern from 0.96 correlation with truth to 0.49).

Integrating $K_1$ out removes the conditioning. With a Gaussian prior $K_1 \sim
\mathcal{N}(\mu_1, \sigma_1^2)$, both models are marginalized over the *same* prior,

$$
D(K_2) = 2\left[\log \sum_a w_a\, p(y \,|\, K_1^{(a)}, K_2)
              - \log \sum_a w_a\, p(y \,|\, K_1^{(a)}, \text{no companion})\right],
$$

with $\{K_1^{(a)}, w_a\}$ a Gauss–Hermite rule, $K_1^{(a)} = \mu_1 + \sqrt2\,\sigma_1 x_a$ and
$w_a = \tilde w_a/\sqrt\pi$, exact for polynomials of degree $\le 2n-1$, and the same
quadrature family the LSF's $h_3$ already uses (§1.3). Using the same prior in numerator and
denominator is what keeps $D$ a ratio of two marginal likelihoods rather than a comparison of
differently-conditioned ones. The cost is a factor $n$ in solves, which is why the scan is
evaluated as one batched `lax.map` over the $(K_1, K_2)$ grid rather than a Python loop.

The recovered spectra at the peak stay *conditional* on the best node
($\hat K_1 = \arg\max_a \log p(y | K_1^{(a)}, \hat K_2)$), a profile rather than a marginal:
there is no closed form for the $K_1$-marginalized spectrum, and averaging the nodes'
spectra would blur the lines instead of widening their error bars.

### 6.2 Calibrating $D$

$D$ has no closed-form null distribution. Wilks' theorem does not apply, because the companion
is a boundary hypothesis whose nuisance parameter is a prior-regularized *function*, so the
effective degrees of freedom depend on $(\tau_2, \eta_2)$, on the epoch sampling, and on the
masks. albireo therefore measures the distribution instead of assuming one, by parametric
bootstrap through the observed data's own operators: with $z = y - r(R\mathbf 1)$ the only
place the fluxes enter,

$$
z'_j = r_j \odot \big(R_j B_j \textstyle\sum_i \ell_{ij} T(\delta_{ij}) d_i\big) + n_j,
\qquad n_j \sim \mathcal N(0, W_j^{-1}),
$$

reuses every operator, weight and mask and costs one forward apply per trial
(`forward.with_data`, `simulate.resimulate`). Scanning $N$ such draws with no companion gives
the null distribution of $\max_{K_2} D$, the maximum being the statistic a search
reports. Repeating with a companion injected at a ladder of $\ell_2$ gives
completeness, and the crossing at 95% is the quoted limit.

Two properties are enforced. The threshold is the smallest $D$ whose
estimated false-alarm probability $(1 + \#\{{\rm null} \ge D\})/(N+1)$ is within budget, so
the realized null exceedance never exceeds the nominal rate; an interpolating sample
quantile does not have that property and errs in the anti-conservative direction. And no FAP
below $1/(N+1)$ is reported: with finitely many trials, an absence of comparable null values
is evidence for a small rate rather than for a zero rate.

The procedure does not check the model. The null trials are drawn at the same $K_1$,
orbit and light fractions the scan assumes, so the threshold is self-consistent with those
assumptions and insensitive to their being wrong, which is why §6.1 accompanies it.

Implementation notes (M4, `albireo.scan.k2_scan`): the no-companion model is the
single-component fit with $\ell_1 = 1$ and the primary's prior; the companion's light
fraction $\ell_2$ must be chosen explicitly (§5.2: the observable is $\ell_2 d_2$).
Because both log-marginals carry their $\tfrac12\log\det$ Occam terms, the extra
marginalized component lowers the likelihood unless coherent signal compensates for it: on a
companion-free dataset $D(K_2)$ is negative at every trial (measured in the closed-loop
test), which is the expected baseline for the empirical calibration. One caveat is
inherited from §5.1: at small $\ell_2$ the companion's smooth envelope (continuum level,
mean line blanketing) is prior-dominated, since an error $\Delta$ in the bright primary's
envelope maps to $-(\ell_1/\ell_2)\Delta$ in the companion, an amplification of ~10 at
$\ell_2 = 0.1$, so the recovered $\hat d_2$ carries its line *pattern* rather than a
reliable absolute depth scale, unless eclipses or photometry pin the envelope.

---

## 7. Joint inference over θ (M3)

### 7.1 The marginal posterior and its static computation graph

With the spectra marginalized (§3), inference over the nonlinear parameters is ordinary
low-dimensional Bayes: $p(\theta \mid y) \propto p(y \mid \theta)\, p(\theta)$, where every
$\theta$-evaluation rebuilds only the shifts $\delta_{ij}(\theta)$ (Kepler velocities → §1.2
frame composition) and reuses every static operator. Under `jax.jit` the solver bandwidth
must be independent of $\theta$, so the probing/factorization pipeline is built once for a
declared velocity budget $v_{\rm rel}^{\max}$:

$$
b_{\rm bound} = \left\lceil \xi(v_{\rm rel}^{\max})/\Delta x \right\rceil + 1 + 2r_B + s_R,
$$

with $r_B$ the LSF kernel radius and $s_R$ the rebin row support (§4.1). Probing with any
$b \ge b_{\rm true}$ is exact, so the bound costs time rather than accuracy. The failure mode
is an underestimate: comb probing then aliases band entries and the likelihood is silently
wrong. The sampler is therefore protected by a bandwidth guard: the numpyro model
computes the realized $\max_{j,i,i'} |\delta_{ij} - \delta_{i'j}|$ and adds a $-\infty$
factor whenever it exceeds the budget implied by $b_{\rm bound}$. A prior wider than the
budget slows mixing near the boundary but cannot corrupt the posterior.

### 7.2 Parameterization

Sampled sites: $P$, $T_{\rm conj}$, $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$, and
$K_i$ (γ ≡ 0 by §5.3 / D14). The $\sqrt{e}$-pair is smooth through $e = 0$, where $\omega$
and $T_{\rm peri}$ are undefined, and a uniform prior on the unit disk maps to
$e \sim \mathcal{U}(0,1)$, $\omega \sim \mathcal{U}(-\pi,\pi)$; the disk constraint enters
as a $-\infty$ factor with $e$ clipped at $e_{\max} = 0.95$ (the Kepler solver's verified
range) before the solve, so out-of-support proposals stay finite and rejectable. The single
non-smooth point is the origin $\sqrt{e}\cos\omega = \sqrt{e}\sin\omega = 0$ (an
`arctan2` branch point of measure zero); circular-orbit initializations should sit slightly
off it. $T_{\rm conj}$ (the §1.2 convention $\nu + \omega = \pi/2$) replaces $T_{\rm peri}$,
which degenerates with $\omega$ as $e \to 0$.

One smoothness caveat: with linear (2-tap) shift interpolation (D3), $A(\theta)$ is
piecewise-linear in each shift, so $\log p(y\mid\theta)$ is piecewise-$C^1$ in the
velocities with derivative kinks where a shift crosses an integer pixel. On an oversampled
model grid the kink amplitudes are set by sub-pixel spectral curvature and are far below
the posterior scale; NUTS (Hoffman & Gelman 2014) treats them as it treats any
leapfrog-scale roughness. The 4-tap
cubic kernel (D3, flagged) is the smoothing upgrade if a dataset ever exposes them.

### 7.3 Hyperparameters: ML-II by default

The prior scales $(\tau_i, \eta_i)$ control exactly the part of spectrum space the data
cannot constrain (§5.1: sub-LSF modes, low-$k$ anchoring), so they must be chosen
explicitly rather than defaulted. Because the spectra are already integrated out,
maximizing the marginal posterior jointly over $(\theta, \log\tau, \log\eta)$ is ML-II
or empirical Bayes (up to weak hyperpriors that keep the optimization proper). The MAP
pipeline does this with L-BFGS in numpyro's unconstrained space; NUTS then runs with the
hyperparameters conditioned at their ML-II values (default), or samples them, trading the
usual mild underestimation of hyperparameter uncertainty for extra
dimensions. Both are supported, and the choice is recorded in the fit metadata. The marginal
likelihood already contains the $\tfrac12\log\det\boldsymbol\Lambda_p$ Occam term, so ML-II
is well-posed: $\tau \to \infty$ is penalized by data misfit, $\tau \to 0$ by the
determinant.

### 7.4 Posterior spectra

$p(d \mid y) = \int p(d \mid y, \theta)\, p(\theta \mid y)\, d\theta$, a mixture of the
§3.3 conditional Gaussians over posterior $\theta$ draws. Draws propagate orbital
uncertainty into the spectra; the pointwise bands from a single $\hat\theta$ (§3.3) are the
*conditional* uncertainty only. Both are exposed and documented as different objects.

### 7.5 Realism extensions in θ (M4)

**Hierarchical SB3.** A triple is two nested Keplerians: inner pair (A, B) and the
outer orbit of their center of mass against the tertiary C,

$$
v_A = v^{\rm in}(t;\, \omega_{\rm in}, K_1) + v^{\rm out}(t;\, \omega_{\rm out}, K_{AB}),
\qquad
v_B = v^{\rm in}(t;\, \omega_{\rm in} + \pi, K_2) + v^{\rm out}(t;\, \omega_{\rm out}, K_{AB}),
$$
$$
v_C = v^{\rm out}(t;\, \omega_{\rm out} + \pi, K_C),
$$

with the light-time effect across the outer orbit neglected (v2 seam; relevant only for
$P_{\rm out}$ measured to seconds). Sites: the five outer analogues
(`period_out`, `t_conj_out`, `secosw_out`, `sesinw_out`, `k_out` $= (K_{AB}, K_C)$),
same $\sqrt{e}$-disk parameterization and constraint, with the outer conjunction
convention applied to the inner pair's center of mass. The tertiary is one more linear
component; nothing downstream changes.

**Per-epoch light fractions.** $\ell_{ij}$ enters $\mathbf{A}(\theta)$ *linearly*
(§1.3), so a `light` site ($(n_{\rm stellar},)$ constant or
$(J, n_{\rm stellar})$ per-epoch, rows on the simplex via Dirichlet priors) swaps into
the static graph exactly like the shifts, and the §5.2 eclipse breaker becomes an
*inferred* quantity. The likelihood is smooth (indeed quadratic-in-$\ell$ per epoch
pre-marginalization), so MAP/NUTS handle it natively.

**LSF widths.** A Gaussian kernel's *values* at fixed integer offsets are smooth in
$\sigma$, so `lsf_sigma` (one width per un-anchored instrument; one per LSF anchor for
an instrument built with `lsf_anchors_angstrom`, D37, where the traced per-anchor bank is
re-interpolated through the same static tables the build used) is traced while the
kernel *radius* stays fixed at build time by the construction-time widths, which thereby
become strict per-entry upper bounds: a realized $\sigma$ above them would be silently
truncated by the fixed radius, so the model rejects it with a $-\infty$ factor (the same
guard-not-silent-corruption pattern as the bandwidth budget, §7.1). Identifiability is
§5.4's caveat sharpened by §1.3: absolute widths require one anchored reference instrument,
and fitted per-anchor profiles are diagnostics rather than measurements. The asymmetry
site ``lsf_h3`` (D38) follows the same pattern (per-anchor Gauss–Hermite $h_3$ for
anchored instruments only, the static radius untouched, clipped and guarded at
$|h_3| \le 0.2$), with the still-sharper identifiability caveat of §1.3.

**Per-epoch response (D33, post-M5).** The multiplicative response enters the
likelihood in three places: the target, $z_j = y_j - r_j \odot (\mathbf{R}\mathbf{1})$;
the sandwich weights, since $\mathbf{A}_j = \mathrm{diag}(r_j)\mathbf{R}_j\mathbf{B}_j\cdots$
folds $r_j^2$ into $\mathbf{A}^\top W \mathbf{A}$; and the right-hand side through
$r_j z_j$. That is why D7 kept its coefficients as build-time constants through M4: a
`response` swap is not a pure operator swap like the shifts. It is nonetheless cheap,
because $\mathbf{R}\mathbf{1}$ (the rebinned unit continuum, stored per group) is
*response-independent*:

$$
z^{\rm new}_j \;=\; z^{\rm old}_j + \left(r^{\rm old}_j - r^{\rm new}_j\right) \odot \mathbf{R}\mathbf{1}
$$

rebuilds the target exactly without carrying the raw fluxes (and re-masking keeps
zero-weight pixels at exactly zero, so the D30 ``0·nan`` trap cannot resurface), while
the $\sum \log w$ term is untouched, since the noise lives on the data rather than on
response-divided data. `response` is a θ site: $(n_{\rm coef},)$ shared or
$(J, n_{\rm coef})$ per-epoch, $r = 1 + \sum_m c_m T_m(x)$ on each group's native
abscissa. Identifiability is §5's response row, sharpened by measurement: the
epoch-to-epoch *differences* of the coefficients are well constrained (closed loop:
recovered to $\sim 10^{-3}$ against injected $3\times 10^{-2}$), while the epoch-shared
mode trades against the components' broad features and lands at its zero-centered prior
rather than at truth. The order should be kept low and the priors tight; the common mode is
a normalization convention rather than a measurement.

### 7.6 Free per-epoch velocities (the RV table)

In this diagnostic mode the Keplerian is replaced by a `velocity` site of shape
$(n_{\rm stellar}, n_{\rm epochs})$, so each epoch's velocity is its own parameter. The
resulting table is the product most binary-star work reports, the point of contact for users
of cross-correlation codes, and, because a Keplerian is a strong constraint, the model check
for §7.

**Identifiability: one arbitrary zero point per component.** With no orbit tying the
components together, component $i$'s deviation spectrum $d_i$ is free, so translating it
absorbs a constant added to that component's shifts and the likelihood cannot tell:

$$
T(\delta_{ij} + \Delta_i)\, d_i \;=\; T(\delta_{ij})\, \big[T(\Delta_i)\, d_i\big].
$$

This is $\gamma$ (§5.3, D14) once per star rather than once in total. The equality is
exact for whole-pixel $\Delta_i$; for fractional ones the linear-interpolation shift
operator blurs slightly as well as translating, so the likelihood is left with a weak
preference that is a property of the *operator*, not of the data. Measured: a one-pixel
common shift of one component costs $4\times10^{-9}$ of the log-likelihood in relative
terms, a 0.1-pixel one costs 7.3 nats. An uncentered table would therefore have its
absolute level set by interpolation error.

albireo removes the zero points in pixel space, which is where the removal is exact:
with $\xi = \operatorname{artanh}(v/c)$ (§1.1, chosen so shifts compose and invert
exactly) a constant *pixel* offset is relativistic velocity addition, not ordinary
addition, so

$$
\tilde\delta_{ij} = \frac{\xi(v_{ij})}{\Delta x}
  - \frac{1}{N}\sum_{j'} \frac{\xi(v_{ij'})}{\Delta x},
\qquad
v^{\rm rel}_{ij} = c \tanh\!\big(\tilde\delta_{ij}\,\Delta x\big)
$$

is exactly invariant under $v_{ij} \mapsto v_{ij} \oplus u_i$, while centering the
velocities would be right only to $O(v^2/c^2)$.

**What survives, and what does not.** Each component's velocity variation, and hence its
semi-amplitude, and every epoch-to-epoch difference are identified, as is the slope of
$v_2$ against $v_1$, which is $-K_2/K_1$: the Wilson mass ratio is a slope and is therefore
unaffected by both zero points. The systemic velocity and either star's absolute velocity are not,
and must be measured afterwards from the disentangled spectra exactly as §5.3 prescribes.

**Uncertainties need the same projection.** Each zero point is an exactly flat likelihood
direction, so its posterior width is the prior width and every epoch's marginal variance
inherits it: on the D42 fixture the raw Laplace diagonal returns $120/\sqrt{10} = 37.95$
km/s on every entry, against a per-epoch error of 0.059 km/s. Projecting each
component's mean out of the covariance leaves exactly $n_{\rm stellar}$ null directions
and gives the identified errors. Posterior samples of the `velocity_rel` deterministic
need no projection at all.

**The mode needs a warm start.** From a cold start with every epoch at one velocity the
components are indistinguishable, and the optimizer lands in a wrong basin, at a
potential far worse than the warm-started one, so the failure is detectable rather
than silent. The intended workflow is to fit a Keplerian first, then free the velocities
and difference the two tables (both centered, differenced in pixel space so the zero
points cancel exactly). Structured residuals, phase-correlated or with one epoch far out,
are the signature of a wrong period, an unseen third body, or line-profile variability.

---

## 8. What the tests assert (traceability)

| Claim | Test |
|---|---|
| $\mathbf{T}, \mathbf{R}$ are exact adjoint pairs | inner-product identity vs. `jax.linear_transpose`, float64, rtol $10^{-12}$ |
| shift gradient correct | `jax.grad` vs. central finite differences at non-integer shifts |
| shift is linear, interior-exact on constants, sum-preserving | direct assertions |
| relativistic shifts compose/invert exactly | $\xi(-v) = -\xi(v)$; round-trip shift |
| rebin conserves flux | $\sum \Delta\lambda_{\rm out} f_{\rm out} = \sum \Delta\lambda_{\rm in} f_{\rm in}$ on covered ranges |
| marginal likelihood correct | brute-force dense Gaussian marginalization on tiny problems, rtol $10^{-10}$ |
| conditional spectra + covariance correct | closed-loop recovery on simulated data; whitened residual $z$-scores $\sim \mathcal{N}(0,1)$ |
| posterior calibration | SBC / coverage on injections (M3; `scripts/m3_coverage.py`, results in benchmarks.md) |
| degeneracy analysis (§5.1) | measured posterior variance of difference modes vs. $\lambda_-(k)^{-1}$ prediction |
| θ-path equals fixed-parameter path (§7.1) | jitted `MarginalOrbitModel.log_likelihood` vs. `build_problem` + `marginal_loglikelihood`, rtol $10^{-12}$ |
| θ-gradients correct through probing + Cholesky | `jax.grad` vs. central finite differences per site, rtol $10^{-4}$ |
| bandwidth guard (§7.1) | out-of-budget orbit ⇒ non-finite model log-density; in-budget ⇒ finite |
| velocity conventions match the simulator | `orbit_velocities(θ)` ≡ `OrbitParams.component_velocities` |
| ML-II sanity (§7.3) | MAP over (θ, log τ, log η) recovers K's and sane hyperscales |
| **M3 gate**: $K_1, K_2$ to <1% with valid posteriors | closed-loop NUTS: posterior means within 1%, truth in central 95%, zero divergences |
| light/LSF θ-paths equal fresh builds (§7.5) | `with_light_fractions` / `with_lsf` vs. `build_problem` at matched kernel radius, rtol $10^{-12}$; FD gradients through both sites |
| response θ-path equals fresh builds (§7.5, D33) | `with_response` vs. `build_problem(response_coeffs=...)`: `r` bitwise, marginal rtol $10^{-12}$; FD gradients; replace-not-compound; masked pixels inert |
| **D33 gate**: per-epoch response closed loop | difference-mode coefficients to $5\times10^{-3}$ against injected $3\times10^{-2}$; K's <1%; common mode prior-pinned (asserted at prior scale, not at truth) |
| AR(1) chain closed forms exact (§1.4a, D34) | marginal vs. dense brute force under the correlated covariance — chain correlation built independently and LAPACK-inverted — with masked gaps ($\rho = \phi^{g}$) and jitter composed, rtol $10^{-10}$; FD gradients, including at $\phi = 0$ (the `pow` nan-grad trap); $\phi = 0$ ≡ diagonal model |
| **D34 gate**: correlated closed loop | injected $\phi = 0.45$ and ivar scale error $\alpha = 1.5$ recovered ($\pm 0.05$, $\pm 5\%$) jointly with K's <1%; chain whitener removes the lag-1 autocorrelation the diagonal whitener exposes ($\approx\phi$ vs $\approx 0$), while both report unit *scale* |
| correlated band assembly (§4.5a, D35) | band $=$ probe on the gapped+jittered fixture (loglike rtol $10^{-12}$, `validate` operator check); $\partial/\partial\phi$ band $=$ probe to $10^{-9}$; `epoch_chunk` batching invariant (the AR weight tuple pads and slices together); the D34 dense gold test runs the band path |
| SB3 velocity law (§7.5) | `orbit_velocities` with outer sites ≡ hand-composed nested Keplerians, atol $10^{-12}$ |
| LSF bound + outer-disk guards | width above build bound / outer $e > e_{\max}$ ⇒ non-finite model log-density |
| telluric constant exchange (§5.4) | closed loop: the two $k=0$ offsets cancel in the light-weighted sum to $<5\times10^{-3}$ |
| **M4 gate**: closed loop per realism feature | telluric joint MAP; SB3 MAP (inner and outer $K$'s <2%); per-epoch light inferred (ℓ rms <0.01, components individually recovered); LSF width vs. reference instrument <3%; $K_2$ scan (peak at truth, negative $D$ under null) |
| TODCOR identities (§10.2) | free-amplitude surface $=$ Zucker & Mazeh's symmetric $R^2$, fixed-ratio/free-scale surface $=$ their $R^2(s_1, s_2; \alpha)$, held-fraction surface $=$ the pinned least squares, each to $10^{-10}$ against a NumPy transcription on a uniform-weight grid epoch |
| **D56 gate**: velocities and calibrated errors (§10.1, §10.4) | simulated SB2 through LSF, rebin, cosmics, gaps and barycentric motion: every velocity within $5\sigma$, rms $<0.1$ km/s, pull rms in $[0.6, 1.6]$; both frames agree; mixed instruments; one and three components; free and global light fractions recover the injected ones; profiled $=$ ivar errors $\times\sqrt{\chi^2_\nu}$ |
| zero points and flags (§10.4, §10.5) | a template offset composes relativistically to $10^{-9}$; an unknown offset is reported as differential; twin stars at one velocity are flagged blended and at 120 km/s are not; a minimum at the range edge is flagged; a continuum offset is absorbed by the nuisance and biases the light without it |
| orbit from the table (§10.6) | injected $P, e, \omega, K_1, K_2, \gamma$ recovered from a noisy table within $3\sigma$; `predict` $\equiv$ `orbit_velocities(to_theta())` $+ \gamma$; per-component $\gamma$ recovers offset zero points while a forced shared one corrupts $K$; the periodogram finds $P$ to 1% |

Sections 1–2 and the operator rows are implemented and tested in M0; §3–4 landed in M2;
§7.1–7.4 landed in M3 (with §5 diagnostics); §6 and §7.5 landed in M4, except the §7.5
response swap, which landed post-M5 (D33), as did the §1.4a correlated-noise chain
(D34) and its §4.5a band assembly (D35). §9 landed with D52–D55, §10 with D56–D57.

## 9. Stellar labels from disentangled components (D52–D55)

Everything above returns component *spectra*. This section is the forward model that turns one
of those into four labels ($T_{\mathrm{eff}}$, $\log g$, [M/H], $v\sin i$) against a
published synthetic grid, so the component can be rendered as a template for epoch radial
velocities elsewhere. It is implemented in `albireo.library` (grids and their interpolation)
and `albireo.match` (the fit).

The scope is narrow, and §9.6 states the accuracy it requires. This mode is not an atmospheric
analysis; `albireo.handoff` remains the route to GSSP, iSpec, Korg.jl and PySME for anything
needing abundances or bespoke synthesis.

### 9.1 What a disentangled component actually is

Write $s_i$ for star $i$'s own-continuum normalized spectrum and $t_i = s_i - 1$ for its
deviation. The observed composite, in the system's continuum, is

$$F(\lambda) = 1 + \sum_i w_i(\lambda)\, t_i(\lambda), \qquad \sum_i w_i(\lambda) = 1,$$

with $w_i$ the true, wavelength-dependent continuum light fraction. But §1.3's model fitted

$$F(\lambda) = 1 + \sum_i \ell^0_i \, d_i(\lambda)$$

with $\ell^0$ assumed and constant. Equating the two, what the disentangler recovered is

$$\hat d_i \;=\; \frac{w_i(\lambda)}{\ell^0_i}\; t_i \;+\; n_i ,$$

where $n_i$ collects the null-space contamination of §5.1: the $\eta$-anchored $k=0$ constant
and the low-$k$ exchange modes, which are additive and live in the continuum.

Three consequences follow, and they determine the design:

1. Only the ratio $w_i/\ell^0_i$ is identified, never $w_i$ alone, which is §5.2's exact
   $(\ell, d) \to (\ell/\alpha, \alpha d)$ degeneracy restated. An error in the assumed
   $\ell^0$ rescales every line depth, which is indistinguishable from a change in
   $T_{\mathrm{eff}}$ unless the fit has another parameter that can absorb it.
2. $n_i$ is additive, so the nuisance absorbing it must be additive too. A multiplicative
   continuum polynomial is identically zero wherever $t_i$ is zero, which is exactly where
   $n_i$ lives, so it cannot represent it.
3. The light fractions' *wavelength dependence* carries the light-ratio information. A
   wavelength-independent dilution factor discards it.

### 9.2 The forward model

Per star, on the model grid of §1.1, with labels $\phi_i = (T_i, g_i, Z, \varsigma_i, v_i)$
and one shared dilution scalar per companion:

$$
\begin{aligned}
(N_i, \ln C_i) &= \mathcal{I}_i(T_i, g_i, Z) && \text{grid interpolation (§9.3)}\\
t_i &= N_i - 1 \\
t_i' &= K_{\mathrm{rot}}(\varsigma_i) \star K_{\mathrm{macro}} \star t_i && \text{intrinsic broadening}\\
t_i'' &= B\, t_i' && \text{instrument profile, matched mode only}\\
t_i''' &= T(\xi(v_i)/\Delta)\, t_i'' && \text{Doppler shift (§1.1)}\\
w_i &= \frac{A_i\, e^{\ln C_i}}{\sum_j A_j\, e^{\ln C_j}}, \quad A_1 = 1,\; A_i = r_i^2 && \text{dilution}\\
m_i &= \frac{w_i}{\ell^0_i}\, t_i''' \;+\; \sum_{m=0}^{M} a_{im} T_m(\tilde x) && \text{additive nuisance}
\end{aligned}
$$

The dilution line carries the light-ratio information. Written as a softmax over
$\ln C_i + \ln A_i$, it enforces $\sum_i w_i(\lambda) = 1$ at every pixel by construction,
with no constraint site and no penalty term, and its wavelength dependence comes from the
grids' own continua rather than from a fitted polynomial. This is GSSP's `gssp_binary`
parameterization (Tkachenko 2015); treating the dilution as wavelength-independent instead
was measured there to move a secondary's $T_{\mathrm{eff}}$ by 275 K. The single-component fallback replaces the
softmax with one free scalar per component, which is `gssp_single`, and is strictly weaker.

$a_{i0}$ is the unconstrained $k=0$ zero point of §5.1. It is fitted and reported rather than
absorbed: a large fitted value indicates that the disentangling zero point was biased.

**Rotational broadening.** $K_{\mathrm{rot}}$ is the Gray (2005) limb-darkened profile,
*integrated over each pixel* rather than point-sampled. The profile has a square-root edge, so
point sampling puts a kink of unbounded slope wherever the support boundary crosses a pixel;
the pixel integral is $C^1$ in $v\sin i$ because $g(\pm 1) = 0$, which is what L-BFGS and NUTS
need. It is not $C^2$ at half-integer $v\sin i/\Delta v$, where the edge lands exactly on a
pixel boundary and that tap picks up a $|\delta|^{3/2}$ term. Both properties are checked in
`tests/test_operators.py`.

**Why the comparison is at native resolution.** $\hat d$ is not quite the
intrinsic spectrum: §1.3 applies the LSF *inside* the epoch model, so $\hat d$ is a regularized
partial deconvolution, faithful where the data had signal and shrunk toward zero where the
smoothness prior dominated. That argues for convolving *both sides once* with the declared $B$
and comparing in the space the data constrained, and `compare="matched"` did that, as the
default, until AI Phe was fitted (D55).

The argument is right about the deconvolution and wrong about what it costs. Convolving the
residuals correlates them over the kernel width, while the likelihood of §9.1 stays diagonal.
For a unit-sum kernel $k$ the resulting over-count is

$$\frac{\chi^2_{\text{matched}}}{\chi^2_{\text{native}}} \;\approx\; \Big(\sum_p k_p^2\Big)^{-1},$$

the usual effective-sample-size factor. On AI Phe (HARPS, $R = 115{,}000$, $\Delta v = 0.8$
km s$^{-1}$, so $\sigma_{\mathrm{LSF}} = 1.38$ px) that predicts 4.91 and the fit measured
4.26, which accounts for the difference between the two modes. A mis-specified likelihood does
not only inflate $\chi^2$: $v\sin i$ absorbs it, and both components went to the floor of their
prior (0.14 and 0.46 km s$^{-1}$), where native returned 2.2 km s$^{-1}$ for
both. The default is therefore `"native"`; `"matched"` is retained, and becomes appropriate
only once a residual-covariance model can carry the correlation it creates.

The closed-loop test cannot decide between the two modes: its injected rows never
pass through an LSF or a disentangling, so they are intrinsic spectra and both modes recover
them. Only real data with a deconvolution behind it separates the two. The LSF width
itself is not fitted here in either mode, since §1.3's identifiability argument continues to
apply.

### 9.3 Interpolation, and why not an emulator (yet)

$\mathcal{I}$ interpolates the *flux*, never the model atmosphere, and the continuum in the
log. On the 250 K / 0.5 dex spacing of the BOSZ grid, Mészáros & Allende Prieto (2013)
measured 0.19% scatter for atmosphere interpolation against 0.051% for linear flux
interpolation and 0.031% for a cubic, while a Payne-style network reaches about 0.1%. On a
well-sampled FGK grid a differentiable cubic is therefore more accurate than the emulator,
with no training cost and nothing to host.

There are two interpolants, chosen by geometry: separable Catmull-Rom on a complete axis
product, and barycentric interpolation over a Delaunay triangulation when physics has cut the
corners off the grid, as it has for the OB libraries. Both reproduce a node exactly, which is
what allows the warm-start node scan and the continuous fit to be compared on the same footing.
The cubic's phantom end nodes are extrapolated linearly rather than clamped: clamping destroys
linear reproduction in the edge cells, and on a grid with a few values per axis it performs
worse than plain multilinear over a third of the range (measured).

Whether a learned emulator is worth building is therefore an empirical question about a
particular grid, and `crossval_library` provides the measurement, as in D49 and D50.

### 9.4 Degeneracies, extending the §5.4 ledger

| Degeneracy | Exact/approx | Broken by | albireo policy |
|---|---|---|---|
| $T_{\mathrm{eff}}$ vs. $\log g$ | approx, $\rho \approx 0.98$ with both free | an external $\log g$ | eclipsing binaries give $\log g$ to 0.01 dex from $M$ and $R$ — declare `logg=Fixed(...)`. Non-eclipsing: run free, fixed, and fixed-with-dilution, and report the spread. The correlation is *reported*, not hidden |
| assumed $\ell^0$ vs. line depth vs. $T_{\mathrm{eff}}$ | exact for constant $\ell$ (§5.2) | joint fit of both components with $\sum w = 1$, wavelength-dependent | `RadiusRatio` is the default; `FixedDilution` is the diagnostic that shows what it was worth |
| $v\sin i$ vs. instrumental width | near-exact (§5.4, restated) | nothing, within one instrument | LSF fixed at its declared value; a fitted $v\sin i$ below it is reported as not a measurement |
| $v\sin i$ vs. macroturbulence | near-exact (widths add in quadrature) | line-shape detail at high $R$ | macroturbulence is *fixed*, not fitted; with the default 0 a fitted $v\sin i$ means "all broadening beyond the instrument", which is what a template needs |
| [M/H] vs. microturbulence | approx | an external $\xi$ | $\xi$ is a property of the grid, echoed in `assumptions`, never silently defaulted — fixing it at 2 km/s when the truth is 10 costs ~0.5 dex in [M/H] (ZETA-PAYNE) |
| $k=0$ zero point vs. line depth | exact (§5.1) | the additive nuisance | $a_{i0}$ fitted and reported; removing it exists only as the control that shows it matters |
| $T_{\mathrm{eff}}$ vs. He abundance (OB) | approx | a He axis in the grid | not offered — the public OB grids fix He. Stated as a known systematic, since in 3900–4600 Å the He lines *are* the temperature diagnostic |

### 9.5 Uncertainties, and why the formal one is not enough

The Laplace covariance is the curvature of the potential at the MAP, projected to the
constrained parameterization by the delta method. It measures the sharpness of the optimum,
which is not the quantity needed here: the residuals of a disentangled component are correlated
rather than white, because disentangling artifacts are structured across wavelength by
construction (§5.1). Codes that have checked report the same gap: Gebruers et al. (2022) give
70 K formal against 425 K realistic for B stars at S/N 150, and Czekala et al. (2015) find
5–10× once correlated residuals are modelled.

albireo therefore quotes two numbers side by side and labels them. The second comes from
refitting the labels once per joint posterior draw of the component spectra (`refit_draws`),
which propagates the correlations the formal error cannot see, including the low-$k$ exchange
modes that trade flux between components. This internalizes the loop
`albireo.handoff.export_draws` documents, without a round trip through an external code.

A per-component jitter site is enabled by default and bounded. Its maximum-likelihood point is
the RMS residual, so on an unusually good fit it runs to zero scale and takes the gradient norm
with it. The bound states that the quoted per-pixel errors are wrong by at most a factor of
five.

### 9.6 The accuracy this has to reach

A label from this mode is a template coordinate. The relevant question is not whether it is the
star's temperature but whether a better label would change the epoch radial velocities, and the
literature sets a loose tolerance:

$$T_{\mathrm{eff}} \lesssim 2\text{–}3\%,\quad \log g \lesssim 0.15\ \mathrm{dex},\quad
[\mathrm{M/H}] \lesssim 0.15\ \mathrm{dex},\quad v\sin i \lesssim 10\%.$$

Posbic et al. (2012) measure a template 400–1000 K too warm to bias solar-type RVs by
$\approx 0.2$ km/s (about FWHM/60) with no loss of precision; Tkachenko et al. (2022) find
LSD profile shapes insensitive to $\pm5\%$ in $T_{\mathrm{eff}}$ and $\pm0.3$–0.4 dex in
$\log g$ and [M/H]. Published methods clear this tolerance, as does the
closed loop in `tests/test_match.py`.

What a wrong template does cost is a per-component constant velocity zero point: the CfA
SB2 orbits' $\gamma_1 - \gamma_2 = 0.35 \pm 0.55$ km/s and, in the extreme case, Gaia DR3's
$-20$ km/s for hot stars before mitigation (Blomme et al. 2023). That is the same
one-zero-point-per-component quantity §5.3 and §7.6 already track. This mode therefore fixes
zero points, flux ratios and gross template mismatch; it does not improve RV precision.

## 10. Epoch velocities by N-dimensional correlation (D56–D57)

Everything above infers the orbit from the composite spectra and never measures a per-epoch
velocity. This section describes the complementary measurement: given the component spectra
(from a library, a label match, or the disentangling itself), each component's velocity is
measured in every epoch separately, by the two-dimensional correlation of Zucker & Mazeh
(1994) generalized to $N$ components and to weighted, masked, multi-instrument data. It is
implemented in `albireo.todcor`, and `albireo.rvorbit` fits a Keplerian to what it produces.

### 10.1 The estimator

Fix the templates $t_i$ (deviation spectra on the model grid, normalized to their own
continua) and write the epoch model of §1.4 with the spectra *given*:

$$
y_j = 1 + \sum_{i=1}^{N} a_{ij}\, \mathbf{R}_j \mathbf{B}_j \mathbf{T}(\delta_{ij})\, t_i
      + \mathbf{P}_j c_j + n_j ,
\qquad n_j \sim \mathcal{N}(0, \mathbf{W}_j^{-1}),
$$

with $\mathbf{P}_j$ an additive low-order (Chebyshev) nuisance basis on the native pixels,
additive for the reason §9.1 gives: what it absorbs lives in the continuum, where a
multiplicative term is identically zero. Write $A_i(\delta) = \mathbf{R}\mathbf{B}\mathbf{T}(\delta) t_i$
for a template shifted, convolved and projected onto the epoch's pixels, $z = y - 1$, and for a
set of shifts $\mathbf{s} = (\delta_1, \dots, \delta_N)$

$$
b_i(\delta_i) = A_i^{\!\top} \mathbf{W} z, \qquad
G_{ik}(\delta_i, \delta_k) = A_i^{\!\top} \mathbf{W} A_k, \qquad
r_P(\mathbf{s}) = \mathbf{P}^{\!\top}\mathbf{W} z - \sum_i a_i\, \mathbf{P}^{\!\top}\mathbf{W} A_i .
$$

The chi-square with the nuisance profiled out is, for **held** amplitudes $a = \ell$,

$$
\chi^2(\mathbf{s}) = z^{\!\top}\mathbf{W}z - 2\,\ell^{\!\top} b + \ell^{\!\top} G\, \ell
  - r_P^{\!\top} (\mathbf{P}^{\!\top}\mathbf{W}\mathbf{P})^{-1} r_P ,
$$

for a **free overall scale** on held ratios ($a = \alpha\ell$) the Schur complement of the
same system in $\alpha$, and for **free** amplitudes the block solve
$\chi^2 = z^{\!\top}\mathbf{W}z - \beta^{\!\top} \mathbf{K}^{-1} \beta$ with
$\mathbf{K} = \begin{pmatrix} G & (\mathbf{P}^{\!\top}\mathbf{W}A)^{\!\top} \\ \mathbf{P}^{\!\top}\mathbf{W}A & \mathbf{P}^{\!\top}\mathbf{W}\mathbf{P}\end{pmatrix}$
and $\beta = (b;\ \mathbf{P}^{\!\top}\mathbf{W}z)$. The velocities are the minimizer over $\mathbf{s}$,
which is searched on the integer shifts of the template grid (one segment-sum builds every
shifted, projected column, and one matrix product per template pair gives every $G_{ik}$)
and then refined below a pixel (§10.3). The search costs $O(N^2 S^2 n_{\rm pix})$ for $S$ shifts
per component, so the global pass strides the grid by the narrowest LSF sigma, which cannot
step over a correlation peak, and the fine pass runs at full resolution around its minimum.

### 10.2 Relation to TODCOR

On a uniform grid with uniform weights, the data on the model grid ($\mathbf{R} = \mathbf{B} = \mathbf{I}$)
and no nuisance, define the classic one-dimensional correlations
$c_i = b_i / (\lVert z\rVert\,\lVert A_i\rVert)$ and $c_{12} = G_{12} / (\lVert A_1\rVert\,\lVert A_2\rVert)$.
Then the free-amplitude chi-square satisfies

$$
1 - \frac{\chi^2(s_1, s_2)}{\lVert z\rVert^2}
  = \frac{c_1^2 - 2 c_1 c_2 c_{12} + c_2^2}{1 - c_{12}^2} = R^2(s_1, s_2),
$$

Zucker & Mazeh's symmetric expression with the light ratio maximized out, and the
fixed-ratio, free-scale chi-square satisfies

$$
1 - \frac{\chi^2(s_1, s_2)}{\lVert z\rVert^2}
  = \left[\frac{c_1 + \alpha' c_2}{\sqrt{1 + 2\alpha' c_{12} + \alpha'^2}}\right]^2,
\qquad \alpha' = \frac{\ell_2}{\ell_1}\,\frac{\lVert A_2\rVert}{\lVert A_1\rVert},
$$

their original $R(s_1, s_2; \alpha)$. Both identities hold to $10^{-10}$ in the suite. The
three- and four-component extensions (Zucker et al. 1995; Torres et al. 2007) are the same
block solve with a larger $G$; nothing in the formulation is specific to
$N = 2$. What the least-squares form adds is that masks, chip gaps, cosmic rays, per-pixel
weights, mixed instruments and mixed samplings enter through $\mathbf{W}$ and $\mathbf{R}_j$
and change no formula, following the same D4 convention as the rest of the package, and that
the templates can be intrinsic spectra with each instrument's LSF applied in quadrature above
the resolution they already carry.

Two differences from the published practice are deliberate. Continuum-normalized data pin
the composite's scale, so the default holds the light fractions exactly rather than leaving
the overall amplitude free; `scale="free"` restores the classic scale-invariant form.
And multi-order spectra are pooled through their declared weights into one chi-square rather
than combined through the per-order maximum-likelihood product of Zucker (2003), which profiles
a separate noise level per order; an `ivar` declared per order (as the readers do), or
splitting the dataset by order, covers that case.

### 10.3 Fractional shifts are exact, and the pixel-locking bound

The shift operator is linear in the template and, for a fractional shift $n + f$,
$\mathbf{T}(n + f)\, t = (1 - f)\,\mathbf{T}(n)\, t + f\,\mathbf{T}(n + 1)\, t$ (§1.1, D3). Every inner
product above is therefore **bilinear in the fractional parts**, and $\chi^2(\mathbf{s})$ with
held amplitudes is an exact quadratic in $f \in [0, 1]^N$ inside each unit cell of the
integer grid, reconstructed from $3^N$ exact evaluations and minimized in closed form, with
the amplitude solve alternating when they are profiled. The sub-pixel minimum and the
curvature at it are thus *computed* for the same operator the forward model uses, not read
off a parabola through three grid points.

That operator has a known artifact: at $f = \tfrac12$ the two-tap interpolation is a
$[\tfrac12, \tfrac12]$ smoothing, which lowers a Gaussian line of width $\sigma_{\rm px}$ by
$\approx 1/(8\sigma_{\rm px}^2)$ of its depth and so adds a one-pixel-periodic ripple to the
chi-square whose pull on the minimum is of order $2\pi / (64\sigma_{\rm px}^2) \approx
0.1/\sigma_{\rm px}^2$ pixels. That is an estimate rather than a bound: measured on noiseless
data simulated at
four times the template resolution, the largest error is 0.03 px at one pixel per sigma, 0.015
at two, 0.006 at five and 0.002 at ten (benchmarks.md, D56). Three pixels per LSF sigma puts it
below a hundredth of a pixel; `Fit.templates()` upsamples the components to that, and `todcor`
warns below two.

### 10.4 Uncertainties and detection

Zucker (2003) showed the correlation peak is a maximum-likelihood estimate with the noise
level profiled out, $\log L = -\tfrac{N}{2}\log[1 - C^2(\hat s)]$, and derived its error from
the Hessian. In the chi-square form the same profiling gives

$$
\operatorname{Cov}(\hat{\mathbf{s}}) = \frac{\chi^2_{\min}}{n_{\rm pix} - p}\; 2\,\mathbf{H}^{-1},
\qquad \mathbf{H} = \left.\frac{\partial^2 \chi^2}{\partial \mathbf{s}\,\partial \mathbf{s}^{\!\top}}\right|_{\hat{\mathbf{s}}},
$$

the curvature error rescaled by the reduced chi-square (what `errors="profiled"` reports),
against $2\mathbf{H}^{-1}$ with the declared weights taken at face value (`errors="ivar"`). The
off-diagonal of $\mathbf{H}^{-1}$ is the blending diagnostic: near conjunction the two shifts
are measured along a ridge, their correlation approaches one, and the table flags the epoch.
Each component's detection statistic is $\Delta\chi^2_i = \chi^2_{\min}(\text{without } i) -
\chi^2_{\min}$ with the remaining amplitudes refitted, which is small for a companion the epoch
does not constrain, the case a batch run has to detect.

### 10.5 Frames and zero points

Velocities are reported barycentric whatever the data's frame: for topocentric data the shift
searched is $\xi(v) - \xi(v_{\rm bary})$ in log-wavelength (§1.2), and the composition is
exact because log-shifts add. A template's rest frame enters the same way: a template at
velocity $v_0$ relative to the star's true rest frame gives $v = v_{\rm meas} \oplus v_0$,
relativistic velocity addition (§7.6). A synthetic template has $v_0 = 0$ and yields absolute
velocities. A disentangled component has an unknown $v_0$, since its zero point is the
unidentified constant of §5.3, so the velocities measured against it are differential, one
arbitrary constant per component; `VelocityTable.absolute` records which is which, and the
label match of §9 is what pins the constant.

### 10.6 The orbit from the table

`albireo.rvorbit` fits the Keplerian of §7.2 (period, conjunction time,
$(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$, one $K_i$ per component, and a systemic velocity)
to the table by weighted nonlinear least squares with the Jacobian from JAX, using the same
Kepler solver and angle conventions as `orbit_velocities`, so the two routes compare element
for element. The systemic velocity is one number when every component is absolute and one
per component otherwise, since a shared $\gamma$ across two different zero points would be
absorbed into the semi-amplitudes. Errors are the curvature errors rescaled by the reduced
chi-square, because a template fit's per-epoch errors never include template mismatch. The
minimum masses follow from $M_{1,2}\sin^3 i = 1.0361\times10^{-7}\,(1 - e^2)^{3/2}\,(K_1 + K_2)^2 K_{2,1}\,P$
in solar masses with $K$ in km/s and $P$ in days.

## References

- Blomme, R. et al. 2023, A&A, 674, A7
- Chaloner, K. & Verdinelli, I. 1995, Statistical Science, 10, 273
- Czekala, I. et al. 2015, ApJ, 812, 128
- Czekala, I. et al. 2017, ApJ, 840, 49
- Gebruers, S. et al. 2022, A&A, 665, A36
- Gray, D. F. 2005, The Observation and Analysis of Stellar Photospheres, 3rd ed. (Cambridge
  University Press)
- Hadrava, P. 1995, A&AS, 114, 393
- Hoffman, M. D. & Gelman, A. 2014, Journal of Machine Learning Research, 15, 1593
- Mészáros, Sz. & Allende Prieto, C. 2013, MNRAS, 430, 3285
- Posbic, H. et al. 2012, A&A, 544, A154
- Simon, K. P. & Sturm, E. 1994, A&A, 281, 286
- Takahashi, K., Fagan, J. & Chin, M.-S. 1973, in Proc. 8th PICA Conference, 63
- Tamajo, E., Pavlovski, K. & Southworth, J. 2011, A&A, 526, A76
- Tkachenko, A. 2015, A&A, 581, A129
- Tkachenko, A. et al. 2022, A&A, 666, A180
- Torres, G., Latham, D. W. & Stefanik, R. P. 2007, ApJ, 662, 602
- van der Marel, R. P. & Franx, M. 1993, ApJ, 407, 525
- Zucker, S. 2003, MNRAS, 342, 1291
- Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
- Zucker, S., Torres, G. & Mazeh, T. 1995, ApJ, 452, 863

The science overview, `docs/science.md`, carries the complete bibliography with ADS links.
