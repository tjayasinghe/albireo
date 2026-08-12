# Mathematical foundations

This document defines the albireo forward model, derives the analytically-marginalized
likelihood that the whole package is built around, analyzes its computational structure, and
works out the degeneracy theory that drives the API design. Everything implemented in the code
must trace back to an equation here; everything here must eventually be covered by a test.

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
whose logarithm is $\operatorname{artanh}\beta$. We make it the default for two reasons:
(i) at $|v| \sim 600\ \mathrm{km\,s^{-1}}$ the classical form is wrong by
$\sim 0.6\ \mathrm{km\,s^{-1}}$, which is far above our RV error budget; (ii)
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
$M_j = 2\pi (t_j - T_{\rm p})/P_{\rm orb}$. We support both $T_{\rm p}$ (periastron) and
$T_0$ (conjunction) parameterizations, and sample in
$(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$ to behave well at low eccentricity. For SB3, a
hierarchical outer orbit adds its contribution to the inner pair. The Kepler solver is a
JAX-differentiable Newton iteration with a fixed iteration count (gradients via implicit
differentiation).

**Free-velocity mode (diagnostic).** $v_{ij}$ free parameters with optional priors — used to
detect non-Keplerian residuals and validate the orbit model.

**Frames and tellurics.** Let $u_j$ be the barycentric correction velocity at epoch $j$
(defined so that $v^{\rm bary} = v^{\rm topo} \oplus u_j$). All shifts are composed in log-shift
space, where composition is exact addition:

| | stellar component $i$ | telluric component |
|---|---|---|
| data in topocentric frame | $\xi(v_{ij}) - \xi(u_j)$ | $0$ |
| data barycentric-corrected | $\xi(v_{ij})$ | $+\xi(u_j)$ |

i.e. the telluric spectrum is just another linear component whose "velocity law" is the
(known) topocentric one — no new machinery.

**Telluric linearity approximation.** Physically telluric transmission is multiplicative:
$(1 + d_\star)\,(1 + d_{\rm tell})$. We model it additively,
$1 + d_\star + d_{\rm tell}$, which is first-order accurate with error
$d_\star d_{\rm tell}$ — of order $10^{-2}$ only where a deep stellar line overlaps a deep
telluric line, and standard practice is to mask deep tellurics anyway. This keeps the model
linear in all component spectra. Exact multiplicative tellurics (alternating solves) are a v2
candidate; the approximation is documented and testable in the simulator.

### 1.3 LSF, light fractions, response

**LSF.** Per-instrument Gaussian in velocity space with width $\sigma_v$: on the uniform
log-$\lambda$ grid this is a stationary discrete convolution $\mathbf{B}_j$ (Toeplitz, banded,
kernel truncated at $\pm 4\sigma$), consistent with a constant-resolving-power spectrograph.
Because $\mathbf{B}_j$ is stationary on the same uniform grid, it commutes with $\mathbf{T}$
(up to edges); we apply it after shifting, matching the physical picture (instrument acts in
the observed frame). Tabulated / wavelength-dependent LSFs become banded non-stationary
matrices in v2 with no structural change to anything below.

**Light fractions.** $\ell_{ij} \ge 0$ with $\sum_i \ell_{ij} = 1$ over the stellar components
(continuum-normalized data), telluric fixed at $\ell = 1$. Constant per component by default;
per-epoch (eclipse) mode is first-class (§5.2 explains why).

**Response.** Per-epoch multiplicative Chebyshev polynomial on the native grid,
$r_j(\lambda) = \sum_{m=0}^{M} c_{jm}\,\phi_m(\lambda)$ (default $M=2$), absorbing
continuum-normalization errors. Coefficients live in $\theta$.

### 1.4 The stacked linear model

Putting it together, the model for epoch $j$ is

$$
m_j(\theta, d) \;=\; \operatorname{diag}\!\big(r_j\big)\,
\mathbf{R}_j \Big[\mathbf{1} + \sum_{i=1}^{N_c} \ell_{ij}\, \mathbf{B}_j\, \mathbf{T}(\delta_{ij})\, d_i \Big],
$$

and the crucial property is that **conditional on $\theta$, $m_j$ is affine in the stacked
deviation vector** $d = (d_1^\top,\dots,d_{N_c}^\top)^\top \in \mathbb{R}^{N_c P}$:

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

---

## 2. Priors on the component spectra

We place independent Gaussian priors on the deviation spectra,
$d_i \sim \mathcal{N}(0, \boldsymbol\Lambda_i^{-1})$, specified by **banded precision** matrices —
never dense kernels (that is the design error that killed PSOAP's scalability):

$$
\boldsymbol\Lambda_i \;=\; \tau_i\, \mathbf{D}_2^\top \mathbf{D}_2 \;+\; \eta_i\, \mathbf{I},
$$

where $\mathbf{D}_2$ is the second-difference operator. Interpretation:

- $\tau_i$ penalizes curvature — the continuum limit is an integrated Wiener process, i.e. a
  smoothness prior with correlation length set by $(\tau_i/\eta_i)^{1/4}$ pixels. Matérn-class
  priors via their SPDE/state-space banded precision are a drop-in v1.x extension.
- $\mathbf{D}_2^\top\mathbf{D}_2$ has an affine nullspace (constant + slope per component).
  These are *exactly* the directions of the low-frequency separation degeneracy (§5.1), so they
  must be proper: the weak ridge $\eta_i$ anchors them to the continuum ($d_i = 0$) with a
  large but finite variance. This is a deliberate, documented choice: it sets the scale of the
  unavoidable low-frequency uncertainty instead of hiding it.

Hyperparameters $\tau_i, \eta_i$ are part of $\theta$: they can be fixed, optimized (ML-II —
free, since the marginal likelihood is what we compute anyway), or sampled. The prior mean is
$0$ in deviation space; emission-line components need no special treatment.

$\log\det\boldsymbol\Lambda_i$ is cheap (banded Cholesky, bandwidth 2).

---

## 3. Marginalizing the spectra analytically

This is the core of the package. Conditional on $\theta$, the model is linear-Gaussian in $d$,
so $d$ integrates out in closed form; the sampler only ever sees the low-dimensional
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
$\tfrac12(\log\det\boldsymbol\Lambda - \log\det\tilde{\boldsymbol\Lambda})$ — the "Laplace
correction", which is *exact* here because the model is truly linear-Gaussian in $d$. This
identifies the three computational strategies of §4 as the *same estimator* with different
$\log\det$ treatments: exact (A), stochastic (B), or frozen/dropped (C). Dropping the
$\log\det$'s $\theta$-dependence is *not* innocuous: it biases exactly the parameters that
change the information geometry (light ratios, LSF widths, prior hyperparameters), which is why
strategy C is quick-look only.

### 3.2a What profiling the jitter estimates

The noise-inflation factor of §1.4 is worth a separate line, because the marginal gives it a
better denominator than the obvious hand calculation. Take one shared $\alpha$ for clarity, so
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
fitted smoothness prior has far fewer data-determined modes than pixels — on HR 6819,
$p_{\text{eff}} \approx 2900$ against $N_c P = 19{,}876$, roughly the number of *resolution
elements* rather than pixels, so the correction was 0.4%. Inverting the two estimators gives
$p_{\text{eff}}$ for free: $p_{\text{eff}} = N\,[1 - (\text{residual sd}/\hat\alpha)^2]$, which
is a cheap and otherwise awkward diagnostic of how much of the spectrum the data actually
constrain.

What this does not buy: $\mathbf{W}$ stays diagonal, so a jitter can only rescale a residual,
never decorrelate one. Against systematics — continuum errors, LSF mismatch, intrinsically
variable line profiles — $\hat\alpha$ widens the intervals around an unchanged, still-biased
point estimate, and it silences the residual-scale diagnostic while doing so.

### 3.3 Recovering the spectra and their uncertainties

Conditional on $\theta$, the posterior of the spectra is Gaussian:

$$
d \,|\, y, \theta \;\sim\; \mathcal{N}\!\left(\hat d(\theta),\; \tilde{\boldsymbol\Lambda}(\theta)^{-1}\right).
$$

The full posterior of the spectra marginalizes over the $\theta$ posterior — in practice, for
each NUTS draw $\theta^{(t)}$ we draw one spectrum realization

$$
d^{(t)} = \hat d(\theta^{(t)}) + \mathbf{L}^{-\top} z, \qquad z \sim \mathcal{N}(0,\mathbf{I}),
$$

giving samples from $p(d\,|\,y)$ that include *both* the linear-Gaussian pixel noise *and* the
orbit/calibration uncertainty. This is the headline product: disentangled spectra with honest
uncertainties, which no incumbent code provides. Pointwise error bars come from the sample
variance and/or the diagonal of $\tilde{\boldsymbol\Lambda}^{-1}$ (computable without dense
inversion via Takahashi selected-inversion recursions on the banded/block factor). The
posterior covariance between pixels — especially the inflated low-frequency modes of §5.1 — is
available from the same factor and is part of the standard output, not an afterthought.

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
$m \sim 10$–$30$ px). The key observation: $\mathbf{T}(\delta)^\top \mathbf{M}\, \mathbf{T}(\delta')$
for banded $\mathbf{M}$ is banded *around the offset diagonal* $\delta' - \delta$. Therefore:

- **Diagonal blocks** ($i = i'$): offset $0$ for every epoch → half-bandwidth $\approx m$. Narrow.
- **Off-diagonal blocks**: offsets range over the epoch-by-epoch relative shifts, so the union
  is a band of half-width $b_{ii'} = \max_j |\delta_{ij} - \delta_{i'j}| + m$, set by the
  **relative RV excursion** — for an SB2, $b \approx (K_1+K_2)(1+e)/\delta v + m$ pixels.

Interleaving the component index gives a single banded matrix of dimension $N_c P$ and
half-bandwidth $p \approx N_c\,(\max_{ii'} b_{ii'} + 1)$. It is *near*-block-Toeplitz (it would
be exactly Toeplitz-structured for stationary weights; masks and response breaks exactness,
which is why we factorize rather than FFT).

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
shifted-product accumulation — subdominant. Two orthogonal levers batch this further:
`vmap` over independent wavelength chunks (echelle orders / natural mask gaps make chunks
*exactly* independent; otherwise chunking is an explicit, benchmarked approximation) and
`vmap` over systems (survey mode). NUTS with $\sim 10^3$–$10^4$ gradient evaluations then
lands in the "minutes on one GPU" budget of the definition of done.

### 4.3 Strategy B: matrix-free CG + stochastic log-det

Apply $\tilde{\boldsymbol\Lambda}$ as operators (shift–conv–rebin chains,
$\mathcal{O}(J N_c P)$ per matvec), solve $\tilde{\boldsymbol\Lambda}\hat d = b$ by
preconditioned CG (circulant/Toeplitz preconditioner from the epoch-averaged stationary
operator, applied by FFT), and estimate $\log\det$ by stochastic Lanczos quadrature.
Assessment: the matvec is cheap but hundreds of CG iterations × probes generally lose to
Strategy A at our bandwidths, and **SLQ log-det estimates are biased and stochastic** — usable
inside MAP optimization, but they violate the exactness NUTS needs (a noisy log-density is not
a valid target; pseudo-marginal MCMC needs unbiased *likelihood*, not log-likelihood,
estimates). Role: fallback for pathological bandwidths (extreme $K_1+K_2$, very fine grids),
and cross-validation of Strategy A.

### 4.4 Strategy C: profile likelihood with frozen log-det

Compute the profile term only (CG solve, no factorization), freezing
$\log\det\tilde{\boldsymbol\Lambda}$ at a reference $\theta_0$ (or dropping it). By §3.2 this
biases light ratios, LSF and hyperparameter inference. Role: fast MAP quick-look and
initialization only, never final inference.

**Decision:** implement A as the default engine; B behind the same interface for benchmarks;
C powers `fit_map(quick=True)`. The A-vs-B crossover is measured at M2/M3 on the design-target
benchmark and recorded in `docs/benchmarks.md`.

### 4.5 Direct band assembly and the closed-form gradient (D28)

Through M5 the band of $\tilde{\boldsymbol\Lambda}$ was assembled by comb *probing*:
$2p+1$ applications of the matrix-free operator, paying for the union of all epochs'
band offsets. The shipped engine now assembles the band directly from its analytic
per-epoch structure (the "shifted-product accumulation" anticipated in §4.2), keeping
probing as the reference implementation and validation oracle.

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
entries with the interpolation tent weights. Concretely, column $q$ of $\mathbf{T}(\delta)$
has exactly two entries — rows $q + \lfloor\delta\rfloor$ (weight $1-\mathrm{frac}\,\delta$)
and $q + \lfloor\delta\rfloor + 1$ (weight $\mathrm{frac}\,\delta$) — so each block band is
a four-term tent-weighted combination of row-translated copies of $\mathbf{G}_j$. The
computation is: (i) $\mathbf{R}^\top\mathbf{W}'\mathbf{R}$ by one `segment_sum` over
static *pair tables* precomputed from the rebin sparsity; (ii) the kernel sandwich as two
unrolled diagonal-shifted accumulations on the band image; (iii) translation + tent
mixing + accumulation into a global band tensor. Cost per epoch is
$\mathcal{O}(P \cdot w)$ with $w = 2(s + 2r) + \mathcal{O}(1)$, versus probing's
$\mathcal{O}(p)$ operator applications — an order of magnitude at survey bandwidths
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
*banded part* of $\tilde{\boldsymbol\Sigma}$ is ever contracted — exactly what the block
Takahashi recursion delivers ($\Sigma_{k+1,k} = -\Sigma_{k+1,k+1}W_k$,
$\Sigma_{kk} = L_{kk}^{-\top}L_{kk}^{-1} + W_k^\top\Sigma_{k+1,k+1}W_k$,
$W_k = L_{k+1,k}L_{kk}^{-1}$). Cross-block cotangents carry a factor 2 (the stored lower
block represents both triangles); the within-block cotangent is left *unsymmetrized*,
which is exact end-to-end because `diag[k]` stores both triangles of a symmetric block
and the band packing reads each entry exactly once, so mirrored entries receive the two
halves of $u\hat d^\top + \hat d u^\top$ separately and sum to the same parameter
gradient. Each $\Sigma$ block is contracted at the step of the recursion that produces
it (`solver.selected_inverse_cotangent`), so the selected inverse is never materialized
— $2K-1$ blocks, 3.1 GB at the design target (D29); `selected_inverse_blocks` remains as
the reference form and test oracle. Gradient contract: gradients flow through the
log-likelihood, the quadratic form, and $\hat d$. The Cholesky factor is deliberately
*not* an output of the custom-VJP stage — a cotangent on it cannot be honoured by this
rule, since propagating one is precisely the reverse pass through the factorization the
rule exists to avoid — so `MarginalResult` rebuilds it outside the boundary where plain
autodiff applies. Verified against plain autodiff to $10^{-13}$ relative and by finite
differences.

**Grid boundaries.** $\mathbf{G}$ is a matrix on the model grid, so entry $(x,y)$ exists
only for $y \in [0, n)$. $\mathbf{H} = \mathbf{R}^\top\mathbf{W}'\mathbf{R}$ is exactly
zero outside the grid (no rebin pairs there), but the $\mathbf{K}$ convolutions smear
in-grid mass *outward*, populating band-image entries at absolute columns that do not
correspond to grid pixels. The T-sandwich reads column $c + \lfloor\delta_j\rfloor + b$,
which leaves the grid whenever an epoch's shift places a component's support against an
edge — and $T(\delta_j)$ has no row there, so the contribution is zero. Those columns
are therefore masked once per group. (Left unmasked this cost $6\times10^{-4}$ relative
in the assembled matrix and $2\times10^{-7}$ in $\log p$ for data spanning the grid;
with any margin between data and grid edge the weights vanish there and it is invisible
— D29.)

**Second derivatives.** Hessians are taken reverse-over-reverse
(`jacrev(jacrev(...))`), which is exact here *because* the forward rule recomputes its
primal inline: the second reverse pass then walks plain graphs instead of re-entering
the custom boundary, where the un-propagated Cholesky cotangent would silently lose the
factor-mediated second-order terms (measured $8\times10^{-3}$ relative before that fix;
equal to plain autodiff at $10^{-15}$ after). Forward mode applied *directly* to the
marginal is impossible — JAX rejects `jvp` of a `custom_vjp` function — but
`jax.hessian` is forward-over-*reverse* and does run, since the inner `jacrev` resolves
the custom boundary first. It nonetheless produces an appreciably *asymmetric* Hessian
on this stack, and does so on the plain-autodiff path too, so the cause is the solver
scans rather than the custom rule; reverse-over-reverse matches central finite
differences of the gradient to 8 digits where forward-over-reverse does not.
`laplace_inverse_mass` uses reverse-over-reverse, fixing a defect present since M3.

---

## 5. Degeneracies and identifiability

These are properties of the *problem*, not bugs in a method. albireo's stance: derive them,
regularize them explicitly, report them in the posterior, and force conscious user choices
where only external information can break them.

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
cannot be attributed to either star from the data alone. This is precisely the low-frequency
"undulation" artifact familiar from KOREL/fd3 output — not a numerical quirk but the nullspace
of the problem, excited by noise. Consequences baked into the design:

1. The prior (§2) makes these directions proper and *sets their scale explicitly* ($\eta_i$).
2. The posterior covariance of §3.3 *reports* the inflation instead of hiding it.
3. $\operatorname{Var}_j(\Delta)$ is an **observing-strategy diagnostic**: albireo exposes
   $\lambda_-(k)$ forecasts from planned epochs (`sensitivity_forecast`), telling observers
   which phase sampling actually pays for separation quality.
4. Per-epoch response polynomials deliberately absorb the per-epoch near-constant modes;
   their order is kept low (default 2) so they cannot eat genuine broad features, and the
   response–low-$k$ covariance is visible in the posterior.

### 5.2 Light ratio ↔ line depth

With constant light fractions the composite is $1 + \sum_i \ell_i\,\mathbf{T}_{ij} d_i$: the
likelihood depends on the *products* $\ell_i d_i$ only. Therefore $(\ell_i, d_i)$ are exactly
degenerate along $\ell_i \to \ell_i/\alpha$, $d_i \to \alpha d_i$ — the data cannot measure the
continuum light ratio, only each star's *contribution* to the composite lines. The degeneracy
is broken only by:

1. **Per-epoch light variation** (eclipses): $\ell_{ij}$ varying with known/parameterized
   geometry makes the products epoch-dependent while $d_i$ is shared — this is why per-epoch
   light fractions are a first-class feature, not an add-on.
2. **External photometric priors** on $\ell_i$ (from light-curve solutions or SED fits).
3. **Physicality floor**: $s_i \ge 0 \Rightarrow \ell_i d_i \ge -\ell_i$; a saturated line in
   the composite bounds $\ell_i$ from below. Weak, but real — available as an optional
   constraint.
4. **Fixing $\ell$** by assumption.

The API refuses to guess: `light_ratio=` must be given explicitly as `Fixed(values)`,
`Free(prior=...)` (requires 1–3 to be informative, and the docs say so), or
`PerEpoch(...)`.

### 5.3 Systemic velocity / zero-point

A change $\gamma \to \gamma + \epsilon$ composed with translating every $d_i$ by
$-\xi(\epsilon)$ leaves the likelihood invariant (up to grid edges), and the stationary priors
of §2 are translation-invariant too — so $\gamma$ is unidentified by disentangling itself
(a known property, inherited from the physics, of all disentangling methods). Default:
$\gamma \equiv 0$; recovered spectra live in the systemic frame, and $\gamma$ is measured
afterwards by template cross-correlation *of the disentangled spectra* — outside the sampler.
Users with genuine rest-frame information can free $\gamma$ with an informative prior. $K_i$,
$e$, $\omega$, $P_{\rm orb}$, $T_{\rm p}$ are unaffected.

### 5.4 Degeneracy ledger

| Degeneracy | Exact/approx | Broken by | albireo policy |
|---|---|---|---|
| low-$k$ mode exchange between components | exact at $k=0$, $\propto 1/k$ | phase coverage ($\operatorname{Var}\Delta$), priors | proper priors; covariance reported; forecast tool |
| $\ell_i$ vs. line depth | exact (constant $\ell$) | eclipses, photometry, saturation floor, assumption | explicit `light_ratio=` choice required |
| $\gamma$ vs. common shift | exact up to edges | external rest-frame info | $\gamma \equiv 0$ default, post-hoc measurement |
| per-epoch constants vs. response | approx | low poly order | order $\le 2$ default, covariance reported |
| telluric constant vs. common stellar constant | exact up to edges | ridge anchors ($\eta$) on both | measured in the telluric closed loop: the two offsets cancel in the sum to $\lesssim 10^{-3}$; report both |
| LSF width vs. intrinsic line widths | near-exact per instrument | cross-instrument spectrum sharing | absolute widths need a *reference instrument* anchor (tight prior); only relative widths are data-identified (M4, benchmarks.md) |

Two of these deserve a sentence. **Telluric constant exchange:** with $\sum_i \ell_i = 1$
and a telluric component of light fraction 1, adding a constant $a$ to the telluric
deviation while subtracting $a$ from *every* stellar deviation changes no epoch's
prediction (constants are shift-invariant away from the grid edges) — a second exact
$k = 0$ mode, split only by the $\eta$ ridges. **LSF ↔ intrinsic widths:** for one
instrument a wider Gaussian kernel composed with intrinsically narrower lines is
observationally near-identical (Gaussian widths add in quadrature), so a template-free
model cannot measure an absolute LSF width; empirically, ML-II with all widths free
inflates them by tens of percent while leaving the orbit untouched. Multiple
instruments *sharing the same spectra* identify the width differences; the absolute
scale must come from one instrument whose LSF is known.

---

## 6. SB1 + faint companion mode ($K_2$ scan)

The dormant-compact-object workflow. Given an SB1 solution (fixed $P_{\rm orb}, e, \omega,
T_{\rm p}, K_1$; primary spectrum either fixed from a single-component fit or left free), scan
a grid of trial $K_2$ (optionally × light fraction $\ell_2$): for each trial, the secondary
deviation spectrum $d_2$ is a linear component and marginalizes analytically, so the detection
statistic

$$
D(K_2) = 2\left[\log p(y \,|\, K_2) - \log p(y \,|\, \text{no companion})\right]
$$

costs one linear solve per grid point (and vmaps over the grid). This is the optimal matched
filter *marginalized over the unknown companion spectrum* — strictly more sensitive than CCF
grid searches with assumed templates, and it returns the recovered companion spectrum
$\hat d_2$ with covariance at the peak. Because $d_2$'s prior scale enters, $D$ is calibrated
empirically by injection–recovery (same simulator as M1) rather than by an asymptotic $\chi^2$
claim; the docs will be explicit that the null distribution is estimated, not assumed.

Implementation notes (M4, `albireo.scan.k2_scan`): the no-companion model is the
single-component fit with $\ell_1 = 1$ and the primary's prior; the companion's light
fraction $\ell_2$ must be chosen explicitly (§5.2 — the observable is $\ell_2 d_2$).
Because both log-marginals carry their $\tfrac12\log\det$ Occam terms, the extra
marginalized component *costs* likelihood unless coherent signal pays for it: on a
companion-free dataset $D(K_2)$ is negative at every trial (measured in the closed-loop
test), which is the sane baseline for the empirical calibration. One honest caveat,
inherited from §5.1: at small $\ell_2$ the companion's smooth envelope (continuum level,
mean line blanketing) is prior-dominated — an error $\Delta$ in the bright primary's
envelope maps to $-(\ell_1/\ell_2)\Delta$ in the companion, an amplification of ~10 at
$\ell_2 = 0.1$ — so the recovered $\hat d_2$ carries its line *pattern*, not a
trustworthy absolute depth scale, unless eclipses or photometry pin the envelope.

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
$b \ge b_{\rm true}$ is exact, so the bound costs time, never accuracy. The failure mode is
an *underestimate* — comb probing then aliases band entries and the likelihood is silently
wrong. The sampler is therefore protected by a **bandwidth guard**: the numpyro model
computes the realized $\max_{j,i,i'} |\delta_{ij} - \delta_{i'j}|$ and adds a $-\infty$
factor whenever it exceeds the budget implied by $b_{\rm bound}$. A prior wider than the
budget slows mixing near the boundary but cannot corrupt the posterior.

### 7.2 Parameterization

Sampled sites: $P$, $T_{\rm conj}$, $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$, and
$K_i$ (γ ≡ 0 by §5.3 / D14). The $\sqrt{e}$-pair is smooth through $e = 0$ — where $\omega$
and $T_{\rm peri}$ are undefined — and a uniform prior on the unit disk maps to
$e \sim \mathcal{U}(0,1)$, $\omega \sim \mathcal{U}(-\pi,\pi)$; the disk constraint enters
as a $-\infty$ factor with $e$ clipped at $e_{\max} = 0.95$ (the Kepler solver's verified
range) before the solve, so out-of-support proposals stay finite and rejectable. The single
non-smooth point is the origin $\sqrt{e}\cos\omega = \sqrt{e}\sin\omega = 0$ (an
`arctan2` branch point of measure zero) — circular-orbit initializations should sit slightly
off it. $T_{\rm conj}$ (the §1.2 convention $\nu + \omega = \pi/2$) replaces $T_{\rm peri}$,
which degenerates with $\omega$ as $e \to 0$.

One smoothness caveat: with linear (2-tap) shift interpolation (D3), $A(\theta)$ is
piecewise-linear in each shift, so $\log p(y\mid\theta)$ is piecewise-$C^1$ in the
velocities with derivative kinks where a shift crosses an integer pixel. On an oversampled
model grid the kink amplitudes are set by sub-pixel spectral curvature and are far below
the posterior scale; NUTS treats them as it treats any leapfrog-scale roughness. The 4-tap
cubic kernel (D3, flagged) is the smoothing upgrade if a dataset ever exposes them.

### 7.3 Hyperparameters: ML-II by default

The prior scales $(\tau_i, \eta_i)$ control exactly the part of spectrum space the data
cannot constrain (§5.1: sub-LSF modes, low-$k$ anchoring), so they must be *chosen*
deliberately rather than defaulted. Because the spectra are already integrated out,
maximizing the marginal posterior jointly over $(\theta, \log\tau, \log\eta)$ **is** ML-II
/ empirical Bayes (up to weak hyperpriors that keep the optimization proper). The MAP
pipeline does this with L-BFGS in numpyro's unconstrained space; NUTS then runs with the
hyperparameters conditioned at their ML-II values (default), or sampling them (at the cost
of the usual mild underestimation-of-hyperparameter-uncertainty trade swapped for extra
dimensions — both supported, the choice is recorded in the fit metadata). The marginal
likelihood already contains the $\tfrac12\log\det\boldsymbol\Lambda_p$ Occam term, so ML-II
is well-posed: $\tau \to \infty$ is penalized by data misfit, $\tau \to 0$ by the
determinant.

### 7.4 Posterior spectra

$p(d \mid y) = \int p(d \mid y, \theta)\, p(\theta \mid y)\, d\theta$ — a mixture of the
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
component — nothing downstream changes.

**Per-epoch light fractions.** $\ell_{ij}$ enters $\mathbf{A}(\theta)$ *linearly*
(§1.3), so a `light` site — $(n_{\rm stellar},)$ constant or
$(J, n_{\rm stellar})$ per-epoch, rows on the simplex via Dirichlet priors — swaps into
the static graph exactly like the shifts, and the §5.2 eclipse breaker becomes an
*inferred* quantity. The likelihood is smooth (indeed quadratic-in-$\ell$ per epoch
pre-marginalization), so MAP/NUTS handle it natively.

**LSF widths.** A Gaussian kernel's *values* at fixed integer offsets are smooth in
$\sigma$, so `lsf_sigma` (one width per instrument) is traced while the kernel *radius*
stays fixed at build time by the construction-time width, which thereby becomes a strict
upper bound: a realized $\sigma$ above it would be silently truncated by the fixed
radius, so the model rejects it with a $-\infty$ factor (the same
guard-not-silent-corruption pattern as the bandwidth budget, §7.1). Identifiability is
§5.4's caveat: anchor one reference instrument.

**Per-epoch response (D33, post-M5).** The multiplicative response enters the
likelihood in three places — the target, $z_j = y_j - r_j \odot (\mathbf{R}\mathbf{1})$;
the sandwich weights, since $\mathbf{A}_j = \mathrm{diag}(r_j)\mathbf{R}_j\mathbf{B}_j\cdots$
folds $r_j^2$ into $\mathbf{A}^\top W \mathbf{A}$; and the right-hand side through
$r_j z_j$ — which is why D7 kept its coefficients as build-time constants through M4: a
`response` swap is not a pure operator swap like the shifts. It is nonetheless cheap,
because $\mathbf{R}\mathbf{1}$ (the rebinned unit continuum, stored per group) is
*response-independent*:

$$
z^{\rm new}_j \;=\; z^{\rm old}_j + \left(r^{\rm old}_j - r^{\rm new}_j\right) \odot \mathbf{R}\mathbf{1}
$$

rebuilds the target exactly without carrying the raw fluxes (and re-masking keeps
zero-weight pixels at exactly zero, so the D30 ``0·nan`` trap cannot resurface), while
the $\sum \log w$ term is untouched — the noise lives on the data, not on
response-divided data. `response` is a θ site: $(n_{\rm coef},)$ shared or
$(J, n_{\rm coef})$ per-epoch, $r = 1 + \sum_m c_m T_m(x)$ on each group's native
abscissa. Identifiability is §5's response row, sharpened by measurement: the
epoch-to-epoch *differences* of the coefficients are well constrained (closed loop:
recovered to $\sim 10^{-3}$ against injected $3\times 10^{-2}$), while the epoch-shared
mode trades against the components' broad features and lands at its zero-centered prior
rather than at truth. Keep the order low and the priors tight; read the common mode as
a normalization convention, not a measurement.

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
| SB3 velocity law (§7.5) | `orbit_velocities` with outer sites ≡ hand-composed nested Keplerians, atol $10^{-12}$ |
| LSF bound + outer-disk guards | width above build bound / outer $e > e_{\max}$ ⇒ non-finite model log-density |
| telluric constant exchange (§5.4) | closed loop: the two $k=0$ offsets cancel in the light-weighted sum to $<5\times10^{-3}$ |
| **M4 gate**: closed loop per realism feature | telluric joint MAP; SB3 MAP (inner and outer $K$'s <2%); per-epoch light inferred (ℓ rms <0.01, components individually recovered); LSF width vs. reference instrument <3%; $K_2$ scan (peak at truth, negative $D$ under null) |

Sections 1–2 and the operator rows are implemented and tested in M0; §3–4 landed in M2;
§7.1–7.4 landed in M3 (with §5 diagnostics); §6 and §7.5 landed in M4, except the §7.5
response swap, which landed post-M5 (D33).
