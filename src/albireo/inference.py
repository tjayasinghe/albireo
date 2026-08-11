"""Joint Bayesian inference of the orbit with the spectra marginalized (M3).

The nonlinear parameter vector ``theta`` is a dict of JAX arrays with sites

- ``period``      orbital period [d]
- ``t_conj``      time of conjunction of component 1 (``nu + omega = pi/2``) [d]
- ``secosw``      ``sqrt(e) cos(omega)``
- ``sesinw``      ``sqrt(e) sin(omega)``
- ``k``           RV semi-amplitudes ``(K_1, K_2, ...)`` [km/s], one per stellar
  component; even components use ``omega``, odd use ``omega + pi``
- ``log_tau``, ``log_eta`` (optional) — log spectral-prior hyperparameters, one per
  model component (including the telluric component when enabled)

``gamma`` is identically zero (design decision D14: a systemic velocity is exactly
degenerate with a common shift of the component spectra). The ``(secosw, sesinw)``
parameterization is smooth through ``e = 0`` — where ``omega`` and a time of
periastron are undefined — and carries a uniform-on-the-unit-disk prior to a uniform
prior on ``e``; the disk constraint ``e < 1`` enters the model as a ``-inf`` factor,
with ``e`` clipped to ``ecc_max`` before the Kepler solve so the likelihood stays
finite (and rejectable) outside it. The map is non-differentiable only at the exact
point ``secosw = sesinw = 0`` — initialize circular orbits slightly off the origin.

Hyperparameters follow empirical Bayes by default (``docs/math.md`` §5.1: the prior
curvature scale *is* information the data cannot supply below the LSF scale, so it
must be estimated deliberately): :func:`run_map` maximizes the joint over
``(theta, log_tau, log_eta)`` — the marginal likelihood is already integrated over
the spectra, so this is ML-II up to the weak hyperpriors — and NUTS then runs with
the hyperparameters held at those values (pass them via ``fixed=``), or sampled, at
the user's choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model

from albireo.data import Dataset
from albireo.forward import build_problem, with_velocities
from albireo.grids import LogGrid
from albireo.kepler import radial_velocity, t_peri_from_t_conj
from albireo.likelihood import MarginalResult, draw_spectra, marginal_loglikelihood
from albireo.priors import SmoothnessPrior

__all__ = [
    "MAPResult",
    "MarginalOrbitModel",
    "laplace_inverse_mass",
    "orbit_parameters",
    "orbit_velocities",
    "posterior_spectra",
    "run_map",
    "run_nuts",
]

_ECC_MAX_DEFAULT = 0.95  # the Kepler solver is verified up to e = 0.95
_THETA_SITES = ("period", "t_conj", "secosw", "sesinw", "k", "log_tau", "log_eta")


def _max_relative_shift(problem) -> jax.Array:
    """Max over epochs/component pairs of |delta_i - delta_i'| in pixels (traced)."""
    rel = jnp.asarray(0.0)
    for g in problem.groups:
        s = g.shifts
        rel = jnp.maximum(rel, jnp.max(jnp.abs(s[:, :, None] - s[:, None, :])))
    return rel


def orbit_parameters(theta: Mapping, *, ecc_max: float = _ECC_MAX_DEFAULT) -> dict:
    """Physical orbit parameters from a ``theta`` dict (differentiable).

    Returns ``{"period", "t_conj", "ecc", "omega", "k"}`` with ``ecc`` clipped to
    ``ecc_max`` (the unclipped value is enforced separately by the model's disk
    constraint).
    """
    h, s = jnp.asarray(theta["secosw"]), jnp.asarray(theta["sesinw"])
    return {
        "period": jnp.asarray(theta["period"]),
        "t_conj": jnp.asarray(theta["t_conj"]),
        "ecc": jnp.minimum(h * h + s * s, ecc_max),
        "omega": jnp.arctan2(s, h),
        "k": jnp.atleast_1d(jnp.asarray(theta["k"])),
    }


def orbit_velocities(theta: Mapping, bjd, *, ecc_max: float = _ECC_MAX_DEFAULT):
    """Stellar radial velocities, shape ``(n_stellar, n_epochs)``, barycentric frame.

    Same convention as :class:`albireo.simulate.OrbitParams`: component ``i`` uses
    ``omega + (i % 2) * pi`` and semi-amplitude ``k[i]``; ``gamma = 0`` (D14).
    """
    par = orbit_parameters(theta, ecc_max=ecc_max)
    t_peri = t_peri_from_t_conj(
        par["t_conj"], period=par["period"], ecc=par["ecc"], omega=par["omega"]
    )
    bjd = jnp.asarray(bjd)
    rows = [
        radial_velocity(
            bjd,
            period=par["period"],
            t_peri=t_peri,
            ecc=par["ecc"],
            omega=par["omega"] + (i % 2) * jnp.pi,
            k=par["k"][i],
        )
        for i in range(par["k"].shape[0])
    ]
    return jnp.stack(rows)


class MarginalOrbitModel:
    """The marginal posterior over orbital parameters for one dataset.

    Bundles the static problem structure (rebin operators, kernels, weights — built
    once) with the θ-dependent path (Kepler velocities → shifts → marginal likelihood)
    so that :meth:`log_likelihood` is a single jit-compiled, differentiable function
    of ``theta``. The solver bandwidth is fixed by ``v_rel_max_kms`` (see
    :meth:`albireo.forward.Problem.half_bandwidth_bound`) so the computation graph is
    static; the numpyro model rejects (``-inf``) any configuration whose actual
    relative shifts exceed that budget, so a prior wider than ``v_rel_max_kms`` slows
    mixing near the bound but can never corrupt the result. The direct
    :meth:`log_likelihood` entry point carries no such guard — keep explicit calls
    within the bound.

    Parameters
    ----------
    grid, dataset, light_fractions, lsf_sigma_v, response_coeffs, telluric
        As in :func:`albireo.forward.build_problem`.
    v_rel_max_kms
        Bound on the largest relative velocity between any two model components at
        any epoch (for an SB2: ``(K_1 + K_2)(1 + e)``, plus barycentric motion if a
        telluric component is enabled). Give it headroom — the prior on ``k`` must
        not allow configurations that exceed it.
    prior
        Fixed :class:`SmoothnessPrior`, used whenever ``theta`` carries no
        ``log_tau``/``log_eta`` sites. Optional if the hyperparameters are always in
        ``theta``.
    ecc_max
        Eccentricity clip/constraint (default 0.95, the solver's verified range).
    block_size
        Solver block size passed through to the marginal likelihood.
    """

    def __init__(
        self,
        grid: LogGrid,
        dataset: Dataset,
        *,
        light_fractions,
        lsf_sigma_v: Mapping[str, float],
        v_rel_max_kms: float,
        response_coeffs=None,
        telluric: bool = False,
        prior: SmoothnessPrior | None = None,
        ecc_max: float = _ECC_MAX_DEFAULT,
        block_size: int | None = None,
    ):
        ell = np.asarray(light_fractions, dtype=np.float64)
        n_stellar = ell.shape[0]
        self.problem = build_problem(
            grid,
            dataset,
            velocities=np.zeros((n_stellar, dataset.n_epochs)),
            light_fractions=ell,
            lsf_sigma_v=lsf_sigma_v,
            response_coeffs=response_coeffs,
            telluric=telluric,
        )
        self.bjd = jnp.asarray(dataset.bjd)
        self.half_bandwidth = self.problem.half_bandwidth_bound(v_rel_max_kms)
        # The shift budget inside half_bandwidth (inverse of half_bandwidth_bound);
        # the numpyro model rejects any configuration whose actual relative shifts
        # exceed it, so a prior wider than v_rel_max cannot silently corrupt probing.
        support = max(g.row_support for g in self.problem.groups)
        self._shift_bound = self.half_bandwidth - 1 - 2 * self.problem.kernel_radius - support
        self.block_size = block_size
        self.ecc_max = float(ecc_max)
        self.fixed_prior = prior
        self._marginal_jit = jax.jit(self._marginal)

    @property
    def n_stellar(self) -> int:
        return self.problem.n_stellar

    def _prior(self, theta: Mapping) -> SmoothnessPrior:
        if "log_tau" in theta or "log_eta" in theta:
            if "log_tau" not in theta or "log_eta" not in theta:
                raise ValueError("theta must carry both log_tau and log_eta, or neither")
            return SmoothnessPrior(jnp.exp(theta["log_tau"]), jnp.exp(theta["log_eta"]))
        if self.fixed_prior is None:
            raise ValueError(
                "no spectral prior: pass prior= at construction or include log_tau/log_eta in theta"
            )
        return self.fixed_prior

    def _velocity_problem(self, theta: Mapping):
        vel = orbit_velocities(theta, self.bjd, ecc_max=self.ecc_max)
        return with_velocities(self.problem, vel)

    def _marginal_from_problem(self, problem, theta: Mapping) -> MarginalResult:
        return marginal_loglikelihood(
            problem,
            self._prior(theta),
            block_size=self.block_size,
            half_bandwidth=self.half_bandwidth,
        )

    def _marginal(self, theta: Mapping) -> MarginalResult:
        return self._marginal_from_problem(self._velocity_problem(theta), theta)

    def marginal(self, theta: Mapping) -> MarginalResult:
        """Jit-compiled marginal result (log-likelihood + conditional spectra) at θ."""
        return self._marginal_jit(dict(theta))

    def log_likelihood(self, theta: Mapping):
        """Jit-compiled marginal log-likelihood at θ (differentiable)."""
        return self.marginal(theta).log_likelihood

    def model(self, priors: Mapping[str, dist.Distribution], *, fixed: Mapping | None = None):
        """Build a numpyro model: sample ``priors``, add the marginal likelihood.

        Parameters
        ----------
        priors
            Distribution per sampled site (``period``, ``t_conj``, ``secosw``,
            ``sesinw``, ``k``; optionally ``log_tau``/``log_eta``). The ``k`` site is
            vector-valued: use a distribution with batch shape ``(n_stellar,)``.
        fixed
            Values injected as constants instead of sampled — the empirical-Bayes
            route: fix ``log_tau``/``log_eta`` at their :func:`run_map` values for the
            NUTS run. Keys must not also appear in ``priors``.

        Returns
        -------
        callable
            A no-argument numpyro model, for :func:`run_map` / :func:`run_nuts`.
            Records ``ecc`` and ``omega`` as deterministic sites.
        """
        unknown = [s for s in priors if s not in _THETA_SITES]
        if unknown:
            raise ValueError(f"unknown sites in priors: {unknown} (expected {_THETA_SITES})")
        fixed = dict(fixed or {})
        overlap = set(fixed) & set(priors)
        if overlap:
            raise ValueError(f"sites both fixed and sampled: {sorted(overlap)}")

        def _model():
            theta = {name: numpyro.sample(name, d) for name, d in priors.items()}
            theta.update({name: jnp.asarray(v) for name, v in fixed.items()})
            ecc_raw = theta["secosw"] ** 2 + theta["sesinw"] ** 2
            numpyro.deterministic("ecc", jnp.minimum(ecc_raw, self.ecc_max))
            numpyro.deterministic("omega", jnp.arctan2(theta["sesinw"], theta["secosw"]))
            numpyro.factor("ecc_disk", jnp.where(ecc_raw <= self.ecc_max, 0.0, -jnp.inf))
            problem = self._velocity_problem(theta)
            # Reject configurations whose relative shifts exceed the static bandwidth
            # (the probed marginal likelihood would be silently wrong out there).
            rel = _max_relative_shift(problem)
            numpyro.factor("bandwidth_guard", jnp.where(rel <= self._shift_bound, 0.0, -jnp.inf))
            numpyro.factor(
                "marginal_loglike", self._marginal_from_problem(problem, theta).log_likelihood
            )

        return _model


@dataclass(frozen=True)
class MAPResult:
    """Result of :func:`run_map`.

    ``potential`` and ``grad_norm`` are evaluated at the convergence-check point,
    which (by the check-then-step loop) is one accepted L-BFGS step *behind*
    ``params`` — irrelevant at convergence, stated for exactness.
    """

    params: dict  # constrained values of all sites, including deterministics
    unconstrained: dict
    potential: float  # potential energy (-log joint, up to constants)
    grad_norm: float  # unconstrained-space gradient norm
    converged: bool
    num_steps: int


def run_map(
    model,
    *,
    init: Mapping,
    rng_key=None,
    max_steps: int = 200,
    tol: float = 1e-2,
) -> MAPResult:
    """MAP over all sampled sites of ``model`` via L-BFGS on numpyro's potential.

    Runs in numpyro's unconstrained space (so constrained priors are handled by the
    standard transforms) with a zoom linesearch; with ``log_tau``/``log_eta`` among
    the sampled sites this is the ML-II / empirical-Bayes hyperparameter fit, since
    the spectra are already marginalized out of the likelihood.

    Parameters
    ----------
    model
        A numpyro model (from :meth:`MarginalOrbitModel.model`).
    init
        Constrained initial values for every sampled site. Start circular orbits at
        small nonzero ``(secosw, sesinw)`` — the origin is the one non-smooth point.
    tol
        Convergence threshold on the unconstrained-space gradient norm.
    """
    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    model_info = initialize_model(
        rng_key, model, init_strategy=init_to_value(values=dict(init)), dynamic_args=False
    )
    potential = model_info.potential_fn
    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(potential)

    @jax.jit
    def step(params, state):
        value, grad = value_and_grad(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=potential)
        params = optax.apply_updates(params, updates)
        return params, state, value, grad

    # numpyro may return python-float leaves for scalar sites; optax needs arrays
    params = jax.tree.map(jnp.asarray, model_info.param_info.z)
    state = opt.init(params)
    value, grad_norm, steps_taken = np.inf, np.inf, 0
    for steps_taken in range(1, max_steps + 1):
        params, state, value, grad = step(params, state)
        grad_norm = float(optax.tree.norm(grad))
        if not np.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite gradient at L-BFGS step {steps_taken} (potential {float(value)})"
            )
        if grad_norm < tol:
            break
    constrained = model_info.postprocess_fn(params)
    return MAPResult(
        params={k: v for k, v in constrained.items()},
        unconstrained=dict(params),
        potential=float(value),
        grad_norm=grad_norm,
        converged=grad_norm < tol,
        num_steps=steps_taken,
    )


def laplace_inverse_mass(model, params: Mapping, *, rng_key=None, floor: float = 1e-10):
    """Unconstrained-space Laplace covariance at ``params`` — a NUTS starting mass matrix.

    Evaluates the Hessian of the model potential at the given constrained site values
    (typically :attr:`MAPResult.params`; extra keys are ignored), symmetrizes, floors
    the eigenvalues at ``floor * max_eig``, and returns the inverse as a dense array.
    Pass it as ``inverse_mass_matrix`` to :func:`run_nuts` built from the *same*
    model: with the mass matrix pre-set to (approximately) the posterior covariance,
    warmup only tunes the step size — without it, parameter scales spanning many
    orders of magnitude drive early trajectories to the tree-depth cap and warmup
    costs more than sampling.
    """
    rng_key = jax.random.PRNGKey(0) if rng_key is None else rng_key
    model_info = initialize_model(
        rng_key, model, init_strategy=init_to_value(values=dict(params)), dynamic_args=False
    )
    z = jax.tree.map(jnp.asarray, model_info.param_info.z)
    flat, unravel = ravel_pytree(z)
    hess = jax.hessian(lambda zf: model_info.potential_fn(unravel(zf)))(flat)
    hess = 0.5 * (hess + hess.T)
    eigval, eigvec = jnp.linalg.eigh(hess)
    eigval = jnp.maximum(eigval, floor * jnp.max(eigval))
    return np.asarray(eigvec @ jnp.diag(1.0 / eigval) @ eigvec.T)


def run_nuts(
    model,
    *,
    rng_key,
    init: Mapping | None = None,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    target_accept: float = 0.9,
    dense_mass: bool = True,
    inverse_mass_matrix=None,
    adapt_mass_matrix: bool | None = None,
    max_tree_depth: int = 8,
    progress_bar: bool = False,
) -> MCMC:
    """NUTS over the sampled sites of ``model`` (spectra stay marginalized).

    ``init`` should be the :attr:`MAPResult.params` dict (extra keys are ignored);
    ``dense_mass=True`` is the right default for the low-dimensional, correlated
    orbital posterior, and ``inverse_mass_matrix`` from :func:`laplace_inverse_mass`
    makes warmup cheap. When an explicit mass matrix is supplied, mass adaptation
    defaults to *off* — warmup's early adaptation windows would overwrite the Laplace
    matrix with a poor few-sample estimate and give back the slow, deep-tree warmup
    the matrix was meant to avoid (override via ``adapt_mass_matrix=True``). Returns
    the numpyro ``MCMC`` object (``.get_samples()``, ``.print_summary()``);
    divergences and tree depths are collected as extra fields.
    """
    if adapt_mass_matrix is None:
        adapt_mass_matrix = inverse_mass_matrix is None
    strategy = init_to_value(values=dict(init)) if init is not None else init_to_value()
    kernel = NUTS(
        model,
        init_strategy=strategy,
        target_accept_prob=target_accept,
        dense_mass=dense_mass,
        inverse_mass_matrix=(
            jnp.asarray(inverse_mass_matrix) if inverse_mass_matrix is not None else None
        ),
        adapt_mass_matrix=adapt_mass_matrix,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, extra_fields=("num_steps", "diverging"))
    return mcmc


def posterior_spectra(
    model: MarginalOrbitModel,
    samples: Mapping,
    key,
    *,
    num_draws: int = 32,
    extra: Mapping | None = None,
):
    """Spectra draws from the *joint* posterior, shape ``(num_draws, n_comp, n_pix)``.

    Each draw picks a posterior ``theta`` sample at random and draws once from the
    conditional Gaussian over the spectra — so the returned scatter includes both the
    conditional spectral uncertainty and the orbital uncertainty. ``extra`` supplies
    sites missing from ``samples`` (e.g. ``log_tau``/``log_eta`` when they were fixed
    during sampling).
    """
    extra = {name: jnp.asarray(v) for name, v in (extra or {}).items()}
    site_names = [s for s in _THETA_SITES if s in samples and s not in extra]
    n_samples = np.asarray(samples[site_names[0]]).shape[0]
    key_idx, key_draw = jax.random.split(jnp.asarray(key))
    idx = np.asarray(jax.random.randint(key_idx, (num_draws,), 0, n_samples))
    out = []
    for j, i in enumerate(idx):
        theta = {name: jnp.asarray(samples[name])[i] for name in site_names}
        theta.update(extra)
        result = model.marginal(theta)
        out.append(draw_spectra(result, jax.random.fold_in(key_draw, j), 1)[0])
    return jnp.stack(out)
