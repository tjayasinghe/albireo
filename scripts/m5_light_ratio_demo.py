"""The hand-set light-ratio systematic, quantified (M5 paper asset).

The LB-1 / HR 6819 debates hinged on disentangled component spectra whose light
ratio was set by hand: the recovered deviation spectrum scales as 1/ell, so an
assumed ell that is wrong by a factor alpha rescales every line depth by alpha —
which then feeds directly into surface-gravity / class diagnostics. This script
demonstrates, on a seeded simulation:

1. Fixed-orbit disentangling with a hand-set wrong ell reproduces the systematic:
   measured line-depth scaling equals ell_true / ell_assumed (math.md §5.2).
2. The marginal likelihood over ell — with hyperparameters refit by ML-II at
   every trial ell, so the comparison is like for like — is *flat* with constant
   light fractions (the data genuinely do not choose ell, the hand does) but
   sharply peaked at truth when three partial-eclipse epochs are added. With
   *fixed* hypers the constant-light profile shows spurious prior-mediated
   curvature; that trap is the reason for the per-trial refit.

Run: python scripts/m5_light_ratio_demo.py   (~12 min CPU; 16 small ML-II fits)
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import albireo as ab
from albireo.forward import build_problem
from albireo.inference import MarginalOrbitModel
from albireo.kepler import t_peri_from_t_conj
from albireo.likelihood import marginal_loglikelihood
from albireo.priors import SmoothnessPrior

GRID = ab.LogGrid.from_wavelength_range(5000.0, 5045.0, dv_kms=5.5)
P, TCONJ, ECC, OMEGA = 6.31, 2.05, 0.2, 0.7
K = (30.0, 22.0)
ELL_TRUE = np.array([0.7, 0.3])
N_EP = 12
PRIOR = SmoothnessPrior(jnp.asarray([300.0, 300.0]), jnp.asarray([5.0, 5.0]))


def simulate(eclipse: bool):
    rng = np.random.default_rng(42)
    bjd = np.sort(rng.uniform(0.0, 2.4 * P, N_EP))
    comps = [
        ab.synthetic_deviation_spectrum(
            GRID, n_lines=26, depth_range=(0.1, 0.7), sigma_v_range=(9.0, 20.0), seed=s
        )
        for s in (1, 2)
    ]
    ell = np.repeat(ELL_TRUE[:, None], N_EP, axis=1)
    if eclipse:
        ell[0, [2, 5, 9]] = [0.52, 0.44, 0.57]  # partial eclipses of the primary
        ell[1] = 1.0 - ell[0]
    tperi = float(t_peri_from_t_conj(TCONJ, period=P, ecc=ECC, omega=OMEGA))
    orbit = ab.OrbitParams(period=P, t_peri=tperi, ecc=ECC, omega=OMEGA, k=K)
    spec = ab.InstrumentSpec(wave=np.arange(5003.0, 5042.0, 0.11), sigma_v_lsf=7.0, snr=130.0)
    ds, truth = ab.simulate_dataset(
        GRID,
        comps,
        bjd=bjd,
        instruments={"inst": spec},
        light_fractions=ell,
        orbit=orbit,
        frame="barycentric",
        seed=11,
    )
    return ds, truth, comps


def solve_at(ds, truth, ell_assumed):
    problem = build_problem(
        GRID,
        ds,
        velocities=truth.velocities,
        light_fractions=np.asarray(ell_assumed),
        lsf_sigma_v={"inst": 7.0},
    )
    return np.asarray(marginal_loglikelihood(problem, PRIOR).d_hat)


def depth_scaling(d_hat, d_true):
    """Affine fit d_hat ~ a * d_true + b over interior pixels.

    The slope ``a`` isolates the multiplicative light-ratio systematic
    (prediction: ell_true / ell_assumed); the intercept ``b`` is the separate,
    additive k~0 envelope offset (math.md §5.1) that would otherwise contaminate
    a per-pixel depth ratio.
    """
    sl = slice(int(0.05 * GRID.n), -int(0.05 * GRID.n))
    x, y = d_true[sl], d_hat[sl]
    a = float(np.cov(y, x)[0, 1] / np.var(x))
    b = float(np.mean(y - a * x))
    return a, b


def main():
    ds, truth, comps = simulate(eclipse=False)
    d2_true = np.asarray(comps[1])

    print("1) Hand-set light ratio -> line-depth systematic (constant light)")
    print("   assumed ell_2   predicted slope (ell2_true/ell2_asm)   measured slope   offset")
    for ell2 in (0.2, 0.3, 0.5):
        d_hat = solve_at(ds, truth, [1.0 - ell2, ell2])
        a, b = depth_scaling(d_hat[1], d2_true)
        pred = ELL_TRUE[1] / ell2
        print(f"   {ell2:.2f}            {pred:5.2f}               {a:5.2f}   {b:+.3f}")

    print()
    print("2) Marginal log-likelihood profile over ell_1, hyperparameters refit by")
    print("   ML-II at every trial ell (like with like: with *fixed* hypers the")
    print("   constant-light profile shows spurious prior-mediated curvature, since")
    print("   a wrong ell forces rescaled spectra that the fixed prior penalizes)")
    ell1_grid = np.linspace(0.50, 0.85, 8)
    for label, eclipse in (("constant light", False), ("3 eclipse epochs", True)):
        ds_e, _truth_e, _ = simulate(eclipse=eclipse)
        model = MarginalOrbitModel(
            GRID,
            ds_e,
            light_fractions=ELL_TRUE,
            lsf_sigma_v={"inst": 7.0},
            v_rel_max_kms=(K[0] + K[1]) * (1 + ECC) * 1.35,
        )
        orbit_theta = {
            "period": jnp.asarray(P),
            "t_conj": jnp.asarray(TCONJ),
            "secosw": jnp.asarray(np.sqrt(ECC) * np.cos(OMEGA)),
            "sesinw": jnp.asarray(np.sqrt(ECC) * np.sin(OMEGA)),
            "k": jnp.asarray(K),
        }

        def nll(hy, light, orbit_theta=orbit_theta, model=model):
            theta = {**orbit_theta, "light": light, "log_tau": hy[:2], "log_eta": hy[2:]}
            return -model._marginal(theta).log_likelihood

        opt = optax.lbfgs()

        @jax.jit
        def ml2_step(hy, state, light, opt=opt, nll=nll):
            value, grad = jax.value_and_grad(nll)(hy, light)
            updates, state = opt.update(
                grad, state, hy, value=value, grad=grad, value_fn=lambda h: nll(h, light)
            )
            return optax.apply_updates(hy, updates), state, value

        def ml2_loglike(light, opt=opt, ml2_step=ml2_step):
            hy = jnp.log(jnp.asarray([300.0, 300.0, 5.0, 5.0]))
            state = opt.init(hy)
            value = jnp.inf
            for _ in range(60):
                hy, state, value = ml2_step(hy, state, light)
            return -float(value)

        lls = []
        for ell1 in ell1_grid:
            ell = np.repeat(np.array([[ell1], [1.0 - ell1]]), N_EP, axis=1)
            if eclipse:
                # keep the *relative* eclipse dips, move the out-of-eclipse level
                ell[0, [2, 5, 9]] = ell1 - ELL_TRUE[0] + np.array([0.52, 0.44, 0.57])
                ell[1] = 1.0 - ell[0]
            lls.append(ml2_loglike(jnp.asarray(ell.T)))
        lls = np.array(lls)
        rel = lls - lls.max()
        peak = ell1_grid[int(np.argmax(lls))]
        print(f"   {label}: peak at ell_1 = {peak:.2f} (truth 0.70)")
        print("     ell_1:  " + "  ".join(f"{x:5.2f}" for x in ell1_grid))
        print("     dlogL:  " + "  ".join(f"{x:6.0f}" for x in rel))
    print()
    print("With ML-II hypers the constant-light profile is nearly flat over a wide")
    print("ell range — the data do not choose ell; whoever sets it by hand sets the")
    print("line depths (part 1). Eclipse epochs turn ell into a measurement.")
    print("(math.md §5.2; per-epoch light inference: M4.)")


if __name__ == "__main__":
    main()
