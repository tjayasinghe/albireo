"""Round-tripping fits to disk, and exporting the disentangled spectra.

The point of these tests is that a saved fit reads back *equal*, not merely readable: the
numbers a user quotes in a paper have to survive the trip. The export tests additionally
check that what is written is the normalized component spectrum ``1 + d``, since writing
the deviation ``d`` instead would be a silently wrong file rather than a failure.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import albireo as ab
from albireo.forward import build_problem
from albireo.inference import MAPResult
from albireo.likelihood import marginal_loglikelihood, spectra_std
from albireo.priors import SmoothnessPrior
from albireo.results import load_fit, save_fit, write_ascii
from albireo.scan import K2ScanResult


@pytest.fixture
def map_result():
    return MAPResult(
        params={"period": np.array(40.37), "k": np.array([12.5, 60.1])},
        unconstrained={"period": np.array(3.698), "k": np.array([0.1, 0.9])},
        potential=-1234.5,
        grad_norm=1e-3,
        converged=True,
        num_steps=57,
    )


@pytest.fixture
def k2_result():
    grid = np.arange(10.0, 60.0, 5.0)
    rng = np.random.default_rng(0)
    return K2ScanResult(
        k2_grid=grid,
        log_likelihood=rng.normal(size=grid.size),
        log_likelihood_null=-10.0,
        detection=rng.normal(size=grid.size),
        k2_peak=35.0,
        primary=rng.normal(size=64),
        primary_std=np.abs(rng.normal(size=64)),
        companion=rng.normal(size=64),
        companion_std=np.abs(rng.normal(size=64)),
        model=None,
    )


@pytest.fixture
def marginal_result(small_grid, small_dataset):
    problem = build_problem(
        small_grid,
        small_dataset,
        velocities=np.zeros((2, len(list(small_dataset)))),
        light_fractions=(0.6, 0.4),
        lsf_sigma_v={"A": 8.25},
    )
    prior = SmoothnessPrior(tau=np.array([300.0, 300.0]), eta=np.array([5.0, 5.0]))
    return marginal_loglikelihood(problem, prior)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_map_result_round_trips(map_result, tmp_path):
    path = save_fit(map_result, tmp_path / "map")
    assert path.suffix == ".npz"
    loaded = load_fit(path)

    assert type(loaded) is type(map_result)
    assert loaded.converged is True
    assert loaded.num_steps == 57
    assert loaded.potential == pytest.approx(map_result.potential)
    assert loaded.grad_norm == pytest.approx(map_result.grad_norm)
    assert set(loaded.params) == set(map_result.params)
    for name, value in map_result.params.items():
        np.testing.assert_allclose(loaded.params[name], value)
    for name, value in map_result.unconstrained.items():
        np.testing.assert_allclose(loaded.unconstrained[name], value)


def test_k2_result_round_trips_without_its_model(k2_result, tmp_path):
    loaded = load_fit(save_fit(k2_result, tmp_path / "scan.npz"))

    assert type(loaded) is type(k2_result)
    # The model holds the dataset and the traced structure; it is deliberately not saved.
    assert loaded.model is None
    assert loaded.k2_peak == pytest.approx(k2_result.k2_peak)
    assert loaded.log_likelihood_null == pytest.approx(k2_result.log_likelihood_null)
    for name in ("k2_grid", "detection", "primary", "companion_std"):
        np.testing.assert_allclose(getattr(loaded, name), getattr(k2_result, name))
    # The derived properties still work, which is what a user actually reads off a scan.
    assert loaded.peak_index == k2_result.peak_index
    assert loaded.detection_peak == pytest.approx(k2_result.detection_peak)


def test_marginal_result_saves_spectra_and_uncertainty_by_default(marginal_result, tmp_path):
    loaded = load_fit(save_fit(marginal_result, tmp_path / "marginal.npz"))

    np.testing.assert_allclose(loaded.d_hat, np.asarray(marginal_result.d_hat))
    assert loaded.log_likelihood == pytest.approx(float(marginal_result.log_likelihood))
    assert loaded.n_components == marginal_result.n_components
    assert loaded.n_pixels == marginal_result.n_pixels
    # The precision is the big object and is not stored, but the uncertainty band derived
    # from it is — otherwise the default save silently drops the differentiator.
    assert loaded.precision is None
    np.testing.assert_allclose(loaded.d_std, np.asarray(spectra_std(marginal_result)), rtol=1e-10)


def test_a_precision_free_result_explains_itself_rather_than_crashing(marginal_result, tmp_path):
    loaded = load_fit(save_fit(marginal_result, tmp_path / "marginal.npz"))

    # Reaching for the factor of a precision that was never saved is a legitimate mistake;
    # it should say what happened and where the numbers went, not raise from inside the
    # block Cholesky on a None.
    with pytest.raises(ValueError, match="precision=True"):
        _ = loaded.chol


def test_marginal_result_can_keep_its_precision(marginal_result, tmp_path):
    loaded = load_fit(save_fit(marginal_result, tmp_path / "full.npz", precision=True))

    assert loaded.precision is not None
    # A precision that round-tripped is a precision that can still be factorized, which is
    # what makes draws and Takahashi variances possible after a reload.
    np.testing.assert_allclose(
        np.asarray(spectra_std(loaded)), np.asarray(spectra_std(marginal_result)), rtol=1e-10
    )


def test_load_rejects_a_foreign_npz(tmp_path):
    path = tmp_path / "not_a_fit.npz"
    np.savez(path, some_array=np.arange(3))
    with pytest.raises(ValueError, match="not an albireo fit file"):
        load_fit(path)


def test_load_rejects_a_future_format_version(map_result, tmp_path):
    path = save_fit(map_result, tmp_path / "future.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    import json

    header = json.loads(str(arrays["__albireo__"]))
    header["format_version"] = 999
    arrays["__albireo__"] = np.array(json.dumps(header))
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="format version"):
        load_fit(path)


def test_save_rejects_an_unknown_type(tmp_path):
    with pytest.raises(TypeError, match="does not know how to save"):
        save_fit({"not": "a result"}, tmp_path / "nope.npz")


# ---------------------------------------------------------------------------
# arviz
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_mcmc():
    """A tiny NUTS run over albireo-shaped site names.

    Deliberately not a real albireo fit: what is being tested is the conversion, and the
    part of it that can actually be wrong is the component labelling of the vector-valued
    ``k`` site. A genuine marginal-likelihood fit would take minutes and exercise the same
    three lines.
    """
    import jax
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model():
        numpyro.sample("period", dist.Normal(40.0, 1.0))
        numpyro.sample("k", dist.Normal(jnp.array([12.0, 60.0]), 1.0))

    mcmc = MCMC(NUTS(model), num_warmup=50, num_samples=100, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))
    return mcmc


def test_to_inference_data_labels_components_rather_than_indices(toy_mcmc):
    pytest.importorskip("arviz")

    idata = ab.to_inference_data(toy_mcmc)

    # Checked by shape rather than by class: arviz 1.x moved from its own InferenceData
    # to xarray's DataTree, and `arviz.InferenceData` now warns. What the caller needs is
    # a `.posterior` group with the right groups and coordinates, in every version.
    assert "period" in idata.posterior
    # The whole point: `k` reads as k[K_1], k[K_2] in a summary table, not k[0], k[1].
    assert list(idata.posterior["k"].coords["component"].values) == ["K_1", "K_2"]
    assert idata.posterior["period"].shape == (1, 100)


def test_to_inference_data_accepts_custom_component_names(toy_mcmc):
    pytest.importorskip("arviz")

    idata = ab.to_inference_data(toy_mcmc, component_names=["primary", "secondary"])

    assert list(idata.posterior["k"].coords["component"].values) == ["primary", "secondary"]


def test_to_inference_data_ignores_wrongly_sized_component_names(toy_mcmc):
    pytest.importorskip("arviz")

    # Three names for two components is a user error that should not raise in the middle
    # of a conversion; fall back to the defaults.
    idata = ab.to_inference_data(toy_mcmc, component_names=["a", "b", "c"])

    assert list(idata.posterior["k"].coords["component"].values) == ["K_1", "K_2"]


def test_inference_data_round_trips_through_netcdf(toy_mcmc, tmp_path):
    az = pytest.importorskip("arviz")
    pytest.importorskip("h5netcdf")

    path = tmp_path / "posterior.nc"
    ab.to_inference_data(toy_mcmc).to_netcdf(path)
    reloaded = az.from_netcdf(path)

    assert list(reloaded.posterior["k"].coords["component"].values) == ["K_1", "K_2"]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_write_ascii_writes_normalized_flux(small_grid, tmp_path):
    d = np.linspace(-0.5, 0.0, small_grid.n)
    path = write_ascii(tmp_path / "one.txt", small_grid, d)

    table = np.loadtxt(path)
    assert table.shape == (small_grid.n, 2)
    np.testing.assert_allclose(table[:, 0], small_grid.wave)
    # 1 + d, not d: an atmosphere code expects a spectrum, not a deviation.
    np.testing.assert_allclose(table[:, 1], 1.0 + d, rtol=1e-9)


def test_write_ascii_splits_components_and_keeps_errors(small_grid, tmp_path):
    d = np.stack([np.zeros(small_grid.n), np.full(small_grid.n, -0.2)])
    std = np.full_like(d, 0.01)
    paths = write_ascii(tmp_path / "spec.txt", small_grid, d, std)

    assert [p.name for p in paths] == ["spec_1.txt", "spec_2.txt"]
    second = np.loadtxt(paths[1])
    assert second.shape == (small_grid.n, 3)
    np.testing.assert_allclose(second[:, 1], 0.8, rtol=1e-9)
    np.testing.assert_allclose(second[:, 2], 0.01, rtol=1e-9)


def test_write_ascii_rejects_a_mismatched_grid(small_grid, tmp_path):
    with pytest.raises(ValueError, match="pixels but the grid has"):
        write_ascii(tmp_path / "bad.txt", small_grid, np.zeros(small_grid.n + 5))


def test_write_spectra_fits_round_trips(small_grid, tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    d = np.stack([np.zeros(small_grid.n), np.full(small_grid.n, -0.2)])
    std = np.full_like(d, 0.01)

    path = ab.write_spectra(
        tmp_path / "spectra.fits",
        small_grid,
        d,
        std,
        light_fractions=(0.6, 0.4),
        prior=SmoothnessPrior(tau=np.array([300.0, 300.0]), eta=np.array([5.0, 5.0])),
    )

    with fits.open(path) as hdul:
        assert hdul[0].header["NCOMP"] == 2
        # The light fractions are recorded because the recovered depths are only
        # interpretable next to them.
        assert hdul[0].header["LIGHTFR1"] == pytest.approx(0.6)
        assert hdul[0].header["TAU1"] == pytest.approx(300.0)
        assert [h.name for h in hdul[1:]] == ["COMP1", "COMP2"]
        np.testing.assert_allclose(hdul["COMP1"].data["WAVE"], small_grid.wave)
        np.testing.assert_allclose(hdul["COMP2"].data["FLUX"], 0.8, rtol=1e-9)
        np.testing.assert_allclose(hdul["COMP2"].data["ERR"], 0.01, rtol=1e-9)


def test_write_spectra_ecsv_round_trips(small_grid, tmp_path):
    table_mod = pytest.importorskip("astropy.table")
    d = np.stack([np.zeros(small_grid.n), np.full(small_grid.n, -0.2)])

    path = ab.write_spectra(tmp_path / "spectra.ecsv", small_grid, d, format="ecsv")
    table = table_mod.Table.read(path)

    assert table.colnames == ["wave", "flux_1", "flux_2"]
    assert str(table["wave"].unit) == "Angstrom"
    np.testing.assert_allclose(table["flux_2"], 0.8, rtol=1e-9)


def test_write_spectra_rejects_an_unknown_format(small_grid, tmp_path):
    pytest.importorskip("astropy.io.fits")
    with pytest.raises(ValueError, match="format must be"):
        ab.write_spectra(tmp_path / "x.txt", small_grid, np.zeros(small_grid.n), format="votable")
