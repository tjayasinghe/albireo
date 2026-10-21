"""Tests for templates fitted to the epochs (``albireo.templatefit``).

A double-lined binary is simulated from a toy library at known labels, and the fit is
started from labels in error, as a classification would give them. The tests check what
the fit is used for: the labels, the rotation and the light fractions are recovered, the
velocities measured with the fitted templates are closer to the injected ones, the
recorded sensitivities are the derivatives of the measured velocities, and a component
that the library does not have is fitted with the labels.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import albireo as ab
from albireo.simulate import library_component, synthetic_library
from albireo.templatefit import TemplateDeviation, fit_templates, line_deviation

LIBRARY_WINDOW = (5100.0, 5300.0)
WINDOW = (5150.0, 5250.0)
SIM_GRID = ab.LogGrid.from_wavelength_range(5140.0, 5260.0, dv_kms=1.0)
LSF_KMS = 6.0
INJECTED = {
    "A": {"teff": 5500.0, "logg": 4.2, "mh": -0.1, "vsini": 28.0},
    "B": {"teff": 4900.0, "logg": 4.5, "mh": -0.1, "vsini": 14.0},
}
LIGHT = (0.62, 0.38)
V_RANGE = (-200.0, 200.0)
TRUTH = {
    "teff_A": 5500.0,
    "logg_A": 4.2,
    "vsini_A": 28.0,
    "teff_B": 4900.0,
    "logg_B": 4.5,
    "vsini_B": 14.0,
    "mh": -0.1,
    "light_B": 0.38,
}
TEFF = tuple(float(t) for t in range(4500, 6751, 250))
AXES = {
    "n_pix": 4000,
    "n_lines": 40,
    "logg": (3.5, 4.0, 4.5, 5.0),
    "mh": (-0.5, -0.25, 0.0, 0.25),
    "medium": "vacuum",
}


@pytest.fixture(scope="module")
def library():
    return synthetic_library(LIBRARY_WINDOW, teff=TEFF, **AXES)


def _component(library, labels, extra=None):
    """One star on the simulation grid, with ``extra`` added before its rotation."""
    from albireo.operators import _rotational_kernel_numpy

    point = {axis: labels[axis] for axis in ("teff", "logg", "mh")}
    if extra is None:
        return library_component(
            library, point, SIM_GRID, medium="vacuum", vsini_kms=labels["vsini"]
        )
    deviation = library_component(library, point, SIM_GRID, medium="vacuum", vsini_kms=0.0)
    kernel = _rotational_kernel_numpy(labels["vsini"] / SIM_GRID.dv_kms, epsilon=0.6)
    return np.convolve(deviation + extra, kernel, mode="same")


def _simulate(library, *, n_epochs=12, snr=120.0, seed=3, extra_b=None):
    rng = np.random.default_rng(seed)
    bjd = 2459000.0 + np.sort(rng.uniform(0.0, 40.0, n_epochs))
    phase = 2.0 * np.pi * (bjd - bjd[0]) / 5.3
    velocity = np.stack([12.0 + 70.0 * np.sin(phase), 12.0 - 95.0 * np.sin(phase)])
    components = [
        _component(library, INJECTED["A"]),
        _component(library, INJECTED["B"], extra_b),
    ]
    spec = {
        "a": ab.InstrumentSpec(
            wave=np.arange(WINDOW[0] + 6.0, WINDOW[1] - 6.0, 0.07), sigma_v_lsf=LSF_KMS, snr=snr
        )
    }
    dataset, _ = ab.simulate_dataset(
        SIM_GRID,
        components,
        bjd=bjd,
        instruments=spec,
        light_fractions=LIGHT,
        velocities=velocity,
        frame="barycentric",
        seed=seed,
    )
    return dataset, velocity


@pytest.fixture(scope="module")
def binary(library):
    dataset, velocity = _simulate(library)
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=260.0, lsf_sigma_kms=LSF_KMS)
    return dataset, velocity, grid


def _declared(library, *, offsets=(150.0, -0.2, 1.3), mh=0.05):
    """The stars as a classification in error declares them, with its errors as priors."""
    d_teff, d_logg, f_vsini = offsets
    stars = {}
    for sign, name in zip((1.0, -1.0), ("A", "B"), strict=True):
        truth = INJECTED[name]
        stars[name] = ab.StarLabels(
            library=library,
            teff=ab.Known(truth["teff"] + sign * d_teff, 200.0),
            logg=ab.Known(truth["logg"] + sign * d_logg, 0.3),
            vsini=ab.Known(truth["vsini"] * f_vsini, 0.4 * truth["vsini"]),
        )
    return stars, ab.Known(mh, 0.2)


def _fit(binary, library, **changes):
    dataset, _, grid = binary
    stars, mh = _declared(library)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    options.update(changes)
    stars = options.pop("stars", stars)
    return fit_templates(dataset, stars, **options)


@pytest.fixture(scope="module")
def fit(binary, library):
    return _fit(binary, library)


def test_the_labels_the_rotation_and_the_light_are_recovered(fit):
    assert fit.converged
    assert 0.8 < fit.reduced_chi2 < 1.2
    for name in ("A", "B"):
        for label, tolerance in (("teff", 60.0), ("logg", 0.12), ("mh", 0.05), ("vsini", 1.5)):
            value, error = fit.labels[name][label], fit.errors[name][label]
            assert abs(value - INJECTED[name][label]) < max(tolerance, 4.0 * error), (name, label)
            assert np.isfinite(error) and error > 0.0
    fractions = fit.light["a"]
    np.testing.assert_allclose(fractions, LIGHT, atol=0.02)
    assert fractions.sum() == pytest.approx(1.0)
    assert np.all(fit.light_errors["a"] > 0.0)
    # The epochs constrain every label far better than the classification did.
    assert max(fit.prior_fraction.values()) < 0.2
    assert fit.parameter_names[-1] == "light_B"
    assert "mh" in fit.parameter_names and "mh_A" not in fit.parameter_names


def test_the_fitted_templates_measure_the_velocities_better(binary, fit):
    _, velocity, _ = binary
    use = fit.table.good & fit.start.good
    assert use.sum() >= 8
    before = np.sqrt(np.mean((fit.start.velocity[:, use] - velocity[:, use]) ** 2))
    after = np.sqrt(np.mean((fit.table.velocity[:, use] - velocity[:, use]) ** 2))
    assert after < 0.6 * before
    pull = (fit.table.velocity[:, use] - velocity[:, use]) / fit.table.sigma[:, use]
    assert np.sqrt(np.mean(pull**2)) < 1.6
    assert fit.table.light_mode == "fitted with the templates"
    assert [t.name for t in fit.templates] == ["A", "B"]
    assert fit.template("B").meta["labels"]["teff"] == pytest.approx(fit.labels["B"]["teff"])


def test_the_sensitivities_are_the_derivatives_of_the_measured_velocities(binary, library, fit):
    dataset, _, grid = binary
    table = fit.table
    # A table of the correlation alone carries none.
    assert fit.start.sensitivity is None and fit.start.template_sigma is None
    assert table.sensitivity.shape == (2, table.n_epochs, len(fit.parameter_names))
    assert table.parameter_names == fit.parameter_names
    np.testing.assert_array_equal(table.parameter_covariance, fit.covariance)
    index = fit.parameter_names.index("vsini_A")
    step = 1.0
    pair = []
    for name in ("A", "B"):
        labels = dict(fit.labels[name])
        vsini = labels.pop("vsini") + (step if name == "A" else 0.0)
        pair.append(
            ab.Template.from_library(
                name, library, labels, grid=grid, medium="vacuum", vsini_kms=vsini
            )
        )
    moved = ab.todcor(
        dataset, pair, light=tuple(fit.light["a"]), v_range=V_RANGE, lsf_sigma_v={"a": LSF_KMS}
    )
    use = table.good & moved.good
    measured = (moved.velocity[:, use] - table.velocity[:, use]) / step
    predicted = table.sensitivity[:, use, index]
    # The derivative is of order 0.1 km/s per km/s at the blended epochs and changes sign
    # with the order of the lines; the two agree to a small fraction of that.
    assert np.max(np.abs(predicted)) > 0.02
    np.testing.assert_allclose(measured, predicted, atol=0.15 * np.max(np.abs(predicted)))
    sigma = table.template_sigma
    assert sigma.shape == table.velocity.shape
    assert np.all((sigma[:, use] > 0.0) & (sigma[:, use] < table.sigma[:, use]))
    # The metallicity, which both templates share, and the light fraction.
    for name, steps in (("mh", {"mh_step": 0.02}), ("light_B", {"light_step": 0.005})):
        other = _remeasured(dataset, grid, library, fit, **steps)
        both = table.good & other.good
        step = next(iter(steps.values()))
        measured = (other.velocity[:, both] - table.velocity[:, both]) / step
        predicted = table.sensitivity[:, both, fit.parameter_names.index(name)]
        scale = np.max(np.abs(predicted))
        assert scale > 0.0, name
        np.testing.assert_allclose(measured, predicted, atol=0.2 * scale, err_msg=name)


def test_an_exchange_of_the_pair_carries_the_sensitivities(fit):
    from albireo.rvorbit import _exchanged, _with_alternative

    table = fit.table
    swap = np.zeros(table.n_epochs, dtype=bool)
    swap[[1, 4]] = True
    exchanged = _exchanged(table, swap)
    np.testing.assert_array_equal(exchanged.sensitivity[0, 1], table.sensitivity[1, 1])
    np.testing.assert_array_equal(exchanged.sensitivity[1, 4], table.sensitivity[0, 4])
    np.testing.assert_array_equal(exchanged.sensitivity[:, 0], table.sensitivity[:, 0])
    other = _with_alternative(table, swap)
    assert np.all(np.isnan(other.sensitivity[:, swap]))
    np.testing.assert_array_equal(other.sensitivity[:, ~swap], table.sensitivity[:, ~swap])


def test_held_labels_and_held_light_are_not_fitted(binary, library):
    stars = {
        name: ab.StarLabels(
            library=library,
            teff=INJECTED[name]["teff"],
            logg=INJECTED[name]["logg"],
            vsini=ab.Between(0.0, 80.0, start_at=1.4 * INJECTED[name]["vsini"]),
        )
        for name in ("A", "B")
    }
    held = _fit(binary, library, stars=stars, mh=-0.1, light=LIGHT)
    assert held.parameter_names == ("vsini_A", "vsini_B")
    np.testing.assert_allclose(held.light["a"], LIGHT)
    np.testing.assert_array_equal(held.light_errors["a"], 0.0)
    assert held.labels["A"]["teff"] == INJECTED["A"]["teff"]
    assert np.isnan(held.errors["A"]["teff"]) and np.isnan(held.errors["B"]["mh"])
    for name in ("A", "B"):
        assert held.labels[name]["vsini"] == pytest.approx(INJECTED[name]["vsini"], abs=1.0)
    assert held.bounds["vsini_A"] == (0.0, 80.0)
    assert held.priors == {}
    assert held.table.light_mode == "fixed"
    # The held values are the ones rendered: at the injected labels the fit is as
    # good as the noise allows, and 400 K away it is not.
    assert held.reduced_chi2 < 1.1
    stars["A"] = dataclasses.replace(stars["A"], teff=INJECTED["A"]["teff"] + 400.0)
    away = _fit(binary, library, stars=stars, mh=-0.1, light=LIGHT)
    assert away.chi2 > held.chi2 + 100.0
    text = held.summary()
    assert "vsini" in text and "light fractions (a)" in text


def test_a_component_the_library_does_not_have_is_fitted_with_the_labels(library):
    lines = (5178.0, 5204.0, 5229.0)
    emission = 0.35 * line_deviation(SIM_GRID, lines, 0.3)
    dataset, _ = _simulate(library, extra_b=emission, seed=8)
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=260.0, lsf_sigma_kms=LSF_KMS)
    stars, mh = _declared(library)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    plain = fit_templates(dataset, stars, **options)
    deviation = TemplateDeviation(
        "emission", line_deviation(grid, lines, 0.3), ab.Between(0.0, 2.0, start_at=0.05)
    )
    fitted = fit_templates(dataset, stars, deviations={"B": [deviation]}, **options)
    assert "emission_B" in fitted.parameter_names
    amplitude, error = fitted.amplitudes["B"][0], fitted.amplitude_errors["B"][0]
    assert amplitude == pytest.approx(0.35, abs=max(0.05, 4.0 * error))
    assert fitted.chi2 < plain.chi2 - 100.0
    assert "A" not in fitted.amplitudes


def test_two_library_boxes_are_one_label_space(binary):
    dataset, _, grid = binary
    cool = synthetic_library(LIBRARY_WINDOW, teff=TEFF[:4], **AXES)
    warm = synthetic_library(LIBRARY_WINDOW, teff=TEFF[3:], **AXES)
    stars = {
        "A": ab.StarLabels(
            library=[warm, cool], teff=ab.Known(5650.0, 200.0), logg=4.2, vsini=28.0
        ),
        # Declared in the warm box, 550 K above the star, which is in the cool one.
        "B": ab.StarLabels(
            library=[warm, cool], teff=ab.Known(5450.0, 300.0), logg=4.5, vsini=14.0
        ),
    }
    crossed = fit_templates(
        dataset,
        stars,
        grid=grid,
        medium="vacuum",
        mh=-0.1,
        v_range=V_RANGE,
        lsf_sigma_v={"a": LSF_KMS},
    )
    assert crossed.bounds["teff_B"] == (4500.0, 6750.0)
    assert crossed.labels["B"]["teff"] == pytest.approx(INJECTED["B"]["teff"], abs=80.0)
    assert crossed.labels["A"]["teff"] == pytest.approx(INJECTED["A"]["teff"], abs=80.0)


def test_the_declarations_are_checked(binary, library):
    dataset, _, grid = binary
    stars, mh = _declared(library)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    with pytest.raises(ValueError, match="v_kms"):
        bad = dict(stars)
        bad["A"] = dataclasses.replace(stars["A"], v_kms=5.0)
        fit_templates(dataset, bad, **options)
    with pytest.raises(ValueError, match="light must be"):
        fit_templates(dataset, stars, light="free", **options)
    with pytest.raises(ValueError, match="not declared"):
        vector = np.zeros(grid.n)
        fit_templates(dataset, stars, deviations={"C": [TemplateDeviation("x", vector)]}, **options)
    with pytest.raises(ValueError, match="nothing to fit"):
        held = {
            name: ab.StarLabels(library=library, teff=5800.0, logg=4.4, vsini=10.0)
            for name in ("A", "B")
        }
        fit_templates(dataset, held, light=LIGHT, **{**options, "mh": 0.0})
    with pytest.raises(ValueError, match="outside the range"):
        outside = dict(stars)
        outside["A"] = dataclasses.replace(stars["A"], teff=7400.0)
        fit_templates(dataset, outside, **options)
    with pytest.raises(ValueError, match="medium"):
        fit_templates(dataset, stars, **{**options, "medium": "ether"})
    with pytest.raises(TypeError, match="Gaussian"):
        import numpyro.distributions as dist

        odd = dict(stars)
        odd["A"] = dataclasses.replace(
            stars["A"], teff=ab.Sampled(dist.LogNormal(8.7, 0.1), start_at=6000.0)
        )
        fit_templates(dataset, odd, **options)


def test_a_rejected_declaration_names_its_parameter(binary, library):
    """A declaration that the specification rejects is reported with the parameter.

    The error of a ``Known`` with zero width named the declaration but not the star and
    the label it was declared for.
    """
    import numpyro.distributions as dist

    dataset, _, grid = binary
    stars, mh = _declared(library)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    zero = {**stars, "B": dataclasses.replace(stars["B"], logg=ab.Known(4.3, 0.0))}
    with pytest.raises(ValueError, match=r"logg of 'B': Known\(4.3, 0.0\): sigma must be"):
        fit_templates(dataset, zero, **options)
    empty = {**stars, "A": dataclasses.replace(stars["A"], vsini=ab.Between(30.0, 10.0))}
    with pytest.raises(ValueError, match=r"vsini of 'A': Between\(30.0, 10.0\): the lower"):
        fit_templates(dataset, empty, **options)
    # numpyro does not validate the arguments of a distribution declared through Sampled.
    flat = ab.Sampled(dist.Normal(5800.0, 0.0), start_at=5800.0)
    unchecked = {**stars, "A": dataclasses.replace(stars["A"], teff=flat)}
    with pytest.raises(ValueError, match="teff of 'A': the prior width must be positive"):
        fit_templates(dataset, unchecked, **options)


def test_a_line_deviation_has_the_area_of_its_strengths():
    grid = ab.LogGrid.from_wavelength_range(5000.0, 5100.0, dv_kms=1.0)
    vector = line_deviation(grid, (5030.0, 5060.0), 0.4, strengths=(1.0, 0.5))
    wave = np.asarray(grid.wave)
    assert np.trapezoid(vector, wave) == pytest.approx(1.5, rel=1e-3)
    assert wave[np.argmax(vector)] == pytest.approx(5030.0, abs=0.02)
    with pytest.raises(ValueError, match="one value per line"):
        line_deviation(grid, (5030.0, 5060.0), 0.4, strengths=(1.0,))


def _remeasured(dataset, grid, library, fit, *, vsini_step=0.0, mh_step=0.0, light_step=0.0):
    """The velocities of the fit's epochs with one parameter of its templates moved."""
    pair = []
    for name in ("A", "B"):
        labels = dict(fit.labels[name])
        vsini = labels.pop("vsini") + (vsini_step if name == "A" else 0.0)
        labels["mh"] += mh_step
        pair.append(
            ab.Template.from_library(
                name, library, labels, grid=grid, medium="vacuum", vsini_kms=vsini
            )
        )
    light = fit.light["a"] + np.array([-light_step, light_step])
    return ab.todcor(dataset, pair, light=tuple(light), v_range=V_RANGE, lsf_sigma_v={"a": LSF_KMS})


def test_the_errors_describe_the_scatter_of_the_fits(library):
    # Six realisations of the noise: the pulls of the eight fitted parameters have unit
    # variance. An error twice too large or too small gives 0.5 or 2.
    pulls = []
    for seed in range(20, 26):
        dataset, _ = _simulate(library, seed=seed)
        grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=260.0, lsf_sigma_kms=LSF_KMS)
        stars, mh = _declared(library)
        result = fit_templates(
            dataset,
            stars,
            grid=grid,
            medium="vacuum",
            mh=mh,
            v_range=V_RANGE,
            lsf_sigma_v={"a": LSF_KMS},
        )
        sigma = np.sqrt(np.diag(result.covariance))
        for value, error, name in zip(result.values, sigma, result.parameter_names, strict=True):
            pulls.append((value - TRUTH[name]) / error)
    rms = float(np.sqrt(np.mean(np.square(pulls))))
    assert len(pulls) == 48 and 0.7 < rms < 1.4, rms


def test_a_declared_noise_correlation_widens_the_errors_of_the_labels(library):
    # Noise correlated along the pixels at a lag-one coefficient of 0.6 leaves the
    # estimator as it is and scatters it by sqrt((1 + 0.6) / (1 - 0.6)) = 2 times more.
    rng = np.random.default_rng(31)
    bjd = 2459000.0 + np.sort(rng.uniform(0.0, 40.0, 12))
    phase = 2.0 * np.pi * (bjd - bjd[0]) / 5.3
    velocity = np.stack([12.0 + 70.0 * np.sin(phase), 12.0 - 95.0 * np.sin(phase)])
    spec = {
        "a": ab.InstrumentSpec(
            wave=np.arange(WINDOW[0] + 6.0, WINDOW[1] - 6.0, 0.07), sigma_v_lsf=LSF_KMS, snr=120.0
        )
    }
    dataset, _ = ab.simulate_dataset(
        SIM_GRID,
        [_component(library, INJECTED["A"]), _component(library, INJECTED["B"])],
        bjd=bjd,
        instruments=spec,
        light_fractions=LIGHT,
        velocities=velocity,
        frame="barycentric",
        ar1_phi=0.6,
        seed=31,
    )
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=260.0, lsf_sigma_kms=LSF_KMS)
    stars, mh = _declared(library)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    white = fit_templates(dataset, stars, **options)
    declared = fit_templates(dataset, stars, noise_correlation=0.6, **options)
    np.testing.assert_allclose(declared.values, white.values, rtol=1e-6)
    ratio = np.sqrt(np.diag(declared.covariance) / np.diag(white.covariance))
    assert np.all((ratio > 1.5) & (ratio < 2.5)), ratio
    widened = declared.table.template_sigma / white.table.template_sigma
    assert np.nanmedian(widened) == pytest.approx(np.median(ratio), rel=0.3)


def test_two_instruments_have_their_own_light_fractions(library):
    rng = np.random.default_rng(41)
    bjd = 2459000.0 + np.sort(rng.uniform(0.0, 40.0, 14))
    phase = 2.0 * np.pi * (bjd - bjd[0]) / 5.3
    velocity = np.stack([12.0 + 70.0 * np.sin(phase), 12.0 - 95.0 * np.sin(phase)])
    names = ["a" if j % 2 == 0 else "b" for j in range(bjd.size)]
    second = np.array([0.38 if name == "a" else 0.50 for name in names])
    spec = {
        "a": ab.InstrumentSpec(wave=np.arange(5156.0, 5244.0, 0.07), sigma_v_lsf=6.0, snr=120.0),
        "b": ab.InstrumentSpec(wave=np.arange(5160.0, 5240.0, 0.09), sigma_v_lsf=9.0, snr=120.0),
    }
    dataset, _ = ab.simulate_dataset(
        SIM_GRID,
        [_component(library, INJECTED["A"]), _component(library, INJECTED["B"])],
        bjd=bjd,
        instruments=spec,
        epoch_instruments=names,
        light_fractions=np.stack([1.0 - second, second]),
        velocities=velocity,
        frame="barycentric",
        seed=41,
    )
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=260.0, lsf_sigma_kms=6.0)
    stars, mh = _declared(library)
    result = fit_templates(
        dataset,
        stars,
        grid=grid,
        medium="vacuum",
        mh=mh,
        v_range=V_RANGE,
        lsf_sigma_v={"a": 6.0, "b": 9.0},
    )
    assert result.parameter_names[-2:] == ("light_B[a]", "light_B[b]")
    assert result.light["a"][1] == pytest.approx(0.38, abs=0.02)
    assert result.light["b"][1] == pytest.approx(0.50, abs=0.02)
    assert result.labels["B"]["vsini"] == pytest.approx(14.0, abs=1.5)


def test_a_free_scale_is_a_parameter_of_each_epoch(binary, library, fit):
    free = _fit(binary, library, scale="free")
    # The data are normalized, so the scale is one and the labels are those of the fit
    # that holds it.
    np.testing.assert_allclose(np.sum(free.table.light, axis=0), 1.0, atol=0.02)
    for name in ("A", "B"):
        assert free.labels[name]["teff"] == pytest.approx(fit.labels[name]["teff"], abs=40.0)
        assert free.labels[name]["vsini"] == pytest.approx(fit.labels[name]["vsini"], abs=0.5)
    # One more parameter per epoch, give or take the shifts that end on a pixel of the
    # template grid, which are not counted.
    assert abs(free.n_parameters - fit.n_parameters - int(free.start.good.sum())) <= 2
    assert free.table.settings["scale"] == "free"


def test_a_single_star_is_fitted_without_a_light_fraction(library):
    rng = np.random.default_rng(51)
    bjd = 2459000.0 + np.sort(rng.uniform(0.0, 40.0, 6))
    velocity = 30.0 * np.sin(2.0 * np.pi * (bjd - bjd[0]) / 5.3)[None, :]
    spec = {"a": ab.InstrumentSpec(wave=np.arange(5156.0, 5244.0, 0.07), sigma_v_lsf=6.0, snr=80.0)}
    dataset, _ = ab.simulate_dataset(
        SIM_GRID,
        [_component(library, INJECTED["A"])],
        bjd=bjd,
        instruments=spec,
        light_fractions=(1.0,),
        velocities=velocity,
        frame="barycentric",
        seed=51,
    )
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=120.0, lsf_sigma_kms=6.0)
    star = ab.StarLabels(
        library=library,
        teff=ab.Known(5650.0, 200.0),
        logg=ab.Known(4.0, 0.3),
        vsini=ab.Known(36.0, 12.0),
    )
    single = fit_templates(
        dataset,
        {"A": star},
        grid=grid,
        medium="vacuum",
        mh=ab.Known(0.0, 0.2),
        v_range=(-100.0, 100.0),
        lsf_sigma_v={"a": 6.0},
    )
    assert single.parameter_names == ("teff_A", "logg_A", "vsini_A", "mh")
    np.testing.assert_array_equal(single.light["a"], [1.0])
    assert single.labels["A"]["vsini"] == pytest.approx(28.0, abs=1.0)
    assert single.labels["A"]["teff"] == pytest.approx(5500.0, abs=4.0 * single.errors["A"]["teff"])
    np.testing.assert_allclose(single.table.velocity[0], velocity[0], atol=1.0)


def test_a_rotation_started_below_half_a_pixel_can_leave(binary, library):
    # Below half a pixel of the template grid (1 km/s here) the rotation kernel is one
    # pixel and the template does not depend on the rotation.
    stars, _ = _declared(library)
    stars["A"] = dataclasses.replace(stars["A"], vsini=ab.Known(0.5, 15.0))
    slow = _fit(binary, library, stars=stars)
    assert slow.labels["A"]["vsini"] == pytest.approx(28.0, abs=1.5)
    assert np.isfinite(slow.errors["A"]["vsini"])
    assert 0.8 < slow.reduced_chi2 < 1.2


def test_the_weights_as_given_scale_the_errors_and_the_residuals_do_not(binary, library, fit):
    dataset, _, grid = binary
    # The same data with every error declared half as large.
    tighter = ab.Dataset(
        epochs=tuple(dataclasses.replace(e, ivar=4.0 * e.ivar) for e in dataset.epochs),
        frame=dataset.frame,
    )
    stars, mh = _declared(library)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    profiled = fit_templates(tighter, stars, **options)
    as_given = fit_templates(tighter, stars, errors="ivar", **options)
    reference = np.sqrt(np.diag(fit.covariance))
    np.testing.assert_allclose(np.sqrt(np.diag(profiled.covariance)), reference, rtol=0.05)
    np.testing.assert_allclose(np.sqrt(np.diag(as_given.covariance)), 0.5 * reference, rtol=0.1)
    assert profiled.reduced_chi2 == pytest.approx(4.0 * fit.reduced_chi2, rel=0.02)


def test_the_other_order_of_a_pair_is_found_against_the_table_before(fit):
    from albireo.rvorbit import _exchanged
    from albireo.templatefit import _nearer_exchanged, _velocity_change

    table = fit.table
    swap = np.zeros(table.n_epochs, dtype=bool)
    swap[[0, 3, 7]] = True
    exchanged = _exchanged(table, swap)
    # As the correlation returns alike stars: the same pairs, some in the other order.
    np.testing.assert_array_equal(_nearer_exchanged(exchanged, table), swap)
    assert not _nearer_exchanged(table, table).any()
    # A pair left in the other order is a change of the velocities as written: the pass
    # that follows is fitted with it, so the alternation has not converged.
    assert _velocity_change(table, table) == 0.0
    assert _velocity_change(table, exchanged) > 30.0


def test_distinct_stars_are_not_exchanged_to_follow_a_table_in_error(binary, library, fit):
    """A starting table with both velocities near the lines of one star at some epochs,
    as mismatched templates leave a blended epoch. The velocities measured with the fitted
    templates are nearer that table in the other order of the pair, and the two stars are
    distinct, so the chi-square of the two orders decides and the pair is not exchanged."""
    _, velocity, _ = binary
    start = fit.start
    wrong = np.zeros(start.n_epochs, dtype=bool)
    separation = velocity[1] - velocity[0]
    wrong[np.argsort(-np.abs(separation))[:3]] = True
    sign = np.sign(separation[wrong])
    in_error = np.array(start.velocity, dtype=np.float64)
    in_error[0, wrong] = velocity[0, wrong] + 9.0 * sign
    in_error[1, wrong] = velocity[0, wrong] - 5.0 * sign
    refit = _fit(binary, library, start=dataclasses.replace(start, velocity=in_error))
    assert refit.table.good[wrong].all()
    assert np.max(np.abs(refit.table.velocity - velocity)[:, wrong]) < 3.0
    assert "exchanged_to_keep_the_order" not in refit.table.settings
    np.testing.assert_allclose(refit.light["a"], LIGHT, atol=0.02)


def test_distinct_stars_are_not_exchanged_at_blended_epochs(library):
    """Three epochs near conjunction, where the lines of the two stars overlap, with a
    starting table in error there. A minimum followed from the exchanged velocities is the
    one returned, so the chi-square of the other order is taken at the exchanged
    velocities themselves, and the brighter star does not get the velocity of the fainter."""
    rng = np.random.default_rng(11)
    bjd = 2459000.0 + np.sort(rng.uniform(0.0, 40.0, 13))
    phase = 2.0 * np.pi * (bjd - bjd[0]) / 5.3
    velocity = np.stack([12.0 + 70.0 * np.sin(phase), 12.0 - 95.0 * np.sin(phase)])
    blended = np.zeros(bjd.size, dtype=bool)
    blended[[2, 6, 10]] = True
    velocity[:, blended] = 12.0 + np.array([[-4.0, 4.0, -5.0], [6.0, -5.0, 7.0]])
    spec = {
        "a": ab.InstrumentSpec(
            wave=np.arange(WINDOW[0] + 6.0, WINDOW[1] - 6.0, 0.07), sigma_v_lsf=LSF_KMS, snr=120.0
        )
    }
    dataset, _ = ab.simulate_dataset(
        SIM_GRID,
        [_component(library, INJECTED["A"]), _component(library, INJECTED["B"])],
        bjd=bjd,
        instruments=spec,
        light_fractions=LIGHT,
        velocities=velocity,
        frame="barycentric",
        seed=11,
    )
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=260.0, lsf_sigma_kms=LSF_KMS)
    stars, mh = _declared(library)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    start = fit_templates(dataset, stars, max_passes=1, **options).start
    sign = np.sign(velocity[1, blended] - velocity[0, blended])
    in_error = np.array(start.velocity, dtype=np.float64)
    in_error[0, blended] = velocity[0, blended] + 2.0 * sign
    in_error[1, blended] = velocity[0, blended] - 3.0 * sign
    fit = fit_templates(
        dataset, stars, start=dataclasses.replace(start, velocity=in_error), **options
    )
    assert "exchanged_to_keep_the_order" not in fit.table.settings
    # The brighter star is measured at every epoch, blended or not.
    assert np.max(np.abs(fit.table.velocity[0] - velocity[0])) < 3.0
    separated = ~blended & fit.table.good
    assert separated.sum() >= 8
    assert np.max(np.abs(fit.table.velocity - velocity)[:, separated]) < 3.0


def test_a_fit_started_with_the_stars_exchanged_ends_in_the_declared_order(binary, library, fit):
    """A starting table with the two stars exchanged at every epoch, as a correlation with
    alike templates returns it. The labels then move to those of the other star. The
    chi-square is the same with the labels, the light fractions and the rows of the table
    exchanged between the stars, and the declared priors are lower there, so the fit is
    moved to that image."""
    from albireo.rvorbit import _exchanged

    _, velocity, _ = binary
    start = _exchanged(fit.start, np.ones(fit.start.n_epochs, dtype=bool))
    refit = _fit(binary, library, start=start)
    for name in ("A", "B"):
        assert abs(refit.labels[name]["teff"] - INJECTED[name]["teff"]) < 80.0, name
        assert abs(refit.labels[name]["vsini"] - INJECTED[name]["vsini"]) < 2.0, name
    np.testing.assert_allclose(refit.light["a"], LIGHT, atol=0.02)
    use = refit.table.good
    assert use.sum() >= 8
    assert np.max(np.abs(refit.table.velocity - velocity)[:, use]) < 3.0
    assert any("exchanged" in note for note in refit.notes)
    assert refit.passes[0]["stars_exchanged"]
    # The same solution as from the table in the declared order.
    np.testing.assert_allclose(refit.values, fit.values, rtol=2e-3, atol=2e-3)
    assert not any("exchanged" in note for note in fit.notes)


def test_the_exchange_of_the_stars_needs_parameters_that_pair_up(binary, library, fit):
    """The exchange is defined for two stars with the same parameters fitted, and it is
    made only where it is asked for. What the stars differ in beside their fitted
    parameters, a limb darkening or light fractions held to them, changes the chi-square
    of the image, which is fitted and compared."""
    from albireo.rvorbit import _exchanged

    stars, _ = _declared(library)
    start = _exchanged(fit.start, np.ones(fit.start.n_epochs, dtype=bool))
    for changes in (
        # A label fitted for one star alone: no exchange is defined.
        {"stars": {**stars, "B": dataclasses.replace(stars["B"], logg=4.5)}},
        # Light fractions held in the order of the table: the image fits worse.
        {"light": (0.38, 0.62)},
        # The starting table names the stars.
        {"exchange_stars": False},
    ):
        refit = _fit(binary, library, start=start, max_passes=1, **changes)
        assert not refit.passes[0]["stars_exchanged"], changes
        assert not any("exchanged" in note for note in refit.notes), changes
    # Left with the table's names, the labels end at those of the other star, and the
    # notes say how far that is from the declaration.
    assert abs(refit.labels["A"]["teff"] - INJECTED["B"]["teff"]) < 80.0
    assert any("prior widths" in note for note in refit.notes)
    refit = _fit(binary, library, start=start, epsilon={"A": 0.6, "B": 0.4})
    assert refit.passes[0]["stars_exchanged"]
    assert abs(refit.labels["A"]["teff"] - INJECTED["A"]["teff"]) < 80.0
    # Light fractions held in the declared order: the image fits better, and is taken.
    refit = _fit(binary, library, start=start, light=LIGHT)
    assert any(record["stars_exchanged"] for record in refit.passes)
    assert abs(refit.labels["A"]["teff"] - INJECTED["A"]["teff"]) < 80.0


def test_alike_stars_keep_the_order_of_the_starting_table(library):
    """Two stars of the same labels and light: the correlation returns each pair in the
    order of the lower chi-square, which the noise decides, and the fit keeps the order of
    the table it started from."""
    from albireo.rvorbit import _exchanged

    labels = {"teff": 5500.0, "logg": 4.2, "mh": -0.1, "vsini": 20.0}
    rng = np.random.default_rng(5)
    bjd = 2459000.0 + np.sort(rng.uniform(0.0, 40.0, 14))
    phase = 2.0 * np.pi * (bjd - bjd[0]) / 5.3
    velocity = np.stack([12.0 + 80.0 * np.sin(phase), 12.0 - 80.0 * np.sin(phase)])
    spec = {
        "a": ab.InstrumentSpec(
            wave=np.arange(WINDOW[0] + 6.0, WINDOW[1] - 6.0, 0.07), sigma_v_lsf=LSF_KMS, snr=60.0
        )
    }
    dataset, _ = ab.simulate_dataset(
        SIM_GRID,
        [_component(library, labels), _component(library, labels)],
        bjd=bjd,
        instruments=spec,
        light_fractions=(0.5, 0.5),
        velocities=velocity,
        frame="barycentric",
        seed=5,
    )
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=260.0, lsf_sigma_kms=LSF_KMS)
    stars = {
        name: ab.StarLabels(
            library=library,
            teff=ab.Known(5500.0 + offset, 150.0),
            logg=4.2,
            vsini=ab.Known(20.0, 4.0),
        )
        for name, offset in (("A", 40.0), ("B", -40.0))
    }
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": -0.1,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    first = fit_templates(dataset, stars, max_passes=1, **options).start
    separated = np.abs(velocity[1] - velocity[0]) > 40.0
    assert separated.sum() >= 8

    def other_order(table):
        direct = np.abs(table.velocity - velocity).sum(axis=0)
        return np.abs(table.velocity[::-1] - velocity).sum(axis=0) < direct

    start = _exchanged(first, other_order(first))
    fit = fit_templates(dataset, stars, start=start, **options)
    use = separated & fit.table.good
    assert use.sum() >= 8
    assert not other_order(fit.table)[use].any()
    # The correlation alone, with the fitted templates, does not keep it.
    alone = ab.todcor(
        dataset,
        fit.templates,
        light=tuple(fit.light["a"]),
        v_range=V_RANGE,
        lsf_sigma_v={"a": LSF_KMS},
    )
    assert other_order(alone)[use].any()


def _pair(library, first, second, *, light, k, n_epochs, snr, seed, ar1_phi=0.0, frame=None):
    """A double-lined binary of two given stars; ``(dataset, velocity, grid)``."""
    rng = np.random.default_rng(seed)
    bjd = 2459000.0 + np.sort(rng.uniform(0.0, 40.0, n_epochs))
    phase = 2.0 * np.pi * (bjd - bjd[0]) / 5.3
    velocity = np.stack([12.0 + k[0] * np.sin(phase), 12.0 - k[1] * np.sin(phase)])
    spec = {
        "a": ab.InstrumentSpec(
            wave=np.arange(WINDOW[0] + 6.0, WINDOW[1] - 6.0, 0.07), sigma_v_lsf=LSF_KMS, snr=snr
        )
    }
    dataset, _ = ab.simulate_dataset(
        SIM_GRID,
        [_component(library, first), _component(library, second)],
        bjd=bjd,
        instruments=spec,
        light_fractions=light,
        velocities=velocity,
        frame=frame or "barycentric",
        ar1_phi=ar1_phi,
        seed=seed,
    )
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=290.0, lsf_sigma_kms=LSF_KMS)
    return dataset, velocity, grid


def _in_the_other_order(table, velocity):
    direct = np.abs(table.velocity - velocity).sum(axis=0)
    return np.abs(table.velocity[::-1] - velocity).sum(axis=0) < direct


def _options(grid, mh):
    return {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }


def _alike(library):
    """Two stars declared 80 K apart, for a pair of identical ones at 5500 K."""
    return {
        name: ab.StarLabels(
            library=library,
            teff=ab.Known(5500.0 + offset, 150.0),
            logg=4.2,
            vsini=ab.Known(20.0, 4.0),
        )
        for name, offset in (("A", 40.0), ("B", -40.0))
    }


def test_a_pass_that_changes_the_order_of_an_epoch_has_not_converged(library):
    """Two stars 100 K apart and a starting table with the pair exchanged at three epochs.
    The first pass is fitted with those epochs in the other order. The table measured
    after it has them in the order of the lower chi-square, which the fit has not read
    yet: the alternation goes on, and ends where the fit from the table in order ends."""
    from albireo.rvorbit import _exchanged

    first = {"teff": 5550.0, "logg": 4.3, "mh": -0.1, "vsini": 20.0}
    second = {"teff": 5450.0, "logg": 4.3, "mh": -0.1, "vsini": 20.0}
    dataset, velocity, grid = _pair(
        library, first, second, light=(0.52, 0.48), k=(70.0, 80.0), n_epochs=24, snr=60.0, seed=6
    )
    stars = {
        name: ab.StarLabels(
            library=library,
            teff=ab.Known(labels["teff"], 150.0),
            logg=4.3,
            vsini=ab.Known(20.0, 4.0),
        )
        for name, labels in (("A", first), ("B", second))
    }
    options = _options(grid, -0.1)
    measured = fit_templates(dataset, stars, max_passes=1, **options).start
    start = _exchanged(measured, _in_the_other_order(measured, velocity))
    separation = np.where(start.good, np.abs(velocity[1] - velocity[0]), 0.0)
    wrong = np.zeros(start.n_epochs, dtype=bool)
    wrong[np.argsort(-separation)[:3]] = True
    reference = fit_templates(dataset, stars, start=start, **options)
    refit = fit_templates(dataset, stars, start=_exchanged(start, wrong), **options)
    assert reference.converged and refit.converged
    # The table is put right ...
    assert not _in_the_other_order(refit.table, velocity)[wrong].any()
    # ... and the labels and the light fraction are those of the table put right.
    sigma = np.sqrt(np.diag(reference.covariance))
    assert np.max(np.abs(refit.values - reference.values) / sigma) < 0.3
    assert refit.chi2 == pytest.approx(reference.chi2, abs=5.0)


def test_alike_stars_keep_the_order_under_a_declared_noise_correlation(library):
    """Noise correlated along the pixels scatters the difference of two chi-squares more,
    up to 9 times at a lag-one coefficient of 0.8, and the two orders of two identical
    stars then differ by more than 9 times the reduced chi-square at one epoch in ten.
    The threshold follows the declared correlation."""
    from albireo.rvorbit import _exchanged

    labels = {"teff": 5500.0, "logg": 4.2, "mh": -0.1, "vsini": 20.0}
    stars = _alike(library)
    lost = total = 0
    for seed in (0, 1, 2):
        dataset, velocity, grid = _pair(
            library,
            labels,
            labels,
            light=(0.5, 0.5),
            k=(80.0, 80.0),
            n_epochs=24,
            snr=60.0,
            seed=seed,
            ar1_phi=0.8,
        )
        options = {**_options(grid, -0.1), "noise_correlation": 0.8}
        measured = fit_templates(dataset, stars, max_passes=1, **options).start
        start = _exchanged(measured, _in_the_other_order(measured, velocity))
        result = fit_templates(dataset, stars, start=start, **options)
        use = result.table.good & (np.abs(velocity[1] - velocity[0]) > 40.0)
        lost += int(_in_the_other_order(result.table, velocity)[use].sum())
        total += int(use.sum())
    assert total >= 40
    assert lost == 0, (lost, total)


def test_alike_stars_keep_the_order_whatever_the_frame_and_the_scale_of_the_weights(library):
    """The threshold of the exchange is 9 times the reduced chi-square under
    ``errors="profiled"``, so errors declared half as large change nothing, and the shifts
    the two orders are compared at carry the barycentric term of a topocentric dataset.
    The largest rise of an exchanged epoch is a third of the threshold here, so a
    threshold of 9 that did not follow the reduced chi-square of 4 would leave that epoch."""
    from albireo.rvorbit import _exchanged

    labels = {"teff": 5500.0, "logg": 4.2, "mh": -0.1, "vsini": 20.0}
    dataset, velocity, grid = _pair(
        library,
        labels,
        labels,
        light=(0.5, 0.5),
        k=(80.0, 80.0),
        n_epochs=16,
        snr=60.0,
        seed=8,
        frame="topocentric",
    )
    assert np.ptp(dataset.v_bary) > 20.0
    tighter = ab.Dataset(
        epochs=tuple(dataclasses.replace(e, ivar=4.0 * e.ivar) for e in dataset.epochs),
        frame=dataset.frame,
    )
    stars = _alike(library)
    options = _options(grid, -0.1)
    measured = fit_templates(tighter, stars, max_passes=1, **options).start
    start = _exchanged(measured, _in_the_other_order(measured, velocity))
    result = fit_templates(tighter, stars, start=start, **options)
    use = result.table.good & (np.abs(velocity[1] - velocity[0]) > 40.0)
    assert use.sum() >= 8
    assert result.reduced_chi2 == pytest.approx(4.0, rel=0.15)
    assert result.table.settings.get("exchanged_to_keep_the_order", 0) >= 2
    assert not _in_the_other_order(result.table, velocity)[use].any()


def test_an_image_that_fits_worse_is_not_taken(binary, library, fit, monkeypatch):
    """The declarations of the two stars interchanged over a table in order, with the
    gravities held at those of the stars in the table's rows. The priors are lower at the
    image and its chi-square is higher by the two gravities in error, so the image is
    fitted, compared and left, and it is not tried again in the passes that follow."""
    import sys

    module = sys.modules["albireo.templatefit"]
    minimise = module._levenberg_marquardt
    runs = []

    def counted(*args, **kwargs):
        runs.append(minimise(*args, **kwargs))
        return runs[-1]

    monkeypatch.setattr(module, "_levenberg_marquardt", counted)
    stars, _ = _declared(library)
    crossed = {
        "A": dataclasses.replace(stars["B"], logg=INJECTED["A"]["logg"]),
        "B": dataclasses.replace(stars["A"], logg=INJECTED["B"]["logg"]),
    }
    refit = _fit(binary, library, stars=crossed, start=fit.start, max_passes=2)
    # The image was fitted once, in the first pass, and its objective is the higher one.
    assert len(runs) == 3 and runs[1].objective > runs[0].objective + 100.0
    assert not any(record["stars_exchanged"] for record in refit.passes)
    assert not any("exchanged" in note for note in refit.notes)
    for name in ("A", "B"):
        assert abs(refit.labels[name]["teff"] - INJECTED[name]["teff"]) < 80.0, name
    np.testing.assert_allclose(refit.light["a"], LIGHT, atol=0.02)
    # The declarations are 3.7 prior widths from the stars, and the notes say so.
    assert any("teff_A ended" in note for note in refit.notes)


def test_the_exchange_of_the_stars_takes_every_kind_of_parameter(library):
    """A metallicity per star, a deviation on both stars and a light fraction per
    instrument: started with the stars exchanged, the fit ends where the fit from the
    table in order ends."""
    from albireo.rvorbit import _exchanged

    rng = np.random.default_rng(41)
    bjd = 2459000.0 + np.sort(rng.uniform(0.0, 40.0, 14))
    phase = 2.0 * np.pi * (bjd - bjd[0]) / 5.3
    velocity = np.stack([12.0 + 70.0 * np.sin(phase), 12.0 - 95.0 * np.sin(phase)])
    names = ["a" if j % 2 == 0 else "b" for j in range(bjd.size)]
    second = np.array([0.38 if name == "a" else 0.50 for name in names])
    spec = {
        "a": ab.InstrumentSpec(wave=np.arange(5156.0, 5244.0, 0.07), sigma_v_lsf=6.0, snr=120.0),
        "b": ab.InstrumentSpec(wave=np.arange(5160.0, 5240.0, 0.09), sigma_v_lsf=9.0, snr=120.0),
    }
    dataset, _ = ab.simulate_dataset(
        SIM_GRID,
        [_component(library, INJECTED["A"]), _component(library, INJECTED["B"])],
        bjd=bjd,
        instruments=spec,
        epoch_instruments=names,
        light_fractions=np.stack([1.0 - second, second]),
        velocities=velocity,
        frame="barycentric",
        seed=41,
    )
    grid = ab.LogGrid.covering(dataset, 2.0, v_margin_kms=260.0, lsf_sigma_kms=6.0)
    stars, _ = _declared(library)
    vector = line_deviation(grid, (5178.0, 5204.0, 5229.0), 0.3)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": {"A": ab.Known(0.05, 0.2), "B": ab.Known(0.0, 0.2)},
        "deviations": {
            name: [TemplateDeviation("emission", vector, ab.Between(0.0, 2.0, start_at=0.05))]
            for name in ("A", "B")
        },
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": 6.0, "b": 9.0},
    }
    reference = fit_templates(dataset, stars, **options)
    assert not any(record["stars_exchanged"] for record in reference.passes)
    start = _exchanged(reference.start, np.ones(reference.start.n_epochs, dtype=bool))
    refit = fit_templates(dataset, stars, start=start, **options)
    assert refit.passes[0]["stars_exchanged"]
    assert sum(record["stars_exchanged"] for record in refit.passes) == 1
    assert refit.parameter_names == reference.parameter_names
    assert {"mh_A", "mh_B", "emission_A", "emission_B", "light_B[a]", "light_B[b]"} <= set(
        refit.parameter_names
    )
    sigma = np.sqrt(np.diag(reference.covariance))
    np.testing.assert_allclose(refit.values / sigma, reference.values / sigma, atol=0.05)
    assert refit.light["b"][1] == pytest.approx(0.50, abs=0.02)
    use = refit.table.good & reference.table.good
    np.testing.assert_allclose(
        refit.table.velocity[:, use], reference.table.velocity[:, use], atol=0.05
    )
    np.testing.assert_allclose(
        refit.table.sensitivity[:, use], reference.table.sensitivity[:, use], atol=1e-3
    )


def test_more_declarations_are_checked(binary, library):
    dataset, _, grid = binary
    stars, mh = _declared(library)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    vector = line_deviation(grid, (5204.0,), 0.3)
    with pytest.raises(ValueError, match="mh names star"):
        fit_templates(dataset, stars, **{**options, "mh": {"a": -0.1, "b": -0.1}})
    with pytest.raises(ValueError, match="epsilon has no entry"):
        fit_templates(dataset, stars, epsilon={"A": 0.6}, **options)
    with pytest.raises(ValueError, match="names of labels"):
        fit_templates(
            dataset, stars, deviations={"B": [TemplateDeviation("light", vector)]}, **options
        )
    with pytest.raises(ValueError, match="two fitted parameters are named"):
        pair = [TemplateDeviation("emission", vector), TemplateDeviation("emission", vector)]
        fit_templates(dataset, stars, deviations={"B": pair}, **options)
    with pytest.raises(ValueError, match="must adjoin"):
        low = synthetic_library(LIBRARY_WINDOW, teff=TEFF[:2], **AXES)
        high = synthetic_library(LIBRARY_WINDOW, teff=TEFF[4:], **AXES)
        gapped = dict(stars)
        gapped["B"] = dataclasses.replace(stars["B"], library=[low, high])
        fit_templates(dataset, gapped, **options)
    with pytest.raises(ValueError, match="one or two stars"):
        three = {**stars, "C": stars["B"]}
        fit_templates(dataset, three, **options)
    with pytest.raises(ValueError, match="start has the components"):
        fit_templates(
            dataset,
            {"P": stars["A"], "S": stars["B"]},
            start=_fit(binary, library).start,
            **options,
        )


def _without_mh(library, mh=0.0):
    """The library at one metallicity, as a grid computed at a fixed [M/H] without that axis."""
    keep = library.nodes[:, 2] == mh
    return ab.SpectralLibrary(
        label_names=("teff", "logg"),
        nodes=library.nodes[keep][:, :2],
        normalized=library.normalized[keep],
        log_continuum=library.log_continuum[keep],
        wave=library.wave,
        medium=library.medium,
        meta=dict(library.meta),
    )


def test_a_library_box_is_checked_before_it_is_used(binary, library):
    """A box without a temperature axis, and an object that is not a library, are named as
    such rather than failing where the boxes are put in order of temperature."""
    dataset, _, grid = binary
    stars, mh = _declared(library)
    options = {
        "grid": grid,
        "medium": "vacuum",
        "mh": mh,
        "v_range": V_RANGE,
        "lsf_sigma_v": {"a": LSF_KMS},
    }
    keep = library.nodes[:, 0] == 5500.0
    no_teff = ab.SpectralLibrary(
        label_names=("logg", "mh"),
        nodes=library.nodes[keep][:, 1:],
        normalized=library.normalized[keep],
        log_continuum=library.log_continuum[keep],
        wave=library.wave,
        medium=library.medium,
    )
    with pytest.raises(ValueError, match="without a teff axis"):
        fit_templates(
            dataset, {**stars, "B": dataclasses.replace(stars["B"], library=no_teff)}, **options
        )
    with pytest.raises(TypeError, match="must be a SpectralLibrary"):
        fit_templates(
            dataset, {**stars, "B": dataclasses.replace(stars["B"], library=None)}, **options
        )


def test_a_star_whose_library_has_no_metallicity_takes_no_part_in_it(binary, library):
    """The metallicity shared by the stars is a label of those whose library has that axis.
    A star rendered from a grid at one fixed metallicity reports none, and no error."""
    stars, _ = _declared(library)
    stars["B"] = dataclasses.replace(stars["B"], library=_without_mh(library))
    mixed = _fit(binary, library, stars=stars, max_passes=1)
    assert "mh" in mixed.parameter_names
    assert np.isfinite(mixed.labels["A"]["mh"]) and np.isfinite(mixed.errors["A"]["mh"])
    assert np.isnan(mixed.labels["B"]["mh"]) and np.isnan(mixed.errors["B"]["mh"])
    assert np.isnan(mixed.template("B").meta["labels"]["mh"])
    assert np.isfinite(mixed.labels["B"]["teff"]) and np.isfinite(mixed.errors["B"]["teff"])


def test_the_derivative_in_temperature_stays_in_the_box_of_the_star(binary):
    """Within a step below the shared temperature of two boxes, the difference in temperature
    is taken in the star's own box. The upper box here starts at a higher gravity, so a
    template rendered from it at the star's gravity is moved to its edge, and a difference
    between the two boxes is a difference of gravity, not of temperature."""
    import jax.numpy as jnp

    from albireo.templatefit import _LabelSpace, _Layout, _Objective, _Parameter, _Star, _State

    dataset, _, grid = binary
    cool = synthetic_library(LIBRARY_WINDOW, teff=TEFF[:4], **AXES)
    warm = synthetic_library(LIBRARY_WINDOW, teff=TEFF[3:], **{**AXES, "logg": (4.0, 4.5, 5.0)})
    space = _LabelSpace([warm, cool], grid, "vacuum", "test")
    star = _Star("A", space, grid, 0.6, 0.0, [])
    labels = {"logg": 3.7, "mh": 0.0, "vsini": 10.0}
    own = ab.library_interpolator(cool.resampled_to(grid, medium="vacuum"))

    def rendered(teff):
        return np.asarray(own(jnp.asarray([teff, labels["logg"], labels["mh"]]))[0]) - 1.0

    seam = TEFF[3]
    for teff in (seam - 50.0, seam - 2.0):
        layout = _Layout()
        layout.add(_Parameter("teff_A", "label", (0,), "teff"), teff, TEFF[0], TEFF[-1], None)
        held = _State(
            labels=[{"teff": teff, **labels}], amplitudes=[np.zeros(0)], light=np.ones((1, 1))
        )
        objective = _Objective(
            dataset,
            [star],
            layout,
            held,
            groups=("a",),
            shifts=np.zeros((1, dataset.n_epochs)),
            scales=np.ones(dataset.n_epochs),
            correlation={},
            lsf_sigma_v={"a": LSF_KMS},
            lsf_anchors_angstrom=None,
            nuisance_order=0,
            free_scale=False,
        )
        phi = np.array([teff])
        vectors, _, slots = objective._vectors(phi, layout.unpack(phi, held), jacobian=True)
        up, down = min(teff + 5.0, seam), teff - 5.0
        expected = star.broaden((rendered(up) - rendered(down)) / (up - down), labels["vsini"])
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(vectors[slots[(0, 0)]], expected, rtol=0, atol=1e-9 * scale)


def test_a_free_scale_epoch_returns_the_residual_at_its_shifts(binary, library, monkeypatch):
    """With the scale free, the walk of an epoch's shifts and the solution for its scale
    alternate until the scale settles, for at most a fixed number of rounds. An epoch that
    reaches that number returns the residual at the shifts it returns, and the scale
    solved there."""
    import sys

    module = sys.modules["albireo.templatefit"]
    monkeypatch.setattr(module, "_MAX_SCALE_ROUNDS", 1)
    epoch = module._Objective._epoch
    seen = []

    def recorded(self, stack, work, static, shift, scale, light):
        out = epoch(self, stack, work, static, shift, scale, light)
        if self.move and len(seen) < 24:
            seen.append((self, stack, work, static, light, out))
        return out

    monkeypatch.setattr(module._Objective, "_epoch", recorded)
    _fit(binary, library, scale="free", max_passes=1, max_iterations=1)
    assert len(seen) == 24
    for objective, stack, work, static, light, out in seen:
        shift, scale, residual = out[0], out[1], out[2]
        objective.move = False  # the epoch at the shifts as given, with its scale solved
        again = epoch(objective, stack, work, static, shift, scale, light)
        objective.move = True
        assert again[1] == pytest.approx(scale, rel=1e-11)
        assert float(again[2] @ again[2]) == pytest.approx(float(residual @ residual), rel=1e-11)


def test_stars_on_one_library_share_its_projection_and_epochs_are_prepared_once(
    binary, library, monkeypatch
):
    """Two stars on one library render from one projection of it, which the kept copies of
    a library do not provide for a writable one or an optical grid, and the work of an
    epoch is prepared once for every objective of the fit."""
    import sys

    module = sys.modules["albireo.templatefit"]
    built, prepared = [], []
    interpolator, prepare = module.library_interpolator, module._prepare_epoch

    def counted_interpolator(lib, **options):
        built.append(lib)
        return interpolator(lib, **options)

    def counted_prepare(index, *args):
        prepared.append(index)
        return prepare(index, *args)

    monkeypatch.setattr(module, "library_interpolator", counted_interpolator)
    monkeypatch.setattr(module, "_prepare_epoch", counted_prepare)
    # synthetic_library returns read-only arrays, which the library caches keep between
    # calls; a writable copy of one array leaves nothing of the library kept.
    library = dataclasses.replace(library, normalized=np.array(library.normalized))
    assert library.normalized.flags.writeable
    _fit(binary, library, max_passes=2)
    assert len(built) == 1
    assert len(prepared) == len(set(prepared)) <= binary[0].n_epochs


def test_an_accepted_trial_point_takes_the_epochs_of_its_trial(binary, library, monkeypatch):
    """The Levenberg-Marquardt evaluates a trial point without the Jacobian, and with it
    where it accepts the point. The second evaluation takes the epochs of the first rather
    than following each epoch's minimum again, and its result is the same bit for bit."""
    import sys

    module = sys.modules["albireo.templatefit"]
    minimise, objectives = module._levenberg_marquardt, []

    def kept(objective, *args, **kwargs):
        objectives.append(objective)
        return minimise(objective, *args, **kwargs)

    monkeypatch.setattr(module, "_levenberg_marquardt", kept)
    fit = _fit(binary, library, max_passes=1)
    objective, phi = objectives[0], np.array(fit.values)
    epoch, calls = module._Objective._epoch, []

    def counted(self, *args):
        calls.append(1)
        return epoch(self, *args)

    monkeypatch.setattr(module._Objective, "_epoch", counted)
    objective.value(phi)
    n_epochs = len(calls)
    reused = objective.evaluate(phi, jacobian=True)
    assert len(calls) == n_epochs > 0
    objective._epochs_at = None
    again = objective.evaluate(phi, jacobian=True)
    assert len(calls) == 2 * n_epochs
    for key in ("chi2", "shifts", "scales", "gradient", "hessian", "scatter", "sensitivity"):
        assert np.array_equal(reused[key], again[key]), key
