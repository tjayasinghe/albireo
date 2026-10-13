# Templates fitted to the epochs

!!! warning "Experimental"

    The names and return shapes of this module may change.

[`todcor`](todcor.md) measures velocities against templates that are given, and its accuracy
is bounded by the agreement between templates and stars. `fit_templates` fits the templates
to the same epochs: the effective temperature, surface gravity and rotation of every star,
a metallicity, and the light fractions, together with the velocities of every epoch. The
templates are interpolated in a published grid ([`albireo.library`](library.md)); nothing is
synthesised. The joint optimisation of templates and epoch velocities over a multi-epoch
dataset is the approach of Nachmani, Faigler & Mazeh (2026).

On 1,200 simulated eclipsing binaries in the Gaia RVS, templates fitted from a
classification give both semi-amplitudes within 10 percent for 84 percent of the systems at
G_RVS 6 to 9, where the classified templates give 62 and the injected ones 86, and quoted
errors of the semi-amplitudes whose pulls have widths of 1.03 and 1.12 against 1.57 and
2.54. Fainter than G_RVS = 11 the noise of the transits limits the orbit and the fit
changes little. The numbers are in
[the benchmark record](../benchmarks.md#gaia-rvs-templates-fitted-to-the-epochs-2026-10-13).

```python
import albireo as ab

library = ab.fetch_library("bosz2024-fgk-rvs")
grid = ab.LogGrid.covering(dataset, 4.0, v_margin_kms=350.0, lsf_sigma_kms=11.7)
fit = ab.fit_templates(
    dataset,
    {
        "A": ab.StarLabels(library, teff=ab.Known(6200, 190), logg=ab.Known(4.2, 0.2),
                           vsini=ab.Known(45, 9)),
        "B": ab.StarLabels(library, teff=ab.Known(5500, 165), logg=ab.Known(4.4, 0.2),
                           vsini=ab.Known(30, 6)),
    },
    grid=grid,
    medium="vacuum",
    mh=ab.Known(0.0, 0.15),
    lsf_sigma_v={"rvs": 11.7},
)
print(fit.summary())
table = fit.table                      # the velocities measured with the fitted templates
orbit = ab.fit_rv_orbit(table, period=3.2)
```

## What is declared

Each star is an [`albireo.StarLabels`](match.md) with the declaration vocabulary of the
`Disentangler` interface. A float holds a label, `Between` gives it a range, and `Known`
gives it a Gaussian prior: the classification of the star with its error. The starting
templates are rendered at the starting values, so these should be those of a
classification. The metallicity is shared by the stars unless a mapping gives one per star.
`library` is one grid, or a sequence of grids that adjoin in temperature (the BOSZ boxes of
the registry, or an [`albireo.eclipsing.LibrarySet`](eclipsing.md)), which are then one label
space: a star is rendered from the box that contains its temperature and can move from one
box to the next during the fit.

The light fractions of one or two stars are fitted by default, one set per instrument,
started at the fractions the correlation measures. They are measurable beside the line
strengths because the data are normalized: the fractions are held to sum to one, so a
change of a fraction is not absorbed by the metallicity of both stars at once. A sequence
of fractions holds them, and three or more stars need them held.
Epochs at which the fractions differ from the rest, such as those in eclipse, are left out
by the caller and measured afterwards with the fitted templates and `light="free"`.

## The two steps

The model of an epoch is that of the correlation
([§10.1](../math.md#101-the-estimator)), with the templates rendered at labels that are
parameters. The shifts and the nuisance of an epoch are its own parameters, and the labels
and light fractions belong to the whole dataset
([§10.7](../math.md#107-templates-fitted-to-the-epochs)).

1. With the velocities started at those of the current table, a bounded
   Levenberg-Marquardt minimises the sum of the chi-squares of the usable epochs over the
   labels and light fractions. Every epoch's shifts are at their own minimum for each trial
   (variable projection), found locally from the current ones: the exact minimum of a pixel
   of the template grid, followed from pixel to pixel as in the correlation's own
   refinement.
2. `todcor` measures every epoch again with the fitted templates, over the whole velocity
   range.

The steps alternate until no velocity changes by more than a tenth of its error. The second
step is needed because the first follows each epoch's minimum locally, and a template far
from the star can leave an epoch in the wrong minimum of its surface.

## The order of the two stars

**At one epoch.** The correlation returns each pair of velocities in the order of the lower
chi-square. For two distinct stars that order is measured. For two alike stars the two
orders are equally deep and the noise decides between them, epoch by epoch, while the fit
reads each star's shift from its own row of the table. A table measured between two passes
is therefore compared with the one before it. An epoch is exchanged where its other order
is nearer the velocities it had and the chi-square with the two velocities exchanged
between the fitted templates is above the one returned by less than the threshold of the
correlation's blend flag: nine times the reduced chi-square, or 9 with the weights as
given. Under a declared noise correlation the threshold is multiplied by the factor by
which the correlation widens the scatter of a difference of two chi-squares, up to
$`(1 + \varphi)/(1 - \varphi)`$. Stars the epochs do not tell apart thereby keep the order
of the starting table, and a starting table in error at some epochs does not exchange two
distinct stars there. An epoch for which the starting table has no pair of velocities is
left in the order returned. A pair measured in another order than the pass was fitted with
counts as a change of the velocities, so the alternation continues.

**At every epoch.** A starting table can also have the two stars in the other order
throughout: a correlation with alike templates returns either. The labels and the light
fractions then converge to those of the other star. That solution and its image under the
exchange of the two stars have the same chi-square, and the epochs cannot tell rows in the
other order from priors given to the other star. `exchange_stars` says which of the two
names the stars. With `True`, the default, the declared priors do: where their sum is
lower at the image by more than 9, the image is fitted with the rows of the table
exchanged, and it is kept if its objective is the lower. The objectives differ beyond the
priors only by what is held to each star (a limb darkening, a held gravity or light
fraction), so for two stars with nothing held to either the priors alone decide. `notes`
records the exchange, and the table returned is then in the other order than `start`. With
`False` the starting table names the stars, and `notes` reports a label that ended more
than three prior widths from its declared value.

Where the declared priors do not tell the stars apart the order of the starting table
stays, and it is an orbit that names the stars: with a photometric ephemeris the table is
fitted in both orders of its components and the one of lower chi-square kept, as
[the simulated survey](survey.md) does.

## Errors and sensitivities

The curvature of the reduced objective gives the covariance of the fitted parameters with
the velocities of every epoch marginalised (`TemplateFit.covariance`, `errors`), rescaled by
the reduced chi-square under `errors="profiled"` as the velocity errors of the correlation
are. A declared `noise_correlation` widens them as it widens the velocity errors: the
covariance is then the sandwich through that curvature. `prior_fraction` is the ratio of
the posterior to the prior variance of each label with a Gaussian prior: a value near one
marks a label the epochs did not constrain. A parameter that ended on a bound, and a
rotation below half a pixel of the template grid, which does not change the template, are
conditioned on and have no error.

The same matrices give the derivative of every velocity with respect to every fitted
parameter, recorded on the table as `VelocityTable.sensitivity`. An error of a template
moves the velocities of every epoch together, by an amount that depends on how far apart
the lines are at that epoch, so the part of the velocity error that comes from the templates
is correlated between epochs. `VelocityTable.template_sigma` is its size per velocity, and
`VelocityTable.sigma` does not contain it. The orbit fits of [`albireo.rvorbit`](rvorbit.md)
use `sigma` alone. The numbers measured for both are in
[the benchmark record](../benchmarks.md).

## Components the library does not have

A `TemplateDeviation` is a vector added to one star's spectrum before its rotation, with an
amplitude fitted beside the labels: emission that fills line cores, for instance, which no
library spectrum has. `line_deviation` builds a vector of Gaussian profiles at given
wavelengths whose amplitude is an equivalent width in ångström.

```python
ca_triplet = (8500.35, 8544.44, 8664.52)                       # vacuum
emission = ab.TemplateDeviation(
    "emission", ab.line_deviation(grid, ca_triplet, 0.25, strengths=(0.6, 1.0, 0.8)),
    ab.Between(0.0, 3.0, start_at=0.1),
)
fit = ab.fit_templates(dataset, stars, deviations={"B": [emission]}, grid=grid, medium="vacuum")
```

## Limits

- The fit starts from a velocity table. Where the starting templates are so far from the
  stars that the table is wrong at most epochs, the fit does not recover: it follows each
  epoch's minimum locally, and the alternation re-measures with templates fitted to wrong
  velocities.
- One template for both stars leaves the order of the pair at each epoch open. Such a
  table is put in order first, by an orbit
  ([`assign_components`](rvorbit.md), or `assign_by_ephemeris` with a photometric
  ephemeris), and passed as `start`.
- Where the minimum of an epoch lies on a pixel of the template grid the chi-square has a
  kink, the pixel-locking of the linear shift operator
  ([§10.3](../math.md#103-exact-fractional-shifts-and-pixel-locking)). The shift there does
  not respond to a small change of the templates: its sensitivity is zero, and the reduced
  objective is continuous with a gradient that is not. The optimiser can end with the
  status `"stalled"` at a point that is a minimum to within that ripple.
- The exchange of the two stars is defined where their fitted parameters pair up: the same
  library boxes and deviations, and the same labels fitted for both. Otherwise a fit
  started with the stars in the other order can end with the two templates exchanged
  between them. `notes` reports a label that ended more than three prior widths from its
  declared value, and nothing else marks it.
- The order of a pair is kept from the table before wherever the exchanged pair is within
  the threshold, which at low S/N also holds for stars that differ. A pair that the
  starting table had in error can then stay in error: 3 of 435 usable epochs in a test at
  S/N 40 with starting temperatures 300 K off, and none of 442 at S/N 120 with 150 K.
- With `scale="free"` the minimum of an epoch is found by alternating its shifts and its
  scale, which at a kink of the surface can stop a few units of chi-square above the joint
  minimum.
- Two library boxes agree at the temperature they share only for the other labels both
  hold. A gravity inside one box and outside the other is moved to the edge of the box in
  use, and the template then jumps at the shared temperature.
- The errors exclude the error of the library itself: its interpolation, its line list and
  the physics it leaves out.

The theory is in [§10.7](../math.md#107-templates-fitted-to-the-epochs).

Background and references: [science overview](../science.md).

::: albireo.templatefit
