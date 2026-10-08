# Eclipsing-binary populations

!!! warning "Experimental"

    The names, the draw parameters and the keys of a system's `meta` may change.

`albireo.population` draws double-lined binaries from the field's distributions on a mean
dwarf sequence. `albireo.eclipsing` draws the detached eclipsing binaries of a catalogue such
as Gaia's `vari_eclipsing_binary`, for which four further things matter. The stars have
evolved. A system is listed in proportion to its eclipse probability and to the volume its
light reaches. An eclipse changes the light fractions of the epochs that fall in it. And
the two stars can lie in different boxes of a spectral library.

The module is used by the [correlation survey](survey.md), whose results are in the
[report](../reports/gaia-rvs-eclipsing-binaries.md).

```python
from albireo.eclipsing import LibrarySet, StellarTracks, draw_eclipsing_population, eclipse_light

systems = draw_eclipsing_population(10, seed=1)      # 10 per cell: 600 systems
system = systems[0]
light = eclipse_light(system, system.t_conj + system.period * np.linspace(0.0, 1.0, 200))
light["fractions"]      # (2, 200): the light fractions through one orbit
light["in_eclipse"]     # (200,)
```

The default draw needs the three BOSZ boxes of the RVS band in the cache
(`albireo.fetch_library`), 190 to 530 MB of downloads each. A `LibrarySet` built from other
libraries is passed as `libraries=`.

## Evolutionary tracks

`StellarTracks.load()` reads a table of 136 MIST v1.2 tracks of solar composition, from 0.10
to 6.0 solar masses, that is kept in the package (219 kB). Each track is sampled at the same
120 equivalent evolutionary points, from the start of the pre-main-sequence contraction to
the tip of the red giant branch. Such a point marks the same stage of evolution on every
track (Dotter 2016), so a quantity at a mass between two tracks is interpolated linearly in
log mass at a fixed point, the age included, and then along the resulting track in log age.
`scripts/build_mist_tracks.py` reduces the published archive to the table and compares the
two with `--check`.

A star of 1.0 solar mass has a radius of 1.03 solar radii and 5850 K at 4.57 Gyr, and
reaches the terminal-age main sequence at 9.9 Gyr.

## The geometry of an eclipse

The projected separation of the two centres at true anomaly $`\nu`$ is

```math
d = r \sqrt{1 - \sin^2 i \, \sin^2(\nu + \omega)}, \qquad
r = \frac{a (1 - e^2)}{1 + e \cos \nu},
```

and the primary is the farther star where $`\sin(\nu + \omega) > 0`$. Its superior
conjunction, $`\nu + \omega = \pi / 2`$, is the conjunction of `BinarySystem.t_conj` and of
[`RVOrbit.t_conj`](rvorbit.md).

`hidden_fraction` gives the share of the farther disc's light that the nearer disc hides,
for the linear limb-darkening law $`I(\mu) = 1 - u (1 - \mu)`$. A ring of radius $`r`$ about
the centre of the farther disc lies inside the nearer one, of radius $`R_f`$, over the
half-angle $`\alpha`$ with

```math
\cos \alpha = \frac{r^2 + d^2 - R_f^2}{2 r d}.
```

The rings with $`r \le R_f - d`$ are hidden whole and their light has a closed form. Those
with $`|d - R_f| < r < d + R_f`$ are hidden in part, and their integral is taken by
Gauss-Legendre quadrature in $`\phi`$ with $`r = c + h \sin \phi`$ between the two limits,
which removes the square-root behaviour of $`\alpha`$ at the contact radii. On 4000 random
configurations of a uniform disc the result is within $`10^{-12}`$ of the area of the
intersection of two circles, and a total eclipse gives exactly one.

`limb_darkening` holds the linear coefficients at 858 nm, interpolated between the Cousins
I and Sloan z' values of Claret & Bloemen (2011), since no published table has a G_RVS
column: 0.55 at 4000 K, 0.47 at 5750 K and 0.29 at 10,000 K.

`roche_radius` is the volume-equivalent radius of Eggleton (1983) and
`pseudo_synchronous_ratio` the rotation rate of Hut (1981) at which the tidal torque
averaged over an eccentric orbit vanishes.

## Library boxes as one label space

A published grid is downloaded in boxes, and a pair such as a 9000 K primary with a 5500 K
secondary lies in two of them. A `LibrarySet` renders each star from the box that contains
its temperature and takes the flux ratio of the two as the ratio of the continua times the
square of the radius ratio. The continua must therefore be on one scale. Those of the three
BOSZ boxes of the registry are: at the nodes the boxes share, 4000 and 7000 K, they are
equal. A label outside its box is moved to the edge and the star is flagged.

## The design sample and its weights

`draw_eclipsing_population` returns the same number of systems in every cell of a grid of
G_RVS bins by classes of the primary's effective temperature. Within a class the systems
follow the model of its docstring with no further selection. G_RVS is assigned, not derived
from a distance. `catalogue_weights` then maps the sample onto the counts of a catalogue,
by cell and, as a second set of weights, raked to the catalogue's periods in the bins the
model populates.

The companion distributions are those of Moe & Di Stefano (2017) with one change. Their
excess of twins, placed on $`0.95 < q < 1`$ with 0.30 of the companions above
$`q = 0.3`$, makes 53 percent of a magnitude-limited eclipsing sample twins, where the
compilation of Eker et al. (2018) has 30 percent and the Gaia DR3 catalogue's detached
classes at most 28 percent of equal eclipse depths. The default takes 0.3 of the published
excess, rising linearly from $`q = 0.85`$. `twin_excess=1.0, twin_from=0.95,
twin_shape="uniform"` restores the published form.

::: albireo.eclipsing
