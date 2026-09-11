# How much does the physicality floor constrain the light? (D64, 2026-09-11)

`docs/math.md` section 5.2 lists four things that break the exact degeneracy between a
component's light fraction and its line depths, the third being the physicality floor: since
a normalised component spectrum cannot go negative, $`s_j \ge 0`$ gives
$`\ell_j \ge \max_q\left(-\ell_j d_j(q)\right)`$, a lower bound on the fraction set by the
depth of that star's own contribution to the composite. The docs described the bound as
"weak but real, and available as an optional constraint". It was measured here on the
archived products of the third run, and the accompanying claim that it is available was
checked against the source.

**It is not implemented.** Nothing in `src/albireo` imposes $`s_j \ge 0`$, bounds a light
fraction from below, or exposes such a constraint, at either the façade or the low-level
path. The sentence has been corrected, along with two others in the same section that named
an API (`light_ratio=` taking `Fixed(values)`, `Free(prior=...)` or `PerEpoch(...)`) which
the D46 façade review considered and rejected in favour of `light=` on the `Star`.

**As a bound it is real and it is never violated.** Over the 66 components of the oracle
tier of both populations, where the light is declared at the injected value so that the
identified product $`\ell_j d_j`$ is the true one, the bound reaches a median 0.608 of the
true fraction, with a 16th to 84th percentile spread of 0.355 to 0.680 and a maximum of
0.909, and it exceeds the truth on none of the 66. It is not an artefact of single noisy
pixels: recomputed at the 99.9th percentile of the depth rather than its maximum the median
moves only from 0.608 to 0.598. Because the two stellar fractions sum to one, the two floors
bracket, and for a near-equal pair the bracket is about a factor two: on
`gaia-47620265212420096` the floors of 0.368 and 0.314 give $`\ell_A \in [0.368, 0.686]`$
against a truth of 0.541, on `mixed-0014` $`[0.347, 0.671]`$ against 0.515. Set against a
profile of the marginal likelihood over the same quantity, where a factor of two costs about
0.2 nats once the smoothness hyperparameters are fitted (`d63_light_profile.md`), a factor
of two from geometry alone is the difference between unidentifiable and bounded.

**It does not detect a wrong declared light, which is what it would have to do to be useful
as a check.** On the 66 orbit-tier and blind-tier runs, where the fraction is measured by
correlation against library templates and was wrong by more than a factor 1.5 on eight
systems (`d63_light_fractions.md`), the fitted component is driven negative on only two
(`gaia-53290511099783296__orbit`, minimum normalised flux -0.406, and `mixed-0010__orbit`,
-0.977). A wrong declared light is absorbed by reshaping $`d_j`$ within the physical range
rather than by leaving it, so negativity is not a usable detector. The bound itself is
roughly invariant under the declared value, as the exact invariance requires: on
`gaia-51574864941234048` the floor on the primary is 0.549, 0.536 and 0.506 in the oracle,
orbit and blind tiers. But it excludes none of the wrong declarations. On that star the floor
implies $`\ell_B \le 0.464`$, and the declared 0.184 and the true 0.396 both lie inside it.

**And it weakens exactly where it would be needed.** The bound is a depth, so a component
whose lines the disentangling never recovered has no depth to bound with. On the same star
the floor on the secondary is 0.040, from a fitted depth of 0.216 at a declared fraction of
0.184. The systems whose light is measured worst are the systems whose secondaries are
recovered worst, and on those the constraint is vacuous.

**What is left.** The floor is worth reporting beside a declared light and is not worth
trusting as a check on one, and that is how the corrected documentation now describes it.
Two caveats bound this measurement. The depths it uses are the archived posterior means,
which `d64_posterior_smoothing.md` shows are not the injected components: the coupling
between the two stars moves an equivalent width by a median 0.952 of the archived departure,
and on `mixed-0008` the secondary's archived equivalent width has the opposite sign to the
injected one. A floor computed from a component the fit got wrong is a valid bound on
nothing in particular, which is consistent with its never being violated and rarely biting.
And imposing the floor during the fit, rather than reading it off afterwards, was not tried;
it is a different thing from measuring it, since a constraint that is slack at the optimum
changes nothing and one that is tight changes the fit.
