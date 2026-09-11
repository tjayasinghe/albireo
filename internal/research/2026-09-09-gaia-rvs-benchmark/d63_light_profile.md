# Can the marginal likelihood measure a binary's light fractions (D63, 2026-09-10)

Four D62 systems whose light stage measured the secondary's fraction badly, reproduced at
the facade level (the simulation records match the manifests to four figures; note the
seed is the index among the systems that survived `min_transits`, one behind the file's
index for field3 from index 2 on), re-declared over a grid of trial secondary fractions, and
the coarse scan model's marginal log-likelihood read as a profile over that fraction at
four orbits (injected; the archived scan start; the archived final; the orbit the new
sequential scan locates at the measured light). Scripts and raw values: the session
scratchpad `wp-j/`.

| system | true l_B | light stage | profile peak at the injected orbit | peak / true | light stage / true |
|---|---|---|---|---|---|
| gaia-17170287112790656 | 0.0673 | 0.3992 | 0.05 | 0.74 | 5.93 |
| mixed-0015 | 0.0731 | 0.1242 | 0.04 | 0.55 | 1.70 |
| mixed-0019 | 0.4242 | 0.2678 | 0.19 | 0.45 | 0.63 |
| gaia-53290511099783296 | 0.4445 | 0.2116 | 0.12 | 0.27 | 0.48 |

**The light fractions are not baked into the compiled problem.** Re-declaring at a new
fraction costs a fresh trace (0.6 to 0.9 s, the same as re-declaring at the old one), and
the model already accepts `theta["light"]` (one trace, then 0.04 to 0.07 s per evaluation);
the two paths agree bit for bit.

**What the profile measures.** The spectrum depends on component j only through
$`l_j s_j`$, and $`s_j`$ carries a Gaussian prior of precision $`\tau_j C + \eta_j I`$, so
$`(l_j, \tau_j, \eta_j)`$ and $`(c\,l_j, c^2 \tau_j, c^2 \eta_j)`$ are the same model: moving
$`l_B`$ while scaling both hyperparameters by $`c^2`$ changes the marginal by at most
4.5e-10 nats out of 17,000 to 65,000 on all four systems, while the same moves at fixed
hyperparameters change it by 11 to 409 nats. The profile exists only because the
hyperparameters are held at their starts ($`\tau = 300`$, $`\eta = 5`$ for both stars); it is
a profile over the prior amplitude ratio between the components, and the two coincide only
when both components' line structure matches the prior equally well. Against a fitted
smoothness (the ML-II the pipeline runs, hyperprior width 3 in the log) the light fraction
is unidentifiable for practical purposes: a factor two costs about 0.2 nats. The bias runs
toward a fainter secondary, worse the brighter the secondary really is: the prior gives
every component the same freedom, and the fit hands the smaller share to the component the
data constrain least. The one system whose profile lands near the truth is the one whose
components separate by ten line widths; the worst is the one whose secondary rotates at
87 km/s.

**The peaks are sharp and stable against a wrong orbit.** The curves span 30 to 409 nats
on the grid; the 5-nat interval is a factor 1.5 to 2 wide in $`l_B`$, the 25-nat interval a
factor 3 to 5; on gaia-17170287112790656 all four orbits tried, including one with a static
secondary at 19 km/s against an injected 193, peak between 0.04 and 0.07, and the light
stage's 0.40 sits 119 to 184 nats below the peak at every one of them.

**The sequence: scan at the measured light, profile at the located orbit, re-scan.**

| system | truth K | scan at the injected light | step 1 (measured light) | profile | step 3 | archived pipeline |
|---|---|---|---|---|---|---|
| gaia-17170287112790656 | 103.0, 192.7 | 100.6, 207.4 | 100.6, 16.0 | 0.05 | 100.6, 207.3 | 116.3, 19.4 |
| mixed-0015 | 12.0, 21.9 | 9.7, 7.0 | 10.3, 7.0 | 0.035 | 9.7, 7.0 | 10.1, 18.5 |
| mixed-0019 | 21.6, 23.7 | 17.0, 31.3 | 10.3, 33.1 | 0.4242 | 17.0, 31.3 | 7.7, 69.8 |
| gaia-53290511099783296 | 33.2, 35.5 | 31.1, 7.0 | 25.0, 7.0 | 0.15 | 31.1, 46.2 | 21.3, 110.7 |

(7.0 km/s is the floor of the scan's grid, a static component.) The round closes cleanly
once: on the faint secondary the scan at the measured light reproduces the archived
failure, the profile rejects the declared amplitude by 151 nats, and the re-scan returns
100.6 and 207.3 against 103.0 and 192.7 where the archived pipeline had 116.3 and 19.4; a
second profile at that orbit does not oscillate. On mixed-0019 the profile recovers the
light to the grid spacing and the re-scan converges to the coarse model's own answer rather
than to the truth; on gaia-53290511099783296 the re-scan lifts the secondary off the floor
(46 against an injected 35.5) although the profile prefers 0.15 against a true 0.44; on
mixed-0015 nothing moves at any light. Judged against the archived pipeline the sequence
improves the semi-amplitudes on three of four.

**Verdict.** Worth having between two scan passes, as an adaptation of the prior amplitude
that a companion's row is allowed, and not as a light measurement. The whole experiment
(one simulation, four scans, thirteen re-declarations, five profiles) runs in about 70 s,
and inside the scan the profile needs no re-declaration: `theta["light"]` is accepted, and
every trial after the first costs an ordinary evaluation. Implemented in D63 as
`Disentangler._profile_prior_amplitudes` and the second scan pass in `fit()`, applied to the
hyperparameter starts so the declared light is untouched.
