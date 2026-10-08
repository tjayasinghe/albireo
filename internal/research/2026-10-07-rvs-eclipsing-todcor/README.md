# Evidence behind the plan for the eclipsing-binary TODCOR experiment (2026-10-07)

[`plan.md`](plan.md) plans a simulation of 10^3 to 10^4 eclipsing binaries as Gaia DR4 RVS
epoch spectra, measured by `albireo.todcor` alone. It cites three notes by letter. The plan
was executed on 2026-10-07: its section 15 records what was done, the report is
`docs/reports/gaia-rvs-eclipsing-binaries.md`, and the runs are under `runs/`, which is not
in the repository.

| letter | file | content |
|---|---|---|
| A | `A_gaia_archive.md` | The scope of DR4 from the ESA pages and the draft data model; the DR3 eclipsing-binary catalogue by G_RVS, colour, temperature, period, light-curve class, eclipse depth and sky position; the per-transit velocity scatter of single stars in DR3; the non-single-star solutions |
| B | `B_eb_population.md` | The literature on eclipsing-binary samples and on the ingredients of a population: class shares, selection, rotation, circularisation, mass ratio, radii, Roche limit, third light, eclipses in spectroscopy, limb darkening, activity, the precision that masses require |
| C | `C_pilot.md` | Measurements on the code at `77a3a32`: the cost of each stage, the cause of the simulation's cost and memory growth, parallel processes, a first look at the dependence on magnitude, the orbit with a known ephemeris, and the state after the correction of 2026-10-07 |

A and B were written on 2026-10-07 by research agents working in parallel, each from a
self-contained brief. They are evidence and are kept as delivered: the register is the
agents' own, and each marks which statements were measured or read in the source and which
are recalled or constructed. C was written by the session that wrote the plan.

Limits stated by the notes themselves. A ran aggregate queries only, on the public TAP
service. Its 36 query texts and result tables are in `A_queries/` under the names the note
cites (`q*.adql`, `q*.csv`; the `t*` files are its tests of the service's ADQL dialect), and
the catalogue weights of the experiment are to be built from them. The scripts that
composed the queries and rebuilt the note's tables from the results are not kept. The
DR4 draft data model was read from a text extraction of 2026-09-03, which drops the
symbols for "at most" and "at least". The journal versions of two papers refused automated
access and their arXiv versions were read. B read about 55 papers as arXiv text; items
marked `[A]` were read as abstracts only and items marked `[R]` are recalled. Its
catalogue numbers are aggregates run on VizieR. The file paths in A and B refer to the
session's scratch directory and are not meaningful here.
