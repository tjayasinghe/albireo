# ESO archive

Finding and fetching reduced spectra from the ESO Science Archive: an ObsCore/TAP client
and a resumable downloader, covering FEROS, HARPS, UVES, X-shooter, GIRAFFE and ESPRESSO
through one query language. Stdlib only — no dependency is needed to find data, only to
open it.

## BLOeM in two lines

The survey this module was built for: 929 OBAF stars in the Small Magellanic Cloud, about
25 epochs each, an intrinsic binary fraction above 70%, and 59 published double-lined
systems whose disentangling the survey team still lists as future work.

```python
import albireo as ab

star = ab.resolve_bloem("1-002")          # -> Gaia DR3 4690519082313236608, SB1, B0 IV:
records = ab.bloem_spectra(star)          # 26 LR02 epochs
ab.download(records, "data/bloem-1-002")  # ~5 MB
```

`ab.bloem_catalogue(binary_class="SB2")` returns the 59 double-lined targets, which is the
list to start from.

Two facts about this survey are not guessable and are why the resolver exists.

**The archive does not know the survey's names.** BLOeM spectra are filed under
`obs_collection='GIRAFFE'` — there is no BLOeM Phase 3 collection — and `target_name` is the
*Gaia DR3 source id*, not `1-002`. The cross-match is published in VizieR, which speaks the
same TAP dialect as ESO, so [`resolve_bloem`][albireo.archive.resolve_bloem] joins the two
with no new dependency. The source ids exceed 2<sup>53</sup> and are therefore kept as
strings: 809 of the 929 do not survive a float64 round trip.

**A second programme observes the same stars, and must not be pooled with the first.**
`112.25R7` is the survey — four sub-runs, LR02, 3960–4571 Å at *R* = 6300. `115.28A9` is a
follow-up on the same targets at *R* = 17000 and 23000 in two other windows. Passing
`programme=None` returns both, and they are usable together only as separate instruments
with their own line-spread functions. The default is the survey.

Sub-run `.004` releases through 2027-01-15, so `public_only=True` is worth passing until
then; every target already has at least one public epoch.

::: albireo.archive
