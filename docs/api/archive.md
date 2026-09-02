# ESO archive

Finding and fetching reduced spectra from the ESO Science Archive: an ObsCore/TAP client
and a resumable downloader covering FEROS, HARPS, UVES, X-shooter, GIRAFFE and ESPRESSO
through one query language. The module depends only on the standard library; opening the
downloaded files requires astropy (see [reading spectra](io.md)).

## BLOeM

The module was written for the BLOeM survey: 929 OBAF stars in the Small Magellanic Cloud,
observed at about 25 epochs each, with an intrinsic binary fraction above 70% and 59
published double-lined systems whose disentangling the survey team lists as future work.

```python
import albireo as ab

star = ab.resolve_bloem("1-002")          # -> Gaia DR3 4690519082313236608, SB1, B0 IV:
records = ab.bloem_spectra(star)          # 26 LR02 epochs
ab.download(records, "data/bloem-1-002")  # ~5 MB
```

`ab.bloem_catalogue(binary_class="SB2")` returns the 59 double-lined targets.

Two properties of this survey cannot be inferred from the archive, which is why the
resolver is provided.

**Archive naming.** BLOeM spectra are filed under `obs_collection='GIRAFFE'` (there is no
BLOeM Phase 3 collection), and `target_name` is the Gaia DR3 source id, not the survey
identifier such as `1-002`. The cross-match is published in VizieR, which uses the same TAP
dialect as ESO, so [`resolve_bloem`][albireo.archive.resolve_bloem] joins the two without
a new dependency. The source ids exceed 2<sup>53</sup> and are kept as strings: 809 of the
929 do not survive a float64 round trip.

**Two programmes on the same targets.** `112.25R7` is the survey: four sub-runs, LR02,
3960–4571 Å at *R* = 6300. `115.28A9` is a follow-up on the same targets at *R* = 17000 and
23000 in two other windows. The two must not be pooled. Passing `programme=None` returns
both, and they can be used together only as separate instruments with their own
line-spread functions. The default is the survey programme.

Sub-run `.004` releases through 2027-01-15, so `public_only=True` is recommended until
then. Every target already has at least one public epoch.

Background and references: [science overview](../science.md).

::: albireo.archive
