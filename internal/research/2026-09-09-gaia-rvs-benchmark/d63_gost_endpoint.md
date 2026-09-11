# The GOST ObjVisSAP endpoint as measured on 2026-09-10 (D63)

Record of the two live requests made while building `albireo.gaia.gost_transits`, and of
the decisions the implementation rests on. The fixture `tests/data/gost_ra45_dec+20.xml`
is the second response.

## What the endpoint returns

`https://gaia.esac.esa.int/gost/ObjVisSAP/gaiaobjvisap?s_ra=45.0&s_dec=20.0` answers HTTP
200 with `application/xml`, about 32 kB, a VOTable 1.3 with two resources: `results`
(`INFO QUERY_STATUS=OK`, one table of 142 rows for this position, 2014-09-03 to
2025-03-19, one row per field-of-view crossing) and a `meta` resource carrying only the
protocol parameters (`standardID ivo://ivoa.net/std/ObjVisSAP#query-0.5`). `t_min`,
`t_max` (MJD) and `MAXREC` are accepted. ObjVisSAP 22.4.3 answered.

Six fields: `t_validity` (d; the date the forecast will change, constant 60857.05),
`t_start` and `t_stop` (barycentric MJD in TCB), their ISO time stamps, and
`t_visibility` (s; constant 40.5). No CCD row, no field-of-view label, no scan angle: those
exist only in the CSV the interactive GOST form exports. The gap ladder of the scanning law
is reproduced exactly (gaps under seven hours are 1.775 to 1.781 h, the 106.5-minute
basic-angle pair; 4.227 to 4.230 h, the complement of a spin; 6.003 h, a spin).
`t_visibility` says 40.5 s while `t_stop - t_start` is 38.846 s in every row.

## Decisions

- BJD = the midpoint of `t_start` and `t_stop` plus 2400000.5, kept on TCB (Gaia's own
  scale); the midpoint is 20 s from either bound and TCB runs about 20 s ahead of TDB in
  this era, both far below the nominal law's own accuracy.
- The RVS rows (4 to 7 of 7; Cropper et al. 2018 Sect. 3.3.7) are drawn uniformly per
  transit because the service publishes no row: the number of RVS epochs is right in
  distribution, their clumping inside a visibility period is not.
- The usable fraction 0.78 (GOST's own reception rate; Katz et al. 2023 Sect. 2; 22.7
  predicted against 18 published transits for DR3) is applied as a seeded thinning after
  the release span and the row draw; the real DR3 gaps (decontaminations, refocusing, 7.8
  percent of the span) are inside that scalar rather than applied as intervals.
- The response is parsed before it is cached, so an error page is never cached; the cache
  name carries every parameter (`objvissap_ra45_dec+20_t56800-60800_n10000.vot`).
- `draw_population` draws the ecliptic longitude from a generator spawned from the seed so
  populations drawn before positions existed reproduce exactly; the longitude is kept in
  the record's metadata.
- A benchmark under `--cadence gost` makes one network call per distinct position the
  first time it sees it; no test touches the network.
