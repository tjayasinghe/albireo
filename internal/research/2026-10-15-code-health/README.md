# Division of the pipeline module and a smaller top-level namespace (2026-10-15)

An architecture survey of the package on 2026-10-15 listed five changes for code health.
Three were made the same day and are recorded in `CHANGELOG.md`. `chebyshev_response` moved
from `albireo.simulate` to `albireo.forward`, the parameter specifications moved to a new
module, `albireo.specs`, and with these two moves the import graph of the package has no
cycle (`tests/test_import_graph.py` checks it, counting imports made inside functions).
`Disentangler.expert()` returns the fixed sites as a fourth element. This note records the
other two changes, which were planned and not executed: a division of
`src/albireo/pipeline.py` into a subpackage with the same public API, and a smaller
top-level namespace. Each part ends with the decisions it leaves to the maintainer.

The counts below were made with `ast` on the source at 856f6ea plus the three changes, and
the timings on the desktop described in `docs/benchmarks.md`, idle. The scripts were run
from the session scratchpad and are not kept.

## Summary

- **The pipeline module divides into 15 modules without an import cycle.** Its 138
  top-level definitions fall into configuration, the TOML schema, input, the declaration,
  the period decision, the bootstrap, the label and velocity stages, reporting, file
  output, the per-star driver, the batch driver and the demo. With the per-star context
  (`_Context`, `_Log`) placed above the configuration classes and the JSON and version
  helpers below them, the references between the definitions form an acyclic graph.
- **The tests change by import path, and six patch targets need a helper.** Of the 62 test
  functions in `tests/test_pipeline.py`, 34 import or monkeypatch private names in their own
  bodies, 25 more reach one through the fixtures, and 3 use none. They use 41 private
  names. Six are monkeypatched, and after the division each is looked up in the module of
  its caller, which for `_orbit_over_candidates` is three modules. A test helper that
  patches every binding of a name makes the tests independent of where the callers are.
- **A smaller namespace does not make `import albireo` faster.** With compiled bytecode the
  third-party imports take 0.84 s and `import albireo` 0.93 s. Deferring `benchmark`,
  `gaia`, `population` and `pipeline` gives 0.91 s, and importing the model layer alone
  0.89 s. The reason for the change is the content of the namespace: 83 of the 209 names
  in `albireo.__all__` are test machinery, lower-level numerics, operations internal to
  the model, result containers that functions return, or the Python form of the pipeline
  configuration.
- **The namespace can be reduced without breaking code for one release.** The 83 names
  would keep resolving through the module `__getattr__` that already serves `albireo.io`
  and `albireo.plotting`, with a `DeprecationWarning` that names the module path. The
  documentation uses 2 of them as `ab.X`, the examples 9, the scripts 9, the tests 104,
  and the package's own docstrings and messages 27.

## 1. The pipeline module as a subpackage

### Present structure

`src/albireo/pipeline.py` has 5,270 lines and 138 top-level definitions, 13 of them public
(`__all__`). Its banners divide it into declarations (to line 805), the TOML schema (808 to
1061), results (1064 to 1265), one star (1268 to 4948, with the subsections stages,
assessment, describing and products), the batch (4951 to 5159) and the demo (5162 to 5270).
Two objects couple the sections: `_Context`, which 34 functions use, and `_Log`.
`_jsonable` is used by the configuration, the result records and the star driver.

### Proposed layout

| Module | Contents | Lines | Imports within the package |
|---|---|---:|---|
| `common` | `_jsonable`, `_version`, `_safe_name` | 25 | none |
| `config` | `ComponentConfig`, `Analysis`, `StarConfig`, `PipelineConfig`, the coercions `_spec` and `_lsf`, the `_describe_*` helpers of a configuration, the constants of the declaration | 660 | `common` |
| `context` | `_Log`, `_Context` | 47 | `config` |
| `schema` | `_TEMPLATE`, `config_template`, `write_config_template`, `load_config`, `config_from_dict`, `_resolve_path` | 240 | `config` |
| `records` | `StarResult`, `PipelineRun`, `_flat_row`, `_write_csv` | 165 | `common` |
| `inputs` | `_load_dataset`, `_fetch_bloem`, `_with_medium`, `_try_share_grid`, `_resolve_lsf`, `_resolve_library` and its cache, `_read_velocities`, `_checked_velocities` | 193 | `config`, `context` |
| `declare` | `_declare`, `_components`, the orbit and prior construction (`_declared_orbit`, `_table_orbit`, `_ecc_omega_specs`, `_element_starts`, `_k_starts`, `_k_prior`), `_noise_declaration`, `_noise_values` | 218 | `config`, `context`, `period` |
| `period` | `_period_candidates`, `_orbit_over_candidates`, the detection gate (`_detection_gate` and five helpers) and their constants | 422 | `context` |
| `bootstrap` | `_bootstrap`, `_library_table`, the measured light (`_measure_light`, `_apply_measured_light`, `_table_light_fractions`), the decision by the disentangling (`_decide_period_by_disentangling`, `_bootstrap_spec`, `_coarse_marginal`, `_describe_period_decision`), `_usable_nights` | 670 | `config`, `context`, `declare`, `inputs`, `measure`, `period` |
| `measure` | the label stage (`_labels`, `_assess_labels`, `_disowned_zero_points` and helpers), the templates and epoch velocities (`_templates`, `_template_light`, `_exchange_allowed`, `_measure_epoch_velocities`), `_orbit`, `_sample` | 472 | `config`, `context`, `inputs`, `period`, `report` |
| `report` | the assessments (`_assess_fit`, `_assess_table`, `_assess_orbit`, `_refuse_diverged`, `_table_failure`), the `_describe_*` blocks of `result.json`, the comparison with the injected values (`_compare_truth`, `_compare_spectra` and helpers) | 715 | `config`, `context` |
| `products` | `_write_products`, `_write_plots`, `_velocity_header`, `_write_velocity_csv`, `_write_template_table`, `_files_block` | 273 | `common`, `context`, `report` |
| `star` | `run_star`, `_run_star_guarded`, `_run_stages` | 364 | all of the above except `period` and `schema` |
| `batch` | `run_pipeline`, `_worker_init`, `_star_task`, `_WORKER_CONFIG`, `_thread_environment`, `_environment`, `_resolve_jobs`, `_star_directories` | 191 | `common`, `config`, `records`, `schema`, `star` |
| `demo` | `demo_config` | 104 | `config`, `inputs` |
| `__init__` | the present module docstring, `__all__`, and the re-export of the 13 public names | | all public names |

Lines count the definitions only. The module docstring, the imports and the comments between
definitions add about 500 lines. The import column was derived from the references between
the 138 definitions, and the graph it forms has no cycle. `context` needs `config` only for
the annotations of `_Context`. The modules could be fewer. Merging `common`, `config` and
`context` into one module, `inputs` with `declare`, and `report` with `products` gives 11
modules, also without a cycle. Not every merge is possible: `common` with `context`, or
`period` with `bootstrap`, creates one.

### The public API

`albireo/pipeline/__init__.py` re-exports the 13 public names, so `albireo.pipeline.X`,
`from albireo.pipeline import X`, `albireo/__init__.py`, `albireo/cli.py` and
`albireo/benchmark.py` are unchanged. Four properties change:

- The `__module__` of the configuration and result classes becomes the submodule
  (`albireo.pipeline.config.PipelineConfig`). Worker processes are started with `spawn` and
  unpickle `StarConfig`, `PipelineConfig` and `StarResult` by their qualified names, which
  works as long as they stay module-level. No pickle of these classes is written to disk.
- `_worker_init` sets `_WORKER_CONFIG`, which `_star_task` reads, so the three stay in one
  module (`batch`). `_LIBRARY_CACHE` stays a global of `inputs`.
- The API page (`::: albireo.pipeline`) renders the re-exported names in import order
  rather than source order. The same happened to the parameter specifications on the
  page of `albireo.facade` when they moved to `albireo.specs`.
- `tests/test_import_graph.py` reads `src/albireo/*.py`. It must read the subpackage
  (`rglob`, dotted module names), and the submodules must import one another directly,
  never through `albireo.pipeline`, or the package and its submodules form a cycle.

### The tests

| Users of private names | Names |
|---|---|
| `tests/test_pipeline.py`, 34 functions directly and 25 through the fixtures `toy`, `quick_run` and the helper `_simulate` (which call `_with_medium`, imported at module level) | 41 |
| `tests/test_benchmark.py` | `_lsf`, `_spec` |
| `tests/test_cli.py` | `_with_medium` |
| `scripts/pipeline_bench.py` | `_environment`, `_with_medium` |
| `scripts/period_decision_bench.py` | `_Context`, `_Log`, `_load_dataset`, `_resolve_lsf`, `_library_table`, `_table_light_fractions`, `_exchange_allowed`, `_period_candidates`, `_orbit_over_candidates` |

The 41 names of `tests/test_pipeline.py` would be in these modules:

| Module | Private names used by the tests |
|---|---|
| `common` | `_jsonable` |
| `context` | `_Context`, `_Log` |
| `inputs` | `_read_velocities`, `_with_medium` |
| `declare` | `_declare`, `_declared_orbit`, `_ecc_omega_specs`, `_element_starts`, `_k_prior`, `_noise_declaration`, `_noise_values`, `_table_orbit` |
| `period` | `_LEAVE_ONE_OUT_MAX_EPOCHS`, `_antiphase`, `_detection_gate`, `_detection_mask`, `_detection_thresholds`, `_in_antiphase`, `_orbit_over_candidates`, `_period_candidates` |
| `bootstrap` | `_bootstrap`, `_library_table` |
| `measure` | `_assess_labels`, `_exchange_allowed`, `_labels`, `_measure_epoch_velocities`, `_orbit`, `_template_light`, `_templates` |
| `report` | `_assess_table`, `_describe_operator`, `_describe_table`, `_exchange_rms`, `_exchanged_truth` |
| `products` | `_velocity_header`, `_write_products`, `_write_template_table`, `_write_velocity_csv` |
| `star` | `_run_stages`, `_run_star_guarded` |

Four rules move the tests:

1. **Imports name the defining submodule.** The 33 function-local imports of private names
   and the module-level import of `_with_medium` change from `albireo.pipeline` to the
   module of the table above. The change is mechanical and can be made by a script from
   that table.
2. **The package does not re-export private names.** A re-exported private name is a
   second binding. `monkeypatch.setattr(albireo.pipeline, "_period_candidates", spy)`
   would then replace a binding that no stage reads, and the test would pass or fail for a
   reason unrelated to the code under test. An import error at collection is the better
   failure.
3. **A monkeypatched name is replaced wherever a pipeline module binds it.** The 16
   `monkeypatch.setattr(pipeline_module, ...)` calls replace six names. After the division
   each name is read in the module of its caller:

   | Name | Defined in | Read in |
   |---|---|---|
   | `_declare` | `declare` | `bootstrap`, `star` |
   | `_library_table` | `bootstrap` | `bootstrap`, `star` |
   | `_orbit` | `measure` | `star` |
   | `_orbit_over_candidates` | `period` | `bootstrap`, `declare`, `measure` |
   | `_period_candidates` | `period` | `bootstrap`, `measure` |
   | `_write_products` | `products` | `star` |

   A helper in the test module replaces every binding and fails when there is none:

   ```python
   def _patch(monkeypatch, name, value):
       """Replace a pipeline function in every pipeline module that binds it."""
       modules = [
           module
           for key, module in sys.modules.items()
           if key.startswith("albireo.pipeline.") and name in vars(module)
       ]
       assert modules, f"no pipeline module binds {name}"
       for module in modules:
           monkeypatch.setattr(module, name, value)
   ```

   The three reads of `pipeline_module._name` (two keep the real function for a spy, one
   calls `_run_star_guarded`) read from the defining module. The alternative is to call
   these six functions through
   their module in the source (`period._period_candidates(...)`) and to patch the defining
   module. It constrains the style of the source for the benefit of the tests, and a caller
   that later imports the name directly would make a patch ineffective without an error.
4. **The test file is divided, if at all, in a second commit.** In the first commit
   `tests/test_pipeline.py` keeps its 62 functions and changes only in imports and patch
   calls, so that the review of the division is a review of moved code and mechanical test
   edits. The file has four sections: the declaration (11 functions, none with a private
   name in its body), the products (12, through the module-scoped `quick_run` fixture, which
   runs one star end to end), the closed loop (23, of which 19 use private names and 3 are
   slow) and the guards (16, of which 15 use private names). A division along these
   sections keeps `quick_run` in one module, so that it still runs once, and moves `library`,
   `toy`, `_simulate`, `_star` and `_context` into a shared helper module. In a run of the
   fast suite in three parallel streams that took 551 s, the file took 339 s, and a
   division would let the streams balance.

### Order of work and verification

1. Cut the definitions into the submodules by their line ranges with a script, and write
   only the import blocks by hand. No submodule holds half of the old file, so git's rename
   detection sees one deletion and 16 additions whatever the order of the moves.
   `git log -- src/albireo/pipeline.py` keeps the history of the old file, and
   `git blame -C -C` follows the moved lines.
2. Check that every top-level definition of the old module appears exactly once in the
   package with an identical `ast.dump`, which shows that the code was moved and not
   changed.
3. Extend `tests/test_import_graph.py` to the subpackage, then run `ruff check .`,
   `ruff format --check .` and the fast suite.
4. Run the three slow tests of the pipeline and `albireo demo`, and compare `result.json`
   and the velocity tables with a run at the parent commit. They must agree in every field
   except the timings and the version.
5. Run `mkdocs build --strict` and compare the anchors of the pipeline API page.

### Decisions for the maintainer

1. The granularity: 15 modules as above, or 11 with the merges listed.
2. The patch helper, or calls through the module in the source.
3. Whether to divide `tests/test_pipeline.py`, and in which commit.
4. The timing. The change moves every line of `pipeline.py`, so it conflicts with any open
   branch that modifies the module.

## 2. A smaller top-level namespace

### Present state

`albireo.__all__` lists 209 names. 191 are imported with the package from 27 modules, and
18 are resolved on first use by the module `__getattr__`, from `albireo.io` (5) and
`albireo.plotting` (13), which need the optional astropy and matplotlib. By the tiers of
the `CHANGELOG.md` preamble, 112 names come from the supported model layer, 76 from the
experimental modules, and 21 from `benchmark`, `gaia` and `population`, which belong to
no tier. `import albireo` executes 30 of the 36 submodules, among them those three and
`pipeline` (10,285 lines together). Their work at import is small: a tier table, two RVS
product grids and a main-sequence table.

The import costs, measured in fresh processes with compiled bytecode (median of nine), are:

| Import | Seconds | albireo modules loaded |
|---|---:|---:|
| `jax` (with x64), `numpyro`, `optax`, `scipy.linalg` alone | 0.84 | 0 |
| `import albireo` | 0.93 | 30 |
| the same, with `benchmark`, `gaia`, `population` and `pipeline` deferred | 0.91 | 26 |
| the same, with everything outside the model layer deferred | 0.89 | 18 |

The deferred variants were copies of the package whose `__init__` resolved the names of the
deferred modules through `__getattr__`. Import time is therefore not a reason for the
change. The reason is the content of the namespace. `dir(albireo)` and tab completion show
the entry points of the documented workflows together with operators, swaps internal to the
model and test machinery, and since every name is re-exported (as the preamble of
`CHANGELOG.md` notes), the stability tier of a name cannot be read from `ab.X`.

### Proposal

Five categories would leave the top level and stay importable from their modules:

| Category | Names | `ab.X` uses: docs / examples / scripts / tests / src |
|---|---:|---|
| A. Test machinery: all of `benchmark`, `gaia`, `population` | 21 | 2 / 0 / 2 / 7 / 0 |
| B. Lower-level numerics: all of `operators` and `kepler`, and `log_doppler_shift` | 24 | 0 / 2 / 5 / 71 / 0 |
| C. Swaps of a built `Problem`: the nine `with_*` functions and `data_residual_zscores` | 10 | 0 / 0 / 2 / 12 / 0 |
| D. Result containers that functions return and users do not construct | 21 | 0 / 7 / 0 / 14 / 27 |
| E. The Python form of the pipeline configuration | 7 | 0 / 0 / 0 / 0 / 0 |

The API overview (`docs/api/index.md`) already describes `operators` and `kepler` as
lower-level than the rest and documented for their numerics rather than as entry points,
and category B follows it. The column counts `ab.X`, `albireo.X` and
`from albireo import X`. Mentions of a name in the prose of the documentation (78 for the
83 names, 41 of them for result containers) remain correct as names and need a module path
only where they are meant as an import.

The names, by category:

- A: `BenchmarkConfig`, `BenchmarkRun`, `Tier`, `run_benchmark`, `write_report`,
  `RVS_DR3_MEAN`, `RVS_DR4_EPOCH`, `RVSProduct`, `RVSTruth`, `predict_grvs`,
  `rvs_snr_per_pixel`, `rvs_transit_times`, `simulate_rvs_dataset`, `BinarySystem`,
  `MainSequence`, `draw_population`, `from_debcat`, `from_gaia_sb2`, `query_gaia_sb2`,
  `read_population`, `write_population`.
- B: `InterpOperator`, `RebinOperator`, `bin_edges_from_centers`, `convolve_spectrum`,
  `convolve_varying`, `convolve_varying_adjoint`, `gauss_hermite_kernel_traced`,
  `gaussian_kernel`, `gaussian_kernel_traced`, `gaussian_lsf_profiles`, `interp_operator`,
  `lsf_anchor_tables`, `rebin_operator`, `rotational_kernel`, `rotational_kernel_traced`,
  `rotational_radius_for`, `shift_spectrum`, `shift_spectrum_adjoint`, `radial_velocity`,
  `solve_kepler`, `t_conj_from_t_peri`, `t_peri_from_t_conj`, `true_anomaly`,
  `log_doppler_shift`.
- C: `with_ar1`, `with_data`, `with_jitter`, `with_light_fractions`, `with_lsf`,
  `with_nebular_amplitudes`, `with_response`, `with_shifts`, `with_velocities`,
  `data_residual_zscores`.
- D: `Fit`, `Posterior`, `MAPResult`, `MarginalResult`, `LabelMatch`, `EpochStatistics`,
  `TodcorBatch`, `TodcorSurface`, `TemplateFit`, `RVOrbit`, `K2ScanResult`,
  `DetectionLimit`, `SensitivityForecast`, `SimulationTruth`, `StarResult`, `PipelineRun`,
  `BoxInterpolator`, `SimplexInterpolator`, `RawSpectrum`, `ArchiveRecord`, `BloemTarget`.
  `VelocityTable` is not among them: `fit_rv_orbit` takes one, so a user with velocities
  measured elsewhere constructs it.
- E: `Analysis`, `ComponentConfig`, `PipelineConfig`, `StarConfig`, `config_from_dict`,
  `write_config_template`, `run_star`.

The 126 names that remain are the declaration vocabulary of the `Disentangler` interface,
the data containers and the grid, reading and preprocessing, the model and inference
functions, the entry points of the stages (scan, detection limit, forecast, library, label
fit, TODCOR, template fit, orbit, hand-off, archive), saving and plotting, the example
datasets, the generators of simulated datasets, and `run_pipeline`, `load_config` and
`demo_config`.

### Mechanism

1. **Deferred loading, with no name removed.** `__getattr__` is generalized from the two
   sets it has to a table from name to module, and `benchmark`, `gaia`, `population` and
   `pipeline` are loaded through it. `import albireo` no longer executes the test machinery
   and the pipeline, and `from albireo import *` still binds every name of `__all__`. This
   step can be made independently of the others.
2. **One minor release with warnings.** `__all__` and `__dir__` list the 126 names. The 83
   others are resolved by `__getattr__` with a warning such as "albireo.rebin_operator is
   deprecated; import it from albireo.operators". A `DeprecationWarning` is shown for code
   run as a script or in a notebook and hidden for code in other modules; a
   `FutureWarning` is shown everywhere. The documentation, the examples, the scripts and
   the tests change to the module paths in the same release.
3. **Removal in the following minor release.** The module paths do not change.

Every name stays importable from its module, pickles record module paths, and the API
reference is generated per module, so none of these is affected. The sentence of the
`CHANGELOG.md` preamble on top-level re-exports would be replaced by the rule above.

Not proposed: sub-namespaces by tier or purpose (`albireo.experimental`, `albireo.sim`),
which change the documented import of every experimental name; a removal without a release
of warnings; deferred loading of every module outside the model layer, which saves 0.02 s
more than step 1.

### Decisions for the maintainer

1. Step 1 alone, or all three steps.
2. The categories. D is the closest decision: the examples use `ab.K2ScanResult` and
   `ab.SensitivityForecast` in annotations, and keeping result containers at the top level
   is common practice. C is named 13 times in the prose of `docs/benchmarks.md`,
   `docs/math.md` and the real-data tutorial, which calls `data_residual_zscores` from
   `albireo.forward` already.
3. The length of the deprecation period, and the timing relative to the JOSS submission,
   since its reviewers read the API reference.
