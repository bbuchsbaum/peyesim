# peyesim parity audit

This file tracks the Python port against the local R source at `eyesim/`.

## Objective

Port `~/code/eyesim` to idiomatic Python with full behavioral parity where the
R package exposes user-facing functionality, plus Python packaging quality
expected of a scientific library.

## Current evidence

- R `NAMESPACE` exports: 41 functions.
- Python public API now covers all R `NAMESPACE` exports via
  `tests/test_public_api_parity.py`.
- Additional Python surfaces cover R internal-but-behavioral helpers:
  `crqa`, `estimate_scale`, `match_scale`, packaged Wynn datasets, and static
  plotting helpers.
- R test files with no same-named Python test file:
  `test_fixation_entropy.R` is covered by `tests/test_entropy.py`;
  `test_multimatch_path_parity.R` is covered by
  `tests/test_multimatch_parity.py`.
- R source files with no same-named Python module:
  `all_generic.R` and `globals.R` are R-specific scaffolding;
  `data.R` is represented by `peyesim.data`;
  `eye_frame.R` is represented by `eye_table.py`;
  `transform_utils.R` is represented by visualization transform helpers.

## Verified gates

- `python -m pytest -q`
- `pytest -q`
- `python -m build --wheel --no-isolation`
- `python -m build --sdist --wheel --no-isolation`
- Wheel-installed import/data smoke check from `dist/peyesim-0.1.0-py3-none-any.whl`
- Editable install/import smoke checks.
- `python tools/r_oracle_parity.py` for deterministic cross-language checks
  against the local R package loaded with `pkgload::load_all("eyesim")`.

## Remaining parity risks

- The current R-oracle script checks core geometry, scanpath, fixation sampling,
  fixation overlap, density sampling, KDE density generation with and without
  duration weighting, scalar density similarity methods, `template_similarity()`
  workflows with and without deterministic permutation baselines, and
  `template_similarity()` through `latent_pca_transform()`,
  `coral_transform()`, `contract_transform()`, and `affine_transform()`. It also checks
  `template_similarity_cv()` row-score parity without a learned transform and
  with geometric learned transforms on a stable two-group split, plus entropy
  contracts, `template_similarity()` permutation baseline contracts including
  R's sample-then-drop matched-template order, exhaustive within-stratum
  baselines, row-order invariance, and degenerate cosine behavior,
  `fixation_similarity()`/`scanpath_similarity()` temporal-window filtering,
  manual held-out CV equivalence for grouped CORAL and contract transforms,
  plus held-out CV improvement under contract and affine distortions,
  `sample_density_time()` edge cases for custom aggregation, temporal
  forward-fill, within-subject permutation baselines, bin boundaries, empty
  bins, and sampled-series length,
  entropy contracts for point-mass/uniform/scaled densities, named multiscale
  per-sigma entropy, duration-weighted density entropy, automatic sigma
  selection, and omitted-bound padding,
  R regression-fix contracts for `eye_table()` grouped construction,
  `simulate_eye_table()` per-group onset resets, fixation-group normalization,
  `estimate_scale()` source-window filtering, non-positive density sigma
  rejection, and density-grid data-frame conversion,
  multiscale density construction from vector sigma values plus multiscale
  `template_similarity()` and `repetitive_similarity()` aggregation behavior,
  including per-scale pairwise repetitive-similarity vectors when aggregation
  is `"none"`,
  row-wise `add_scanpath()` construction for both ordinary data frames and
  `EyeTable` inputs,
  exported saccade helpers `cart2pol()` and `calcangle()` including exact
  parallel/perpendicular vector behavior,
  MultiMatch simple-path metrics, graph path/cost, translation invariance, and
  direction scale-invariance against R-reference numeric fixtures,
  `sample_density()`/`sample_density_time()` normalization modes (`none`,
  `max`, `sum`, `zscore`) including uniform/zero-density guards,
  time-sampled normalization pass-through, and normalized binned permutation
  summaries,
  `eye_density(..., kde_pkg="ks")` as the R-compatible KDE backend keyword
  accepted by the native Python implementation,
  public `center()`/fixation-group centering behavior,
  `eye_table(..., vars=...)` as the R-compatible alias for Python's
  `extra_vars=...`,
  latent-transform `scale.` compatibility through Python
  `**{"scale.": ...}` calls for direct and fit/apply transform paths,
  weak-area regressions for deterministic fixation-overlap counts,
  `suggest_sigma()` input validation/display clamp, and sequential
  `concat_fixation_groups()` onset/index behavior with invalid-input rejection,
  `template_regression(method="lm")`, robust
  `template_regression(method="rlm")`, and
  `template_multireg(method="lm"|"logistic"|"rlm")` coefficient estimates.
  When optional R packages are available, it also checks
  `template_regression(method="rank")` against `ppcor::pcor(...,
  method="spearman")` and `template_multireg(method="nnls")` against
  `nnls::nnls`; both optional paths pass in an isolated `/tmp` R library. The
  same isolated R library now exercises CRQA summaries against R `crqa::crqa`
  for the eyesim wrapper fixture, including `RR`, `DET`, line metrics,
  `LAM`, `TT`, and `RP`. Static matplotlib visualization helpers now have
  render-level Python coverage for transformed density arrays, blank axes,
  R-style 10% fixation-plot expansion, R-style duration point-size aesthetics,
  onset-colored path segments, raster nonblankness, and `anim_scanpath()`
  time-bin frame drawing. They are still not byte- or pixel-oracled against
  ggplot/gganimate output, since the rendering backends differ.
  CORAL `fit_by` grouping has Python grouped-covariance coverage plus a live
  single-stratum R oracle. Geometric `contract_transform()` and
  `affine_transform()` now use grouped `fit_by` models for direct and CV
  fit/apply paths, with Python coverage for missing strata and live
  single-stratum R oracle checks. `cca_transform()` now also uses grouped
  `fit_by` models for direct and CV fit/apply paths, with Python coverage for
  single-stratum equivalence and missing-stratum fallback, plus live R oracle
  checks against R `stats::cancor` for direct and single-stratum `fit_by`
  template-similarity cases. The oracle is
  still a smoke test, not a complete oracle for every transform,
  cross-validation, regression, and visualization path.
- Private helper parity is structural rather than exact for several optimized
  paths, especially similarity and latent-transform internals.
- Seeded CV fold membership now uses R's exact `sample(seq_len(n), n)` order
  when `Rscript` is available, with a deterministic NumPy fallback otherwise.
  The R oracle resets `RNGkind()` to the R default before CV fixtures and checks
  exact `.cv_fold` membership for the covered paths.
- Bundled Wynn datasets load through packaged Python-readable companions because
  `pyreadr` cannot parse the nested fixation-group list columns in the original
  `.rda` files. The original `.rda` files remain packaged for provenance, and
  an isolated `uv run` check exercises `load_dataset()` without `pyreadr`.
  Other optional dependency paths (`matplotlib`, third-party MultiMatch) are
  tested for availability/failure semantics, but not all optional live paths are
  exercised in the default environment.
- README and Quarto tutorial source now point at the maintained docs surface;
  the latent-transform tutorial demonstrates the current cross-fitted
  `template_similarity_cv(..., similarity_transform=...)` path rather than an
  older no-transform limitation note.
- Source distributions include the maintained docs and notebook sources while
  pruning generated Quarto cache/site output, so README links are not broken in
  the packaged artifact.
