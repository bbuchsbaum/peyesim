# peyesim parity audit

This file tracks the Python port against the R source at `eyesim/` (a symlink
to the R checkout; set `EYESIM_SOURCE_DIR` to point elsewhere).

## Objective

Port `~/code/eyesim` to idiomatic Python with full behavioral parity where the
R package exposes user-facing functionality, plus Python packaging quality
expected of a scientific library. GazeWeave (`gaze_*`,
`fit_gaze_replay_model`) is experimental in R and deliberately not ported.

## Synced to R eyesim 0.3.0 (development, October 2026)

The port was first written against eyesim at `1580ec9` (March 2026). It now
tracks eyesim `9ce29bc` (0.3.0 plus `fixation_revisits()`), excluding
GazeWeave. Changes brought over in that sync:

- **New functions:** `fixation_revisits()` (fixation groups and eye tables,
  with `null = "none" | "subject" | "item"`), `revisit_summary()`,
  `visual_angle_to_px()`, `px_to_visual_angle()`, `eyesim_colours()`,
  `theme_eyesim()`, `theme_eyesim_spatial()`, and the density/time colour
  maps (`eyesim_density_cmap()`, `eyesim_time_cmap()`, the counterparts of
  `scale_fill_eyesim_density()` and `scale_colour_eyesim_time()`).
- **Permutation baselines:** every copy of the true match is removed before
  sampling, each other template counts once, and results carry `n_perm`.
  `template_similarity_cv()` draws its permutations from `seed` and leaves
  NumPy's global random state untouched. `sample_density_time()` uses the same
  candidate rule, and its last time bin is half-open.
- **Similarity:** maps on different lattices raise `LatticeMismatchError`
  (also through multiscale maps and `repetitive_similarity()`); Fisher z
  snaps perfect correlations to one clamped value; `dcov` uses an O(n log n)
  distance correlation; EMD normalises weights and drops zero-mass cells, and
  the saliency EMD is Rubner's partial matching on raw masses;
  `fixation_overlap()` builds its default time grid from both groups and
  leaves time before the first fixation out of `perc`; the similarity overlap
  threshold defaults to 60 and needs `time_samples`.
- **Density and fixations:** `eye_density()` validates `sigma` and `kde_pkg`,
  takes explicit `weights`, returns `None` for zero weight, checks extra
  `ks::kde()` arguments, and rounds like `zapsmall(z, digits = 7)`;
  `suggest_sigma()` returns NaN without spread; entropy rejects signed maps
  and forwards `aggregate`; `sample_fixations()` orders by onset;
  `rep_fixations()` counts without floating-point loss; empty fixation groups
  work; `calcangle()` clamps; `crqa()` honours `delay`, `embed`, `rescale` and
  `metric`.
- **Transforms and regression:** CORAL applies `X Cs^-1/2 Ct^1/2`; CCA zeroes
  the columns after the canonical variates and warns for unfitted groups;
  `mm_position_emd` uses all fixations; `template_regression()` gains
  `density_var` and refuses duplicated or missing baseline keys;
  `template_multireg()` defaults to `lm`; `template_sample()` drops rows with a
  missing template or fixation group.
- **Plots:** the eyesim 0.2.0 visual system in matplotlib (see
  `peyesim.visualization` for what is approximate).

The sync also fixed long-standing divergences the new oracle exposed:
`method="jaccard"` now uses R's extended (Tanimoto) Jaccard;
`method="sinkhorn"` for fixation groups now computes R's duration-weighted
entropic OT distance in (x, y, onset) space with T4transport's Sinkhorn
iterations; `sample_density()` rounds the grid index half-to-even on R's
1-based scale and returns NaN (not an edge cell) before the first onset;
`ks`-path densities now truncate each kernel at `supp = 3.7` SDs as
`ks::kde()` does; and `crqa(rescale=...)` follows `crqa::crqa`.

## Live R parity tests

`tests/r_oracle.py` runs R code against `pkgload::load_all()` of the R source
and returns its results as JSON; each `tests/test_parity_*.py` module (and the
R-backed tests in `test_fixation_revisits.py`, `test_visual_angle.py`,
`test_visualization.py` and `test_public_api_parity.py`) runs one R script and
compares. The tests skip when `Rscript`, `pkgload`/`jsonlite` or the R source
tree are missing. Fixtures are deterministic and permutation baselines are
exhaustive, so no cross-language random stream is involved. Optional R
packages used by the oracle: `crqa`, `emdist`, `energy`, `transport`,
`T4transport`, `ppcor`, `nnls`, `MASS`, `ks`.

```bash
export EYESIM_SOURCE_DIR=/path/to/eyesim   # default: ./eyesim
python -m pytest -q                          # parity tests run when R is found
python tools/r_oracle_parity.py              # older single-script oracle
```

## Known differences from R

- **Random streams.** Permutation subsets, `simulate_eye_table()` and other
  random draws cannot reproduce R's streams; parity is checked with
  exhaustive baselines. CV fold membership still uses R's `sample()` order
  when `Rscript` is available.
- **EMD on large grids.** `emdist` stops after 500 iterations (with a
  warning) and then returns a non-optimal, larger distance. peyesim solves
  exactly; on such grids it agrees with R's exact `transport::wasserstein`,
  not with `emdist`. Where `emdist` converges the two agree to about 1e-9
  (`emdist` works in single precision).
- **Binned ks KDE.** `ks::kde()` bins by default above 500 fixations;
  peyesim always evaluates the exact estimate and raises
  `NotImplementedError` for `binned=True`.
- **Component signs.** PCA/CCA scores may differ in sign from R; parity is
  checked through sign-invariant results (similarities, covariances).
- **`estimate_scale()`** uses SciPy's L-BFGS-B on a non-smooth objective and
  agrees with R's `optim()` to about 1e-3.
- **Plots** are matplotlib, not pixel-compatible with ggplot2; label
  placement and title wrapping are simplified.
- **Kept in Python only:** `density_matrix()` (R removed its empty generic;
  the Python function has a real implementation) and the
  `wynn_study_image` dataset (R dropped it to stay within CRAN's size limit).

## Verified gates

- `python -m pytest -q` with and without R available.
- `python tools/r_oracle_parity.py`.
- `python -m build --sdist --wheel --no-isolation` and wheel/editable
  import smoke checks (from the initial port).

## Earlier audit notes

### Evidence at the initial port (eyesim 1580ec9)

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

### Gates at the initial port

- `python -m pytest -q`
- `pytest -q`
- `python -m build --wheel --no-isolation`
- `python -m build --sdist --wheel --no-isolation`
- Wheel-installed import/data smoke check from `dist/peyesim-0.1.0-py3-none-any.whl`
- Editable install/import smoke checks.
- `python tools/r_oracle_parity.py` for deterministic cross-language checks
  against the local R package loaded with `pkgload::load_all("eyesim")`.

### Parity risks noted at the initial port

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
