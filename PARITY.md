# Parity with R eyesim

peyesim is a Python port of the R package
[eyesim](https://github.com/bbuchsbaum/eyesim). This file records which eyesim
it matches, how that is checked, and every known difference. GazeWeave
(`gaze_*`, `fit_gaze_replay_model`) is experimental in R and deliberately not
ported.

## Reference version

peyesim tracks eyesim **0.3.0 at commit `9ce29bc`** (0.3.0 plus
`fixation_revisits()`). CI pins that commit (`EYESIM_REF` in
`.github/workflows/tests.yml`); move it in its own commit when peyesim is
synced to a newer eyesim.

## How parity is checked

`tests/r_oracle.py` runs an R script against `pkgload::load_all()` of the
eyesim source and returns the results as JSON. Every `tests/test_parity_*.py`
module, and the R-backed tests in a few other modules, compare peyesim with
those results. Three kinds of check run there:

- **Fixture tests.** Deterministic inputs for each exported function and its
  edge cases (empty and single-fixation groups, degenerate maps, missing keys,
  strata of size one).
- **Randomized differential tests.** `test_parity_random_core.py` (314 cases)
  and `test_parity_random_similarity.py` draw inputs from a fixed NumPy seed
  and send them to R as exact literals, so both languages see bit-identical
  doubles; R returns numbers as `%.17g` text. Permutation baselines are always
  exhaustive, so no random stream is involved. When R raises an error peyesim
  must raise, and when R returns `NA` peyesim must return NaN.
- **Bundled data.** `test_parity_datasets.py` checks that `wynn_study` and
  `wynn_test` hold R's data bit for bit (every column, and an md5 of every
  fixation written with `%.17g`). `tools/regenerate_datasets.py` rebuilds the
  JSON copies from R deterministically.

- **Whole analyses on real data.** `test_parity_wynn.py` runs density maps,
  template similarity (Pearson, Spearman, Fisher z, cosine), time-binned
  sampling, MultiMatch, revisitation and entropy on a subset of the Wynn data
  in both languages. A one-off run on the full datasets also matched:
  - 5,158 per-trial maps were identical;
  - template similarity over 1,199 trials agreed to 5e-14, with identical
    `n_perm` and group t statistics to about 1e-14;
  - MultiMatch over 360 scanpath pairs agreed to 7e-16 (2.5e-8 for the
    `emdist`-based position metric);
  - revisitation counts were identical under all three null models.

Tolerances are stated per test. In short: values R computes with the same IEEE
operations are compared exactly; sums and means agree to 1e-13 relative (R
accumulates in long double); densities, after both sides round with
`zapsmall(z, digits = 7)`, were identical in every one of 69,608 random cells.

The CI job *Parity with R eyesim* installs R and the pinned eyesim, sets
`PEYESIM_REQUIRE_R=1` (missing R then fails rather than skips) and runs the
whole suite. Locally:

```bash
export EYESIM_SOURCE_DIR=/path/to/eyesim   # default: ./eyesim
PEYESIM_REQUIRE_R=1 python -m pytest -q -n auto --dist loadfile
python tools/r_oracle_parity.py            # older single-script oracle
```

R packages used by the oracle: `pkgload`, `jsonlite`, eyesim's own imports,
and `crqa`, `emdist`, `energy`, `transport`, `T4transport`, `ppcor`, `nnls`,
`MASS`, `ks`.

## Coverage

eyesim `9ce29bc` exports 50 functions outside GazeWeave. All have a Python
counterpart except `element_text_wrap` (a ggplot2 theme element);
`scale_fill_eyesim_density` and `scale_colour_eyesim_time` are
`eyesim_density_cmap` and `eyesim_time_cmap` in matplotlib
(`tests/test_public_api_parity.py` reads the live `NAMESPACE`). Covered
against R:

- fixation groups, scanpaths, `center`/`rescale`/`normalize`,
  `rep_fixations`, `sample_fixations`, `concat_fixation_groups`, saccade
  helpers;
- `eye_table`, `simulate_eye_table` (structure), `add_scanpath`;
- `eye_density` (ks and MASS backends, weights, time windows, multiscale),
  `density_by`, `suggest_sigma`, entropy;
- `similarity`/`compute_similarity` for every method on vectors, maps,
  saliency EMD and multiscale maps;
- `template_similarity` with permutation baselines, strata and every latent or
  geometric transform; `template_similarity_cv` with seeded folds;
- `sample_density`, `sample_density_time`, `fixation_overlap`,
  `fixation_similarity`, `scanpath_similarity`, `multi_match`;
- `repetitive_similarity`, `template_regression`, `template_multireg`,
  `template_sample`;
- `crqa`, `fixation_revisits`, `revisit_summary`, visual-angle conversion,
  `estimate_scale`.

## Known differences from R

### By design

- **Random streams.** Permutation subsets, `simulate_eye_table()` and other
  random draws cannot reproduce R's streams. With `permutations` below the
  number of candidates the baseline is a different random subset, so
  `perm_sim` and `eye_sim_diff` agree in distribution, not value. Seeded CV
  folds use R's `sample()` order when `Rscript` is available.
- **EMD on large grids.** `emdist` stops after 500 iterations (with a warning)
  and then returns a non-optimal, larger distance. peyesim solves exactly and
  agrees with R's exact `transport::wasserstein`. Where `emdist` converges the
  two agree to about 1e-9 (`emdist` works in single precision).
- **Binned ks KDE.** `ks::kde()` bins by default above 500 fixations; peyesim
  always evaluates the exact estimate and raises `NotImplementedError` for
  `binned=True`. The binning error grows with the bandwidth. On Wynn maps
  pooled over trials it peaks at about 1% of the map maximum at sigma 25,
  0.3% at sigma 50 and 5% at sigma 100. `ks::kde` alone shows the same error,
  so it is not caused by eyesim. At sigma 100, 60 of 1,199 test trials changed
  the sign of their match-minus-mismatch difference, although the group t
  statistic barely moved. Pass `binned = FALSE` in R to get the exact
  estimate, which matches peyesim exactly.
- **Component signs.** PCA/CCA scores may differ in sign from R; parity is
  checked through sign-invariant results.
- **`estimate_scale()`** uses SciPy's L-BFGS-B on a non-smooth objective and
  agrees with R's `optim()` to about 1e-3.
- **Plots** are matplotlib, not pixel-compatible with ggplot2.
- **Return shapes.** R adds a `matchind` column and a `similarity_transform`
  attribute to `template_similarity()` results; peyesim does not. With
  `multiscale_aggregation="none"` peyesim returns per-scale values as an
  unnamed array. Printed summaries differ in wording.
- **Method names** are matched case-insensitively in Python.
- **Kept in Python only:** `density_matrix()` (R removed its empty generic)
  and the `wynn_study_image` dataset (R dropped it to stay within CRAN's size
  limit).

### Where R errors or returns NA and peyesim returns a value

These come from bugs in R eyesim or its dependencies found by the randomized
audit. peyesim returns the value R evidently intends rather than copying the
bug. The linked issues on eyesim give a reproduction and a suggested fix.

- `match_scale()` always errors in R (`pmap` passes the columns through
  `...`) ([#42](https://github.com/bbuchsbaum/eyesim/issues/42)).
- `crqa` with a single embedded point: the crqa package drops a dimension and
  builds a 2×2 plot ([#43](https://github.com/bbuchsbaum/eyesim/issues/43)).
- Combining fixation groups whose first group is empty gives `-Inf` onsets
  in R ([#44](https://github.com/bbuchsbaum/eyesim/issues/44)).
- The fast cosine path for all-zero maps gives 1 only on the `[1, 1]`
  element ([#45](https://github.com/bbuchsbaum/eyesim/issues/45)).
- `template_similarity(multiscale_aggregation="none")` crashes in
  `bind_rows` when a row has no map ([#46](https://github.com/bbuchsbaum/eyesim/issues/46)).
- EMD similarity with a missing permutation template crashes ([#47](https://github.com/bbuchsbaum/eyesim/issues/47)).
- Multiscale permutation baselines are averaged by position, not by sigma, if
  the maps list their sigmas in different orders ([#48](https://github.com/bbuchsbaum/eyesim/issues/48)).
- CCA with a missing `fit_by` value errors ([#49](https://github.com/bbuchsbaum/eyesim/issues/49)).
- MultiMatch duration similarity for two zero-duration scanpaths is 0/0 (`NA`)
  in R and 1 in peyesim, as for any equal durations ([#50](https://github.com/bbuchsbaum/eyesim/issues/50)).
- `similarity.scanpath` in R ignores `method`; peyesim validates it ([#51](https://github.com/bbuchsbaum/eyesim/issues/51)).
- R refuses multiscale EMD; peyesim computes it per scale.
- `sample_density_time()` forwards `NA` values to `aggregate_fun`, and
  `repetitive_similarity(pairwise)` applies `na.omit`, in slightly different
  places from peyesim; results agree whenever there are no missing values.

## Fixed during the October 2026 audit

The randomized audit found, and peyesim fixed, more than 30 Python bugs. Each
has a regression test that runs without R (`tests/test_core_parity_fixes.py`,
`tests/test_similarity_parity_fixes.py`). The ones most likely to change
results in existing analyses:

- `similarity()`/`compute_similarity()` defaulted to Spearman; R and now
  peyesim use Pearson.
- Multiscale similarity sorted the sigmas instead of keeping `x`'s order.
- Missing match keys never matched (R's `match()` pairs `NA` with `NA`), and
  missing `permute_on` values were pooled into one stratum.
- PCA ignored `center=False` and `scale.`; `lm` returned minimum-norm
  coefficients for aliased terms; `rlm` and logistic regression now follow
  `MASS::rlm` and `glm.fit`.
- `template_similarity_cv` used rows with missing split keys.
- crqa vertical-line measures used the transposed plot and `rENTR` the wrong
  normaliser.
- `density_by` ignored extra `eye_density` arguments and dropped `NA` groups.
- The bundled datasets rounded coordinates to 15 significant digits (about 9%
  of x/y values off by up to 1.1e-13); they are now exact.

The full list is in the 0.1.0 entry of `CHANGELOG.md`.

## Sync history

The port was written against eyesim `1580ec9` (March 2026) and synced to
`9ce29bc` in October 2026. That sync added `fixation_revisits()`,
`revisit_summary()`, the visual-angle helpers and the eyesim 0.2.0 colour
system; changed permutation baselines to remove every copy of the true match,
count each other template once and report `n_perm`; and brought over
eyesim's similarity, density, transform and regression fixes (lattice checks,
Fisher z clamping, O(n log n) distance correlation, normalised EMD, extended
Jaccard, duration-weighted Sinkhorn, CORAL `X Cs^-1/2 Ct^1/2`, ks kernel
truncation at 3.7 SD, and others).
