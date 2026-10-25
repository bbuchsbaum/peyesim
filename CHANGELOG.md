# Changelog

## 0.1.0 (unreleased)

First public release. peyesim is a Python port of R
[eyesim](https://github.com/bbuchsbaum/eyesim) 0.3.0 (commit `9ce29bc`),
except the experimental GazeWeave functions. See `PARITY.md` for how parity
is checked and the known differences.

### Features

- Fixation groups, scanpaths and `EyeTable`, with the bundled `wynn_study`
  and `wynn_test` datasets.
- Fixation density maps (`eye_density`, `density_by`, multiscale densities,
  `suggest_sigma`) and fixation entropy.
- Similarity between maps and scanpaths (Pearson, Spearman, Fisher z, cosine,
  Jaccard, distance correlation, EMD, Sinkhorn, MultiMatch, fixation overlap).
- `template_similarity` with permutation baselines and strata, cross-validated
  `template_similarity_cv`, and latent or geometric transforms (PCA, CORAL,
  CCA, contract, affine).
- `sample_density`, `sample_density_time`, `repetitive_similarity`,
  `template_regression`, `template_multireg` and `template_sample`.
- `fixation_revisits` and `revisit_summary`, cross-recurrence (`crqa`),
  visual-angle conversion, and eyesim's colour system in matplotlib.

### Parity fixes from the randomized audit against R

These changed results relative to earlier development snapshots.

- `similarity()` and `compute_similarity()` default to Pearson, as in R (they
  defaulted to Spearman). Scanpath similarity requires `screensize`.
- Multiscale similarity keeps `x`'s sigma order instead of sorting; invalid
  `multiscale_aggregation` values are refused.
- `template_similarity(multiscale_aggregation="none")` keeps per-scale
  `perm_sim` and `eye_sim_diff`.
- Missing match keys match each other, as with R's `match()`; missing
  `permute_on` values are no longer pooled into one stratum.
- Jaccard no longer returns NaN for small values or two zero vectors; the
  vectorised cosine path uses R's thresholds.
- `template_similarity_cv` skips rows with missing split keys; transforms can
  be named by string.
- PCA honours `center=False` and `scale.` like `prcomp`; a missing `fit_by`
  value no longer forms its own stratum.
- `lm` gives `NA` for aliased terms and tolerates NaN cells; `rlm` follows
  `MASS::rlm`; logistic regression follows `glm.fit`; `method="rank"` returns
  `NA` on a constant baseline.
- `repetitive_similarity` no longer counts a missing condition as "other".
- Method names are validated like `match.arg`.
- Printing a table with a `fixgroup` column no longer raises.
- `center`, `rescale` and `normalize` accept integer coordinates; `center`,
  `summary` and `concat_fixation_groups` propagate NaN as R propagates `NA`.
- `rep_fixations` raises on a NaN duration; the scanpath of an empty group is
  empty.
- `eye_table` keeps `NA` group keys and its columns when every fixation is
  clipped; `simulate_eye_table` orders groups numerically.
- `density_by` forwards extra `eye_density` arguments, keeps `NA` groups and
  orders columns as R does; `eye_density` returns `None` for NaN coordinates.
- Multiscale entropy names keep the exact sigma; `fixation_entropy` ignores
  the other method's arguments, as R does.
- `crqa`: vertical-line measures use the right orientation of the recurrence
  plot, `rENTR` uses crqa's normaliser, and two edge cases match crqa.
- The bundled datasets hold R's coordinates exactly (they were rounded to 15
  significant digits); `tools/regenerate_datasets.py` rebuilds them from R.

### Packaging

- Python 3.9 to 3.13; tested at the declared minimum dependency versions
  (`numpy` 1.22, `pandas` 1.4, `scipy` 1.8, `scikit-learn` 1.0, `statsmodels`
  0.13, `POT` 0.9, `networkx` 2.8; `matplotlib` 3.8 for plotting).
- CI runs the whole suite against live R eyesim on every pull request.
- `CITATION.cff`.
