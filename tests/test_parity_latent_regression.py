"""Live R parity for latent transforms, MultiMatch and template regression.

Fixtures are deterministic formulas evaluated identically in both languages.
Latent scores themselves are only defined up to the sign of each PCA/CCA
component, so transforms are compared through sign-invariant quantities
(template-similarity scores, covariances, zero-column structure).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from peyesim import (
    cca_transform,
    coral_transform,
    fixation_group,
    latent_pca_transform,
    multi_match,
    scanpath,
    template_multireg,
    template_regression,
    template_sample,
    template_similarity,
)
from peyesim.density import gen_density
from tests.r_oracle import as_float, requires_r_mark, run_r

pytestmark = requires_r_mark()

R_CODE = r"""
make_map <- function(k, shift = 0) {
  z <- outer(1:4, 1:4, function(i, j) 1.5 + sin(0.7 * k + 1.3 * i + 0.4 * j * k + shift))
  gen_density(c(0, 1, 2, 3), c(0, 1, 2, 3), z)
}
ids <- paste0("i", 1:8)
pids <- rep(c("p1", "p2"), each = 4)
ref <- data.frame(id = ids, pid = pids, stringsAsFactors = FALSE)
ref$density <- lapply(1:8, make_map)
src <- data.frame(id = ids, pid = pids, stringsAsFactors = FALSE)
src$density <- lapply(1:8, function(k) make_map(k, shift = 0.35))

ts <- function(transform, args) {
  res <- suppressWarnings(suppressMessages(template_similarity(
    ref, src, match_on = "id", method = "cosine", permutations = 0,
    similarity_transform = transform, similarity_transform_args = args)))
  res$eye_sim
}

coral_direct <- coral_transform(ref, src, comps = 3, shrink = 1e-3)
coral_scores <- do.call(rbind, coral_direct$source_tab$density)
ref_scores <- do.call(rbind, coral_direct$ref_tab$density)

cca_direct <- cca_transform(ref, src, match_on = "id", comps = 3, shrink = 1e-3)
cca_src <- do.call(rbind, cca_direct$source_tab$density)

# A source group with no reference rows has no CCA model.
src_p3 <- src
src_p3$pid[7:8] <- "p3"
cca_unfitted_warning <- tryCatch({
  cca_transform(ref, src_p3, match_on = "id", comps = 3, shrink = 1e-3, fit_by = "pid")
  ""
}, warning = function(w) conditionMessage(w))

# MultiMatch fixtures: unequal lengths and total durations.
mm_fg <- function(n, a, b, dscale) {
  i <- seq_len(n)
  fixation_group(x = 300 + 120 * sin(a * i), y = 250 + 90 * cos(b * i),
                 duration = dscale * (0.15 + 0.05 * (i %% 3)),
                 onset = cumsum(0.2 + 0.03 * i))
}
mm_pairs <- list(
  list(mm_fg(5, 0.9, 1.1, 1), mm_fg(6, 1.0, 1.2, 1.4)),
  list(mm_fg(4, 0.5, 0.7, 1), mm_fg(7, 0.6, 0.4, 0.8)),
  list(mm_fg(6, 1.7, 0.3, 2), mm_fg(6, 1.6, 0.35, 1))
)
mm <- lapply(mm_pairs, function(p) as.list(multi_match(scanpath(p[[1]]), scanpath(p[[2]]), c(640, 480))))

# Template regression with a non-default density column.
reg_ref <- data.frame(id = ids[1:4], stringsAsFactors = FALSE)
reg_ref$dens <- lapply(1:4, make_map)
reg_src <- data.frame(id = ids[1:4], base = c("b1", "b2", "b1", "b2"), stringsAsFactors = FALSE)
reg_src$dens <- lapply(1:4, function(k) make_map(k, shift = 0.35))
reg_base <- data.frame(base = c("b1", "b2"), stringsAsFactors = FALSE)
reg_base$dens <- list(make_map(11), make_map(12))
reg_lm <- template_regression(reg_ref, reg_src, "id", reg_base, "base", method = "lm", density_var = "dens")
reg_rlm <- template_regression(reg_ref, reg_src, "id", reg_base, "base", method = "rlm", density_var = "dens")
dup_base <- rbind(reg_base, reg_base[1, ])
err_dup <- tryCatch(template_regression(reg_ref, reg_src, "id", dup_base, "base", density_var = "dens"),
                    error = function(e) conditionMessage(e))
err_missing <- tryCatch(template_regression(reg_ref, reg_src, "id", reg_base[1, ], "base", density_var = "dens"),
                        error = function(e) conditionMessage(e))

mr_tab <- data.frame(id = ids[1:3], stringsAsFactors = FALSE)
mr_tab$resp <- lapply(1:3, function(k) make_map(k, shift = 0.35))
mr_tab$c1 <- lapply(1:3, make_map)
mr_tab$c2 <- lapply(4:6, make_map)
mr_default <- template_multireg(mr_tab, "resp", c("c1", "c2"))

ts_tab <- data.frame(id = ids[1:3], stringsAsFactors = FALSE)
ts_tab$tmpl <- list(make_map(1), NULL, make_map(3))
ts_tab$fixgroup <- list(
  fixation_group(x = c(0.2, 1.6, 2.9), y = c(0.1, 2.2, 1.4), duration = c(1, 1, 1), onset = c(0, 1, 2)),
  fixation_group(x = c(1, 2), y = c(1, 2), duration = c(1, 1), onset = c(0, 1)),
  NULL
)
ts_tab$fixgroup[[3]] <- fixation_group(x = c(3, 0.4), y = c(0, 2.6), duration = c(1, 1), onset = c(0, 1))
ts_warning <- ""
ts_res <- withCallingHandlers(template_sample(ts_tab, "tmpl"),
  warning = function(w) { ts_warning <<- conditionMessage(w); invokeRestart("muffleWarning") })

out <- list(
  pca = ts(latent_pca_transform, list(comps = 3)),
  coral = ts(coral_transform, list(comps = 3, shrink = 1e-3)),
  coral_fit_by = ts(coral_transform, list(comps = 3, shrink = 1e-3, fit_by = "pid")),
  cca = ts(cca_transform, list(comps = 3, shrink = 1e-3)),
  cca_fit_by = ts(cca_transform, list(comps = 3, shrink = 1e-3, fit_by = "pid")),
  coral_src_cov = cov(coral_scores),
  coral_ref_cov = cov(ref_scores),
  cca_width = ncol(cca_src),
  cca_zero_cols = which(colSums(abs(cca_src)) == 0),
  cca_unfitted_warning = cca_unfitted_warning,
  mm = mm,
  reg_lm = reg_lm[, c("beta_baseline", "beta_source")],
  reg_rlm = reg_rlm[, c("beta_baseline", "beta_source")],
  err_dup = err_dup,
  err_missing = err_missing,
  mr_default = as.data.frame(mr_default$multireg[[1]][, c("term", "estimate")]),
  ts_n = nrow(ts_res),
  ts_ids = ts_res$id,
  ts_warning = ts_warning,
  ts_values = lapply(ts_res$sample_out, function(s) s$z)
)
"""


def _make_map(k, shift=0.0):
    i = np.arange(1, 5)[:, None]
    j = np.arange(1, 5)[None, :]
    z = 1.5 + np.sin(0.7 * k + 1.3 * i + 0.4 * j * k + shift)
    grid = np.arange(4, dtype=float)
    return gen_density(grid, grid, z)


def _tables():
    ids = [f"i{k}" for k in range(1, 9)]
    pids = ["p1"] * 4 + ["p2"] * 4
    ref = pd.DataFrame({"id": ids, "pid": pids, "density": [_make_map(k) for k in range(1, 9)]})
    src = pd.DataFrame({"id": ids, "pid": pids,
                        "density": [_make_map(k, 0.35) for k in range(1, 9)]})
    return ref, src


def _mm_fg(n, a, b, dscale):
    i = np.arange(1, n + 1)
    return fixation_group(
        x=300 + 120 * np.sin(a * i),
        y=250 + 90 * np.cos(b * i),
        duration=dscale * (0.15 + 0.05 * (i % 3)),
        onset=np.cumsum(0.2 + 0.03 * i),
    )


@pytest.fixture(scope="module")
def r():
    return run_r(R_CODE)


def _template_sim(transform, args):
    ref, src = _tables()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = template_similarity(ref, src, match_on="id", method="cosine", permutations=0,
                                  similarity_transform=transform, similarity_transform_args=args)
    return res["eye_sim"].to_numpy(dtype=float)


@pytest.mark.parametrize(
    "key, transform, args",
    [
        ("pca", latent_pca_transform, {"comps": 3}),
        ("coral", coral_transform, {"comps": 3, "shrink": 1e-3}),
        ("coral_fit_by", coral_transform, {"comps": 3, "shrink": 1e-3, "fit_by": "pid"}),
        ("cca", cca_transform, {"comps": 3, "shrink": 1e-3}),
        ("cca_fit_by", cca_transform, {"comps": 3, "shrink": 1e-3, "fit_by": "pid"}),
    ],
)
def test_transformed_template_similarity_matches_r(r, key, transform, args):
    np.testing.assert_allclose(_template_sim(transform, args), as_float(r[key]), rtol=1e-8, atol=1e-10)


def test_coral_adapted_source_takes_reference_covariance(r):
    ref, src = _tables()
    out = coral_transform(ref, src, comps=3, shrink=1e-3)
    src_scores = np.vstack(out["source_tab"]["density"].to_list())
    ref_scores = np.vstack(out["ref_tab"]["density"].to_list())
    # Covariances are sign-invariant up to the PCA sign of each component, so
    # compare absolute values; the diagonals are exact.
    np.testing.assert_allclose(np.abs(np.cov(src_scores, rowvar=False)),
                               np.abs(as_float(r["coral_src_cov"])), rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.cov(ref_scores, rowvar=False), np.cov(src_scores, rowvar=False),
                               rtol=1e-2, atol=1e-3)


def test_cca_output_keeps_pca_width_with_trailing_zero_columns(r):
    ref, src = _tables()
    out = cca_transform(ref, src, match_on="id", comps=3, shrink=1e-3)
    src_scores = np.vstack(out["source_tab"]["density"].to_list())
    assert src_scores.shape[1] == r["cca_width"]
    zero_cols = np.flatnonzero(np.abs(src_scores).sum(axis=0) == 0) + 1
    np.testing.assert_array_equal(zero_cols, np.atleast_1d(as_float(r["cca_zero_cols"])))


def test_cca_unfitted_group_warning_matches_r(r):
    ref, src = _tables()
    src.loc[6:7, "pid"] = "p3"
    with pytest.warns(UserWarning) as record:
        cca_transform(ref, src, match_on="id", comps=3, shrink=1e-3, fit_by="pid")
    messages = [str(w.message) for w in record if "CCA model" in str(w.message)]
    assert messages == [r["cca_unfitted_warning"]]


def test_multimatch_metrics_match_r(r):
    pairs = [
        (_mm_fg(5, 0.9, 1.1, 1), _mm_fg(6, 1.0, 1.2, 1.4)),
        (_mm_fg(4, 0.5, 0.7, 1), _mm_fg(7, 0.6, 0.4, 0.8)),
        (_mm_fg(6, 1.7, 0.3, 2), _mm_fg(6, 1.6, 0.35, 1)),
    ]
    for (a, b), expected in zip(pairs, r["mm"]):
        result = multi_match(scanpath(a), scanpath(b), screensize=(640, 480))
        for key, value in expected.items():
            # emdist solves in single precision
            tol = 1e-6 if key == "mm_position_emd" else 1e-10
            assert result[key] == pytest.approx(value, abs=tol), key


def _regression_tables():
    ids = ["i1", "i2", "i3", "i4"]
    reg_ref = pd.DataFrame({"id": ids, "dens": [_make_map(k) for k in range(1, 5)]})
    reg_src = pd.DataFrame({"id": ids, "base": ["b1", "b2", "b1", "b2"],
                            "dens": [_make_map(k, 0.35) for k in range(1, 5)]})
    reg_base = pd.DataFrame({"base": ["b1", "b2"], "dens": [_make_map(11), _make_map(12)]})
    return reg_ref, reg_src, reg_base


@pytest.mark.parametrize("method, tol", [("lm", 1e-9), ("rlm", 1e-5)])
def test_template_regression_density_var_matches_r(r, method, tol):
    reg_ref, reg_src, reg_base = _regression_tables()
    res = template_regression(reg_ref, reg_src, "id", reg_base, "base", method=method, density_var="dens")
    expected = r[f"reg_{method}"]
    np.testing.assert_allclose(res["beta_baseline"], as_float(expected["beta_baseline"]), rtol=tol, atol=tol)
    np.testing.assert_allclose(res["beta_source"], as_float(expected["beta_source"]), rtol=tol, atol=tol)


def test_template_regression_key_errors_match_r(r):
    reg_ref, reg_src, reg_base = _regression_tables()
    dup = pd.concat([reg_base, reg_base.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError) as dup_err:
        template_regression(reg_ref, reg_src, "id", dup, "base", density_var="dens")
    assert str(dup_err.value) == r["err_dup"]
    with pytest.raises(ValueError) as missing_err:
        template_regression(reg_ref, reg_src, "id", reg_base.iloc[[0]], "base", density_var="dens")
    assert str(missing_err.value) == r["err_missing"]


def test_template_multireg_default_method_matches_r(r):
    tab = pd.DataFrame({
        "id": ["i1", "i2", "i3"],
        "resp": [_make_map(k, 0.35) for k in range(1, 4)],
        "c1": [_make_map(k) for k in range(1, 4)],
        "c2": [_make_map(k) for k in range(4, 7)],
    })
    res = template_multireg(tab, "resp", ["c1", "c2"])
    fit = res["multireg"].iloc[0]
    assert list(fit["term"]) == r["mr_default"]["term"]
    np.testing.assert_allclose(fit["estimate"], as_float(r["mr_default"]["estimate"]), rtol=1e-8, atol=1e-12)


def test_template_sample_drops_null_rows_like_r(r):
    tab = pd.DataFrame({
        "id": ["i1", "i2", "i3"],
        "tmpl": [_make_map(1), None, _make_map(3)],
        "fixgroup": [
            fixation_group(x=[0.2, 1.6, 2.9], y=[0.1, 2.2, 1.4], duration=[1, 1, 1], onset=[0, 1, 2]),
            fixation_group(x=[1, 2], y=[1, 2], duration=[1, 1], onset=[0, 1]),
            fixation_group(x=[3, 0.4], y=[0, 2.6], duration=[1, 1], onset=[0, 1]),
        ],
    })
    with pytest.warns(UserWarning) as record:
        res = template_sample(tab, "tmpl")
    assert [str(w.message) for w in record] == [r["ts_warning"]]
    assert len(res) == r["ts_n"]
    assert list(res["id"]) == r["ts_ids"]
    for sampled, expected in zip(res["sample_out"], r["ts_values"]):
        np.testing.assert_allclose(sampled["z"], as_float(expected), rtol=1e-12)
