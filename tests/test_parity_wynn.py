"""Whole-analysis parity with R eyesim on the bundled Wynn et al. data.

One R script runs each analysis on a small, fixed subset of ``wynn_study`` and
``wynn_test`` (the first two subjects; 20 images each for the permutation
analyses) and writes its results to a temporary directory; the tests run the
same analysis in peyesim and compare. Permutation baselines are exhaustive
(``permutations`` exceeds the number of candidates), so no random stream is
involved.

Two input-level facts are checked first: the trial keys match row for row,
and every fixation is bit-identical (``tests/test_parity_datasets.py`` checks
the full datasets).

Density maps with more than 500 fixations differ by design: ``ks::kde()``
switches to binned estimation there and peyesim evaluates the exact estimate.
``test_pooled_maps_match_exact_ks`` checks agreement with ``binned = FALSE``;
``test_pooled_maps_binned_difference_is_bounded`` pins the size of the binned
difference at sigma = 50 (below 1% of the map maximum).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import peyesim as p
from peyesim.fixations import FixationGroup
from peyesim.revisits import REVISIT_SUMMARY_COLUMNS
from tests.r_oracle import requires_r, run_r

BOUNDS = dict(xbounds=(0, 800), ybounds=(0, 600))
N_IMAGES = 20  # images per subject in the permutation analyses
MM_PAIRS = 100

R_CODE = r"""
suppressPackageStartupMessages(library(dplyr))
od <- "@OUTDIR@"
data(wynn_study); data(wynn_test)
subs <- sort(unique(wynn_study$Subject))[1:2]
st <- wynn_study %>% ungroup() %>% filter(Subject %in% subs)
te <- wynn_test %>% ungroup() %>% filter(Subject %in% subs)
bin <- function(v, f) writeBin(as.double(v), file.path(od, f))
fix_vec <- function(tab) { fx <- do.call(rbind, tab$fixgroup); c(fx$x, fx$y, fx$duration, fx$onset) }
bin(fix_vec(st), "study_fix.bin"); bin(fix_vec(te), "test_fix.bin")
q <- function(expr) suppressWarnings(suppressMessages(expr))
dens <- function(tab, groups, ...) q(density_by(tab, groups = groups, xbounds = c(0, 800),
                                               ybounds = c(0, 600), outdim = c(100, 100), ...))
zs <- function(res) unlist(lapply(res$density, function(d) as.vector(d$z)))

## per-trial maps (< 500 fixations each): study maps of subject 1, test maps of subject 2
d_st <- dens(st %>% filter(Subject == subs[1]), "ImageNumberS", sigma = 50, duration_weighted = TRUE)
d_te <- dens(te %>% filter(Subject == subs[2]), "ImageNumberS", sigma = 25, duration_weighted = FALSE)
bin(zs(d_st), "dens_study.bin"); bin(zs(d_te), "dens_test.bin")

## pooled maps of two images over all subjects (> 3000 fixations each)
pool <- wynn_study %>% ungroup() %>% filter(ImageNumber %in% sort(unique(ImageNumber))[1:2])
bin(zs(dens(pool, "ImageNumber", sigma = 50, duration_weighted = TRUE)), "pool_binned.bin")
pool_small <- pool %>% filter(ImageNumber == sort(unique(ImageNumber))[1], Subject %in% sort(unique(Subject))[1:16])
bin(zs(q(density_by(pool_small, groups = "ImageNumber", sigma = 50, xbounds = c(0, 800),
                     ybounds = c(0, 600), outdim = c(50, 50), binned = FALSE))), "pool_exact.bin")

## template similarity and sample_density_time on 30 images per subject
imgs <- sort(unique(wynn_study$ImageNumber))[1:@NIMG@]
ref <- dens(st %>% filter(ImageNumber %in% imgs), "ImageNumberS", keep_vars = "Subject",
            sigma = 50)
src <- dens(te %>% filter(ImageNumber %in% imgs), "ImageNumberS",
            keep_vars = c("Subject", "probe_type"), sigma = 50)
tsim <- lapply(c(pearson = "pearson", spearman = "spearman", fisherz = "fisherz", cosine = "cosine"),
  function(m) q(template_similarity(ref, src, match_on = "ImageNumberS", permute_on = "Subject",
                                    method = m, permutations = 500)) %>%
                select(ImageNumberS, eye_sim, perm_sim, eye_sim_diff, n_perm))
srcx <- te %>% filter(ImageNumber %in% imgs) %>% select(ImageNumberS, Subject, fixgroup)
sdt <- q(sample_density_time(ref, srcx, match_on = "ImageNumberS", times = seq(0, 3000, by = 50),
                             time_bins = c(0, 250, 500, 1000, 1500, 3000), permutations = 500,
                             permute_on = "Subject"))
sdt <- sdt[, c("ImageNumberS", grep("bin_", names(sdt), value = TRUE))]

## MultiMatch: first study viewing against the test viewing
first_view <- function(fg) fg[cumsum(c(1, diff(fg$onset) <= 0)) == 1, , drop = FALSE]
te_mm <- te[1:@MMPAIRS@, ]
mi <- match(te_mm$ImageNumberS, st$ImageNumberS)
mm <- do.call(rbind, lapply(seq_len(nrow(te_mm)), function(i)
  q(multi_match(scanpath(first_view(st$fixgroup[[mi[i]]])), scanpath(te_mm$fixgroup[[i]]),
                screensize = c(800, 600)))))

## revisits (radius 30 px)
rev <- function(tab, nl) {
  r <- fixation_revisits(as_eye_table(tab), radius = 30, null = nl,
                         subject_var = if (nl != "none") "Subject" else NULL,
                         item_var = if (nl == "item") "ImageNumber" else NULL)
  as.data.frame(ungroup(r)[, revisit_summary_cols])
}
te_sub <- te %>% filter(ImageNumber %in% imgs)
revs <- list(study_none = rev(st, "none"), test_none = rev(te, "none"),
             test_subject = rev(te_sub, "subject"), test_item = rev(te_sub, "item"))

## entropy
ent_fg <- te_sub$fixgroup
ent <- list(
  density = sapply(ent_fg, function(g) q(fixation_entropy(g, method = "density", sigma = 50,
                                                          xbounds = c(0, 800), ybounds = c(0, 600)))),
  density_auto = sapply(ent_fg, function(g) q(fixation_entropy(g, method = "density"))),
  grid = sapply(ent_fg, function(g) q(fixation_entropy(g, method = "grid", xbounds = c(0, 800),
                                                       ybounds = c(0, 600)))),
  maps = sapply(ref$density, fixation_entropy))

out <- list(study_keys = st$ImageNumberS, test_keys = te$ImageNumberS,
            dens_study_keys = d_st$ImageNumberS, dens_test_keys = d_te$ImageNumberS,
            tsim = tsim, sdt = sdt, mm = as.data.frame(mm), revs = revs, ent = ent)
"""


@pytest.fixture(scope="module")
def outdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("wynn_parity")


@pytest.fixture(scope="module")
def r(outdir):
    code = (R_CODE.replace("@OUTDIR@", str(outdir)).replace("@NIMG@", str(N_IMAGES))
            .replace("@MMPAIRS@", str(MM_PAIRS)))
    return run_r(code)


@pytest.fixture(scope="module")
def py():
    study = p.load_dataset("wynn_study")
    test = p.load_dataset("wynn_test")
    subs = np.sort(study.Subject.unique())[:2]
    st = study[study.Subject.isin(subs)].reset_index(drop=True)
    te = test[test.Subject.isin(subs)].reset_index(drop=True)
    imgs = np.sort(study.ImageNumber.unique())[:N_IMAGES]
    return dict(study=study, st=st, te=te, imgs=imgs,
                te_sub=te[te.ImageNumber.astype(str).isin(imgs)].reset_index(drop=True))


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _zs(res) -> np.ndarray:
    return np.stack([d.z.ravel(order="F") for d in res["density"]])


def _assert_maps(r_flat: np.ndarray, maps: np.ndarray) -> None:
    r_maps = r_flat.reshape(maps.shape)
    scale = r_maps.max(axis=1, keepdims=True)
    tol = np.broadcast_to(1e-9 * scale + 1e-15, maps.shape)
    np.testing.assert_array_less(np.abs(r_maps - maps), tol)


def _fix_vec(tab) -> np.ndarray:
    fx = pd.concat([fg.to_pandas() for fg in tab.fixgroup], ignore_index=True)
    return np.concatenate([fx[c].to_numpy(float) for c in ("x", "y", "duration", "onset")])


@requires_r
def test_inputs_identical(r, py, outdir):
    assert list(py["st"].ImageNumberS) == r["study_keys"]
    assert list(py["te"].ImageNumberS) == r["test_keys"]
    for name, tab in (("study", py["st"]), ("test", py["te"])):
        rv = np.fromfile(outdir / f"{name}_fix.bin")
        pv = _fix_vec(tab)
        assert rv.shape == pv.shape
        np.testing.assert_array_equal(rv, pv)


@requires_r
def test_density_by_per_trial_maps(r, py, outdir):
    st, te, subs = py["st"], py["te"], np.sort(py["st"].Subject.unique())
    d_st = _quiet(p.density_by, st[st.Subject == subs[0]], groups="ImageNumberS", sigma=50,
                  duration_weighted=True, **BOUNDS)
    d_te = _quiet(p.density_by, te[te.Subject == subs[1]], groups="ImageNumberS", sigma=25,
                  duration_weighted=False, **BOUNDS)
    assert len(d_st) == 120 and len(d_te) >= 119
    assert list(d_st.ImageNumberS) == r["dens_study_keys"]
    assert list(d_te.ImageNumberS) == r["dens_test_keys"]
    _assert_maps(np.fromfile(outdir / "dens_study.bin"), _zs(d_st))
    _assert_maps(np.fromfile(outdir / "dens_test.bin"), _zs(d_te))


def _pool(py, n_subjects=None):
    study = py["study"]
    pool = study[study.ImageNumber.isin(np.sort(study.ImageNumber.unique())[:2])]
    if n_subjects is not None:
        pool = pool[(pool.ImageNumber == np.sort(pool.ImageNumber.unique())[0])
                    & pool.Subject.isin(np.sort(pool.Subject.unique())[:n_subjects])]
    return _quiet(p.density_by, pool, groups="ImageNumber", sigma=50, duration_weighted=True,
                  outdim=(100, 100) if n_subjects is None else (50, 50), **BOUNDS)


@requires_r
def test_pooled_maps_match_exact_ks(r, py, outdir):
    res = _pool(py, n_subjects=16)
    assert (res.fixgroup.map(len) > 500).all()
    _assert_maps(np.fromfile(outdir / "pool_exact.bin"), _zs(res))


@requires_r
def test_pooled_maps_binned_difference_is_bounded(r, py, outdir):
    # Documented difference: ks::kde() bins above 500 fixations, peyesim is exact.
    maps = _zs(_pool(py))
    r_maps = np.fromfile(outdir / "pool_binned.bin").reshape(maps.shape)
    rel = np.abs(r_maps - maps).max(axis=1) / r_maps.max(axis=1)
    assert np.all(rel > 1e-6)  # really binned in R
    assert np.all(rel < 0.01)
    for a, b in zip(r_maps, maps):
        assert np.corrcoef(a, b)[0, 1] > 0.99999


@pytest.fixture(scope="module")
def py_maps(py):
    imgs = py["imgs"]
    ref = _quiet(p.density_by, py["st"][py["st"].ImageNumber.isin(imgs)], groups="ImageNumberS",
                 keep_vars=["Subject"], sigma=50, **BOUNDS)
    src = _quiet(p.density_by, py["te_sub"], groups="ImageNumberS",
                 keep_vars=["Subject", "probe_type"], sigma=50, **BOUNDS)
    return ref, src


@requires_r
@pytest.mark.parametrize("method", ["pearson", "spearman", "fisherz", "cosine"])
def test_template_similarity(r, py_maps, method):
    ref, src = py_maps
    got = _quiet(p.template_similarity, ref, src, match_on="ImageNumberS", permute_on="Subject",
                 method=method, permutations=500)
    exp = pd.DataFrame(r["tsim"][method])
    assert list(got.ImageNumberS) == list(exp.ImageNumberS)
    np.testing.assert_array_equal(got.n_perm.to_numpy(), exp.n_perm.to_numpy())
    assert (exp.n_perm == N_IMAGES - 1).all()  # exhaustive
    for col in ("eye_sim", "perm_sim", "eye_sim_diff"):
        np.testing.assert_allclose(got[col].to_numpy(float), exp[col].to_numpy(float),
                                   rtol=1e-10, atol=1e-12)


@requires_r
def test_sample_density_time(r, py, py_maps):
    ref, _ = py_maps
    srcx = py["te_sub"][["ImageNumberS", "Subject", "fixgroup"]]
    got = _quiet(p.sample_density_time, ref, srcx, match_on="ImageNumberS",
                 times=np.arange(0, 3001, 50), time_bins=np.array([0, 250, 500, 1000, 1500, 3000]),
                 permutations=500, permute_on="Subject")
    exp = pd.DataFrame(r["sdt"])
    assert list(got.ImageNumberS) == list(exp.ImageNumberS)
    cols = [c for c in exp.columns if "bin_" in c]
    assert len(cols) == 15
    for c in cols:
        e = np.array([np.nan if v is None else v for v in exp[c]], dtype=float)
        np.testing.assert_allclose(got[c].to_numpy(float), e, rtol=1e-10, atol=1e-15)


def _first_view(fg: FixationGroup) -> FixationGroup:
    df = fg.to_pandas()
    view = np.cumsum(np.r_[1, np.diff(df.onset.to_numpy()) <= 0])
    return FixationGroup(df[view == 1].reset_index(drop=True))


@requires_r
def test_multi_match(r, py):
    st, te = py["st"], py["te"].iloc[:MM_PAIRS]
    mi = pd.Index(st.ImageNumberS).get_indexer(te.ImageNumberS)
    got = pd.DataFrame([
        _quiet(p.multi_match, p.scanpath(_first_view(st.fixgroup.iloc[m])), p.scanpath(fg),
               screensize=(800, 600))
        for m, fg in zip(mi, te.fixgroup)
    ])
    exp = pd.DataFrame(r["mm"])
    assert exp.shape == (MM_PAIRS, 6)
    for col in exp.columns:
        e = np.array([np.nan if v is None else v for v in exp[col]], dtype=float)
        # mm_position_emd goes through emdist, which works in single precision.
        tol = 1e-6 if col == "mm_position_emd" else 1e-12
        np.testing.assert_allclose(got[col].to_numpy(float), e, rtol=tol, atol=tol)


@requires_r
@pytest.mark.parametrize("case", ["study_none", "test_none", "test_subject", "test_item"])
def test_fixation_revisits(r, py, case):
    name, null = case.split("_")
    tab = {"study": py["st"], "test": py["te"]}[name]
    if null != "none":
        tab = py["te_sub"]
    got = p.fixation_revisits(tab, radius=30, null=null,
                              subject_var="Subject" if null != "none" else None,
                              item_var="ImageNumber" if null == "item" else None)
    exp = pd.DataFrame(r["revs"][case])
    for col in REVISIT_SUMMARY_COLUMNS:
        e = np.array([np.nan if v is None else v for v in exp[col]], dtype=float)
        np.testing.assert_allclose(got[col].to_numpy(float), e, rtol=1e-10, atol=1e-12,
                                   err_msg=col)


@requires_r
def test_fixation_entropy(r, py, py_maps):
    fgs = list(py["te_sub"].fixgroup)
    got = {
        "density": [_quiet(p.fixation_entropy, g, method="density", sigma=50, **BOUNDS) for g in fgs],
        "density_auto": [_quiet(p.fixation_entropy, g, method="density") for g in fgs],
        "grid": [_quiet(p.fixation_entropy, g, method="grid", **BOUNDS) for g in fgs],
        "maps": [p.fixation_entropy(d) for d in py_maps[0].density],
    }
    for key, vals in got.items():
        e = np.array([np.nan if v is None else v for v in r["ent"][key]], dtype=float)
        np.testing.assert_allclose(np.asarray(vals, float), e, rtol=1e-9, atol=1e-12, err_msg=key)
