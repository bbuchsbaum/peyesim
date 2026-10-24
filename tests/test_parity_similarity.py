"""Live R parity for similarity metrics, permutation baselines and sampling.

Covers the R eyesim 0.3.0 changes: ``n_perm`` and distinct non-matching
permutation candidates, lattice checks, the Fisher z snap, the fast distance
correlation, normalised EMD and partial-matching saliency EMD, the
T4transport-style Sinkhorn fixation similarity, the symmetric overlap
denominator, and half-open ``sample_density_time()`` bins. Every fixture is
deterministic, and permutation baselines are exhaustive so no cross-language
random stream is involved.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from peyesim import (
    eye_density,
    fixation_group,
    fixation_overlap,
    fixation_similarity,
    multi_match,
    repetitive_similarity,
    sample_density,
    sample_density_time,
    scanpath,
    scanpath_similarity,
    similarity,
    template_similarity,
    template_similarity_cv,
)
from peyesim.density import EyeDensityMultiscale, gen_density
from peyesim.similarity import LatticeMismatchError, compute_similarity
from tests.r_oracle import as_float, requires_r, run_r

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

GRID_X = np.linspace(0, 100, 9)
GRID_Y = np.linspace(0, 80, 7)


def _map(cx, cy, s, bump=0.0):
    xx, yy = np.meshgrid(GRID_X, GRID_Y, indexing="ij")
    z = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2)) + bump * (xx / 100.0)
    return gen_density(GRID_X, GRID_Y, z)


MAP_SPECS = [(30, 30, 15, 0.0), (35, 25, 18, 0.05), (70, 50, 12, 0.0), (60, 40, 25, 0.1),
             (20, 60, 10, 0.02), (80, 20, 14, 0.0)]

VECTORS = {
    "a": [1, 2, 3, 4, 5, 6, 7, 8.5],
    "b": [2, 1, 5, 3, 8, 6, 9, 7],
    "c": [0, 0, 1, 3, 0, 2, 0, 5],
    "const1": [2, 2, 2, 2, 2, 2, 2, 2],
}

FIXGROUPS = [
    dict(x=[100, 220, 330, 410, 150], y=[100, 180, 120, 300, 260],
         onset=[0, 210, 450, 700, 990], duration=[190, 220, 230, 270, 160]),
    dict(x=[110, 200, 350, 400], y=[90, 170, 140, 310],
         onset=[0, 250, 520, 800], duration=[230, 250, 260, 200]),
    dict(x=[500, 420, 300, 250, 180], y=[400, 350, 330, 200, 100],
         onset=[0, 180, 400, 650, 900], duration=[160, 200, 230, 230, 150]),
    dict(x=[480, 440, 310, 230], y=[380, 360, 310, 210],
         onset=[30, 260, 540, 760], duration=[210, 260, 200, 240]),
]


def _r_num(v):
    return "c(" + ", ".join(repr(float(t)) for t in np.asarray(v, dtype=float).ravel(order="F")) + ")"


def _r_fg(spec):
    return (f"fixation_group(x = {_r_num(spec['x'])}, y = {_r_num(spec['y'])}, "
            f"onset = {_r_num(spec['onset'])}, duration = {_r_num(spec['duration'])})")


def _py_fg(spec):
    return fixation_group(x=spec["x"], y=spec["y"], onset=spec["onset"], duration=spec["duration"])


R_CODE = f"""
gx <- {_r_num(GRID_X)}
gy <- {_r_num(GRID_Y)}
# Map values are generated in Python and passed exactly (17 significant
# digits), so ties and rounding are identical in both languages.
zs <- list({", ".join(_r_num(_map(*spec).z.ravel(order="F")) for spec in MAP_SPECS)})
maps <- lapply(zs, function(z) gen_density(gx, gy, matrix(z, length(gx), length(gy))))
vecs <- list({", ".join(f"{k} = {_r_num(v)}" for k, v in VECTORS.items())})
fgs <- list({", ".join(_r_fg(s) for s in FIXGROUPS)})

methods <- c("pearson", "spearman", "fisherz", "cosine", "l1", "jaccard", "dcov")
vec_pairs <- list(c("a", "b"), c("a", "c"), c("b", "c"), c("const1", "const1"))
vec_sims <- lapply(vec_pairs, function(p) as.list(sapply(methods, function(m)
  suppressWarnings(compute_similarity(vecs[[p[1]]], vecs[[p[2]]], method = m)))))
map_sims <- lapply(list(c(1, 2), c(1, 3), c(3, 4), c(5, 6)), function(p)
  as.list(sapply(c(methods, "emd"), function(m) suppressWarnings(similarity(maps[[p[1]]], maps[[p[2]]], method = m)))))
fisherz_special <- list(
  identical = similarity(maps[[1]], maps[[1]], method = "fisherz"),
  rescaled = similarity(maps[[1]], {{ m <- maps[[1]]; m$z <- m$z * 3.7; m }}, method = "fisherz"),
  negated = similarity(maps[[1]], {{ m <- maps[[1]]; m$z <- -m$z; m }}, method = "fisherz"),
  constant = similarity(vecs$const1, vecs$const1, method = "fisherz")
)
sal <- maps[[4]]
saliency_emd <- c(similarity(maps[[1]], maps[[2]], method = "emd", saliency_map = sal),
                  similarity(maps[[3]], maps[[5]], method = "emd", saliency_map = sal$z))

# Lattice mismatch: plain and multiscale
other <- gen_density(gx + 1, gy, maps[[1]]$z)
lattice_err <- tryCatch({{ similarity(maps[[1]], other); "none" }},
                        eyesim_lattice_mismatch = function(e) "lattice")
ylen_err <- tryCatch({{ similarity(maps[[1]], 1:5); "none" }}, error = function(e) "error")

# Sinkhorn fixation similarity (T4transport) for several lambdas
sinkhorn <- sapply(list(c(1, 2), c(1, 3), c(2, 4)), function(p)
  similarity(fgs[[p[1]]], fgs[[p[2]]], method = "sinkhorn"))
sinkhorn_lambda <- similarity(fgs[[1]], fgs[[2]], method = "sinkhorn", lambda = 0.05, tweight = 0.3)

# Overlap: default time grid is built from both groups; leading NA not counted
ov_default <- fixation_overlap(fgs[[4]], fgs[[2]])
ov_rev <- fixation_overlap(fgs[[2]], fgs[[4]])
ov_custom <- fixation_overlap(fgs[[1]], fgs[[2]], dthresh = 40, time_samples = seq(-100, 1000, by = 25))
ov_sim <- similarity(fgs[[1]], fgs[[2]], method = "overlap", time_samples = seq(0, 900, by = 30))

# Template similarity with duplicated keys, a NULL density and exhaustive perms
ref_tab <- tibble::tibble(image = 1:6, subject = rep(c("s1", "s2"), each = 3), density = maps)
src_maps <- lapply(c(2, 1, 4, 3, 6, 5), function(i) maps[[i]])
src_tab <- tibble::tibble(image = c(1, 1, 2, 3, 4, 5, 6),
                          subject = c("s1", "s1", "s1", "s1", "s2", "s2", "s2"),
                          density = c(src_maps[1], src_maps[1:6]))
src_tab$density[4] <- list(NULL)
ts_all <- suppressWarnings(template_similarity(ref_tab, src_tab, match_on = "image",
                                               method = "pearson", permutations = 100))
ts_strat <- suppressWarnings(template_similarity(ref_tab, src_tab, match_on = "image",
                                                 permute_on = "subject",
                                                 method = "spearman", permutations = 100))
ts_noperm <- template_similarity(ref_tab, src_tab[-4, ], match_on = "image",
                                 method = "cosine", permutations = 0)

# fixation_similarity / scanpath_similarity with exhaustive perms
fix_ref <- tibble::tibble(trial = 1:4, fixgroup = fgs)
fix_src <- tibble::tibble(trial = c(2, 1, 4, 3), fixgroup = fgs)
fs <- suppressMessages(fixation_similarity(fix_ref, fix_src, match_on = "trial",
                                           method = "sinkhorn", permutations = 10))
sp_ref <- tibble::tibble(trial = 1:4, scanpath = lapply(fgs, scanpath))
sp_src <- tibble::tibble(trial = c(2, 1, 4, 3), scanpath = lapply(fgs, scanpath))
sps <- suppressMessages(scanpath_similarity(sp_ref, sp_src, match_on = "trial",
                                            method = "multimatch", permutations = 10,
                                            screensize = c(600, 500)))

# CV: exhaustive within-fold baselines give deterministic n_perm and perm_sim
cv_ref <- tibble::tibble(image = 1:6, density = maps)
cv_src <- tibble::tibble(image = 1:6, density = lapply(c(2, 3, 1, 5, 6, 4), function(i) maps[[i]]))
cv <- template_similarity_cv(cv_ref, cv_src, match_on = "image", method = "pearson",
                             permutations = 100, n_folds = 2, seed = 7)

# Repetitive similarity refuses mixed lattices
rep_tab <- tibble::tibble(condition = c("a", "a", "b"), density = list(maps[[1]], maps[[2]], other))
rep_err <- tryCatch({{ repetitive_similarity(rep_tab, condition_var = "condition"); "none" }},
                    eyesim_lattice_mismatch = function(e) "lattice")

# sample_density: midpoints (half to even), off-grid clamp, NA before first onset
fg_s <- fixation_group(x = c(12.5, 37.5, 62.5, -10, 140), y = c(20, 40, 60, 75, 5),
                       onset = c(100, 200, 300, 400, 500), duration = rep(90, 5))
sd_norm <- lapply(c("none", "max", "sum", "zscore"), function(n)
  sample_density(maps[[2]], fg_s, normalize = n)$z)
sd_times <- sample_density(maps[[2]], fg_s, times = c(0, 99, 100, 250, 600))$z

# sample_density_time: last break excluded, distinct candidates, exhaustive perms
tmpl <- tibble::tibble(image = 1:4, subject = c("s1", "s1", "s2", "s2"), density = maps[1:4])
sdt_src <- tibble::tibble(image = c(1, 1, 2, 3, 4), subject = c("s1", "s1", "s1", "s2", "s2"),
                          fixgroup = list(fgs[[1]], fgs[[2]], fgs[[3]], fgs[[4]], fgs[[1]]))
tmpl$density <- lapply(tmpl$density, function(m) {{ m$x <- m$x * 6; m$y <- m$y * 6; m }})
sdt <- sample_density_time(tmpl, sdt_src, match_on = "image",
                           times = seq(0, 1000, by = 100), time_bins = c(0, 500, 1000),
                           permutations = 50)
sdt_strat <- sample_density_time(tmpl, sdt_src, match_on = "image", permute_on = "subject",
                                 times = seq(0, 1000, by = 100), time_bins = c(0, 500, 1000),
                                 permutations = 50, normalize = "max")

# A larger map for dcov / emd
big_fg1 <- fixation_group(x = c(100, 220, 330, 410, 150, 260), y = c(100, 180, 120, 300, 260, 220),
                          onset = seq(0, 1000, by = 200), duration = rep(200, 6))
big_fg2 <- fixation_group(x = c(120, 240, 300, 420, 170, 280), y = c(140, 160, 150, 280, 230, 200),
                          onset = seq(0, 1000, by = 200), duration = rep(200, 6))
b1 <- eye_density(big_fg1, sigma = 40, xbounds = c(0, 500), ybounds = c(0, 400), outdim = c(40, 32))
b2 <- eye_density(big_fg2, sigma = 40, xbounds = c(0, 500), ybounds = c(0, 400), outdim = c(40, 32))
# emdist stops at 500 iterations on grids this size (with a warning) and
# returns a non-optimal distance, so the exact reference is transport's solver.
emd_exact <- function(d1, d2) {{
  coords <- as.matrix(expand.grid(x = d1$x, y = d1$y))
  w1 <- as.vector(d1$z); w2 <- as.vector(d2$z)
  k1 <- w1 > 0; k2 <- w2 > 0
  1 / (1 + transport::wasserstein(transport::wpp(coords[k1, ], w1[k1] / sum(w1)),
                                  transport::wpp(coords[k2, ], w2[k2] / sum(w2)), p = 1))
}}
big <- list(dcov = similarity(b1, b2, method = "dcov"), emd = emd_exact(b1, b2),
         jaccard = similarity(b1, b2, method = "jaccard"))
ms1 <- eye_density(big_fg1, sigma = c(30, 60), xbounds = c(0, 500), ybounds = c(0, 400), outdim = c(20, 16))
ms2 <- eye_density(big_fg2, sigma = c(30, 60), xbounds = c(0, 500), ybounds = c(0, 400), outdim = c(20, 16))
ms3 <- eye_density(big_fg2, sigma = c(30, 60), xbounds = c(0, 600), ybounds = c(0, 400), outdim = c(20, 16))
ms_none <- similarity(ms1, ms2, method = "pearson", multiscale_aggregation = "none")
ms_err <- tryCatch({{ similarity(ms1, ms3); "none" }}, eyesim_lattice_mismatch = function(e) "lattice")

cols <- function(tab, nm) as.list(tab[, nm])
out <- list(
  vec_sims = vec_sims, map_sims = map_sims, fisherz_special = fisherz_special,
  saliency_emd = saliency_emd, lattice_err = lattice_err, ylen_err = ylen_err,
  sinkhorn = sinkhorn, sinkhorn_lambda = sinkhorn_lambda,
  ov_default = ov_default, ov_rev = ov_rev, ov_custom = ov_custom, ov_sim = ov_sim,
  ts_all = cols(ts_all, c("eye_sim", "perm_sim", "eye_sim_diff", "n_perm")),
  ts_strat = cols(ts_strat, c("eye_sim", "perm_sim", "eye_sim_diff", "n_perm")),
  ts_noperm = cols(ts_noperm, c("eye_sim")),
  fs = cols(fs, c("eye_sim", "perm_sim", "eye_sim_diff", "n_perm")),
  sps = as.list(sps[, grep("^mm_|^n_perm", names(sps))]),
  cv = cols(cv, c("image", "eye_sim", "perm_sim", "n_perm", ".cv_fold")),
  rep_err = rep_err,
  sd_norm = sd_norm, sd_times = sd_times,
  sdt = list(bins = as.matrix(sdt[, c("bin_1", "bin_2")]),
             perm = as.matrix(sdt[, c("perm_bin_1", "perm_bin_2")]),
             perm_sampled = lapply(sdt$perm_sampled, function(d) d$z),
             sampled = lapply(sdt$sampled, function(d) d$z)),
  sdt_strat = list(bins = as.matrix(sdt_strat[, c("bin_1", "bin_2")]),
                   perm = as.matrix(sdt_strat[, c("perm_bin_1", "perm_bin_2")])),
  big = big, ms_none = ms_none, ms_err = ms_err
)
"""


@pytest.fixture(scope="module")
def r():
    return run_r(R_CODE)


@pytest.fixture(scope="module")
def maps():
    return [_map(*spec) for spec in MAP_SPECS]


@pytest.fixture(scope="module")
def fgs():
    return [_py_fg(spec) for spec in FIXGROUPS]


METHODS = ["pearson", "spearman", "fisherz", "cosine", "l1", "jaccard", "dcov"]


def _close(actual, desired, tol=1e-10):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), as_float(desired),
                               rtol=tol, atol=tol, equal_nan=True)


@requires_r
def test_vector_similarity_methods_match_r(r):
    pairs = [("a", "b"), ("a", "c"), ("b", "c"), ("const1", "const1")]
    for (a, b), r_vals in zip(pairs, r["vec_sims"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            py = [compute_similarity(np.array(VECTORS[a]), np.array(VECTORS[b]), method=m)
                  for m in METHODS]
        _close(py, [r_vals[m] for m in METHODS], tol=1e-12)


@requires_r
def test_density_similarity_methods_match_r(r, maps):
    pairs = [(0, 1), (0, 2), (2, 3), (4, 5)]
    for (i, j), r_vals in zip(pairs, r["map_sims"]):
        py = [similarity(maps[i], maps[j], method=m) for m in METHODS]
        _close(py, [r_vals[m] for m in METHODS], tol=1e-12)
        # emdist solves in single precision
        _close(similarity(maps[i], maps[j], method="emd"), r_vals["emd"], tol=1e-6)


@requires_r
def test_fisherz_snaps_perfect_correlations_like_r(r, maps):
    m = maps[0]
    rescaled = gen_density(m.x, m.y, m.z * 3.7)
    negated = gen_density(m.x, m.y, -m.z)
    py = [
        similarity(m, m, method="fisherz"),
        similarity(m, rescaled, method="fisherz"),
        similarity(m, negated, method="fisherz"),
        compute_similarity(np.array(VECTORS["const1"]), np.array(VECTORS["const1"]), method="fisherz"),
    ]
    fz = r["fisherz_special"]
    _close(py, [fz["identical"], fz["rescaled"], fz["negated"], fz["constant"]], tol=1e-12)
    assert py[0] == py[1] == py[3] == pytest.approx(np.arctanh(1 - np.finfo(float).eps))


@requires_r
def test_saliency_emd_is_partial_matching_like_r(r, maps):
    py = [similarity(maps[0], maps[1], method="emd", saliency_map=maps[3]),
          similarity(maps[2], maps[4], method="emd", saliency_map=maps[3].z)]
    _close(py, r["saliency_emd"], tol=1e-6)


@requires_r
def test_lattice_mismatch_is_refused_like_r(r, maps):
    assert r["lattice_err"] == "lattice"
    assert r["ylen_err"] == "error"
    other = gen_density(maps[0].x + 1, maps[0].y, maps[0].z)
    with pytest.raises(LatticeMismatchError):
        similarity(maps[0], other)
    with pytest.raises(ValueError, match="grid cells"):
        similarity(maps[0], np.arange(1, 6))
    assert r["rep_err"] == "lattice"
    tab = pd.DataFrame({"condition": ["a", "a", "b"], "density": [maps[0], maps[1], other]})
    with pytest.raises(LatticeMismatchError):
        repetitive_similarity(tab, condition_var="condition")


@requires_r
def test_sinkhorn_fixation_similarity_matches_t4transport(r, fgs):
    py = [similarity(fgs[i], fgs[j], method="sinkhorn") for i, j in [(0, 1), (0, 2), (1, 3)]]
    _close(py, r["sinkhorn"], tol=1e-10)
    _close(similarity(fgs[0], fgs[1], method="sinkhorn", lambda_=0.05, tweight=0.3),
           r["sinkhorn_lambda"], tol=1e-10)


@requires_r
def test_fixation_overlap_matches_r(r, fgs):
    for py, key in [(fixation_overlap(fgs[3], fgs[1]), "ov_default"),
                    (fixation_overlap(fgs[1], fgs[3]), "ov_rev"),
                    (fixation_overlap(fgs[0], fgs[1], dthresh=40,
                                      time_samples=np.arange(-100, 1001, 25)), "ov_custom")]:
        assert py["overlap"] == r[key]["overlap"]
        _close(py["perc"], r[key]["perc"], tol=1e-12)
    assert r["ov_default"]["perc"] == pytest.approx(r["ov_rev"]["perc"])
    _close(similarity(fgs[0], fgs[1], method="overlap", time_samples=np.arange(0, 901, 30)),
           r["ov_sim"], tol=1e-12)


def _template_tables(maps):
    ref_tab = pd.DataFrame({"image": np.arange(1, 7), "subject": np.repeat(["s1", "s2"], 3),
                            "density": maps})
    src_maps = [maps[i - 1] for i in (2, 1, 4, 3, 6, 5)]
    src_tab = pd.DataFrame({"image": [1, 1, 2, 3, 4, 5, 6],
                            "subject": ["s1", "s1", "s1", "s1", "s2", "s2", "s2"],
                            "density": [src_maps[0]] + src_maps})
    src_tab.at[3, "density"] = None
    return ref_tab, src_tab


def _compare_frame(py, rcols, columns, tol=1e-10):
    for col in columns:
        _close(py[col].to_numpy(dtype=float), rcols[col], tol=tol)


@requires_r
def test_template_similarity_n_perm_and_candidates_match_r(r, maps):
    ref_tab, src_tab = _template_tables(maps)
    ts_all = template_similarity(ref_tab, src_tab, match_on="image", method="pearson", permutations=100)
    _compare_frame(ts_all, r["ts_all"], ["eye_sim", "perm_sim", "eye_sim_diff", "n_perm"])
    ts_strat = template_similarity(ref_tab, src_tab, match_on="image", permute_on="subject",
                                   method="spearman", permutations=100)
    _compare_frame(ts_strat, r["ts_strat"], ["eye_sim", "perm_sim", "eye_sim_diff", "n_perm"])
    ts_noperm = template_similarity(ref_tab, src_tab.drop(index=3).reset_index(drop=True),
                                    match_on="image", method="cosine", permutations=0)
    _compare_frame(ts_noperm, r["ts_noperm"], ["eye_sim"])
    assert "n_perm" not in ts_noperm.columns
    # duplicated keys: candidates are distinct templates, never the own match
    assert ts_all["n_perm"].tolist()[:3] == [5, 5, 5]
    assert ts_all["n_perm"].iloc[3] == 0


@requires_r
def test_fixation_and_scanpath_similarity_n_perm_match_r(r, fgs):
    ref = pd.DataFrame({"trial": [1, 2, 3, 4], "fixgroup": fgs})
    src = pd.DataFrame({"trial": [2, 1, 4, 3], "fixgroup": fgs})
    fs = fixation_similarity(ref, src, match_on="trial", method="sinkhorn", permutations=10)
    _compare_frame(fs, r["fs"], ["eye_sim", "perm_sim", "eye_sim_diff", "n_perm"])

    sp_ref = pd.DataFrame({"trial": [1, 2, 3, 4], "scanpath": [scanpath(f) for f in fgs]})
    sp_src = pd.DataFrame({"trial": [2, 1, 4, 3], "scanpath": [scanpath(f) for f in fgs]})
    sps = scanpath_similarity(sp_ref, sp_src, match_on="trial", method="multimatch",
                              permutations=10, screensize=(600, 500))
    for col, vals in r["sps"].items():
        _close(sps[col].to_numpy(dtype=float), vals, tol=1e-6)


@requires_r
def test_template_similarity_cv_exhaustive_baseline_matches_r(r, maps):
    cv_ref = pd.DataFrame({"image": np.arange(1, 7), "density": maps})
    cv_src = pd.DataFrame({"image": np.arange(1, 7),
                           "density": [maps[i - 1] for i in (2, 3, 1, 5, 6, 4)]})
    state = np.random.get_state()
    cv = template_similarity_cv(cv_ref, cv_src, match_on="image", method="pearson",
                                permutations=100, n_folds=2, seed=7)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1])
    _compare_frame(cv, r["cv"], ["image", "eye_sim", "perm_sim", "n_perm", ".cv_fold"])


@requires_r
def test_sample_density_matches_r(r, maps):
    fg_s = fixation_group(x=[12.5, 37.5, 62.5, -10, 140], y=[20, 40, 60, 75, 5],
                          onset=[100, 200, 300, 400, 500], duration=[90] * 5)
    for norm, r_vals in zip(["none", "max", "sum", "zscore"], r["sd_norm"]):
        _close(sample_density(maps[1], fg_s, normalize=norm)["z"], r_vals, tol=1e-12)
    _close(sample_density(maps[1], fg_s, times=[0, 99, 100, 250, 600])["z"], r["sd_times"], tol=1e-12)


def _sdt_inputs(maps, fgs):
    scaled = [gen_density(m.x * 6, m.y * 6, m.z) for m in maps[:4]]
    tmpl = pd.DataFrame({"image": [1, 2, 3, 4], "subject": ["s1", "s1", "s2", "s2"], "density": scaled})
    src = pd.DataFrame({"image": [1, 1, 2, 3, 4], "subject": ["s1", "s1", "s1", "s2", "s2"],
                        "fixgroup": [fgs[0], fgs[1], fgs[2], fgs[3], fgs[0]]})
    return tmpl, src


@requires_r
def test_sample_density_time_matches_r(r, maps, fgs):
    tmpl, src = _sdt_inputs(maps, fgs)
    times = np.arange(0, 1001, 100)
    sdt = sample_density_time(tmpl, src, match_on="image", times=times,
                              time_bins=[0, 500, 1000], permutations=50)
    _close(sdt[["bin_1", "bin_2"]].to_numpy(dtype=float), r["sdt"]["bins"], tol=1e-12)
    _close(sdt[["perm_bin_1", "perm_bin_2"]].to_numpy(dtype=float), r["sdt"]["perm"], tol=1e-12)
    for py_df, r_z in zip(sdt["sampled"], r["sdt"]["sampled"]):
        _close(py_df["z"], r_z, tol=1e-12)
    for py_df, r_z in zip(sdt["perm_sampled"], r["sdt"]["perm_sampled"]):
        _close(py_df["z"], r_z, tol=1e-12)

    strat = sample_density_time(tmpl, src, match_on="image", permute_on="subject", times=times,
                                time_bins=[0, 500, 1000], permutations=50, normalize="max")
    _close(strat[["bin_1", "bin_2"]].to_numpy(dtype=float), r["sdt_strat"]["bins"], tol=1e-12)
    _close(strat[["perm_bin_1", "perm_bin_2"]].to_numpy(dtype=float), r["sdt_strat"]["perm"], tol=1e-12)


@requires_r
def test_large_map_dcov_emd_and_multiscale_match_r(r):
    fg1 = fixation_group(x=[100, 220, 330, 410, 150, 260], y=[100, 180, 120, 300, 260, 220],
                         onset=np.arange(0, 1001, 200), duration=[200] * 6)
    fg2 = fixation_group(x=[120, 240, 300, 420, 170, 280], y=[140, 160, 150, 280, 230, 200],
                         onset=np.arange(0, 1001, 200), duration=[200] * 6)
    b1 = eye_density(fg1, sigma=40, xbounds=(0, 500), ybounds=(0, 400), outdim=(40, 32))
    b2 = eye_density(fg2, sigma=40, xbounds=(0, 500), ybounds=(0, 400), outdim=(40, 32))
    # densities agree with R to its 7-significant-digit rounding rule
    _close(similarity(b1, b2, method="dcov"), r["big"]["dcov"], tol=1e-6)
    _close(similarity(b1, b2, method="jaccard"), r["big"]["jaccard"], tol=1e-6)
    _close(similarity(b1, b2, method="emd"), r["big"]["emd"], tol=1e-6)

    ms1 = eye_density(fg1, sigma=[30, 60], xbounds=(0, 500), ybounds=(0, 400), outdim=(20, 16))
    ms2 = eye_density(fg2, sigma=[30, 60], xbounds=(0, 500), ybounds=(0, 400), outdim=(20, 16))
    ms3 = eye_density(fg2, sigma=[30, 60], xbounds=(0, 600), ybounds=(0, 400), outdim=(20, 16))
    assert isinstance(ms1, EyeDensityMultiscale)
    _close(similarity(ms1, ms2, method="pearson", multiscale_aggregation="none"), r["ms_none"], tol=1e-6)
    assert r["ms_err"] == "lattice"
    with pytest.raises(LatticeMismatchError):
        similarity(ms1, ms3)


def test_emdw_edge_cases_follow_r():
    from peyesim._utils import emdw

    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert emdw(pts, [0, 0], pts, [0, 0]) == 0.0
    assert np.isnan(emdw(pts, [1, 0], pts, [0, 0]))
    assert np.isnan(emdw(pts, [np.inf, 0], pts, [1, 0]))
    # normalised: scaling one side does not change the distance
    assert emdw(pts, [1, 1], pts, [3, 1]) == pytest.approx(emdw(pts, [2, 2], pts, [3, 1]))
    # partial matching: only min(total mass) is moved
    assert emdw(pts, [1, 0], pts, [1, 5], normalize=False) == pytest.approx(0.0)


def test_template_similarity_reports_n_perm_without_r(maps):
    ref_tab, src_tab = _template_tables(maps)
    res = template_similarity(ref_tab, src_tab, match_on="image", method="pearson", permutations=2)
    assert res["n_perm"].tolist() == [2, 2, 2, 0, 2, 2, 2]
