"""Live parity checks against R eyesim for density, entropy, fixation and CRQA helpers.

All fixtures are deterministic, so no cross-language random streams are involved.
Density maps are compared cell by cell with ``rtol=1e-9`` and an absolute
tolerance of ``1e-12 * max(z)``: both sides apply R's ``zapsmall(z, digits = 7)``
rounding, so values agree to well below the rounding step.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from peyesim import (
    calcangle,
    crqa,
    density_by,
    eye_density,
    fixation_entropy,
    fixation_group,
    gen_density,
    rep_fixations,
    sample_fixations,
    suggest_sigma,
)
from peyesim.expansion import estimate_scale
from tests.r_oracle import as_float, run_r



R_CODE = r"""
fg <- fixation_group(x = c(10, 30, 80), y = c(10, 20, 40),
                     onset = c(0, 100, 200), duration = c(1, 2, 4))
fg_big <- fixation_group(
  x = c(120, 340, 560, 610, 205, 450, 700, 90, 330, 515),
  y = c(80, 260, 140, 400, 310, 75, 220, 360, 190, 285),
  onset = seq(0, 2250, by = 250),
  duration = c(180, 220, 260, 150, 300, 210, 190, 240, 170, 230)
)
dens <- function(g, ...) suppressMessages(suppressWarnings(
  eye_density(g, sigma = 10, xbounds = c(0, 100), ybounds = c(0, 50), outdim = c(5, 3), ...)))
z_or_null <- function(d) if (is.null(d)) NULL else d$z

big <- function(...) suppressMessages(eye_density(fg_big, xbounds = c(0, 800),
                                                 ybounds = c(0, 450), outdim = c(40, 23), ...))
ms <- big(sigma = c(30, 60, 120), duration_weighted = TRUE)

single <- fixation_group(x = 10, y = 10, onset = 0, duration = 100)
coincident <- fixation_group(x = rep(500, 5), y = rep(500, 5),
                             onset = seq(0, 800, by = 200), duration = rep(200, 5))
offgrid <- fixation_group(x = c(10, -1000), y = c(10, 10), onset = c(0, 100), duration = c(1, 1))

fixations <- data.frame(
  subject = rep(c("s1", "s2"), each = 10),
  x = c(seq(100, 900, length.out = 10), seq(150, 850, length.out = 10)),
  y = c(seq(200, 800, length.out = 10), seq(250, 750, length.out = 10)),
  duration = 200,
  onset = rep(seq(0, 1800, by = 200), 2)
)
et <- eye_table("x", "y", "duration", "onset", groupvar = "subject", data = fixations)
dby <- suppressWarnings(density_by(et, sigma = 80, xbounds = c(0, 1000), ybounds = c(0, 1000),
                                   outdim = c(20, 20)))

reps <- rep_fixations(fixation_group(x = c(1, 2, 3, 4), y = rep(1, 4), onset = c(0, 1, 2, 3),
                                     duration = c(0.29, 0.57, 0.295, 0.001)), 100)
unsorted <- fixation_group(x = c(3, 1, 2), y = c(30, 10, 20), onset = c(200, 0, 100),
                           duration = rep(50, 3))
tied <- fixation_group(x = c(1, 2, 3, 9), y = c(1, 2, 3, 9), onset = c(0, 100, 100, 200),
                       duration = rep(50, 4))
missing_xy <- fixation_group(x = c(1, NA, 3), y = c(1, NA, 3), onset = c(0, 100, 200),
                             duration = rep(50, 3))
na_onset <- fixation_group(x = c(1, 2, 3), y = c(1, 2, 3), onset = c(0, NA, 200),
                           duration = rep(50, 3))

crqa_a <- fixation_group(x = c(10, 200, 400, 30, 250, 410, 60, 220),
                         y = c(500, 20, 300, 80, 40, 310, 470, 60),
                         onset = seq(0, 1400, by = 200), duration = rep(200, 8))
crqa_b <- fixation_group(x = c(15, 210, 390, 600, 240, 420, 50, 230),
                         y = c(505, 25, 310, 900, 30, 300, 480, 75),
                         onset = seq(0, 1400, by = 200), duration = rep(200, 8))
crqa_settings <- list(
  list(radius = 60, delay = 1, embed = 1, rescale = 0, metric = "euclidean"),
  list(radius = 60, delay = 1, embed = 1, rescale = 0, metric = "manhattan"),
  list(radius = 120, delay = 2, embed = 2, rescale = 0, metric = "euclidean"),
  list(radius = 90, delay = 1, embed = 3, rescale = 0, metric = "manhattan"),
  list(radius = 0.4, delay = 1, embed = 1, rescale = 1, metric = "euclidean"),
  list(radius = 0.2, delay = 1, embed = 2, rescale = 2, metric = "euclidean"),
  list(radius = 2, delay = 1, embed = 1, rescale = 3, metric = "manhattan"),
  list(radius = 0.3, delay = 1, embed = 1, rescale = 4, metric = "euclidean"),
  list(radius = 1, delay = 1, embed = 1, rescale = 0, metric = "euclidean")
)
crqa_keys <- c("RR", "DET", "NRLINE", "maxL", "L", "ENTR", "rENTR", "LAM", "TT",
               "max_vertlength")
crqa_out <- lapply(crqa_settings, function(s) {
  r <- suppressMessages(do.call(crqa, c(list(crqa_a, crqa_b), s)))
  vals <- lapply(crqa_keys, function(k) {
    v <- r[[k]]
    # Keep NA, NaN and +/-Inf apart: JSON would turn all of them into null.
    if (is.null(v) || length(v) == 0 || (is.na(v) && !is.nan(v))) NA_character_
    else sprintf("%.17g", as.numeric(v))
  })
  names(vals) <- crqa_keys
  vals$RP <- if (inherits(r$RP, "Matrix") || is.matrix(r$RP)) as.matrix(r$RP) * 1 else NULL
  vals
})

scale_ref <- fixation_group(x = c(100, 300, 500, 200), y = c(100, 50, 300, 250),
                            onset = c(0, 1, 2, 3), duration = rep(1, 4))
scale_src <- fixation_group(x = c(50, 150, 250, 100), y = c(200, 100, 600, 500),
                            onset = c(0, 1, 2, 3), duration = rep(1, 4))

out <- list(
  plain = dens(fg)$z,
  dur = dens(fg, duration_weighted = TRUE)$z,
  weights = dens(fg, weights = c(4, 2, 1))$z,
  weights_and_dur = dens(fg, weights = c(4, 2, 1), duration_weighted = TRUE)$z,
  weights_zero = dens(fg, weights = c(1, 1, 0))$z,
  weights_window = dens(fg, weights = c(4, 2, 1), window = c(50, 300))$z,
  unbinned = dens(fg, binned = FALSE)$z,
  zero_after_window_ks = z_or_null(dens(fg, weights = c(1, 0, 0), window = c(50, 300))),
  zero_after_window_mass = z_or_null(dens(fg, weights = c(1, 0, 0), window = c(50, 300),
                                          kde_pkg = "MASS")),
  mass_plain = suppressMessages(eye_density(fg, sigma = 40, xbounds = c(0, 100),
                                            ybounds = c(0, 50), outdim = c(5, 3),
                                            kde_pkg = "MASS"))$z,
  mass_dur = suppressMessages(eye_density(fg, sigma = 40, xbounds = c(0, 100),
                                          ybounds = c(0, 50), outdim = c(5, 3),
                                          kde_pkg = "MASS", duration_weighted = TRUE))$z,
  big_plain = big(sigma = 50)$z,
  big_unnormalized = big(sigma = 50, normalize = FALSE)$z,
  big_dur = big(sigma = 75, duration_weighted = TRUE)$z,
  big_window = big(sigma = 50, window = c(400, 1800))$z,
  big_ms = lapply(ms, function(d) d$z),
  big_x = big(sigma = 50)$x,
  big_y = big(sigma = 50)$y,
  sigma_big = suggest_sigma(fg_big),
  sigma_big_bounds = suggest_sigma(fg_big, xbounds = c(0, 800), ybounds = c(0, 450)),
  sigma_single = suggest_sigma(single),
  sigma_coincident = suggest_sigma(coincident),
  sigma_coincident_bounds = suggest_sigma(coincident, xbounds = c(0, 1000), ybounds = c(0, 1000)),
  dby_n = nrow(dby),
  dby_z = dby$density[[1]]$z,
  ent_density = fixation_entropy(fg_big),
  ent_density_raw = fixation_entropy(fg_big, normalize = FALSE, base = 2),
  ent_density_sigma = fixation_entropy(fg_big, sigma = 60, xbounds = c(0, 800),
                                       ybounds = c(0, 450)),
  ent_density_dur = fixation_entropy(fg_big, sigma = 60, duration_weighted = TRUE),
  ent_ms_none = as.list(fixation_entropy(fg_big, sigma = c(30, 60, 120), aggregate = "none")),
  ent_ms_mean = fixation_entropy(fg_big, sigma = c(30, 60, 120)),
  ent_grid = fixation_entropy(fg_big, method = "grid"),
  ent_grid_bounds = fixation_entropy(fg_big, method = "grid", grid = c(4, 3),
                                     xbounds = c(0, 800), ybounds = c(0, 450)),
  ent_single = fixation_entropy(single),
  ent_single_grid = fixation_entropy(single, method = "grid", xbounds = c(0, 100),
                                     ybounds = c(0, 100)),
  ent_offgrid = fixation_entropy(offgrid, method = "grid", grid = c(2, 2), xbounds = c(0, 100),
                                 ybounds = c(0, 50), normalize = FALSE),
  ent_zero = fixation_entropy(dens(fg) - dens(fg)),
  reps = as.vector(table(reps$index)),
  sf_unsorted = sample_fixations(unsorted, c(-5, 0, 50, 150, 250)),
  sf_unsorted_slow = sample_fixations(unsorted, c(-5, 0, 50, 150, 250), fast = FALSE),
  sf_tied = sample_fixations(tied, c(50, 100, 150, 250))$x,
  sf_missing = sample_fixations(missing_xy, c(50, 100, 150, 250))$x,
  sf_na_onset = sample_fixations(na_onset, c(50, 100, 150, 250))$x,
  angles = c(calcangle(c(1, 0), c(0, 1)), calcangle(c(1, 1), c(1, 1)),
             calcangle(c(.1, .7), c(.3, 2.1)), calcangle(c(1, 2), c(-2, -4)),
             calcangle(c(3, 1), c(-1, 2))),
  crqa = crqa_out,
  scale_par = estimate_scale(scale_ref, scale_src)$par,
  print_na = capture.output(print(fixation_group(x = c(100, NA, 300), y = c(10, 20, 30),
                                                 onset = c(0, 200, 400),
                                                 duration = c(150, 150, 200)))),
  print_empty = capture.output(print(fixation_group(x = numeric(0), y = numeric(0),
                                                    onset = numeric(0), duration = numeric(0))))
)
"""


@pytest.fixture(scope="module")
def r():
    return run_r(R_CODE)


FG = fixation_group(x=[10, 30, 80], y=[10, 20, 40], onset=[0, 100, 200], duration=[1, 2, 4])
FG_BIG = fixation_group(
    x=[120, 340, 560, 610, 205, 450, 700, 90, 330, 515],
    y=[80, 260, 140, 400, 310, 75, 220, 360, 190, 285],
    onset=np.arange(0, 2251, 250),
    duration=[180, 220, 260, 150, 300, 210, 190, 240, 170, 230],
)


def _dens(fg, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return eye_density(fg, sigma=kwargs.pop("sigma", 10), xbounds=(0, 100), ybounds=(0, 50),
                           outdim=(5, 3), **kwargs)


def _big(**kwargs):
    return eye_density(FG_BIG, xbounds=(0, 800), ybounds=(0, 450), outdim=(40, 23), **kwargs)


def assert_density_close(actual, expected):
    expected = as_float(expected)
    actual = np.asarray(actual, dtype=float)
    assert actual.shape == expected.shape
    scale = float(np.nanmax(np.abs(expected))) or 1.0
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12 * scale)


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("plain", {}),
        ("dur", {"duration_weighted": True}),
        ("weights", {"weights": [4, 2, 1]}),
        ("weights_and_dur", {"weights": [4, 2, 1], "duration_weighted": True}),
        ("weights_zero", {"weights": [1, 1, 0]}),
        ("weights_window", {"weights": [4, 2, 1], "window": (50, 300)}),
        ("unbinned", {"binned": False}),
        ("mass_plain", {"sigma": 40, "kde_pkg": "MASS"}),
        ("mass_dur", {"sigma": 40, "kde_pkg": "MASS", "duration_weighted": True}),
    ],
)
def test_eye_density_small_grid_matches_r(r, key, kwargs):
    assert_density_close(_dens(FG, **kwargs).z, r[key])


@pytest.mark.parametrize("pkg_key, pkg", [("zero_after_window_ks", "ks"),
                                          ("zero_after_window_mass", "MASS")])
def test_zero_weight_after_window_is_none_like_r(r, pkg_key, pkg):
    assert r[pkg_key] is None
    with pytest.warns(UserWarning, match="Sum of weights is zero"):
        res = eye_density(FG, sigma=10, xbounds=(0, 100), ybounds=(0, 50), outdim=(5, 3),
                          weights=[1, 0, 0], window=(50, 300), kde_pkg=pkg)
    assert res is None


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("big_plain", {"sigma": 50}),
        ("big_unnormalized", {"sigma": 50, "normalize": False}),
        ("big_dur", {"sigma": 75, "duration_weighted": True}),
        ("big_window", {"sigma": 50, "window": (400, 1800)}),
    ],
)
def test_eye_density_larger_grid_matches_r(r, key, kwargs):
    assert_density_close(_big(**kwargs).z, r[key])


def test_eye_density_grid_and_multiscale_match_r(r):
    d = _big(sigma=50)
    np.testing.assert_allclose(d.x, r["big_x"], rtol=1e-12)
    np.testing.assert_allclose(d.y, r["big_y"], rtol=1e-12)
    ms = _big(sigma=[30, 60, 120], duration_weighted=True)
    assert len(ms) == len(r["big_ms"]) == 3
    for scale, expected in zip(ms, r["big_ms"]):
        assert_density_close(scale.z, expected)


def test_suggest_sigma_matches_r(r):
    single = fixation_group(x=[10], y=[10], onset=[0], duration=[100])
    coincident = fixation_group(x=[500] * 5, y=[500] * 5, onset=np.arange(0, 801, 200),
                                duration=[200] * 5)
    assert suggest_sigma(FG_BIG) == pytest.approx(r["sigma_big"], rel=1e-12)
    assert suggest_sigma(FG_BIG, xbounds=(0, 800), ybounds=(0, 450)) == pytest.approx(
        r["sigma_big_bounds"], rel=1e-12)
    assert r["sigma_single"] is None and np.isnan(suggest_sigma(single))
    assert r["sigma_coincident"] is None and np.isnan(suggest_sigma(coincident))
    assert suggest_sigma(coincident, xbounds=(0, 1000), ybounds=(0, 1000)) == pytest.approx(
        r["sigma_coincident_bounds"], rel=1e-12)


def test_density_by_without_groups_matches_r(r):
    fixations = pd.DataFrame({
        "subject": ["s1"] * 10 + ["s2"] * 10,
        "x": np.concatenate([np.linspace(100, 900, 10), np.linspace(150, 850, 10)]),
        "y": np.concatenate([np.linspace(200, 800, 10), np.linspace(250, 750, 10)]),
        "duration": 200.0,
        "onset": np.tile(np.arange(0, 1801, 200), 2),
    })
    from peyesim import eye_table

    et = eye_table("x", "y", "duration", "onset", groupvar="subject", data=fixations)
    res = density_by(et, sigma=80, xbounds=(0, 1000), ybounds=(0, 1000), outdim=(20, 20))
    assert len(res) == r["dby_n"] == 1
    assert_density_close(res["density"].iloc[0].z, r["dby_z"])


def test_fixation_entropy_matches_r(r):
    single = fixation_group(x=[10], y=[10], onset=[0], duration=[100])
    offgrid = fixation_group(x=[10, -1000], y=[10, 10], onset=[0, 100], duration=[1, 1])
    close = dict(rel=1e-9)
    assert fixation_entropy(FG_BIG) == pytest.approx(r["ent_density"], **close)
    assert fixation_entropy(FG_BIG, normalize=False, base=2) == pytest.approx(
        r["ent_density_raw"], **close)
    assert fixation_entropy(FG_BIG, sigma=60, xbounds=(0, 800), ybounds=(0, 450)) == pytest.approx(
        r["ent_density_sigma"], **close)
    assert fixation_entropy(FG_BIG, sigma=60, duration_weighted=True) == pytest.approx(
        r["ent_density_dur"], **close)
    per_scale = fixation_entropy(FG_BIG, sigma=[30, 60, 120], aggregate="none")
    assert list(per_scale) == list(r["ent_ms_none"])
    np.testing.assert_allclose(list(per_scale.values()), list(r["ent_ms_none"].values()),
                               rtol=1e-9)
    assert fixation_entropy(FG_BIG, sigma=[30, 60, 120]) == pytest.approx(r["ent_ms_mean"], **close)
    assert fixation_entropy(FG_BIG, method="grid") == pytest.approx(r["ent_grid"], **close)
    assert fixation_entropy(FG_BIG, method="grid", grid=(4, 3), xbounds=(0, 800),
                            ybounds=(0, 450)) == pytest.approx(r["ent_grid_bounds"], **close)
    assert r["ent_single"] is None and np.isnan(fixation_entropy(single))
    assert fixation_entropy(single, method="grid", xbounds=(0, 100), ybounds=(0, 100)) == r[
        "ent_single_grid"] == 0
    assert fixation_entropy(offgrid, method="grid", grid=(2, 2), xbounds=(0, 100), ybounds=(0, 50),
                            normalize=False) == r["ent_offgrid"] == 0
    d = _dens(FG)
    assert r["ent_zero"] is None and np.isnan(fixation_entropy(d - d))


def test_rep_and_sample_fixations_match_r(r):
    fg = fixation_group(x=[1, 2, 3, 4], y=[1] * 4, onset=[0, 1, 2, 3],
                        duration=[0.29, 0.57, 0.295, 0.001])
    counts = rep_fixations(fg, 100)["index"].value_counts().sort_index().tolist()
    assert counts == r["reps"]

    unsorted = fixation_group(x=[3, 1, 2], y=[30, 10, 20], onset=[200, 0, 100], duration=[50] * 3)
    times = [-5, 0, 50, 150, 250]
    for fast, key in [(True, "sf_unsorted"), (False, "sf_unsorted_slow")]:
        got = sample_fixations(unsorted, times, fast=fast)
        for col in ("x", "y", "onset", "duration"):
            np.testing.assert_array_equal(got[col].to_numpy(dtype=float), as_float(r[key][col]))

    tied = fixation_group(x=[1, 2, 3, 9], y=[1, 2, 3, 9], onset=[0, 100, 100, 200],
                          duration=[50] * 4)
    missing = fixation_group(x=[1, np.nan, 3], y=[1, np.nan, 3], onset=[0, 100, 200],
                             duration=[50] * 3)
    na_onset = fixation_group(x=[1, 2, 3], y=[1, 2, 3], onset=[0, np.nan, 200], duration=[50] * 3)
    t2 = [50, 100, 150, 250]
    for fast in (True, False):
        np.testing.assert_array_equal(sample_fixations(tied, t2, fast=fast)["x"], as_float(r["sf_tied"]))
        np.testing.assert_array_equal(sample_fixations(missing, t2, fast=fast)["x"],
                                      as_float(r["sf_missing"]))
        np.testing.assert_array_equal(sample_fixations(na_onset, t2, fast=fast)["x"],
                                      as_float(r["sf_na_onset"]))


def test_calcangle_matches_r(r):
    pairs = [([1, 0], [0, 1]), ([1, 1], [1, 1]), ([.1, .7], [.3, 2.1]), ([1, 2], [-2, -4]),
             ([3, 1], [-1, 2])]
    got = [calcangle(a, b) for a, b in pairs]
    # Near-parallel angles are ill-conditioned (arccos near 1); 1e-5 degrees is R's own
    # test tolerance there.
    np.testing.assert_allclose(got, r["angles"], rtol=1e-12, atol=1e-5)


CRQA_SETTINGS = [
    dict(radius=60, delay=1, embed=1, rescale=0, metric="euclidean"),
    dict(radius=60, delay=1, embed=1, rescale=0, metric="manhattan"),
    dict(radius=120, delay=2, embed=2, rescale=0, metric="euclidean"),
    dict(radius=90, delay=1, embed=3, rescale=0, metric="manhattan"),
    dict(radius=0.4, delay=1, embed=1, rescale=1, metric="euclidean"),
    dict(radius=0.2, delay=1, embed=2, rescale=2, metric="euclidean"),
    dict(radius=2, delay=1, embed=1, rescale=3, metric="manhattan"),
    dict(radius=0.3, delay=1, embed=1, rescale=4, metric="euclidean"),
    dict(radius=1, delay=1, embed=1, rescale=0, metric="euclidean"),
]
CRQA_A = fixation_group(x=[10, 200, 400, 30, 250, 410, 60, 220],
                        y=[500, 20, 300, 80, 40, 310, 470, 60],
                        onset=np.arange(0, 1401, 200), duration=[200] * 8)
CRQA_B = fixation_group(x=[15, 210, 390, 600, 240, 420, 50, 230],
                        y=[505, 25, 310, 900, 30, 300, 480, 75],
                        onset=np.arange(0, 1401, 200), duration=[200] * 8)


@pytest.mark.parametrize("i", range(len(CRQA_SETTINGS)))
def test_crqa_matches_r_crqa(r, i):
    expected = r["crqa"][i]
    got = crqa(CRQA_A, CRQA_B, **CRQA_SETTINGS[i])
    for key in ["RR", "DET", "NRLINE", "maxL", "L", "ENTR", "rENTR", "LAM", "TT", "max_vertlength"]:
        e = expected[key]
        if e is None or float(e) != float(e):  # NA or NaN
            assert np.isnan(got[key]), key
        elif np.isinf(float(e)):
            assert got[key] == float(e), key
        else:
            assert got[key] == pytest.approx(float(e), rel=1e-12, abs=1e-12), key
    if expected.get("RP") is not None:
        np.testing.assert_array_equal(np.asarray(got["RP"], dtype=float), as_float(expected["RP"]))


def test_crqa_requires_rows():
    with pytest.raises(ValueError, match="non-empty"):
        crqa(CRQA_A.iloc[0:0], CRQA_B)


def test_estimate_scale_matches_r_on_exact_scaling(r):
    ref = fixation_group(x=[100, 300, 500, 200], y=[100, 50, 300, 250], onset=[0, 1, 2, 3],
                         duration=[1] * 4)
    src = fixation_group(x=[50, 150, 250, 100], y=[200, 100, 600, 500], onset=[0, 1, 2, 3],
                         duration=[1] * 4)
    # The Hausdorff objective is non-smooth, so the two L-BFGS-B implementations agree
    # only near the exact solution (2, 0.5).
    np.testing.assert_allclose(estimate_scale(ref, src)["par"], r["scale_par"], atol=1e-3)


def test_print_fixation_group_matches_r(r):
    fg = fixation_group(x=[100, np.nan, 300], y=[10, 20, 30], onset=[0, 200, 400],
                        duration=[150, 150, 200])
    lines = repr(fg).splitlines()
    assert lines[0].replace("FixationGroup", "Fixation group") == r["print_na"][0]
    assert [ln.strip() for ln in lines[1:]] == [ln.strip() for ln in r["print_na"][1:]]
    empty = fixation_group(x=[], y=[], onset=[], duration=[])
    assert repr(empty).replace("FixationGroup", "Fixation group") == r["print_empty"]
