"""Ports of R eyesim's 0.3.0 regression tests for density, entropy, fixations and CRQA.

Mirrors tests/testthat/test_audit_regressions.R and test_cran_review_fixes.R
(the parts that live in density/entropy/fixation/saccade/CRQA code).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from peyesim import (
    calcangle,
    crqa,
    eye_density,
    eye_table,
    density_by,
    fixation_entropy,
    fixation_group,
    gen_density,
    rep_fixations,
    sample_fixations,
    scanpath,
    suggest_sigma,
)
from peyesim.density import _zapsmall
from peyesim.expansion import estimate_scale


def audit_fg():
    return fixation_group(x=[10, 30, 80], y=[10, 20, 40], onset=[0, 100, 200], duration=[1, 2, 4])


def audit_density(fg, **kwargs):
    return eye_density(fg, sigma=kwargs.pop("sigma", 10), xbounds=(0, 100), ybounds=(0, 50),
                       outdim=(5, 3), **kwargs)


def test_eye_density_honours_explicit_weights():
    fg = audit_fg()
    d0 = audit_density(fg)
    dw = audit_density(fg, weights=[4, 2, 1])
    assert not np.allclose(d0.z, dw.z)
    np.testing.assert_allclose(audit_density(fg, weights=[1, 2, 4]).z,
                               audit_density(fg, duration_weighted=True).z)
    np.testing.assert_allclose(audit_density(fg, weights=[4, 2, 1], duration_weighted=True).z, dw.z)
    np.testing.assert_allclose(audit_density(fg, weights=[1, 1, 0]).z,
                               audit_density(fg.iloc[0:2]).z)
    np.testing.assert_allclose(audit_density(fg, weights=[4, 2, 1], window=(50, 300)).z,
                               audit_density(fg.iloc[1:3].reset_index(drop=True), weights=[2, 1]).z)

    with pytest.raises(ValueError, match="one value per fixation"):
        audit_density(fg, weights=[1, 2])
    with pytest.raises(ValueError, match="non-negative"):
        audit_density(fg, weights=[1, -1, 2])
    with pytest.raises(ValueError, match="not all be zero"):
        audit_density(fg, weights=[0, 0, 0])

    for pkg in ("ks", "MASS"):
        with pytest.warns(UserWarning, match="Sum of weights is zero"):
            assert audit_density(fg, weights=[1, 0, 0], window=(50, 300), kde_pkg=pkg) is None
    fg0 = fixation_group(x=[10, 30, 80], y=[10, 20, 40], onset=[0, 100, 200], duration=[0, 0, 0])
    with pytest.warns(UserWarning, match="Sum of weights is zero"):
        assert audit_density(fg0, duration_weighted=True) is None


def test_eye_density_validates_ks_arguments():
    fg = audit_fg()
    np.testing.assert_array_equal(audit_density(fg, binned=False).z, audit_density(fg).z)
    np.testing.assert_array_equal(audit_density(fg, **{"compute.cont": False}).z,
                                  audit_density(fg).z)
    ms = eye_density(fg, sigma=[5, 10], xbounds=(0, 100), ybounds=(0, 50), outdim=(5, 3),
                     binned=False)
    np.testing.assert_array_equal(ms[1].z, audit_density(fg).z)

    with pytest.raises(ValueError, match="Unsupported ks::kde"):
        audit_density(fg, not_a_kde_argument=1)
    with pytest.raises(ValueError, match="Unsupported ks::kde"):
        audit_density(fg, gridsize=(2, 2))
    with pytest.raises(ValueError, match="Unsupported ks::kde"):
        audit_density(fg, **{"eval.points": np.array([[10, 20]])})
    with pytest.raises(ValueError, match="not supported when kde_pkg"):
        audit_density(fg, kde_pkg="MASS", binned=False)
    with pytest.raises(NotImplementedError, match="binned"):
        audit_density(fg, binned=True)

    for bad in ("ks ", ["ks", "MASS"], "other"):
        with pytest.raises(ValueError, match='must be "ks" or "MASS"'):
            audit_density(fg, kde_pkg=bad)


def test_ks_kernel_is_truncated_at_supp_standard_deviations():
    fg = audit_fg()
    wide = audit_density(fg, supp=50, normalize=False).z
    default = audit_density(fg, normalize=False).z
    # Grid point (0, 50) is beyond 3.7 SD of the nearest fixations in y.
    assert default[0, 2] < wide[0, 2]
    # Far from every truncation boundary the two estimates agree.
    np.testing.assert_allclose(default[1, 1], wide[1, 1], rtol=1e-6)


def test_weighted_mass_density_on_non_square_grid():
    fg = audit_fg()

    def mass(fg, **kwargs):
        return eye_density(fg, sigma=40, xbounds=(0, 100), ybounds=(0, 50), outdim=(5, 3),
                           kde_pkg="MASS", **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dw = mass(fg, duration_weighted=True)
    assert dw.z.shape == (5, 3)
    assert not np.allclose(dw.z, mass(fg).z)
    np.testing.assert_allclose(mass(fg, weights=[1, 1, 0]).z, mass(fg.iloc[0:2]).z)


def test_under_mass_sigma_is_four_times_the_kernel_sd():
    fg = audit_fg()
    ks_map = audit_density(fg).z

    def mass_map(s):
        return eye_density(fg, sigma=s, xbounds=(0, 100), ybounds=(0, 50), outdim=(5, 3),
                           kde_pkg="MASS").z

    assert np.max(np.abs(mass_map(40) - ks_map)) < 1e-3
    assert np.max(np.abs(mass_map(10) - ks_map)) > 1e-2


def test_zapsmall_follows_r_rounding_rule():
    z = np.array([3.3e-3, 1.234567891e-4, 2.5e-11])
    np.testing.assert_array_equal(_zapsmall(z), [0.0033, 0.000123457, 0.0])
    np.testing.assert_array_equal(_zapsmall(np.zeros(3)), np.zeros(3))
    # Density maps are rounded to 7 significant digits relative to their maximum.
    d = audit_density(audit_fg()).z
    np.testing.assert_array_equal(d, _zapsmall(d))


def test_sigma_must_be_positive_and_finite():
    fg = audit_fg()
    one = fixation_group(x=[10], y=[10], onset=[0], duration=[100])
    for bad in (np.nan, suggest_sigma(one), -1, 0, np.inf, []):
        with pytest.raises(ValueError, match="positive, finite"):
            eye_density(fg, sigma=bad)


def test_single_fixation_entropy_is_defined():
    one = fixation_group(x=[10], y=[10], onset=[0], duration=[100])
    assert np.isnan(fixation_entropy(one))
    assert np.isnan(fixation_entropy(one, method="density"))
    assert np.isnan(fixation_entropy(one, sigma=10))
    assert fixation_entropy(one, sigma=10, min_fixations=1) == pytest.approx(
        fixation_entropy(eye_density(one, sigma=10, xbounds=(9.5, 10.5), ybounds=(9.5, 10.5),
                                     outdim=(50, 50), min_fixations=1)))
    assert fixation_entropy(one, method="grid", xbounds=(0, 100), ybounds=(0, 100)) == 0


def test_fixation_entropy_rejects_signed_maps():
    fg = audit_fg()
    d1 = audit_density(fg)
    d2 = audit_density(fg, weights=[4, 2, 1])
    with pytest.raises(ValueError, match="non-negative mass"):
        fixation_entropy(d1 - d2)
    signed = gen_density(x=[1, 2], y=[1, 2], z=np.array([[2, 1], [-1, 0.5]]))
    with pytest.raises(ValueError, match="non-negative mass"):
        fixation_entropy(signed)
    assert np.isnan(fixation_entropy(d1 - d1))


def test_fixation_entropy_forwards_aggregate_for_fixation_groups():
    rng = np.random.default_rng(1)
    fg = fixation_group(x=rng.uniform(0, 100, 20), y=rng.uniform(0, 100, 20),
                        onset=np.arange(20) * 250.0, duration=[200] * 20)
    sigmas = [5, 10, 20]
    per_scale = fixation_entropy(fg, sigma=sigmas, aggregate="none")
    assert list(per_scale) == [f"sigma_{s}" for s in sigmas]
    assert fixation_entropy(fg, sigma=sigmas) == pytest.approx(np.mean(list(per_scale.values())))
    assert per_scale["sigma_10"] == pytest.approx(fixation_entropy(fg, sigma=10))
    assert fixation_entropy(fg, aggregate="none") == fixation_entropy(fg)
    assert fixation_entropy(fg, method="grid", aggregate="none") == fixation_entropy(fg, method="grid")
    with pytest.raises(ValueError, match="should be one of"):
        fixation_entropy(fg, aggregate="median")


def test_grid_entropy_counts_off_bounds_fixations_in_edge_cell():
    fg = fixation_group(x=[10, -1000], y=[10, 10], onset=[0, 100], duration=[1, 1])
    assert fixation_entropy(fg, method="grid", grid=(2, 2), xbounds=(0, 100), ybounds=(0, 50),
                            normalize=False) == 0


def test_suggest_sigma_is_nan_without_spread_or_bounds():
    fg = fixation_group(x=[500] * 5, y=[500] * 5, onset=np.arange(0, 801, 200), duration=[200] * 5)
    assert np.isnan(suggest_sigma(fg))
    assert suggest_sigma(fg, xbounds=(0, 1000), ybounds=(0, 1000)) > 0


def test_density_by_works_without_groups():
    fixations = pd.DataFrame({
        "subject": ["s1"] * 10 + ["s2"] * 10,
        "x": np.concatenate([np.linspace(100, 900, 10), np.linspace(150, 850, 10)]),
        "y": np.concatenate([np.linspace(200, 800, 10), np.linspace(250, 750, 10)]),
        "duration": 200.0,
        "onset": np.tile(np.arange(0, 1801, 200), 2),
    })
    et = eye_table("x", "y", "duration", "onset", groupvar="subject", data=fixations)
    res = density_by(et, sigma=80, xbounds=(0, 1000), ybounds=(0, 1000))
    assert len(res) == 1
    assert res["density"].iloc[0].z.shape == (100, 100)


def test_empty_fixation_groups_do_not_crash():
    fg = fixation_group(x=[], y=[], onset=[], duration=[])
    assert len(fg) == 0
    assert len(rep_fixations(fg)) == 0
    assert repr(fg) == "FixationGroup: 0 fixations"
    out = sample_fixations(fg, [0, 10])
    assert np.isnan(out["x"]).all() and len(out) == 2


def test_rep_fixations_counts_without_floating_point_loss():
    fg = fixation_group(x=[1, 2, 3, 4], y=[1] * 4, onset=[0, 1, 2, 3],
                        duration=[0.29, 0.57, 0.295, 0.001])
    assert rep_fixations(fg, 100)["index"].value_counts().sort_index().tolist() == [29, 57, 29, 1]
    single = fixation_group(x=[5.5], y=[1], onset=[0], duration=[0.29])
    assert len(rep_fixations(single, 100)) == 29


def test_sample_fixations_holds_last_fixation_on_both_paths():
    fg = fixation_group(x=[0, 1], y=[0, 1], onset=[0, 100], duration=[100, 100])
    times = [-10, 0, 50, 100, 150, 200]
    fast = sample_fixations(fg, times)
    slow = sample_fixations(fg, times, fast=False)
    np.testing.assert_array_equal(fast["x"], [np.nan, 0, 0, 1, 1, 1])
    pd.testing.assert_frame_equal(fast.to_pandas(), slow.to_pandas())
    np.testing.assert_array_equal(fast["duration"], np.ones(6))
    empty_time = sample_fixations(fg, [])
    assert list(empty_time.columns) == ["x", "y", "onset", "duration"]
    pd.testing.assert_frame_equal(empty_time.to_pandas(), sample_fixations(fg, [], fast=False).to_pandas())

    one = fixation_group(x=[5], y=[6], onset=[10], duration=[100])
    for f in (True, False):
        np.testing.assert_array_equal(sample_fixations(one, [0, 10, 500], fast=f)["x"], [np.nan, 5, 5])

    unsorted = fixation_group(x=[3, 1, 2], y=[3, 1, 2], onset=[200, 0, 100], duration=[50] * 3)
    tied = fixation_group(x=[1, 2, 3, 9], y=[1, 2, 3, 9], onset=[0, 100, 100, 200], duration=[50] * 4)
    missing = fixation_group(x=[1, np.nan, 3], y=[1, np.nan, 3], onset=[0, 100, 200], duration=[50] * 3)
    for f in (True, False):
        np.testing.assert_array_equal(sample_fixations(unsorted, [-5, 0, 50, 150, 250], fast=f)["x"],
                                      [np.nan, 1, 1, 2, 3])
        np.testing.assert_array_equal(sample_fixations(tied, [50, 100, 150, 250], fast=f)["x"],
                                      [1, 3, 3, 9])
        np.testing.assert_array_equal(sample_fixations(missing, [50, 100, 150, 250], fast=f)["x"],
                                      [1, np.nan, np.nan, 3])


def test_print_fixation_group_skips_na_and_handles_empty():
    fg = fixation_group(x=[100, np.nan, 300], y=[10, 20, 30], onset=[0, 200, 400],
                        duration=[150, 150, 200])
    out = repr(fg)
    assert out.startswith("FixationGroup: 3 fixations")
    assert "x range: [100, 300]" in out
    assert "duration range: [150, 200]" in out


def test_calcangle_returns_zero_not_nan_for_parallel_vectors():
    assert calcangle([.1, .7], [.3, 2.1]) == pytest.approx(0, abs=1e-5)
    assert calcangle([1, 2], [-2, -4]) == pytest.approx(180, rel=1.5e-8)  # testthat default
    rng = np.random.default_rng(1)
    angles = []
    for _ in range(200):
        v = rng.uniform(size=2)
        angles.append(calcangle(v, rng.uniform(0.1, 10) * v))
    assert not np.isnan(angles).any()


def test_scanpath_keeps_row_order():
    fg = fixation_group(x=[0, 10, 30], y=[0, 0, 0], onset=[200, 0, 100], duration=[50] * 3)
    sp = scanpath(fg)
    np.testing.assert_array_equal(sp["x"], [0, 10, 30])
    np.testing.assert_array_equal(sp["lenx"], [10, 20, 0])


def test_crqa_uses_xy_columns_and_rejects_empty_and_bad_options():
    fga = fixation_group(x=[10, 200, 400, 30], y=[500, 20, 300, 80], onset=[0, 200, 400, 600],
                         duration=[200] * 4)
    fgb = fixation_group(x=[15, 210, 390, 600], y=[505, 25, 310, 900], onset=[0, 200, 400, 600],
                         duration=[200] * 4)
    res = crqa(fga, fgb, radius=60)
    assert res["recurrent_points"] == 3
    with pytest.raises(ValueError, match="non-empty"):
        crqa(fga.iloc[0:0], fgb)
    with pytest.raises(ValueError, match="metric"):
        crqa(fga, fgb, metric="maximum")
    with pytest.raises(ValueError, match="rescale"):
        crqa(fga, fgb, rescale=5)
    # rescale multiplies the radius by a statistic of the distance matrix.
    scaled = crqa(fga, fgb, radius=0.5, rescale=2)
    assert scaled["effective_radius"] == pytest.approx(0.5 * scaled["distance_matrix"].max())


def test_estimate_scale_defaults_and_window():
    ref = fixation_group(x=[100, 300, 500, 200], y=[100, 50, 300, 250], onset=[0, 1, 2, 3],
                         duration=[1] * 4)
    src = fixation_group(x=[50, 150, 250, 100], y=[200, 100, 600, 500], onset=[0, 1, 2, 3],
                         duration=[1] * 4)
    np.testing.assert_allclose(estimate_scale(ref, src)["par"], [2, 0.5], atol=1e-3)
    assert list(estimate_scale(ref, src, window=(10, 20))["par"]) == [1.0, 1.0]
