"""Regression tests for core behaviours fixed to match R eyesim.

Each test pins a difference found by tests/test_parity_random_core.py; the
expected values are what R eyesim 0.3.0 returns (noted per test) and need no R
at run time.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import peyesim as pe
from peyesim.entropy import _r_as_character


def _fg(x, y, onset=None, duration=None):
    onset = np.arange(len(x), dtype=float) * 100 if onset is None else onset
    duration = np.full(len(x), 100.0) if duration is None else duration
    return pe.fixation_group(x, y, duration=duration, onset=onset)


def test_table_with_fixation_groups_prints():
    # pandas treated FixationGroup as a sequence and iterated it through
    # __getitem__(0), so printing any table with a fixgroup column raised KeyError.
    et = pe.simulate_eye_table(20, 2)
    text = repr(et.to_pandas())
    assert "FixationGroup: 10 fixations" in text
    assert "FixationGroup" in pe.load_dataset("wynn_test").head(2).to_string()
    assert not pd.api.types.is_list_like(et["fixgroup"].iloc[0])


def test_center_and_summary_propagate_missing_like_r():
    fg = _fg([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    # R: center(fg)$x is NA NA NA; y is centred normally.
    assert np.isnan(pe.center(fg)["x"]).all()
    np.testing.assert_array_equal(pe.center(fg)["y"], [-1.0, 0.0, 1.0])
    s = fg.summary()
    assert np.isnan(s["cen_x"]) and np.isnan(s["sd_x"])
    assert s["cen_y"] == 2.0 and s["sd_y"] == 1.0


def test_concat_missing_onset_propagates_like_r():
    # R: c(fg_with_NA_onset, fg)$onset is 0 NA NA (max() of an NA onset is NA).
    a = _fg([1.0, 2.0], [1.0, 2.0], onset=[0.0, np.nan], duration=[1.0, np.nan])
    b = _fg([1.0], [1.0], onset=[0.0], duration=[1.0])
    onset = pe.concat_fixation_groups(a, b)["onset"].to_numpy()
    assert onset[0] == 0 and np.isnan(onset[1:]).all()


def test_rep_fixations_rejects_missing_duration():
    # R: rep() fails with "invalid 'times' argument".
    fg = _fg([1.0, 2.0], [1.0, 2.0], duration=[np.nan, 1.0])
    with pytest.raises(ValueError, match="finite durations"):
        pe.rep_fixations(fg, 100)


def test_scanpath_of_empty_group_is_empty():
    # R returns 0 rows; peyesim returned one row of NaN.
    sp = pe.scanpath(pe.fixation_group([], [], duration=[], onset=[]))
    assert len(sp) == 0
    assert {"lenx", "leny", "rho", "theta"} <= set(sp.columns)
    one = pe.scanpath(_fg([1.0], [2.0]))
    assert one[["lenx", "leny", "rho", "theta"]].to_numpy().tolist() == [[0.0, 0.0, 0.0, 0.0]]


def test_fixation_group_requires_onset():
    with pytest.raises(ValueError, match="requires onset"):
        pe.fixation_group([1, 2], [1, 2])


def test_eye_table_keeps_missing_group_keys_last():
    # dplyr::group_by() keeps NA as its own group, sorted last: R gives "a" "b" NA.
    df = pd.DataFrame({"x": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4], "d": 1.0, "o": [0.0, 1, 2, 3],
                       "g": [None, "b", "a", "b"]})
    et = pe.eye_table("x", "y", "d", "o", "g", df)
    assert et["g"].tolist()[:2] == ["a", "b"] and pd.isna(et["g"].iloc[2])
    assert [len(f) for f in et["fixgroup"]] == [1, 2, 1]


def test_eye_table_with_everything_clipped_keeps_columns():
    df = pd.DataFrame({"x": [-5.0], "y": [1.0], "d": [1.0], "o": [0.0], "g": ["a"], "v": [1]})
    et = pe.eye_table("x", "y", "d", "o", "g", df, vars=["v"], clip_bounds=(0, 10, 0, 10))
    assert len(et) == 0 and list(et.columns) == ["g", "fixgroup", "v"]


def test_simulate_eye_table_orders_groups_numerically():
    # R uses factor(g, levels = 1:n_groups).
    et = pe.simulate_eye_table(120, 12)
    assert et["groupvar"].tolist() == [str(g) for g in range(1, 13)]


def _density_table():
    df = pd.DataFrame({
        "x": [100.0, 200, 300, 400, 500, 600, 700, 150],
        "y": [100.0, 150, 200, 250, 300, 350, 400, 120],
        "d": [100.0] * 8, "o": np.arange(8) * 100.0,
        "s": ["s1", "s1", "s2", "s2", None, None, "s1", "s2"],
        "k": np.arange(8),
        "c": [1, 1, 2, 2, 3, 3, 1, 2],
    })
    return pe.eye_table("x", "y", "d", "o", ["s", "k"], df, vars=["c"], clip_bounds=(0, 800, 0, 600))


def test_density_by_passes_extra_arguments_to_eye_density():
    et = _density_table()
    res = pe.density_by(et, groups="s", sigma=200, xbounds=(0, 800), ybounds=(0, 600),
                        outdim=(10, 8), kde_pkg="MASS", normalize=False)
    fg = res["fixgroup"].iloc[0]
    expected = pe.eye_density(fg, sigma=200, xbounds=(0, 800), ybounds=(0, 600), outdim=(10, 8),
                              duration_weighted=True, kde_pkg="MASS", normalize=False)
    np.testing.assert_array_equal(res["density"].iloc[0].z, expected.z)


def test_density_by_column_order_and_missing_groups_like_r():
    et = _density_table()
    res = pe.density_by(et, groups="s", keep_vars=["c"], sigma=80, xbounds=(0, 800),
                        ybounds=(0, 600), outdim=(10, 8))
    # R: names are groups, keep_vars, fixvar, result_name; the NA group is kept, last.
    assert list(res.columns) == ["s", "c", "fixgroup", "density"]
    assert res["s"].tolist()[:2] == ["s1", "s2"] and pd.isna(res["s"].iloc[2])
    assert [len(f) for f in res["fixgroup"]] == [3, 3, 2]


def test_eye_density_missing_coordinates_return_none():
    # R: ks::kde() (or the bounds) fail and eye_density() warns and returns NULL.
    fg = _fg([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    for kw in ({"xbounds": (0, 4), "ybounds": (0, 4)}, {}):
        for pkg in ("ks", "MASS"):
            with pytest.warns(UserWarning, match="missing"):
                assert pe.eye_density(fg, sigma=1, outdim=(3, 3), kde_pkg=pkg, **kw) is None


def test_suggest_sigma_rejects_missing_coordinates():
    # R: IQR() stops with "missing values and NaN's not allowed".
    fg = _fg([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="missing"):
        pe.suggest_sigma(fg)
    assert np.isnan(pe.suggest_sigma(_fg([1.0], [np.nan])))  # n < 2 first, as in R


@pytest.mark.parametrize("value, expected", [
    (30, "30"), (45.5, "45.5"), (12.345678901, "12.345678901"), (1e5, "1e+05"),
    (123456, "123456"), (0.1 + 0.2, "0.3"), (1e-4, "1e-04"), (2.5e-7, "2.5e-07"),
])
def test_multiscale_entropy_names_follow_r_as_character(value, expected):
    assert _r_as_character(value) == expected


def test_multiscale_entropy_names_keep_full_precision():
    fg = _fg([100.0, 500, 900], [100.0, 500, 900])
    ms = pe.eye_density(fg, sigma=[1.23456789, 2], xbounds=(0, 1000), ybounds=(0, 1000), outdim=(5, 5))
    # R: names sigma_1.23456789 and sigma_2
    assert list(pe.fixation_entropy(ms, aggregate="none")) == ["sigma_1.23456789", "sigma_2"]


def test_fixation_entropy_ignores_the_other_methods_arguments():
    fg = _fg([100.0, 300, 500, 700, 200], [100.0, 300, 200, 400, 250])
    grid = pe.fixation_entropy(fg, method="grid")
    assert pe.fixation_entropy(fg, method="grid", sigma=10, outdim=(5, 5),
                               duration_weighted=True) == grid
    dens = pe.fixation_entropy(fg, method="density", sigma=50)
    assert pe.fixation_entropy(fg, method="density", sigma=50, grid=(3, 3)) == dens
    with pytest.raises(TypeError):
        pe.fixation_entropy(fg, method="grid", not_an_argument=1)


# CRQA: values from crqa 2.1.0 via R eyesim's crqa() wrapper.
CRQA_A = _fg([0.0, 0.0, 100.0, 300.0], [0.0, 0.0, 100.0, 300.0])
CRQA_B = _fg([0.0, 100.0, 100.0, 300.0], [0.0, 100.0, 100.0, 300.0])


def test_crqa_vertical_lines_run_down_rp_columns():
    # crqa counts vertical lines down the columns of RP (fg2 along the rows).
    got = pe.crqa(CRQA_A, CRQA_B, radius=1)
    rp = np.asarray(got["RP"], dtype=int)
    assert rp.tolist() == [[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # column 3 of RP holds the only vertical line (length 2): LAM = 2/5, TT = 2
    assert got["LAM"] == pytest.approx(40.0)
    assert got["TT"] == 2.0 and got["max_vertlength"] == 2


def test_crqa_relative_entropy_uses_distinct_line_lengths():
    lengths = [2, 2, 5]
    entr = pe.crqa.__globals__["_line_entropy"](lengths)
    rentr = pe.crqa.__globals__["_relative_line_entropy"](lengths)
    assert rentr == pytest.approx(entr / np.log(2))  # 2 distinct lengths, not 5 - 2 + 1


def test_crqa_recurrence_without_diagonal_lines_matches_crqa():
    a = _fg([0.0, 500.0], [0.0, 500.0])
    b = _fg([900.0, 0.0], [900.0, 0.0])
    got = pe.crqa(a, b, radius=10)
    assert got["RR"] == 25.0 and got["NRLINE"] == 0 and got["maxL"] == 0 and got["L"] == 0
    assert np.isnan(got["DET"]) and np.isnan(got["ENTR"]) and np.isnan(got["rENTR"])
    assert got["LAM"] == 0 and got["TT"] == 0 and np.isnan(got["max_vertlength"])


def test_crqa_rescale_4_with_one_point_divides_by_zero_like_crqa():
    # crqa: radius * |sum(d) / (1 - 1)| = Inf, so the single pair is recurrent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = pe.crqa(_fg([0.0], [0.0]), _fg([50.0], [0.0]), radius=0.5, rescale=4)
    assert got["effective_radius"] == np.inf and got["RR"] == 100.0


def test_coordinate_transforms_accept_integer_columns():
    # An int64 x column (as JSON gives for whole-number coordinates) made
    # center/rescale/normalize raise TypeError: the in-place assignment kept int64.
    g = pe.FixationGroup(pd.DataFrame({"index": [1, 2], "x": [1, 2], "y": [3, 4],
                                       "duration": [1, 1], "onset": [0, 1]}))
    assert pe.center(g)["x"].tolist() == [-0.5, 0.5]
    assert pe.rescale(g, 0.5, 0.5)["x"].tolist() == [0.5, 1.0]
    assert pe.normalize(g, (0, 4), (0, 8))["y"].tolist() == [0.375, 0.5]


def test_bundled_fixation_groups_use_r_column_types():
    # R: x, y, group_index double; index, duration, onset integer.
    for name in ("wynn_study", "wynn_test"):
        data = pe.load_dataset(name)
        for fg in data["fixgroup"]:
            dtypes = fg.to_pandas(copy=False).dtypes
            assert dtypes["x"].kind == dtypes["y"].kind == dtypes["group_index"].kind == "f"
            assert dtypes["index"].kind == dtypes["duration"].kind == dtypes["onset"].kind == "i"
