import numpy as np
import pandas as pd
import pytest

from peyesim.eye_table import EyeTable, simulate_eye_table
from peyesim.fixations import fixation_group
from peyesim.revisits import (
    REVISIT_SUMMARY_COLUMNS,
    RevisitTable,
    fixation_revisits,
    revisit_summary,
)
from tests.r_oracle import as_float, r_unavailable_reason, requires_r, run_r  # noqa: F401


def fg_abca():
    return fixation_group(x=[100, 400, 400, 105], y=[100, 100, 400, 98],
                          onset=[0, 250, 500, 750], duration=[200] * 4)


def _xy(fg):
    return np.column_stack([fg["x"].to_numpy(), fg["y"].to_numpy()])


# --- ports of tests/testthat/test_fixation_revisits.R -------------------------


def test_abca_scores_one_revisit_of_the_first_fixation():
    rv = fixation_revisits(fg_abca(), radius=50)
    assert isinstance(rv, RevisitTable)
    assert list(rv.columns) == ["index", "onset", "opportunity", "revisit", "lag",
                                "match_index", "distance", "expected"]
    assert rv["opportunity"].tolist() == [False, False, True, True]
    assert rv["revisit"].tolist() == [False, False, False, True]
    assert rv["match_index"].iloc[3] == 1
    assert rv["lag"].iloc[3] == 3
    assert rv["distance"].iloc[3] == pytest.approx(np.sqrt(5**2 + 2**2))
    s = revisit_summary(rv)
    assert list(s.columns) == REVISIT_SUMMARY_COLUMNS
    assert s["n_fix"].iloc[0] == 4
    assert s["opportunities"].iloc[0] == 2
    assert s["revisits"].iloc[0] == 1
    assert s["revisit_rate"].iloc[0] == 0.5
    assert np.isnan(s["expected_rate"].iloc[0])
    assert rv.attrs["radius"] == 50
    assert rv.attrs["null"] is False


def test_dwell_cluster_is_a_revisit_only_without_departure_rule():
    fg = fixation_group(x=[100, 105, 110], y=[100, 100, 100], onset=[0, 1, 2], duration=[1, 1, 1])
    strict = fixation_revisits(fg, radius=50)
    assert strict["opportunity"].sum() == 0
    assert strict["revisit"].sum() == 0
    loose = fixation_revisits(fg, radius=50, require_departure=False)
    assert loose["revisit"].tolist() == [False, False, True]
    assert loose["lag"].iloc[2] == 2
    lag1 = fixation_revisits(fg, radius=50, min_lag=1, require_departure=False)
    assert lag1["revisit"].tolist() == [False, True, True]
    assert lag1["lag"].iloc[2] == 1


def test_most_recent_match_is_reported_and_max_lag_limits_lookback():
    fg = fixation_group(x=[0, 300, 5, 300, 10], y=[0, 0, 0, 300, 0], onset=[1, 2, 3, 4, 5],
                        duration=[1] * 5)
    rv = fixation_revisits(fg, radius=50)
    assert rv["revisit"].tolist() == [False, False, True, False, True]
    assert rv["match_index"].iloc[4] == 3
    assert rv["lag"].iloc[4] == 2
    rv2 = fixation_revisits(fg, radius=50, min_lag=2, max_lag=2)
    assert rv2["revisit"].tolist() == [False, False, True, False, True]
    rv3 = fixation_revisits(fg, radius=50, min_lag=3, max_lag=3)
    assert rv3["revisit"].sum() == 0


def test_spatial_null_gives_conditional_proportion_of_reference_locations():
    fg = fg_abca()
    ref = np.array([[102, 101], [398, 102], [401, 399], [800, 800]], dtype=float)
    rv = fixation_revisits(fg, radius=50, reference=ref)
    assert rv["expected"].iloc[2] == pytest.approx(1 / 3)
    assert rv["expected"].iloc[3] == pytest.approx(2 / 3)
    s = revisit_summary(rv)
    assert s["expected_revisits"].iloc[0] == pytest.approx(1)
    assert s["expected_rate"].iloc[0] == pytest.approx(0.5)
    assert s["excess_revisits"].iloc[0] == pytest.approx(0)
    rv_df = fixation_revisits(fg, radius=50, reference=pd.DataFrame({"x": ref[:, 0], "y": ref[:, 1]}))
    np.testing.assert_array_equal(rv_df["expected"], rv["expected"])


def test_results_depend_on_onset_order_not_row_order():
    fg = fg_abca()
    shuffled = type(fg)(fg.to_pandas().iloc[[2, 0, 3, 1]])
    assert (fixation_revisits(shuffled, radius=50)["revisit"].tolist()
            == fixation_revisits(fg, radius=50)["revisit"].tolist())


def test_manhattan_distance_changes_the_location_criterion():
    fg = fixation_group(x=[0, 300, 30], y=[0, 0, 30], onset=[1, 2, 3], duration=[1, 1, 1])
    assert fixation_revisits(fg, radius=50)["revisit"].iloc[2]
    assert not fixation_revisits(fg, radius=50, dist_method="manhattan")["revisit"].iloc[2]


def test_empty_and_single_fixation_episodes_have_no_opportunities():
    empty = fixation_group([], [], [], [])
    rv = fixation_revisits(empty, radius=50)
    assert len(rv) == 0
    s = revisit_summary(rv)
    assert s["opportunities"].iloc[0] == 0
    assert np.isnan(s["revisit_rate"].iloc[0])
    one = fixation_group([100], [100], onset=[0], duration=[200])
    assert revisit_summary(fixation_revisits(one, radius=50))["opportunities"].iloc[0] == 0


def test_invalid_arguments_are_rejected():
    fg = fg_abca()
    with pytest.raises(ValueError, match="radius"):
        fixation_revisits(fg)
    with pytest.raises(ValueError, match="radius"):
        fixation_revisits(fg, radius=-1)
    with pytest.raises(ValueError, match="min_lag"):
        fixation_revisits(fg, radius=50, min_lag=0)
    with pytest.raises(ValueError, match="max_lag"):
        fixation_revisits(fg, radius=50, min_lag=3, max_lag=2)
    with pytest.raises(ValueError, match="two columns"):
        fixation_revisits(fg, radius=50, reference=np.array([[1, 2, 3]]))
    with pytest.raises(TypeError, match="fixation_revisits"):
        revisit_summary(pd.DataFrame())


def _simulated(seed, n_fixations, n_groups):
    np.random.seed(seed)
    return simulate_eye_table(n_fixations=n_fixations, n_groups=n_groups)


def test_eye_table_method_with_leave_episode_out_subject_null():
    et = _simulated(3, 80, 4)
    et["subject"] = ["s1", "s1", "s2", "s2"]
    res = fixation_revisits(et, radius=200, null="subject", subject_var="subject", keep_fixations=True)
    assert isinstance(res, EyeTable)
    assert len(res) == 4
    assert set(REVISIT_SUMMARY_COLUMNS) <= set(res.columns)
    ref = _xy(et["fixgroup"].iloc[1])
    direct = revisit_summary(fixation_revisits(et["fixgroup"].iloc[0], radius=200, reference=ref))
    assert res["expected_revisits"].iloc[0] == pytest.approx(direct["expected_revisits"].iloc[0], nan_ok=True)
    assert res["revisits"].iloc[0] == direct["revisits"].iloc[0]
    assert res["opportunities"].sum() > 0
    rates = res["expected_rate"][res["opportunities"] > 0]
    assert np.all(np.isfinite(rates))
    assert isinstance(res["revisit_fixations"].iloc[0], RevisitTable)
    plain = fixation_revisits(et, radius=200)
    assert plain["expected_rate"].isna().all()
    with pytest.raises(ValueError, match="subject_var"):
        fixation_revisits(et, radius=200, null="subject")


def test_item_null_pools_other_subjects_viewing_the_same_item():
    et = _simulated(4, 120, 6)
    et["subject"] = ["s1", "s1", "s2", "s2", "s3", "s3"]
    et["image"] = ["a", "b", "a", "b", "a", "b"]
    fgs = list(et["fixgroup"])
    res = fixation_revisits(et, radius=200, null="item", item_var="image", subject_var="subject")
    ref = np.vstack([_xy(fgs[2]), _xy(fgs[4])])
    direct = revisit_summary(fixation_revisits(fgs[0], radius=200, reference=ref))
    assert res["expected_revisits"].iloc[0] == pytest.approx(direct["expected_revisits"].iloc[0], nan_ok=True)

    et2 = et.copy()
    et2["subject"] = ["s2", "s1", "s2", "s2", "s3", "s3"]
    with_subj = fixation_revisits(et2, radius=200, null="item", item_var="image", subject_var="subject")
    no_subj = fixation_revisits(et2, radius=200, null="item", item_var="image")
    ref3_subj = _xy(fgs[4])
    ref3_all = np.vstack([_xy(fgs[0]), _xy(fgs[4])])
    e_subj = revisit_summary(fixation_revisits(fgs[2], radius=200, reference=ref3_subj))
    e_all = revisit_summary(fixation_revisits(fgs[2], radius=200, reference=ref3_all))
    assert with_subj["expected_revisits"].iloc[2] == pytest.approx(e_subj["expected_revisits"].iloc[0], nan_ok=True)
    assert no_subj["expected_revisits"].iloc[2] == pytest.approx(e_all["expected_revisits"].iloc[0], nan_ok=True)
    with pytest.raises(ValueError, match="item_var"):
        fixation_revisits(et, radius=200, null="item")
    with pytest.raises(ValueError, match="subject_var"):
        fixation_revisits(et, radius=200, null="item", item_var="image", subject_var="nope")


def test_rescoring_a_scored_eye_table_replaces_the_summary_columns():
    et = _simulated(5, 60, 3)
    first = fixation_revisits(et, radius=100, keep_fixations=True)
    again = fixation_revisits(first, radius=300)
    assert list(again.columns) == [c for c in first.columns if c != "revisit_fixations"]
    np.testing.assert_array_equal(again["revisits"], fixation_revisits(et, radius=300)["revisits"])
    assert again.origin == et.origin
    assert again.attrs["revisit_columns"] == fixation_revisits(et, radius=300).attrs["revisit_columns"]


def test_callers_column_with_a_summary_name_is_never_overwritten():
    et = _simulated(6, 60, 3)
    et["n_fix"] = [99, 98, 97]
    with pytest.raises(ValueError, match="n_fix"):
        fixation_revisits(et, radius=100)
    assert et["n_fix"].tolist() == [99, 98, 97]
    renamed = EyeTable(et.to_pandas().rename(columns={"n_fix": "tracker_n_fix"}), origin=et.origin)
    res = fixation_revisits(renamed, radius=100)
    assert res["tracker_n_fix"].tolist() == [99, 98, 97]


def test_empty_reference_gives_nan_expectations_not_zero():
    rv = fixation_revisits(fg_abca(), radius=50, reference=np.empty((0, 2)))
    s = revisit_summary(rv)
    assert np.isnan(s["expected_revisits"].iloc[0])
    assert np.isnan(s["excess_rate"].iloc[0])
    et = _simulated(7, 40, 2)
    et["subject"] = ["s1", "s2"]
    et["image"] = ["a", "b"]
    res = fixation_revisits(et, radius=200, null="item", item_var="image", subject_var="subject")
    assert res["expected_revisits"].isna().all()


def test_none_fixation_group_is_scored_as_empty():
    tab = pd.DataFrame({"fixgroup": [fg_abca(), None]})
    res = fixation_revisits(tab, radius=50)
    assert res["n_fix"].tolist() == [4, 0]


# --- live R parity ------------------------------------------------------------

_rng = np.random.default_rng(20261023)


def _rand_group(n, spread=300.0):
    x = np.round(_rng.uniform(0, spread, n), 1)
    y = np.round(_rng.uniform(0, spread, n), 1)
    onset = np.round(np.cumsum(_rng.uniform(100, 300, n)), 1)
    return x, y, onset


GROUPS = {
    "abca": ([100, 400, 400, 105], [100, 100, 400, 98], [0, 250, 500, 750]),
    "dwell": ([100, 105, 110, 300, 102, 104], [100, 100, 100, 100, 101, 99], [0, 1, 2, 3, 4, 5]),
    "aba": ([0, 300, 5, 300, 10], [0, 0, 0, 300, 0], [1, 2, 3, 4, 5]),
    # unsorted onsets and a missing coordinate / onset
    "messy": ([105, 400, np.nan, 100, 400, 120, 101], [98, 400, 50, 100, 100, 110, 99],
              [750, 500, 300, 0, 250, np.nan, 900]),
}
for _k in range(6):
    GROUPS[f"rand{_k}"] = _rand_group(18)

REF = np.round(_rng.uniform(0, 300, (40, 2)), 1)

SETTINGS = [
    {"radius": 50},
    {"radius": 50, "require_departure": False},
    {"radius": 60, "min_lag": 1},
    {"radius": 80, "min_lag": 2, "max_lag": 3},
    {"radius": 50, "dist_method": "manhattan"},
    {"radius": 70, "reference": True},
    {"radius": 70, "reference": True, "require_departure": False, "dist_method": "manhattan"},
]

ET_SUBJECT = ["s1", "s1", "s2", "s2", "s3", "s3"]
ET_IMAGE = ["a", "b", "a", "b", "a", "b"]


def _r_vec(values):
    return "c(" + ", ".join("NA" if np.isnan(v) else repr(float(v)) for v in np.asarray(values, float)) + ")"


def _r_setting(s):
    parts = []
    for k, v in s.items():
        if k == "reference":
            parts.append("reference = REF")
        elif isinstance(v, bool):
            parts.append(f"{k} = {'TRUE' if v else 'FALSE'}")
        elif isinstance(v, str):
            parts.append(f'{k} = "{v}"')
        else:
            parts.append(f"{k} = {v}")
    return ", ".join(parts)


def _py_setting(s):
    s = dict(s)
    if s.pop("reference", False):
        s["reference"] = REF
    return s


def _make_fg(key):
    x, y, onset = GROUPS[key]
    return fixation_group(x=x, y=y, onset=onset, duration=[100] * len(x))


def _build_r_code():
    lines = ["REF <- matrix(" + _r_vec(REF.ravel()) + ", ncol = 2, byrow = TRUE)", "G <- list()"]
    for key, (x, y, onset) in GROUPS.items():
        lines.append(
            f"G[['{key}']] <- fixation_group(x = {_r_vec(x)}, y = {_r_vec(y)}, "
            f"onset = {_r_vec(onset)}, duration = rep(100, {len(x)}))"
        )
    lines.append("tab <- function(rv) as.data.frame(unclass_revisit(rv))")
    lines.append("per <- list()")
    for gi, key in enumerate(GROUPS):
        for si, s in enumerate(SETTINGS):
            lines.append(
                f"rv <- fixation_revisits(G[['{key}']], {_r_setting(s)}); "
                f"per[['{gi}_{si}']] <- list(table = tab(rv), summary = as.data.frame(revisit_summary(rv)))"
            )
    rand_keys = [f"rand{k}" for k in range(6)]
    lines.append(
        "et <- as_eye_table(tibble::tibble(subject = " + "c(" + ", ".join(f'"{s}"' for s in ET_SUBJECT) + "), "
        "image = c(" + ", ".join(f'"{s}"' for s in ET_IMAGE) + "), "
        "fixgroup = G[c(" + ", ".join(f'"{k}"' for k in rand_keys) + ")]))"
    )
    lines.append(
        "sumcols <- function(res) as.data.frame(tibble::as_tibble(res)[, revisit_summary_cols])\n"
        "et_out <- list(\n"
        "  none = sumcols(fixation_revisits(et, radius = 60)),\n"
        "  subject = sumcols(fixation_revisits(et, radius = 60, null = 'subject', subject_var = 'subject')),\n"
        "  item = sumcols(fixation_revisits(et, radius = 60, null = 'item', item_var = 'image')),\n"
        "  item_subj = sumcols(fixation_revisits(et, radius = 60, null = 'item', item_var = 'image', "
        "subject_var = 'subject')),\n"
        "  subject_manhattan = sumcols(fixation_revisits(et, radius = 60, null = 'subject', "
        "subject_var = 'subject', dist_method = 'manhattan', require_departure = FALSE, max_lag = 4))\n"
        ")"
    )
    lines.append("out <- list(per = per, et = et_out)")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def r():
    return run_r(_build_r_code())


def _assert_table_equal(py, r_tab, label):
    if len(py) == 0:
        assert r_tab["index"] == [] or r_tab.get("index") in (None, [])
        return
    np.testing.assert_array_equal(np.asarray(py["index"], float), as_float(r_tab["index"]), err_msg=label)
    np.testing.assert_allclose(py["onset"].to_numpy(float), as_float(r_tab["onset"]), err_msg=label)
    for col in ("opportunity", "revisit"):
        assert py[col].tolist() == [bool(v) for v in r_tab[col]], f"{label}: {col}"
    for col in ("lag", "match_index", "distance", "expected"):
        np.testing.assert_allclose(
            py[col].to_numpy(dtype=float, na_value=np.nan), as_float(r_tab[col]),
            rtol=1e-12, equal_nan=True, err_msg=f"{label}: {col}",
        )


def _assert_summary_equal(py, r_sum, label):
    for col in REVISIT_SUMMARY_COLUMNS:
        np.testing.assert_allclose(
            py[col].to_numpy(dtype=float), as_float(r_sum[col]).reshape(-1),
            rtol=1e-12, equal_nan=True, err_msg=f"{label}: {col}",
        )


@requires_r
@pytest.mark.parametrize("gi,key", list(enumerate(GROUPS)))
def test_per_fixation_tables_match_r(r, gi, key):
    for si, s in enumerate(SETTINGS):
        rv = fixation_revisits(_make_fg(key), **_py_setting(s))
        expected = r["per"][f"{gi}_{si}"]
        label = f"{key} {s}"
        _assert_table_equal(rv, expected["table"], label)
        _assert_summary_equal(revisit_summary(rv), expected["summary"], label)


@requires_r
def test_eye_table_summaries_match_r(r):
    et = EyeTable(
        pd.DataFrame({
            "subject": ET_SUBJECT,
            "image": ET_IMAGE,
            "fixgroup": [_make_fg(f"rand{k}") for k in range(6)],
        })
    )
    cases = {
        "none": dict(radius=60),
        "subject": dict(radius=60, null="subject", subject_var="subject"),
        "item": dict(radius=60, null="item", item_var="image"),
        "item_subj": dict(radius=60, null="item", item_var="image", subject_var="subject"),
        "subject_manhattan": dict(radius=60, null="subject", subject_var="subject",
                                  dist_method="manhattan", require_departure=False, max_lag=4),
    }
    for name, kwargs in cases.items():
        res = fixation_revisits(et, **kwargs)
        _assert_summary_equal(res, r["et"][name], name)
