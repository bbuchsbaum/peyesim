"""Port of test_sample_density_time.R"""

import numpy as np
import pandas as pd
import pytest
from peyesim import fixation_group, sample_density_time
from peyesim.density import EyeDensity


def _make_dens(px, py, sigma=2):
    g = np.arange(0, 10.5, 0.5)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    z = np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * sigma**2))
    return EyeDensity(x=g, y=g, z=z, sigma=sigma)


def _make_fg(xs, ys, onsets):
    return fixation_group(xs, ys, onset=onsets, duration=[100]*len(xs))


def test_basic_functionality():
    tt = pd.DataFrame({
        "trial_id": ["A", "B"],
        "density": [_make_dens(2, 2), _make_dens(8, 8)],
    })
    st = pd.DataFrame({
        "trial_id": ["A", "B"],
        "fixgroup": [
            _make_fg([2, 3, 4], [2, 3, 4], [0, 100, 200]),
            _make_fg([8, 7, 6], [8, 7, 6], [0, 100, 200]),
        ],
    })
    result = sample_density_time(tt, st, match_on="trial_id", times=[0, 100, 200])
    assert "sampled" in result.columns
    assert len(result) == 2
    for s in result["sampled"]:
        assert "z" in s.columns
        assert "time" in s.columns


def test_time_bins_aggregation():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({"trial_id": ["A"],
                        "fixgroup": [_make_fg([5]*6, [5]*6,
                                              [0, 50, 100, 150, 200, 250])]})
    result = sample_density_time(tt, st, match_on="trial_id",
                                 times=np.arange(0, 301, 50),
                                 time_bins=[0, 150, 300])
    assert "bin_1" in result.columns
    assert "bin_2" in result.columns
    assert np.isfinite(result["bin_1"].iloc[0])
    assert np.isfinite(result["bin_2"].iloc[0])


def test_permutations_add_baseline():
    tt = pd.DataFrame({
        "trial_id": ["A", "B", "C"],
        "subject": ["S1", "S1", "S1"],
        "density": [_make_dens(2, 2), _make_dens(5, 5), _make_dens(8, 8)],
    })
    st = pd.DataFrame({
        "trial_id": ["A", "B", "C"],
        "subject": ["S1", "S1", "S1"],
        "fixgroup": [
            _make_fg([2, 3], [2, 3], [0, 100]),
            _make_fg([5, 6], [5, 6], [0, 100]),
            _make_fg([8, 7], [8, 7], [0, 100]),
        ],
    })
    result = sample_density_time(tt, st, match_on="trial_id",
                                 times=[0, 100], time_bins=[0, 100, 200],
                                 permutations=10, permute_on="subject")
    assert "perm_sampled" in result.columns
    assert "perm_bin_1" in result.columns
    assert "diff_bin_1" in result.columns


def test_handles_missing_matches():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({
        "trial_id": ["A", "B"],
        "fixgroup": [_make_fg([5], [5], [0]), _make_fg([5], [5], [0])],
    })
    with pytest.warns(UserWarning, match="Did not find matching template"):
        result = sample_density_time(tt, st, match_on="trial_id", times=[0, 100])
    assert len(result) == 1


def test_validates_inputs():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({"trial_id": ["A"], "fixgroup": [_make_fg([5], [5], [0])]})

    with pytest.raises(ValueError, match="not found"):
        sample_density_time(tt, st, match_on="nonexistent", times=[0, 100])

    with pytest.raises(ValueError, match="monotonically increasing"):
        sample_density_time(tt, st, match_on="trial_id", times=[0, 100],
                            time_bins=[100, 50])


def test_preserves_source_columns():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({
        "trial_id": ["A"], "subject": ["S1"], "condition": ["test"],
        "extra_col": [42],
        "fixgroup": [_make_fg([5], [5], [0])],
    })
    result = sample_density_time(tt, st, match_on="trial_id", times=[0, 100])
    assert "subject" in result.columns
    assert "condition" in result.columns
    assert result["extra_col"].iloc[0] == 42


def test_custom_aggregate_fun():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([5] * 4, [5] * 4, [0, 50, 100, 150])],
    })

    def agg_mean(vals, **kw):
        return np.nanmean(vals)

    result_mean = sample_density_time(
        tt,
        st,
        match_on="trial_id",
        times=np.arange(0, 201, 50),
        time_bins=[0, 200],
        aggregate_fun=agg_mean,
    )
    result_max = sample_density_time(
        tt,
        st,
        match_on="trial_id",
        times=np.arange(0, 201, 50),
        time_bins=[0, 200],
        aggregate_fun=lambda vals, **kw: np.nanmax(vals),
    )

    assert result_max["bin_1"].iloc[0] >= result_mean["bin_1"].iloc[0]


def test_sampled_values_correct_for_peaks():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5, sigma=2)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([5, 0, 3], [5, 0, 3], [0, 100, 200])],
    })
    result = sample_density_time(tt, st, match_on="trial_id", times=[0, 100, 200])
    s = result["sampled"].iloc[0]
    val_peak = s["z"].iloc[0]
    val_origin = s["z"].iloc[1]
    val_mid = s["z"].iloc[2]
    assert val_peak > 0.9
    assert val_origin < 0.1
    assert val_mid > val_origin
    assert val_mid < val_peak


def test_permutation_baseline_uses_nonmatching():
    tt = pd.DataFrame({
        "trial_id": ["A", "B"], "subject": ["S1", "S1"],
        "density": [_make_dens(2, 2, sigma=1), _make_dens(8, 8, sigma=1)],
    })
    st = pd.DataFrame({
        "trial_id": ["A", "B"], "subject": ["S1", "S1"],
        "fixgroup": [_make_fg([2], [2], [0]), _make_fg([8], [8], [0])],
    })
    result = sample_density_time(tt, st, match_on="trial_id", times=[0],
                                 time_bins=[0, 100], permutations=10,
                                 permute_on="subject")
    assert result["bin_1"].iloc[0] > 0.9
    assert result["perm_bin_1"].iloc[0] < 0.1
    assert result["diff_bin_1"].iloc[0] > 0.8


def test_null_density_handled():
    tt = pd.DataFrame({
        "trial_id": ["A", "B"],
        "density": [_make_dens(5, 5), None],
    })
    st = pd.DataFrame({
        "trial_id": ["A", "B"],
        "fixgroup": [_make_fg([5], [5], [0]), _make_fg([5], [5], [0])],
    })
    result = sample_density_time(tt, st, match_on="trial_id",
                                 times=[0, 100], time_bins=[0, 200])
    assert np.isfinite(result["bin_1"].iloc[0])
    assert np.isnan(result["bin_1"].iloc[1])


def test_temporal_interpolation_forward_fill():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(2, 2, sigma=2)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([2, 8, 8], [2, 8, 8], [0, 200, 400])],
    })
    result = sample_density_time(tt, st, match_on="trial_id", times=[0, 100, 200, 300])
    s = result["sampled"].iloc[0]

    assert s["z"].iloc[0] == pytest.approx(s["z"].iloc[1], abs=0.01)
    assert s["z"].iloc[0] > 0.9
    assert s["z"].iloc[2] == pytest.approx(s["z"].iloc[3], abs=0.01)
    assert s["z"].iloc[2] < 0.1


def test_multi_subject_permutation_stays_within_subject():
    tt = pd.DataFrame({
        "trial_id": ["S1_A", "S1_B", "S2_A", "S2_B"],
        "subject": ["S1", "S1", "S2", "S2"],
        "density": [
            _make_dens(2, 2, sigma=1),
            _make_dens(3, 3, sigma=1),
            _make_dens(7, 7, sigma=1),
            _make_dens(8, 8, sigma=1),
        ],
    })
    st = pd.DataFrame({
        "trial_id": ["S1_A", "S1_B", "S2_A", "S2_B"],
        "subject": ["S1", "S1", "S2", "S2"],
        "fixgroup": [
            _make_fg([2], [2], [0]),
            _make_fg([3], [3], [0]),
            _make_fg([7], [7], [0]),
            _make_fg([8], [8], [0]),
        ],
    })

    np.random.seed(123)
    result = sample_density_time(
        tt,
        st,
        match_on="trial_id",
        times=[0],
        time_bins=[0, 100],
        permutations=10,
        permute_on="subject",
    )

    assert all(result["bin_1"] > 0.9)
    assert all(result.loc[result["subject"] == "S1", "perm_bin_1"] > 0.3)
    assert all(result.loc[result["subject"] == "S2", "perm_bin_1"] > 0.3)


def test_bin_boundaries_edge_cases():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([5] * 5, [5] * 5, [0, 100, 200, 300, 400])],
    })
    result = sample_density_time(
        tt,
        st,
        match_on="trial_id",
        times=[0, 100, 200, 300, 400],
        time_bins=[0, 200, 400, 500],
    )

    assert "bin_1" in result.columns
    assert "bin_2" in result.columns
    assert "bin_3" in result.columns
    assert np.isfinite(result["bin_1"].iloc[0])
    assert np.isfinite(result["bin_2"].iloc[0])
    assert np.isfinite(result["bin_3"].iloc[0])


def test_empty_time_bins_return_nan():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([5, 5], [5, 5], [0, 50])],
    })
    result = sample_density_time(
        tt,
        st,
        match_on="trial_id",
        times=[0, 50],
        time_bins=[0, 100, 200, 300],
    )

    assert np.isfinite(result["bin_1"].iloc[0])
    assert np.isnan(result["bin_2"].iloc[0])
    assert np.isnan(result["bin_3"].iloc[0])


def test_sampled_time_series_correct_length():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({"trial_id": ["A"], "fixgroup": [_make_fg([5], [5], [0])]})
    times = np.arange(0, 1001, 25)

    result = sample_density_time(tt, st, match_on="trial_id", times=times)

    assert len(result["sampled"].iloc[0]) == len(times)
    np.testing.assert_array_equal(result["sampled"].iloc[0]["time"].values, times)
