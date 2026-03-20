"""Tests for coverage gaps identified in audit: fixation_overlap, suggest_sigma,
concat_fixation_groups, estimate_scale, temporal interpolation, bin edges,
empty bins, multi-subject permutation, custom aggregate_fun, sampled length."""

import numpy as np
import pandas as pd
import pytest
from peyesim import (
    fixation_group, concat_fixation_groups, sample_density, sample_density_time,
)
from peyesim.density import EyeDensity, suggest_sigma
from peyesim.overlap import fixation_overlap


# ---------------------------------------------------------------------------
# fixation_overlap
# ---------------------------------------------------------------------------

def test_fixation_overlap_basic():
    fg1 = fixation_group(
        x=np.random.uniform(0, 100, 50),
        y=np.random.uniform(0, 100, 50),
        duration=np.ones(50),
        onset=np.arange(1, 51, dtype=float),
    )
    fg2 = fixation_group(
        x=np.random.uniform(0, 100, 50),
        y=np.random.uniform(0, 100, 50),
        duration=np.ones(50),
        onset=np.arange(1, 51, dtype=float),
    )
    result = fixation_overlap(fg1, fg2, dthresh=60)
    assert "overlap" in result
    assert "perc" in result
    assert 0 <= result["perc"] <= 1


# ---------------------------------------------------------------------------
# suggest_sigma
# ---------------------------------------------------------------------------

def test_suggest_sigma_requires_y():
    with pytest.raises(ValueError, match="y.*must be provided"):
        suggest_sigma([1, 2, 3])


def test_suggest_sigma_clamps_to_display():
    fg = fixation_group(
        x=np.random.uniform(640, 641, 30),  # tightly clustered
        y=np.random.uniform(512, 513, 30),
        onset=np.cumsum(np.full(30, 200.0)),
        duration=np.full(30, 200.0),
    )
    sigma = suggest_sigma(fg, xbounds=(0, 1280), ybounds=(0, 1024))
    display_scale = np.mean([1280, 1024])
    assert sigma >= display_scale * 0.01  # at least 1%


# ---------------------------------------------------------------------------
# concat_fixation_groups
# ---------------------------------------------------------------------------

def test_concat_fixation_groups():
    fg1 = fixation_group(x=[1, 2], y=[3, 4], onset=[0, 100], duration=[100, 100])
    fg2 = fixation_group(x=[5, 6], y=[7, 8], onset=[0, 100], duration=[100, 100])
    combined = concat_fixation_groups(fg1, fg2)

    assert len(combined) == 4
    # Onsets of fg2 should be shifted
    assert combined["onset"].iloc[2] >= combined["onset"].iloc[1] + combined["duration"].iloc[1]
    # Indices recomputed
    np.testing.assert_array_equal(combined["index_col"].values, [1, 2, 3, 4])


def test_concat_fixation_groups_none_handling():
    fg1 = fixation_group(x=[1], y=[2], onset=[0], duration=[100])
    assert concat_fixation_groups() is None
    result = concat_fixation_groups(fg1)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# sample_density_time: temporal interpolation
# ---------------------------------------------------------------------------

def _make_dens(px, py, sigma=2):
    g = np.arange(0, 10.5, 0.5)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    z = np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * sigma**2))
    return EyeDensity(x=g, y=g, z=z, sigma=sigma)


def _make_fg(xs, ys, onsets):
    return fixation_group(xs, ys, onset=onsets, duration=[100]*len(xs))


def test_temporal_interpolation_forward_fill():
    """At t=100 between fix onsets 0 and 200, gaze should still be at first fixation."""
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(2, 2, sigma=2)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([2, 8, 8], [2, 8, 8], [0, 200, 400])],
    })
    result = sample_density_time(tt, st, match_on="trial_id", times=[0, 100, 200, 300])
    s = result["sampled"].iloc[0]

    # t=0 and t=100 should both be at (2,2) → high value
    assert s["z"].iloc[0] == pytest.approx(s["z"].iloc[1], abs=0.01)
    assert s["z"].iloc[0] > 0.9

    # t=200 and t=300 should both be at (8,8) → low value
    assert s["z"].iloc[2] == pytest.approx(s["z"].iloc[3], abs=0.01)
    assert s["z"].iloc[2] < 0.1


# ---------------------------------------------------------------------------
# sample_density_time: custom aggregate_fun
# ---------------------------------------------------------------------------

def test_custom_aggregate_fun():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([5]*4, [5]*4, [0, 50, 100, 150])],
    })

    def agg_mean(vals, **kw):
        return np.nanmean(vals)

    result_mean = sample_density_time(
        tt, st, match_on="trial_id", times=np.arange(0, 201, 50),
        time_bins=[0, 200], aggregate_fun=agg_mean,
    )
    result_max = sample_density_time(
        tt, st, match_on="trial_id", times=np.arange(0, 201, 50),
        time_bins=[0, 200], aggregate_fun=lambda v, **kw: np.nanmax(v),
    )
    assert result_max["bin_1"].iloc[0] >= result_mean["bin_1"].iloc[0]


# ---------------------------------------------------------------------------
# sample_density_time: multi-subject permutation stays within subject
# ---------------------------------------------------------------------------

def test_multi_subject_permutation_within_subject():
    tt = pd.DataFrame({
        "trial_id": ["S1_A", "S1_B", "S2_A", "S2_B"],
        "subject": ["S1", "S1", "S2", "S2"],
        "density": [
            _make_dens(2, 2, sigma=1), _make_dens(3, 3, sigma=1),
            _make_dens(7, 7, sigma=1), _make_dens(8, 8, sigma=1),
        ],
    })
    st = pd.DataFrame({
        "trial_id": ["S1_A", "S1_B", "S2_A", "S2_B"],
        "subject": ["S1", "S1", "S2", "S2"],
        "fixgroup": [
            _make_fg([2], [2], [0]), _make_fg([3], [3], [0]),
            _make_fg([7], [7], [0]), _make_fg([8], [8], [0]),
        ],
    })
    np.random.seed(123)
    result = sample_density_time(
        tt, st, match_on="trial_id", times=[0], time_bins=[0, 100],
        permutations=10, permute_on="subject",
    )

    # All matched values should be high (at peak)
    assert all(result["bin_1"] > 0.9)

    # S1 perms use other S1 templates (nearby peaks) → moderate, not ~0
    s1_perm = result.loc[result["subject"] == "S1", "perm_bin_1"]
    assert all(s1_perm > 0.3)

    # S2 perms use other S2 templates → moderate
    s2_perm = result.loc[result["subject"] == "S2", "perm_bin_1"]
    assert all(s2_perm > 0.3)


# ---------------------------------------------------------------------------
# sample_density_time: bin boundary edge cases
# ---------------------------------------------------------------------------

def test_bin_boundaries_edge_cases():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([5]*5, [5]*5, [0, 100, 200, 300, 400])],
    })
    result = sample_density_time(
        tt, st, match_on="trial_id", times=[0, 100, 200, 300, 400],
        time_bins=[0, 200, 400, 500],
    )
    assert "bin_1" in result.columns
    assert "bin_2" in result.columns
    assert "bin_3" in result.columns
    assert np.isfinite(result["bin_1"].iloc[0])
    assert np.isfinite(result["bin_2"].iloc[0])


def test_empty_time_bins_return_nan():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([5, 5], [5, 5], [0, 50])],
    })
    result = sample_density_time(
        tt, st, match_on="trial_id", times=[0, 50],
        time_bins=[0, 100, 200, 300],
    )
    assert np.isfinite(result["bin_1"].iloc[0])
    assert np.isnan(result["bin_2"].iloc[0])
    assert np.isnan(result["bin_3"].iloc[0])


def test_sampled_time_series_correct_length():
    tt = pd.DataFrame({"trial_id": ["A"], "density": [_make_dens(5, 5)]})
    st = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_fg([5], [5], [0])],
    })
    times = np.arange(0, 1001, 25)
    result = sample_density_time(tt, st, match_on="trial_id", times=times)
    assert len(result["sampled"].iloc[0]) == len(times)
    np.testing.assert_array_equal(result["sampled"].iloc[0]["time"].values, times)


# ---------------------------------------------------------------------------
# normalize + time_bins + permutations combined
# ---------------------------------------------------------------------------

def test_normalize_with_time_bins_and_permutations():
    tt = pd.DataFrame({
        "trial_id": ["A", "B"], "subject": ["S1", "S1"],
        "density": [_make_dens(2, 2, sigma=1), _make_dens(8, 8, sigma=1)],
    })
    st = pd.DataFrame({
        "trial_id": ["A", "B"], "subject": ["S1", "S1"],
        "fixgroup": [_make_fg([2], [2], [0]), _make_fg([8], [8], [0])],
    })
    result = sample_density_time(
        tt, st, match_on="trial_id", times=[0], time_bins=[0, 100],
        permutations=5, permute_on="subject", normalize="zscore",
    )
    assert "bin_1" in result.columns
    assert "perm_bin_1" in result.columns
    # Matched values at peak → positive z-score
    assert all(result["bin_1"] > 0)
    # Permuted values far from peak → negative z-score
    assert all(result["perm_bin_1"] < 0)
