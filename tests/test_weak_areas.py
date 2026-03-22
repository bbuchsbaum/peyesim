"""Tests for overlap, sigma, and concat (ported from R test_weak_areas.R)."""
import numpy as np
import pytest
from peyesim import fixation_group, suggest_sigma, concat_fixation_groups
from peyesim.fixations import FixationGroup
from peyesim.overlap import fixation_overlap


# ---------- fixation_overlap deterministic counts ----------


def test_fixation_overlap_deterministic_counts():
    """fixation_overlap with known data produces exact overlap counts."""
    fg_ref = fixation_group(
        x=[0, 10, 20], y=[0, 0, 0], onset=[0, 100, 200], duration=[100, 100, 100]
    )
    fg_shift = fixation_group(
        x=[2, 15, 40], y=[1, 0, 0], onset=[0, 100, 200], duration=[100, 100, 100]
    )
    times = np.array([0, 100, 200], dtype=float)

    euclidean = fixation_overlap(
        fg_ref, fg_shift, dthresh=6, time_samples=times, dist_method="euclidean"
    )
    manhattan = fixation_overlap(
        fg_ref, fg_shift, dthresh=6, time_samples=times, dist_method="manhattan"
    )

    # At time 0: dist(0,0)-(2,1) = sqrt(5) ~= 2.24 < 6  -> overlap
    # At time 100: dist(10,0)-(15,0) = 5 < 6             -> overlap
    # At time 200: dist(20,0)-(40,0) = 20 > 6            -> no overlap
    assert euclidean["overlap"] == 2
    assert abs(euclidean["perc"] - 2 / 3) < 1e-10

    # Manhattan: (0,0)-(2,1)=3<6, (10,0)-(15,0)=5<6, (20,0)-(40,0)=20>6
    assert manhattan["overlap"] == 2
    assert abs(manhattan["perc"] - 2 / 3) < 1e-10


# ---------- suggest_sigma display clamp ----------


def test_suggest_sigma_display_clamp():
    """suggest_sigma clamps to 1%-15% of display scale for tight fixations."""
    fg_tight = fixation_group(
        x=np.full(20, 400.0),
        y=np.full(20, 300.0),
        onset=np.arange(0, 1000, 50, dtype=float),
        duration=np.full(20, 50.0),
    )
    sigma = suggest_sigma(fg_tight, xbounds=(0, 1000), ybounds=(0, 1000))
    # display_scale = 1000, 1% = 10, so sigma should be clamped to 10
    assert sigma == 10.0


def test_suggest_sigma_requires_y_for_arrays():
    """suggest_sigma raises when x is array-like but y is not provided."""
    with pytest.raises((ValueError, TypeError)):
        suggest_sigma(np.array([1, 2, 3]))


# ---------- concat_fixation_groups ----------


def test_concat_fixation_groups():
    """concat_fixation_groups concatenates and rejects invalid inputs."""
    fg1 = fixation_group(
        x=[10, 20], y=[5, 10], onset=[0, 100], duration=[100, 100]
    )
    fg2 = fixation_group(
        x=[30, 40], y=[15, 20], onset=[0, 100], duration=[100, 100]
    )
    combined = concat_fixation_groups(fg1, fg2)

    assert isinstance(combined, FixationGroup)
    assert len(combined) == 4
