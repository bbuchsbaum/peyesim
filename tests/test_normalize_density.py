"""Port of test_normalize_density.R"""

import numpy as np
import pandas as pd
import pytest
from peyesim import fixation_group, sample_density, sample_density_time
from peyesim.density import EyeDensity


def _make_test_density(peak_x, peak_y, sigma=2):
    x_grid = np.arange(0, 10.5, 0.5)
    y_grid = np.arange(0, 10.5, 0.5)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    z = np.exp(-((xx - peak_x) ** 2 + (yy - peak_y) ** 2) / (2 * sigma ** 2))
    return EyeDensity(x=x_grid, y=y_grid, z=z, sigma=sigma)


def _make_test_fixgroup(x_coords, y_coords, onsets):
    return fixation_group(x_coords, y_coords, onset=onsets,
                          duration=[100] * len(x_coords))


# --- sample_density tests ---

def test_normalize_none_returns_raw():
    dens = _make_test_density(5, 5, sigma=2)
    fix = fixation_group(x=[5], y=[5], onset=[0], duration=[100])
    res_default = sample_density(dens, fix)
    res_none = sample_density(dens, fix, normalize="none")
    assert res_default["z"].iloc[0] == pytest.approx(res_none["z"].iloc[0])
    assert res_default["z"].iloc[0] == pytest.approx(1.0)


def test_normalize_max_scales_to_unit():
    dens = _make_test_density(5, 5, sigma=2)
    fix = fixation_group(x=[5, 0], y=[5, 0], onset=[0, 100], duration=[100, 100])
    res = sample_density(dens, fix, normalize="max")
    assert res["z"].iloc[0] == pytest.approx(1.0)
    assert 0 < res["z"].iloc[1] < 1


def test_normalize_sum_produces_probability():
    dens = _make_test_density(5, 5, sigma=2)
    fix = fixation_group(x=[5], y=[5], onset=[0], duration=[100])
    res = sample_density(dens, fix, normalize="sum")
    assert res["z"].iloc[0] > 0
    assert res["z"].iloc[0] < 1


def test_normalize_zscore_centres():
    dens = _make_test_density(5, 5, sigma=2)
    fix_peak = fixation_group(x=[5], y=[5], onset=[0], duration=[100])
    res_peak = sample_density(dens, fix_peak, normalize="zscore")
    assert res_peak["z"].iloc[0] > 0

    fix_corner = fixation_group(x=[0], y=[0], onset=[0], duration=[100])
    res_corner = sample_density(dens, fix_corner, normalize="zscore")
    assert res_corner["z"].iloc[0] < 0


def test_normalize_works_with_times():
    dens = _make_test_density(5, 5, sigma=2)
    fix = fixation_group(x=[5, 0], y=[5, 0], onset=[0, 200], duration=[100, 100])
    res_raw = sample_density(dens, fix, times=[0, 100, 200], normalize="none")
    res_max = sample_density(dens, fix, times=[0, 100, 200], normalize="max")
    assert res_max["z"].iloc[0] == pytest.approx(1.0)
    assert len(res_raw) == len(res_max)


def test_normalize_handles_uniform_density():
    x_grid = np.arange(0, 11, dtype=float)
    y_grid = np.arange(0, 11, dtype=float)
    z_mat = np.full((len(x_grid), len(y_grid)), 0.5)
    dens = EyeDensity(x=x_grid, y=y_grid, z=z_mat, sigma=1)
    fix = fixation_group(x=[5], y=[5], onset=[0], duration=[100])

    # zscore with sd=0: should return original value (no division)
    res_z = sample_density(dens, fix, normalize="zscore")
    assert res_z["z"].iloc[0] == pytest.approx(0.5)

    # max should give 1.0
    res_m = sample_density(dens, fix, normalize="max")
    assert res_m["z"].iloc[0] == pytest.approx(1.0)

    # sum
    res_s = sample_density(dens, fix, normalize="sum")
    assert res_s["z"].iloc[0] == pytest.approx(0.5 / z_mat.sum())


def test_normalize_handles_zero_density():
    x_grid = np.arange(0, 11, dtype=float)
    y_grid = np.arange(0, 11, dtype=float)
    z_mat = np.zeros((len(x_grid), len(y_grid)))
    dens = EyeDensity(x=x_grid, y=y_grid, z=z_mat, sigma=1)
    fix = fixation_group(x=[5], y=[5], onset=[0], duration=[100])

    assert sample_density(dens, fix, normalize="max")["z"].iloc[0] == 0
    assert sample_density(dens, fix, normalize="sum")["z"].iloc[0] == 0
    assert sample_density(dens, fix, normalize="zscore")["z"].iloc[0] == 0


# --- sample_density_time tests ---

def test_sample_density_time_passes_normalize_through():
    template_tab = pd.DataFrame({
        "trial_id": ["A"],
        "density": [_make_test_density(5, 5, sigma=2)],
    })
    source_tab = pd.DataFrame({
        "trial_id": ["A"],
        "fixgroup": [_make_test_fixgroup([5, 0], [5, 0], [0, 200])],
    })
    res_max = sample_density_time(template_tab, source_tab, match_on="trial_id",
                                  times=[0, 200], normalize="max")
    assert res_max["sampled"].iloc[0]["z"].iloc[0] == pytest.approx(1.0)


def test_sample_density_time_zscore_makes_comparable():
    dens_strong = _make_test_density(5, 5, sigma=1)
    dens_weak = _make_test_density(5, 5, sigma=1)
    dens_weak.z = dens_weak.z * 0.1

    template_tab = pd.DataFrame({
        "trial_id": ["A", "B"],
        "density": [dens_strong, dens_weak],
    })
    source_tab = pd.DataFrame({
        "trial_id": ["A", "B"],
        "fixgroup": [
            _make_test_fixgroup([5], [5], [0]),
            _make_test_fixgroup([5], [5], [0]),
        ],
    })

    res_z = sample_density_time(template_tab, source_tab, match_on="trial_id",
                                times=[0], normalize="zscore")
    z_A = res_z["sampled"].iloc[0]["z"].iloc[0]
    z_B = res_z["sampled"].iloc[1]["z"].iloc[0]
    assert z_A > 0
    assert z_B > 0
    assert z_A == pytest.approx(z_B, abs=0.01)


def test_normalize_argument_validates():
    dens = _make_test_density(5, 5)
    fix = fixation_group(x=[5], y=[5], onset=[0], duration=[100])
    with pytest.raises(ValueError, match="should be one of"):
        sample_density(dens, fix, normalize="invalid")
