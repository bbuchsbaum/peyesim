"""Port of test_sample_density.R"""

import numpy as np
import pandas as pd
from peyesim import fixation_group, sample_density
from peyesim.density import EyeDensity


def test_times_argument_matches_direct_sampling():
    x_grid = np.arange(0, 11, dtype=float)
    y_grid = np.arange(0, 11, dtype=float)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
    z_mat = xx + yy
    dens = EyeDensity(x=x_grid, y=y_grid, z=z_mat, sigma=1)

    fix = fixation_group(x=[1, 5, 10], y=[2, 5, 9],
                         onset=[0, 50, 100], duration=[1, 1, 1])

    direct = sample_density(dens, fix)
    timed = sample_density(dens, fix, times=fix["onset"].values)
    pd.testing.assert_frame_equal(direct, timed)


def test_zscore_normalization_uses_sample_standard_deviation_like_r():
    x_grid = np.arange(0, 3, dtype=float)
    y_grid = np.arange(0, 3, dtype=float)
    z_mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    dens = EyeDensity(x=x_grid, y=y_grid, z=z_mat, sigma=1)
    fix = fixation_group(x=[0, 1, 2], y=[0, 1, 2],
                         onset=[0, 50, 100], duration=[1, 1, 1])

    sampled = sample_density(dens, fix, normalize="zscore")
    expected = (z_mat - z_mat.mean()) / z_mat.ravel().std(ddof=1)

    np.testing.assert_allclose(sampled["z"].to_numpy(), [expected[0, 0], expected[1, 1], expected[2, 2]])
