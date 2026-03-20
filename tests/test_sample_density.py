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
