"""Port of test_density_by.R"""

import numpy as np
import pandas as pd
import pytest
from peyesim import (
    fixation_group, density_by, template_similarity, eye_density,
)


def test_density_by_produces_perfect_similarity_for_identical_patterns():
    np.random.seed(42)
    fgs = []
    for i in range(100):
        x = np.random.uniform(size=10)
        y = np.random.uniform(size=10)
        onset = np.arange(1, 10 * 50 + 1, 50, dtype=float)
        duration = np.ones(10)
        fgs.append(fixation_group(x, y, onset=onset, duration=duration))

    g1 = pd.DataFrame({"fixgroup": fgs, "image": np.arange(1, 101)})
    dens = density_by(g1, "image", xbounds=(0, 1), ybounds=(0, 1))
    dens2 = density_by(g1, "image", xbounds=(0, 1), ybounds=(0, 1))
    tsim = template_similarity(dens, dens2, match_on="image",
                               method="spearman", permutations=30)
    np.testing.assert_allclose(tsim["eye_sim"].values, 1.0, atol=1e-10)


def test_weighted_and_unweighted_density_maps_are_highly_correlated():
    np.random.seed(123)
    centers = np.array([[250, 250], [500, 500], [750, 750]])
    n_per = 20
    x2, y2 = [], []
    for c in centers:
        x2.extend(np.random.normal(c[0], 50, n_per))
        y2.extend(np.random.normal(c[1], 50, n_per))

    duration2 = np.random.uniform(50, 500, len(x2))
    onset2 = np.arange(1, len(x2) * 50 + 1, 50, dtype=float)
    fg2 = fixation_group(x2, y2, duration2, onset2)

    wd = eye_density(fg2, sigma=50, xbounds=(0, 1000), ybounds=(0, 1000),
                     outdim=(100, 100), duration_weighted=True)
    ud = eye_density(fg2, sigma=50, xbounds=(0, 1000), ybounds=(0, 1000),
                     outdim=(100, 100), duration_weighted=False)

    corr = np.corrcoef(wd.z.ravel(), ud.z.ravel())[0, 1]
    assert corr > 0.95


def test_density_by_handles_min_fixations_correctly():
    fg_single = fixation_group(x=[100], y=[100], onset=[1], duration=[1])
    tab = pd.DataFrame({"fixgroup": [fg_single], "grp": [1]})

    with pytest.warns(UserWarning, match="Removing rows"):
        dens_default = density_by(tab, groups="grp",
                                  xbounds=(0, 200), ybounds=(0, 200))
    assert len(dens_default) == 0

    dens_allowed = density_by(tab, groups="grp",
                              xbounds=(0, 200), ybounds=(0, 200),
                              min_fixations=1)
    assert len(dens_allowed) == 1
    from peyesim.density import EyeDensity
    assert isinstance(dens_allowed["density"].iloc[0], EyeDensity)
