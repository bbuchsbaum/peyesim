"""Port of test_multiscale_density.R"""

import numpy as np
import pandas as pd
from peyesim import (
    fixation_group, eye_table, density_by, template_similarity,
)
from peyesim.density import EyeDensity, EyeDensityMultiscale
from peyesim.repetitive_similarity import repetitive_similarity


def test_density_by_creates_multiscale_objects_with_vector_sigma():
    np.random.seed(123)
    n_trials = 6
    sigmas_vec = [30, 60, 90]

    fix_data = pd.DataFrame({
        "trial": np.repeat(np.arange(1, n_trials + 1), 10),
        "condition": np.repeat(["A", "B"], n_trials // 2 * 10),
        "x": np.random.uniform(0, 1000, n_trials * 10),
        "y": np.random.uniform(0, 1000, n_trials * 10),
        "duration": np.random.normal(300, 50, n_trials * 10),
        "onset": np.tile(np.arange(0, 500, 50), n_trials) +
                 np.repeat(np.arange(n_trials) * 500, 10),
    })

    et = eye_table("x", "y", "duration", "onset",
                   groupvar=["trial", "condition"], data=fix_data,
                   clip_bounds=(0, 1000, 0, 1000))

    ms = density_by(et, groups=["trial", "condition"], sigma=sigmas_vec,
                    xbounds=(0, 1000), ybounds=(0, 1000),
                    result_name="multiscale_map")

    assert "multiscale_map" in ms.columns
    assert len(ms) == n_trials

    first_map = ms["multiscale_map"].iloc[0]
    assert isinstance(first_map, EyeDensityMultiscale)
    assert len(first_map) == len(sigmas_vec)
    assert first_map.sigmas == sigmas_vec
    for s in first_map:
        assert isinstance(s, EyeDensity)


def test_template_similarity_with_multiscale_and_aggregation():
    np.random.seed(456)
    n_trials = 4
    sigmas_vec = [40, 80]

    fix1 = pd.DataFrame({
        "trial": np.repeat(np.arange(1, n_trials + 1), 15),
        "x": np.random.uniform(0, 800, n_trials * 15),
        "y": np.random.uniform(0, 600, n_trials * 15),
        "duration": np.random.normal(250, 40, n_trials * 15),
        "onset": np.tile(np.arange(0, 750, 50), n_trials) +
                 np.repeat(np.arange(n_trials) * 750, 15),
    })
    fix2 = fix1.copy()
    fix2["x"] = fix2["x"] + np.random.normal(0, 50, len(fix2))
    fix2["y"] = fix2["y"] + np.random.normal(0, 50, len(fix2))

    et1 = eye_table("x", "y", "duration", "onset", groupvar="trial",
                    data=fix1, clip_bounds=(0, 800, 0, 600))
    et2 = eye_table("x", "y", "duration", "onset", groupvar="trial",
                    data=fix2, clip_bounds=(0, 800, 0, 600))

    dens1 = density_by(et1, "trial", sigma=sigmas_vec,
                       xbounds=(0, 800), ybounds=(0, 600), result_name="ms_dens")
    dens2 = density_by(et2, "trial", sigma=sigmas_vec,
                       xbounds=(0, 800), ybounds=(0, 600), result_name="ms_dens")

    # mean aggregation
    tsim_mean = template_similarity(dens1, dens2, match_on="trial",
                                    refvar="ms_dens", sourcevar="ms_dens",
                                    method="cosine", permutations=0,
                                    multiscale_aggregation="mean")
    assert "eye_sim" in tsim_mean.columns
    assert len(tsim_mean) == n_trials
    assert all(isinstance(v, (float, np.floating)) for v in tsim_mean["eye_sim"])

    # none aggregation
    tsim_none = template_similarity(dens1, dens2, match_on="trial",
                                    refvar="ms_dens", sourcevar="ms_dens",
                                    method="cosine", permutations=0,
                                    multiscale_aggregation="none")
    assert "eye_sim" in tsim_none.columns
    assert len(tsim_none) == n_trials
    # Each eye_sim should be an array with len = n sigmas
    first_sim = tsim_none["eye_sim"].iloc[0]
    assert hasattr(first_sim, "__len__")
    assert len(first_sim) == len(sigmas_vec)


def test_repetitive_similarity_with_multiscale():
    np.random.seed(789)
    n_cond = 2
    n_per = 3
    n_total = n_cond * n_per
    sigmas_vec = [25, 50]

    fix_data = pd.DataFrame({
        "trial_id": np.repeat(np.arange(1, n_total + 1), 12),
        "condition": np.repeat(["A", "B"], n_per * 12),
        "x": np.random.uniform(0, 1200, n_total * 12),
        "y": np.random.uniform(0, 900, n_total * 12),
        "duration": np.random.normal(200, 30, n_total * 12),
        "onset": np.tile(np.arange(0, 600, 50), n_total) +
                 np.repeat(np.arange(n_total) * 600, 12),
    })
    mask_b = fix_data["condition"] == "B"
    fix_data.loc[mask_b, "x"] += 50
    fix_data.loc[mask_b, "y"] += 30

    et = eye_table("x", "y", "duration", "onset",
                   groupvar=["trial_id", "condition"], data=fix_data,
                   clip_bounds=(0, 1200, 0, 900))

    dens = density_by(et, groups=["trial_id", "condition"], sigma=sigmas_vec,
                      xbounds=(0, 1200), ybounds=(0, 900), result_name="ms_map")

    rep_mean = repetitive_similarity(dens, density_var="ms_map",
                                     condition_var="condition",
                                     method="spearman",
                                     multiscale_aggregation="mean",
                                     pairwise=True)

    assert "repsim" in rep_mean.columns
    assert "othersim" in rep_mean.columns
    assert "pairwise_repsim" in rep_mean.columns
    assert len(rep_mean) == n_total
    assert all(isinstance(v, (float, np.floating)) for v in rep_mean["repsim"])
    assert all(isinstance(v, (float, np.floating)) for v in rep_mean["othersim"])
