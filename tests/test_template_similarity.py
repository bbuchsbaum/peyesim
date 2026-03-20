"""Port of test_template_similarity.R"""

import numpy as np
import pandas as pd
from peyesim import fixation_group, density_by, template_similarity


def test_template_similarity_produces_perfect_for_identical():
    np.random.seed(42)
    fgs1 = []
    fgs2 = []
    for _ in range(10):
        x = np.random.uniform(size=10)
        y = np.random.uniform(size=10)
        onset = np.arange(1, 10 * 50 + 1, 50, dtype=float)
        dur = np.ones(10)
        fgs1.append(fixation_group(x, y, onset=onset, duration=dur))
        fgs2.append(fixation_group(np.random.uniform(size=10),
                                   np.random.uniform(size=10),
                                   onset=onset, duration=dur))

    g1 = pd.DataFrame({"fixgroup": fgs1, "image": np.arange(1, 11)})
    g2 = pd.DataFrame({"fixgroup": fgs2, "image": np.arange(1, 11)})

    dens = density_by(g1, "image", xbounds=(0, 1), ybounds=(0, 1))
    dens2 = density_by(g2, "image", xbounds=(0, 1), ybounds=(0, 1))
    tsim = template_similarity(dens, dens2, match_on="image",
                               method="spearman", permutations=3)
    assert all(tsim["eye_sim"] <= 1)
    assert all(tsim["eye_sim"] >= -1)


def test_template_similarity_works_for_permute_on():
    np.random.seed(42)
    fgs = []
    for _ in range(100):
        x = np.random.uniform(size=10)
        y = np.random.uniform(size=10)
        onset = np.arange(1, 10 * 50 + 1, 50, dtype=float)
        dur = np.ones(10)
        fgs.append(fixation_group(x, y, onset=onset, duration=dur))

    g1 = pd.DataFrame({
        "fixgroup": fgs,
        "image": np.arange(1, 101),
        "subject": np.repeat(np.arange(1, 11), 10),
    })

    dens = density_by(g1, "image", keep_vars=["subject"],
                      xbounds=(0, 1), ybounds=(0, 1), duration_weighted=True)
    dens2 = density_by(g1, "image", keep_vars=["subject"],
                       xbounds=(0, 1), ybounds=(0, 1), duration_weighted=True)
    tsim = template_similarity(dens, dens2, match_on="image",
                               method="pearson", permute_on="subject",
                               permutations=6)
    assert all(tsim["eye_sim"] > 0.99)


def test_compute_density_with_custom_fixvar():
    np.random.seed(42)
    fgs = []
    for _ in range(100):
        x = np.random.uniform(size=10)
        y = np.random.uniform(size=10)
        onset = np.arange(1, 10 * 50 + 1, 50, dtype=float)
        dur = np.ones(10)
        fgs.append(fixation_group(x, y, onset=onset, duration=dur))

    g1 = pd.DataFrame({
        "fg": fgs,
        "image": np.arange(1, 101),
        "subject": np.repeat(np.arange(1, 11), 10),
    })
    dens = density_by(g1, "image", keep_vars=["subject"],
                      xbounds=(0, 1), ybounds=(0, 1), fixvar="fg")
    assert dens["fg"] is not None
