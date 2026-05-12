"""Port of test_template_similarity.R"""

import numpy as np
import pandas as pd
from peyesim import fixation_group, density_by, template_similarity
from peyesim.density import EyeDensity
from peyesim.similarity import similarity


def _tiny_density(vals):
    return EyeDensity(
        x=np.array([1.0, 2.0]),
        y=np.array([1.0, 2.0]),
        z=np.array(vals, dtype=float).reshape(2, 2),
        sigma=50.0,
    )


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


def test_template_similarity_cosine_permutations_match_manual_baseline():
    ref_tab = pd.DataFrame({
        "image": np.arange(1, 7),
        "subject": np.repeat([1, 2], 3),
        "density": [
            _tiny_density([1, 2, 3, 4]),
            _tiny_density([2, 3, 4, 5]),
            _tiny_density([3, 4, 5, 6]),
            _tiny_density([6, 5, 4, 3]),
            _tiny_density([5, 4, 3, 2]),
            _tiny_density([4, 3, 2, 1]),
        ],
    })
    source_tab = pd.DataFrame({
        "row_id": np.arange(1, 7),
        "image": np.arange(1, 7),
        "subject": np.repeat([1, 2], 3),
        "density": [
            _tiny_density([1.1, 2.1, 3.1, 4.1]),
            _tiny_density([2.1, 3.1, 4.1, 5.1]),
            _tiny_density([3.1, 4.1, 5.1, 6.1]),
            _tiny_density([6.1, 5.1, 4.1, 3.1]),
            _tiny_density([5.1, 4.1, 3.1, 2.1]),
            _tiny_density([4.1, 3.1, 2.1, 1.1]),
        ],
    })

    matchind = [ref_tab.index[ref_tab["image"] == image][0] for image in source_tab["image"]]
    match_split = {
        str(subject): [matchind[i] for i in np.where(source_tab["subject"].to_numpy() == subject)[0]]
        for subject in source_tab["subject"].unique()
    }

    np.random.seed(42)
    fast = template_similarity(
        ref_tab,
        source_tab,
        match_on="image",
        permute_on="subject",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=2,
    )

    np.random.seed(42)
    manual_rows = []
    for i, mi in enumerate(matchind):
        d2 = source_tab["density"].iloc[i]
        eye_sim = similarity(ref_tab["density"].iloc[mi], d2, method="cosine")
        candidates = list(match_split[str(source_tab["subject"].iloc[i])])
        if 2 < len(candidates):
            candidates = list(np.random.choice(candidates, 2, replace=False))
        candidates = [candidate for candidate in candidates if candidate != mi]
        perm_vals = [similarity(ref_tab["density"].iloc[j], d2, method="cosine") for j in candidates]
        perm_sim = np.nan if len(perm_vals) == 0 else float(np.nanmean(perm_vals))
        manual_rows.append({
            "row_id": source_tab["row_id"].iloc[i],
            "eye_sim": eye_sim,
            "perm_sim": perm_sim,
            "eye_sim_diff": eye_sim - perm_sim,
        })
    manual = pd.DataFrame(manual_rows)

    fast = fast.sort_values("row_id")[["row_id", "eye_sim", "perm_sim", "eye_sim_diff"]].reset_index(drop=True)
    manual = manual.sort_values("row_id").reset_index(drop=True)
    pd.testing.assert_series_equal(fast["row_id"], manual["row_id"], check_names=False)
    np.testing.assert_allclose(fast["eye_sim"], manual["eye_sim"], atol=1e-10)
    np.testing.assert_allclose(fast["perm_sim"], manual["perm_sim"], atol=1e-10)
    np.testing.assert_allclose(fast["eye_sim_diff"], manual["eye_sim_diff"], atol=1e-10)


def test_template_similarity_cosine_exhaustive_within_stratum_baseline():
    ref_tab = pd.DataFrame({
        "image": np.arange(1, 7),
        "subject": np.repeat([1, 2], 3),
        "density": [
            _tiny_density([1, 0, 0, 1]),
            _tiny_density([0, 1, 1, 0]),
            _tiny_density([1, 1, 0, 0]),
            _tiny_density([0, 0, 1, 1]),
            _tiny_density([1, 0, 1, 0]),
            _tiny_density([0, 1, 0, 1]),
        ],
    })
    source_tab = pd.DataFrame({
        "row_id": np.arange(1, 7),
        "image": np.arange(1, 7),
        "subject": np.repeat([1, 2], 3),
        "density": [
            _tiny_density([0.9, 0.1, 0.1, 0.9]),
            _tiny_density([0.1, 0.9, 0.9, 0.1]),
            _tiny_density([0.9, 0.9, 0.1, 0.1]),
            _tiny_density([0.1, 0.1, 0.9, 0.9]),
            _tiny_density([0.9, 0.1, 0.9, 0.1]),
            _tiny_density([0.1, 0.9, 0.1, 0.9]),
        ],
    })

    manual_rows = []
    for _, row in source_tab.iterrows():
        match_idx = ref_tab.index[ref_tab["image"] == row["image"]][0]
        candidates = ref_tab[(ref_tab["subject"] == row["subject"]) & (ref_tab["image"] != row["image"])]
        perm_vals = [similarity(d1, row["density"], method="cosine") for d1 in candidates["density"]]
        eye_sim = similarity(ref_tab["density"].iloc[match_idx], row["density"], method="cosine")
        perm_sim = float(np.nanmean(perm_vals))
        manual_rows.append({
            "row_id": row["row_id"],
            "eye_sim": eye_sim,
            "perm_sim": perm_sim,
            "eye_sim_diff": eye_sim - perm_sim,
        })
    manual = pd.DataFrame(manual_rows)

    res = template_similarity(
        ref_tab,
        source_tab,
        match_on="image",
        permute_on="subject",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=99,
    )

    res = res.sort_values("row_id")[["row_id", "eye_sim", "perm_sim", "eye_sim_diff"]].reset_index(drop=True)
    manual = manual.sort_values("row_id").reset_index(drop=True)
    pd.testing.assert_series_equal(res["row_id"], manual["row_id"], check_names=False)
    np.testing.assert_allclose(res["eye_sim"], manual["eye_sim"], atol=1e-10)
    np.testing.assert_allclose(res["perm_sim"], manual["perm_sim"], atol=1e-10)
    np.testing.assert_allclose(res["eye_sim_diff"], manual["eye_sim_diff"], atol=1e-10)


def test_template_similarity_cosine_exhaustive_permutation_is_row_order_invariant():
    ref_tab = pd.DataFrame({
        "image": np.arange(1, 7),
        "subject": np.repeat([1, 2], 3),
        "density": [
            _tiny_density([1, 2, 3, 4]),
            _tiny_density([2, 3, 4, 5]),
            _tiny_density([3, 4, 5, 6]),
            _tiny_density([6, 5, 4, 3]),
            _tiny_density([5, 4, 3, 2]),
            _tiny_density([4, 3, 2, 1]),
        ],
    })
    source_tab = pd.DataFrame({
        "row_id": np.arange(1, 7),
        "image": np.arange(1, 7),
        "subject": np.repeat([1, 2], 3),
        "density": [
            _tiny_density([1.1, 2.1, 3.1, 4.1]),
            _tiny_density([2.1, 3.1, 4.1, 5.1]),
            _tiny_density([3.1, 4.1, 5.1, 6.1]),
            _tiny_density([6.1, 5.1, 4.1, 3.1]),
            _tiny_density([5.1, 4.1, 3.1, 2.1]),
            _tiny_density([4.1, 3.1, 2.1, 1.1]),
        ],
    })
    shuffled_ref = ref_tab.sample(frac=1, random_state=11).reset_index(drop=True)
    shuffled_source = source_tab.sample(frac=1, random_state=12).reset_index(drop=True)

    ordered = template_similarity(
        ref_tab,
        source_tab,
        match_on="image",
        permute_on="subject",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=99,
    )
    shuffled = template_similarity(
        shuffled_ref,
        shuffled_source,
        match_on="image",
        permute_on="subject",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=99,
    )

    ordered = ordered.sort_values("row_id")[["row_id", "eye_sim", "perm_sim", "eye_sim_diff"]].reset_index(drop=True)
    shuffled = shuffled.sort_values("row_id")[["row_id", "eye_sim", "perm_sim", "eye_sim_diff"]].reset_index(drop=True)
    pd.testing.assert_series_equal(ordered["row_id"], shuffled["row_id"], check_names=False)
    np.testing.assert_allclose(ordered["eye_sim"], shuffled["eye_sim"], atol=1e-10)
    np.testing.assert_allclose(ordered["perm_sim"], shuffled["perm_sim"], atol=1e-10)
    np.testing.assert_allclose(ordered["eye_sim_diff"], shuffled["eye_sim_diff"], atol=1e-10)


def test_template_similarity_cosine_preserves_degenerate_zero_vector_behavior():
    ref_tab = pd.DataFrame({
        "image": np.arange(1, 5),
        "density": [
            _tiny_density([0, 0, 0, 0]),
            _tiny_density([1, 0, 0, 1]),
            _tiny_density([0, 1, 1, 0]),
            _tiny_density([1, 1, 1, 1]),
        ],
    })
    source_tab = pd.DataFrame({
        "row_id": np.arange(1, 5),
        "image": np.arange(1, 5),
        "density": [
            _tiny_density([0, 0, 0, 0]),
            _tiny_density([1, 0, 0, 1]),
            _tiny_density([0, 1, 1, 0]),
            _tiny_density([0, 0, 0, 0]),
        ],
    })

    res = template_similarity(
        ref_tab,
        source_tab,
        match_on="image",
        refvar="density",
        sourcevar="density",
        method="cosine",
        permutations=0,
    )
    manual = [
        similarity(ref_tab["density"].iloc[i], source_tab["density"].iloc[i], method="cosine")
        for i in range(len(source_tab))
    ]

    np.testing.assert_allclose(res["eye_sim"], manual, atol=1e-10)
    assert res["eye_sim"].iloc[0] == 1
    assert res["eye_sim"].iloc[3] == 0


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
