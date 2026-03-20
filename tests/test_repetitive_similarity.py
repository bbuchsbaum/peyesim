"""Port of test_repetitive_similarity.R"""

import numpy as np
import pandas as pd
from peyesim import fixation_group, eye_table, density_by
from peyesim.repetitive_similarity import repetitive_similarity


def test_repetitive_similarity_with_standard_density():
    np.random.seed(101)
    n_cond = 2
    n_per = 4
    n_total = n_cond * n_per
    sigma = 50

    rows = []
    for trial in range(n_total):
        cond = "Cond1" if trial < n_per else "Cond2"
        for _ in range(20):
            x = np.random.uniform(0, 1024)
            y = np.random.uniform(0, 768)
            if cond == "Cond2":
                x -= 40
                y -= 20
            rows.append({
                "trial_id": trial + 1,
                "condition": cond,
                "x": x,
                "y": y,
                "duration": np.random.normal(220, 25),
                "onset": float(trial * 1000 + len([r for r in rows if r["trial_id"] == trial + 1]) * 50),
            })

    fix_data = pd.DataFrame(rows)
    # Ensure onset is monotonically increasing per group by recomputing
    for tid in fix_data["trial_id"].unique():
        mask = fix_data["trial_id"] == tid
        fix_data.loc[mask, "onset"] = np.arange(mask.sum()) * 50.0

    et = eye_table("x", "y", "duration", "onset",
                   groupvar=["trial_id", "condition"], data=fix_data,
                   clip_bounds=(0, 1024, 0, 768))

    dens = density_by(et, groups=["trial_id", "condition"], sigma=sigma,
                      xbounds=(0, 1024), ybounds=(0, 768),
                      result_name="standard_density")

    rep_sim = repetitive_similarity(
        dens, density_var="standard_density",
        condition_var="condition", pairwise=True, method="pearson",
    )

    assert isinstance(rep_sim, pd.DataFrame)
    assert len(rep_sim) == n_total
    assert all(c in rep_sim.columns for c in ["repsim", "othersim", "pairwise_repsim"])
    assert all(isinstance(v, (float, np.floating)) for v in rep_sim["repsim"])
    assert all(isinstance(v, (float, np.floating)) for v in rep_sim["othersim"])
    assert all(isinstance(v, list) for v in rep_sim["pairwise_repsim"])

    # Check pairwise counts
    for i in range(n_total):
        pw = rep_sim["pairwise_repsim"].iloc[i]
        assert len(pw) == n_per - 1
