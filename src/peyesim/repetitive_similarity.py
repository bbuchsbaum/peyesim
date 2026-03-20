"""Within- and between-condition repetitive similarity."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from peyesim.similarity import similarity


def repetitive_similarity(
    tab: pd.DataFrame,
    density_var: str = "density",
    condition_var: str = "condition",
    method: str = "spearman",
    pairwise: bool = False,
    multiscale_aggregation: str = "mean",
    **kwargs,
) -> pd.DataFrame:
    """Compute within-condition and between-condition similarity.

    For each density map (trial), calculates its average similarity to all
    other maps within the same condition (``repsim``) and its average
    similarity to all maps from different conditions (``othersim``).
    """
    if density_var not in tab.columns:
        raise ValueError(f"Density variable {density_var} not found in input table.")
    if condition_var not in tab.columns:
        raise ValueError(f"Condition variable {condition_var} not found in input table.")

    tab = tab.reset_index(drop=True)
    n = len(tab)
    densities = tab[density_var].values
    conditions = tab[condition_var].values

    # Pre-compute pairwise similarity matrix (upper triangle)
    sim_matrix = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        d1, d2 = densities[i], densities[j]
        if d1 is None or d2 is None:
            val = np.nan
        else:
            try:
                val = similarity(d1, d2, method=method,
                                 multiscale_aggregation=multiscale_aggregation,
                                 **kwargs)
            except Exception as e:
                warnings.warn(f"Error in similarity calculation for row {i} vs {j}: {e}")
                val = np.nan
        sim_matrix[i, j] = val
        sim_matrix[j, i] = val

    def _mean_sim(vals):
        """Mean of a list of similarity values (which may be scalars or arrays)."""
        if len(vals) == 0:
            return np.nan
        means = [float(np.nanmean(v)) for v in vals]
        return float(np.nanmean(means))

    repsim_vec = np.zeros(n)
    othersim_vec = np.zeros(n)
    pairwise_list = [None] * n

    for i in range(n):
        cond_i = conditions[i]
        same_idx = [j for j in range(n) if j != i and conditions[j] == cond_i]
        other_idx = [j for j in range(n) if conditions[j] != cond_i]

        repsim_vals = [sim_matrix[i, j] for j in same_idx]
        othersim_vals = [sim_matrix[i, j] for j in other_idx]

        repsim_vec[i] = _mean_sim(repsim_vals)
        othersim_vec[i] = _mean_sim(othersim_vals)

        if pairwise:
            pairwise_list[i] = repsim_vals

    result = tab.copy()
    result["repsim"] = repsim_vec
    result["othersim"] = othersim_vec
    if pairwise:
        result["pairwise_repsim"] = pairwise_list

    return result
