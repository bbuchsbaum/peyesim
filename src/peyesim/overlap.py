"""Fixation overlap measure."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from peyesim.fixations import FixationGroup


def fixation_overlap(
    x: FixationGroup,
    y: FixationGroup,
    dthresh: float = 60,
    time_samples: np.ndarray | None = None,
    dist_method: str = "euclidean",
) -> dict:
    """Calculate the proportion of overlapping fixations between two groups.

    Parameters
    ----------
    x, y : FixationGroup
        The two fixation sequences.
    dthresh : float
        Distance threshold for overlap.
    time_samples : array-like or None
        Time points at which to evaluate.  Defaults to
        ``np.arange(0, max(x.onset), 20)``.
    dist_method : str
        ``'euclidean'`` or ``'cityblock'`` (Manhattan).

    Returns
    -------
    dict
        ``{'overlap': int, 'perc': float}``
    """
    if time_samples is None:
        time_samples = np.arange(0, x["onset"].max() + 1, 20)

    method_map = {"euclidean": "euclidean", "manhattan": "cityblock"}
    dm = method_map.get(dist_method, dist_method)

    fx1 = x.sample_fixations(time_samples)
    fx2 = y.sample_fixations(time_samples)

    coords1 = fx1[["x", "y"]].values
    coords2 = fx2[["x", "y"]].values

    # Pairwise (element-wise) distances
    d = np.sqrt(((coords1 - coords2) ** 2).sum(axis=1)) if dm == "euclidean" else \
        np.abs(coords1 - coords2).sum(axis=1)

    valid = ~np.isnan(d)
    overlap = int((d[valid] < dthresh).sum())
    perc = overlap / len(d) if len(d) > 0 else 0.0

    return {"overlap": overlap, "perc": perc}
