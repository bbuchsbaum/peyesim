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
        ``seq(0, max(onset of x and y), by = 20)``, built from both groups so
        the measure is symmetric in ``x`` and ``y``.
    dist_method : str
        ``'euclidean'`` or ``'manhattan'``.

    Returns
    -------
    dict
        ``{'overlap': int, 'perc': float}``. ``perc`` is the proportion of
        compared time points at which the groups overlap; time points before
        either group's first fixation are not compared (``nan`` if none are).
    """
    if time_samples is None:
        top = max(np.nanmax(np.asarray(x["onset"], dtype=float)),
                  np.nanmax(np.asarray(y["onset"], dtype=float)))
        # R seq(0, top, by = 20), including its 1e-10 tolerance at the end.
        time_samples = 20.0 * np.arange(int(np.floor(top / 20 + 1e-10)) + 1)

    method_map = {"euclidean": "euclidean", "manhattan": "cityblock"}
    if dist_method not in method_map:
        raise ValueError("dist_method must be one of {'euclidean', 'manhattan'}")
    dm = method_map[dist_method]

    fx1 = x.sample_fixations(time_samples)
    fx2 = y.sample_fixations(time_samples)

    coords1 = fx1[["x", "y"]].values
    coords2 = fx2[["x", "y"]].values

    # Pairwise (element-wise) distances
    d = np.sqrt(((coords1 - coords2) ** 2).sum(axis=1)) if dm == "euclidean" else \
        np.abs(coords1 - coords2).sum(axis=1)

    valid = ~np.isnan(d)
    overlap = int((d[valid] < dthresh).sum())
    # Time points before either group's first fixation have no gaze position
    # to compare, so they are left out of the denominator.
    n_compared = int(valid.sum())
    perc = overlap / n_compared if n_compared > 0 else np.nan

    return {"overlap": overlap, "perc": perc}
