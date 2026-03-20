"""Shared utilities."""

from __future__ import annotations

import os
import warnings

import numpy as np


def emdw(x, wx, y, wy):
    """Weighted Earth Mover's Distance between two 2-D point clouds using POT.

    Mirrors ``emdist::emdw`` from the R package: when total masses differ, it
    computes the minimum-cost partial transport and normalizes by the smaller
    total mass.
    """
    os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
    os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
    os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")
    os.environ.setdefault("POT_BACKEND_DISABLE_CUPY", "1")

    import ot

    wx = np.asarray(wx, dtype=float)
    wy = np.asarray(wy, dtype=float)
    sw = wx.sum()
    sy = wy.sum()
    if sw <= 0 or sy <= 0:
        return 0.0

    transported_mass = min(sw, sy)
    M = ot.dist(np.asarray(x, dtype=float), np.asarray(y, dtype=float), metric="euclidean")
    dist = ot.partial.partial_wasserstein2(wx, wy, M, m=transported_mass)
    return float(dist / transported_mass)


def match_keys(source_keys, ref_keys):
    """Build match-index list mapping source → reference by key equality.

    Returns a list of (int | None), one per source key.
    """
    key_to_idx = {}
    for i, k in enumerate(ref_keys):
        if k not in key_to_idx:
            key_to_idx[k] = i
    return [key_to_idx.get(sk) for sk in source_keys]


def filter_unmatched(source_tab, matchind, warn_msg=None):
    """Remove rows from *source_tab* where matchind is None.

    Returns ``(filtered_source_tab, filtered_matchind)``.
    """
    keep = [i for i, m in enumerate(matchind) if m is not None]
    if len(keep) < len(matchind) and warn_msg:
        warnings.warn(warn_msg)
    return (
        source_tab.iloc[keep].reset_index(drop=True),
        [matchind[i] for i in keep],
    )
