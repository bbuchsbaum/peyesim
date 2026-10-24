"""Shared utilities."""

from __future__ import annotations

import os
import warnings

import numpy as np


def emdw(x, wx, y, wy, normalize: bool = True):
    """Weighted Earth Mover's Distance (Wasserstein-1) between 2-D point clouds.

    Mirrors the internal ``emdw()`` helper of R eyesim. By default the weights
    need not sum to one: each side is normalised to unit mass, so the result is
    the distance between the two distributions. Zero-mass points are dropped
    first (they do not change the distance), which keeps dense density grids
    cheap. With ``normalize=False`` the raw masses are kept and unequal totals
    give Rubner's partial-matching EMD (as ``emdist::emdw`` computes): the
    minimum-cost transport of ``min(sum(wx), sum(wy))`` mass, divided by that
    mass.

    Returns ``nan`` when either total is non-finite or when exactly one side
    has no mass, and ``0`` when neither side has mass.
    """
    os.environ.setdefault("POT_BACKEND_DISABLE_PYTORCH", "1")
    os.environ.setdefault("POT_BACKEND_DISABLE_TENSORFLOW", "1")
    os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")
    os.environ.setdefault("POT_BACKEND_DISABLE_CUPY", "1")

    import ot

    x = np.asarray(x, dtype=float).reshape(-1, 2)
    y = np.asarray(y, dtype=float).reshape(-1, 2)
    wx = np.asarray(wx, dtype=float).ravel()
    wy = np.asarray(wy, dtype=float).ravel()
    sx = wx.sum()
    sy = wy.sum()
    if not (np.isfinite(sx) and np.isfinite(sy)):
        return float("nan")
    if sx <= 0 or sy <= 0:
        return 0.0 if (sx <= 0 and sy <= 0) else float("nan")

    kx = wx > 0
    ky = wy > 0
    x, wx = x[kx], wx[kx]
    y, wy = y[ky], wy[ky]
    M = ot.dist(x, y, metric="euclidean")

    if not normalize:
        transported_mass = min(wx.sum(), wy.sum())
        dist = ot.partial.partial_wasserstein2(wx, wy, M, m=transported_mass)
        return float(dist / transported_mass)

    wx = wx / wx.sum()
    wy = wy / wy.sum()
    return float(ot.emd2(wx, wy, M, numItermax=10_000_000))


def sinkhorn_distance(D, wx, wy, p: float = 2.0, lam: float = 0.1,
                      maxiter: int = 496, abstol: float = 1e-10) -> float:
    """Entropic optimal-transport distance, as ``T4transport::sinkhornD``.

    Runs the same log-domain Sinkhorn iterations on the cost ``D ** p``
    (stopping when the L1 marginal error drops below ``abstol``) and returns
    ``(sum(plan * D ** p)) ** (1 / p)``.
    """
    from scipy.special import logsumexp

    D = np.asarray(D, dtype=float)
    a = np.asarray(wx, dtype=float).ravel()
    b = np.asarray(wy, dtype=float).ravel()
    p = max(1.0, float(p))
    lam = max(np.sqrt(np.finfo(float).eps), float(lam))
    maxiter = max(1, int(round(maxiter)))
    abstol = max(np.sqrt(np.finfo(float).eps), float(abstol))

    C = D ** p
    f = np.zeros(C.shape[0])
    g = np.zeros(C.shape[1])
    with np.errstate(divide="ignore"):
        log_a = np.log(a)
        log_b = np.log(b)
    for _ in range(maxiter):
        f = lam * (log_a - logsumexp((-C + g[None, :]) / lam, axis=1))
        g = lam * (log_b - logsumexp((-C + f[:, None]) / lam, axis=0))
        plan = np.exp((f[:, None] + g[None, :] - C) / lam)
        err = np.abs(plan.sum(axis=1) - a).sum() + np.abs(plan.sum(axis=0) - b).sum()
        if err < abstol:
            break
    plan = np.exp((f[:, None] + g[None, :] - C) / lam)
    return float(np.sum(plan * C) ** (1.0 / p))


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
