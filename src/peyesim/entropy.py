"""Entropy measures for fixation density maps and fixation groups."""

from __future__ import annotations

import warnings

import numpy as np

from peyesim.fixations import FixationGroup
from peyesim.density import (
    EyeDensity,
    EyeDensityMultiscale,
    eye_density,
    suggest_sigma,
)


def entropy_from_mass(
    mass: np.ndarray,
    normalize: bool = True,
    base: float = np.e,
) -> float:
    """Compute Shannon entropy from a probability/mass array.

    Parameters
    ----------
    mass : array-like
        Non-negative values (counts or probabilities).  Will be
        normalised to sum to 1 internally.
    normalize : bool
        If *True* divide the raw entropy by ``log(N, base)`` so the
        result lies in [0, 1].
    base : float
        Logarithm base (default *e* for nats).

    Returns
    -------
    float
        Entropy value, or ``nan`` if the input is empty / all-zero.
    """
    arr = np.asarray(mass, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    total = arr.sum()
    if total <= 0:
        return np.nan

    n_cells = len(arr)
    p = arr / total
    # Only sum over positive entries (0*log(0) == 0 by convention)
    pos = p[p > 0]

    log_fn = np.log if base == np.e else lambda v: np.log(v) / np.log(base)
    ent = -np.sum(pos * log_fn(pos))

    if normalize:
        if n_cells <= 1:
            # Single cell: entropy is 0 by convention (no uncertainty)
            return 0.0
        max_ent = log_fn(n_cells)
        ent = ent / max_ent

    return float(ent)


def _entropy_eye_density(ed: EyeDensity, normalize: bool, base: float) -> float:
    return entropy_from_mass(ed.z, normalize=normalize, base=base)


def _entropy_multiscale(
    edm: EyeDensityMultiscale,
    normalize: bool,
    base: float,
    aggregate: str = "mean",
) -> float | list[float]:
    per_scale = [
        _entropy_eye_density(s, normalize=normalize, base=base)
        for s in edm.scales
    ]
    if aggregate == "none":
        return per_scale
    return float(np.nanmean(per_scale))


def _entropy_fixation_group_density(
    fg: FixationGroup,
    normalize: bool,
    base: float,
    sigma: float | None = None,
    outdim: tuple[int, int] = (100, 100),
    **kwargs,
) -> float:
    x_vals = fg["x"].to_numpy(dtype=float)
    y_vals = fg["y"].to_numpy(dtype=float)

    if len(x_vals) < 2:
        return np.nan

    # Auto-resolve bounds with 5% padding
    xmin, xmax = float(x_vals.min()), float(x_vals.max())
    ymin, ymax = float(y_vals.min()), float(y_vals.max())
    xpad = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
    ypad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
    xbounds = kwargs.pop("xbounds", (xmin - xpad, xmax + xpad))
    ybounds = kwargs.pop("ybounds", (ymin - ypad, ymax + ypad))

    if sigma is None:
        sigma = suggest_sigma(fg, xbounds=xbounds, ybounds=ybounds)
        if np.isnan(sigma) or sigma <= 0:
            sigma = 50.0

    ed = eye_density(
        fg,
        sigma=sigma,
        xbounds=xbounds,
        ybounds=ybounds,
        outdim=outdim,
        normalize=True,
        **kwargs,
    )
    if ed is None:
        return np.nan

    return _entropy_eye_density(ed, normalize=normalize, base=base)


def _entropy_fixation_group_grid(
    fg: FixationGroup,
    normalize: bool,
    base: float,
    grid_size: tuple[int, int] = (10, 10),
    **kwargs,
) -> float:
    x_vals = fg["x"].to_numpy(dtype=float)
    y_vals = fg["y"].to_numpy(dtype=float)

    if len(x_vals) < 1:
        return np.nan

    xbounds = kwargs.get("xbounds", None)
    ybounds = kwargs.get("ybounds", None)

    if xbounds is None:
        xbounds = (float(x_vals.min()), float(x_vals.max()))
    if ybounds is None:
        ybounds = (float(y_vals.min()), float(y_vals.max()))

    # Bin fixations into a grid
    x_edges = np.linspace(xbounds[0], xbounds[1], grid_size[0] + 1)
    y_edges = np.linspace(ybounds[0], ybounds[1], grid_size[1] + 1)

    counts, _, _ = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])

    return entropy_from_mass(counts, normalize=normalize, base=base)


def fixation_entropy(
    x,
    normalize: bool = True,
    base: float = np.e,
    method: str = "density",
    **kwargs,
) -> float:
    """Compute Shannon entropy of fixation patterns.

    Parameters
    ----------
    x : EyeDensity, EyeDensityMultiscale, or FixationGroup
        The fixation data to compute entropy for.
    normalize : bool
        If *True*, return normalised entropy in [0, 1].
    base : float
        Logarithm base (default *e*).
    method : str
        For ``FixationGroup`` input only: ``"density"`` (default) builds a
        KDE density map first; ``"grid"`` bins fixations into an occupancy
        grid.
    **kwargs
        Extra arguments forwarded to the density or grid computation
        (e.g. ``sigma``, ``outdim``, ``grid_size``, ``xbounds``,
        ``ybounds``, ``aggregate``).

    Returns
    -------
    float (or list[float] when ``aggregate="none"`` for multiscale)
    """
    if isinstance(x, EyeDensity):
        return _entropy_eye_density(x, normalize=normalize, base=base)

    if isinstance(x, EyeDensityMultiscale):
        aggregate = kwargs.pop("aggregate", "mean")
        return _entropy_multiscale(
            x, normalize=normalize, base=base, aggregate=aggregate,
        )

    if isinstance(x, FixationGroup):
        if len(x) == 0:
            return np.nan

        if method == "density":
            return _entropy_fixation_group_density(
                x, normalize=normalize, base=base, **kwargs,
            )
        elif method == "grid":
            return _entropy_fixation_group_grid(
                x, normalize=normalize, base=base, **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'density' or 'grid'."
            )

    raise TypeError(
        f"fixation_entropy() does not support type {type(x).__name__}. "
        "Expected EyeDensity, EyeDensityMultiscale, or FixationGroup."
    )
