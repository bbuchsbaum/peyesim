"""Scanpath construction and polar coordinate utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from peyesim.fixations import FixationGroup


class Scanpath(FixationGroup):
    """A fixation group augmented with saccade vectors (lenx, leny, rho, theta)."""


def cart2pol(x, y):
    """Convert Cartesian to polar coordinates.

    Returns
    -------
    ndarray
        (n, 2) array with columns ``[rho, theta]``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return np.column_stack([rho, theta])


def calcangle(x1, x2):
    """Angle (degrees) between two vectors."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    # R's norm(x, "2") is the largest singular value; computing it the same way
    # keeps near-parallel angles (where arccos is ill-conditioned) bitwise close.
    def norm2(v):
        return np.linalg.svd(v.reshape(-1, 1), compute_uv=False)[0]

    cos_val = np.sum(x1 * x2) / (norm2(x1) * norm2(x2))
    # Clamp: rounding can push the cosine of (anti)parallel vectors just past
    # +/-1, where arccos() returns NaN (as in R's calcangle()).
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def scanpath(fg: FixationGroup) -> Scanpath:
    """Create a :class:`Scanpath` from a :class:`FixationGroup`.

    Adds columns ``lenx``, ``leny``, ``rho``, ``theta`` computed from
    successive fixation differences. Saccades join consecutive rows, so rows
    should be in temporal order; they are not re-sorted by onset (a group that
    pools several viewings keeps each viewing's fixations together).
    """
    xs = fg["x"].to_numpy()
    ys = fg["y"].to_numpy()

    lenx = np.diff(xs)
    leny = np.diff(ys)

    polar = cart2pol(lenx, leny)

    out = fg.to_pandas(copy=True)
    out["lenx"] = np.append(lenx, 0.0)
    out["leny"] = np.append(leny, 0.0)
    out["rho"] = np.append(polar[:, 0], 0.0)
    out["theta"] = np.append(polar[:, 1], 0.0)

    return Scanpath(out)


def add_scanpath(table: pd.DataFrame, outvar: str = "scanpath",
                 fixvar: str = "fixgroup") -> pd.DataFrame:
    """Add a scanpath column to *table* (mirrors R ``add_scanpath``)."""
    table = table.copy()
    table[outvar] = [scanpath(fg) for fg in table[fixvar]]
    return table
