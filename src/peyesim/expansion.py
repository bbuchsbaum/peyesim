"""Scale matching helpers for fixation coordinate sets."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import directed_hausdorff

from peyesim.fixations import FixationGroup


def _as_frame(x) -> pd.DataFrame:
    if isinstance(x, FixationGroup):
        return x.to_pandas(copy=False)
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(np.asarray(x))


def _xy_matrix(x) -> np.ndarray:
    frame = _as_frame(x)
    if {"x", "y"}.issubset(frame.columns):
        return frame.loc[:, ["x", "y"]].to_numpy(dtype=float)
    if frame.shape[1] < 2:
        raise ValueError("scale estimation inputs must have at least two coordinate columns.")
    return frame.iloc[:, :2].to_numpy(dtype=float)


def _hausdorff_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(max(directed_hausdorff(x, y)[0], directed_hausdorff(y, x)[0]))


def estimate_scale(
    x,
    y,
    lower: tuple[float, float] = (0.1, 0.1),
    upper: tuple[float, float] = (10.0, 10.0),
    window: tuple[float, float] | None = None,
) -> dict:
    """Estimate x/y scale factors that align ``y`` to ``x``.

    This mirrors the R helper in ``expansion.R``: an optional ``window`` filters
    the source fixations ``y`` by onset, and the optimizer minimizes the
    Hausdorff distance between scaled source coordinates and reference
    coordinates.
    """
    y_frame = _as_frame(y)
    if window is not None:
        if window[1] <= window[0]:
            raise ValueError("window upper bound must be greater than lower bound.")
        if "onset" not in y_frame.columns:
            raise ValueError("window filtering requires an 'onset' column in y.")
        y_frame = y_frame.loc[(y_frame["onset"] >= window[0]) & (y_frame["onset"] < window[1])]

    cx = _xy_matrix(x)
    cy = _xy_matrix(y_frame)
    if len(cx) == 0 or len(cy) == 0:
        return {"par": np.array([1.0, 1.0]), "value": np.nan, "success": True}

    bounds = list(zip(lower, upper))

    def objective(par):
        scaled = cy @ np.diag(par)
        return _hausdorff_distance(scaled, cx)

    result = minimize(
        objective,
        x0=np.array([1.0, 1.0]),
        bounds=bounds,
        method="L-BFGS-B",
    )
    return {
        "par": np.asarray(result.x, dtype=float),
        "value": float(result.fun),
        "success": bool(result.success),
        "message": result.message,
        "n_ref": int(len(cx)),
        "n_source": int(len(cy)),
    }


def match_scale(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str,
    refvar: str = "fixgroup",
    sourcevar: str = "fixgroup",
    window: tuple[float, float] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Estimate per-row source scale factors from matched reference rows."""
    if window is not None and window[1] <= window[0]:
        raise ValueError("window upper bound must be greater than lower bound.")

    ref_lookup = {}
    for idx, key in enumerate(ref_tab[match_on]):
        ref_lookup.setdefault(key, idx)

    rows = []
    missing = []
    for _, row in source_tab.iterrows():
        match_idx = ref_lookup.get(row[match_on])
        if match_idx is None:
            missing.append(row[match_on])
            continue
        ref_fix = ref_tab.iloc[match_idx][refvar]
        src_fix = row[sourcevar]
        scale = estimate_scale(ref_fix, src_fix, window=window, **kwargs)["par"]
        out = row.copy()
        out["scale_x"] = scale[0]
        out["scale_y"] = scale[1]
        rows.append(out)

    if missing:
        warnings.warn(
            "did not find matching template map for all source maps. "
            "Removing non-matching elements."
        )
    return pd.DataFrame(rows).reset_index(drop=True)
