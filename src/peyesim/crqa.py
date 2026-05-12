"""Cross-recurrence quantification analysis for fixation coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from peyesim.fixations import FixationGroup


def _coordinate_matrix(x) -> np.ndarray:
    if isinstance(x, FixationGroup):
        frame = x.to_pandas(copy=False)
    elif isinstance(x, pd.DataFrame):
        frame = x
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("CRQA inputs must have at least two coordinate columns.")
        return arr[:, :2]

    if {"x", "y"}.issubset(frame.columns):
        return frame.loc[:, ["x", "y"]].to_numpy(dtype=float)
    if frame.shape[1] < 2:
        raise ValueError("CRQA inputs must have at least two coordinate columns.")
    return frame.iloc[:, :2].to_numpy(dtype=float)


def _embed_series(coords: np.ndarray, delay: int, embed: int) -> np.ndarray:
    if delay < 1:
        raise ValueError("delay must be >= 1.")
    if embed < 1:
        raise ValueError("embed must be >= 1.")
    if embed == 1:
        return coords

    n_rows = coords.shape[0] - (embed - 1) * delay
    if n_rows < 1:
        raise ValueError("embed and delay leave no observations to analyze.")
    return np.column_stack([coords[i * delay:i * delay + n_rows] for i in range(embed)])


def _diagonal_run_lengths(recurrence: np.ndarray, min_length: int) -> list[int]:
    lengths: list[int] = []
    n_rows, n_cols = recurrence.shape
    for offset in range(-n_rows + 1, n_cols):
        run = 0
        for value in np.diagonal(recurrence, offset=offset):
            if value:
                run += 1
            elif run:
                if run >= min_length:
                    lengths.append(run)
                run = 0
        if run >= min_length:
            lengths.append(run)
    return lengths


def _vertical_run_lengths(recurrence: np.ndarray, min_length: int) -> list[int]:
    lengths: list[int] = []
    for col in range(recurrence.shape[1]):
        run = 0
        for value in recurrence[:, col]:
            if value:
                run += 1
            elif run:
                if run >= min_length:
                    lengths.append(run)
                run = 0
        if run >= min_length:
            lengths.append(run)
    return lengths


def _line_entropy(lengths: list[int]) -> float:
    if not lengths:
        return np.nan
    values, counts = np.unique(lengths, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def _relative_line_entropy(lengths: list[int]) -> float:
    if not lengths:
        return np.nan
    entropy = _line_entropy(lengths)
    n_bins = max(lengths) - min(lengths) + 1
    if not np.isfinite(entropy) or n_bins <= 1:
        return np.nan
    return float(entropy / np.log(n_bins))


def crqa(
    fg1,
    fg2,
    radius: float = 60,
    delay: int = 1,
    embed: int = 1,
    rescale: int | bool = 0,
    metric: str = "euclidean",
    min_line: int = 2,
) -> dict:
    """Compute multidimensional cross-recurrence summaries.

    The default mirrors the R wrapper in ``eyesim``: both inputs are truncated
    to their shared length, the two coordinate columns are analyzed, and points
    within ``radius`` are recurrent.
    """
    coords1 = _coordinate_matrix(fg1)
    coords2 = _coordinate_matrix(fg2)
    n_rows = min(len(coords1), len(coords2))
    if n_rows == 0:
        raise ValueError("CRQA inputs must contain at least one row.")

    coords1 = coords1[:n_rows]
    coords2 = coords2[:n_rows]

    if rescale:
        pooled = np.vstack([coords1, coords2])
        center = np.nanmean(pooled, axis=0)
        scale = np.nanstd(pooled, axis=0, ddof=0)
        scale = np.where(scale == 0, 1.0, scale)
        coords1 = (coords1 - center) / scale
        coords2 = (coords2 - center) / scale

    coords1 = _embed_series(coords1, delay=delay, embed=embed)
    coords2 = _embed_series(coords2, delay=delay, embed=embed)
    n_rows = min(len(coords1), len(coords2))
    coords1 = coords1[:n_rows]
    coords2 = coords2[:n_rows]

    metric_map = {"euclidean": "euclidean", "manhattan": "cityblock"}
    dist = cdist(coords1, coords2, metric=metric_map.get(metric, metric))
    recurrence = dist <= radius
    recurrent_points = int(recurrence.sum())
    total_points = int(recurrence.size)

    diag_lengths = _diagonal_run_lengths(recurrence, min_length=min_line)
    vert_lengths = _vertical_run_lengths(recurrence, min_length=min_line)
    diag_points = int(sum(diag_lengths))
    vert_points = int(sum(vert_lengths))
    recurrence_rate = recurrent_points / total_points if total_points else np.nan
    determinism = diag_points / recurrent_points if recurrent_points else np.nan
    laminarity = vert_points / recurrent_points if recurrent_points else np.nan
    max_vertlength = int(max(vert_lengths)) if vert_lengths else -np.inf

    return {
        "recurrence_matrix": recurrence,
        "distance_matrix": dist,
        "radius": float(radius),
        "n_points": int(n_rows),
        "recurrent_points": recurrent_points,
        "recurrence_rate": recurrence_rate,
        "determinism": determinism,
        "mean_line": float(np.mean(diag_lengths)) if diag_lengths else np.nan,
        "max_line": int(max(diag_lengths)) if diag_lengths else 0,
        "line_entropy": _line_entropy(diag_lengths),
        "laminarity": laminarity,
        "trapping_time": float(np.mean(vert_lengths)) if vert_lengths else np.nan,
        "RR": recurrence_rate * 100 if np.isfinite(recurrence_rate) else np.nan,
        "DET": determinism * 100 if np.isfinite(determinism) else np.nan,
        "NRLINE": int(len(diag_lengths)),
        "maxL": int(max(diag_lengths)) if diag_lengths else 0,
        "L": float(np.mean(diag_lengths)) if diag_lengths else np.nan,
        "ENTR": _line_entropy(diag_lengths),
        "rENTR": _relative_line_entropy(diag_lengths),
        "LAM": laminarity * 100 if np.isfinite(laminarity) else np.nan,
        "TT": float(np.mean(vert_lengths)) if vert_lengths else np.nan,
        "catH": np.nan,
        "max_vertlength": max_vertlength,
        "RP": recurrence,
    }
