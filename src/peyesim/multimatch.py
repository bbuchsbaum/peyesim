"""MultiMatch scanpath comparison algorithm."""

from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

from peyesim._utils import emdw as _emdw
from peyesim.saccades import Scanpath


def install_multimatch(upgrade: bool = False) -> None:
    """Install the optional ``multimatch_gaze`` reference package via ``pip``."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append("multimatch_gaze")
    subprocess.run(cmd, check=True)


def _emd_position_similarity(fg1, fg2, screensize):
    """Order-insensitive EMD-based position similarity."""
    pts1 = fg1[["x", "y"]].values
    pts2 = fg2[["x", "y"]].values
    emd_val = _emdw(pts1, fg1["duration"].values, pts2, fg2["duration"].values)
    max_dist = np.sqrt(screensize[0] ** 2 + screensize[1] ** 2)
    return 1.0 - emd_val / max_dist


def _create_graph(sacx, sacy):
    """Build directed graph and find shortest path (mirrors R ``create_graph``)."""
    M = cdist(
        np.column_stack([sacx["lenx"].values, sacx["leny"].values]),
        np.column_stack([sacy["lenx"].values, sacy["leny"].values]),
    )

    nr, nc = M.shape
    # Node IDs laid out row-major (like numpy reshape default)
    M_assignment = np.arange(nr * nc).reshape(nr, nc)

    G = nx.DiGraph()

    # Right edges
    if nc > 1:
        for i in range(nr):
            for j in range(nc - 1):
                G.add_edge(M_assignment[i, j], M_assignment[i, j + 1],
                           weight=M[i, j + 1])

    # Down edges
    if nr > 1:
        for i in range(nr - 1):
            for j in range(nc):
                G.add_edge(M_assignment[i, j], M_assignment[i + 1, j],
                           weight=M[i + 1, j])

    # Diagonal edges
    if nr > 1 and nc > 1:
        for i in range(nr - 1):
            for j in range(nc - 1):
                G.add_edge(M_assignment[i, j], M_assignment[i + 1, j + 1],
                           weight=M[i + 1, j + 1])

    # Terminal self-edge
    last = M_assignment[nr - 1, nc - 1]
    G.add_edge(last, last, weight=0)

    start = M_assignment[0, 0]
    end = last

    path = nx.shortest_path(G, start, end, weight="weight")

    return {"g": G, "vpath": path, "M": M, "M_assignment": M_assignment}


def _vector_diff_2d(x, y, v1, v2, cds):
    x1 = x[v1].values[cds[:, 0]]
    x2 = y[v1].values[cds[:, 1]]
    y1 = x[v2].values[cds[:, 0]]
    y2 = y[v2].values[cds[:, 1]]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _vector_diff_1d(x, y, v1, cds):
    x1 = x[v1].values[cds[:, 0]]
    x2 = y[v1].values[cds[:, 1]]
    return x1 - x2


def _angle_diff_1d(theta1_arr, theta2_arr, cds):
    t1 = theta1_arr[cds[:, 0]].copy()
    t2 = theta2_arr[cds[:, 1]].copy()
    t1 = np.where(t1 < 0, np.pi + (np.pi + t1), t1)
    t2 = np.where(t2 < 0, np.pi + (np.pi + t2), t2)
    adiff = np.abs(t1 - t2)
    adiff = np.where(adiff > np.pi, 2 * np.pi - adiff, adiff)
    return adiff


def _duration_diff_1d(dur1, dur2, cds):
    d1 = dur1[cds[:, 0]]
    d2 = dur2[cds[:, 1]]
    adiff = np.abs(d1 - d2)
    denom = np.maximum(d1, d2)
    # avoid /0
    denom = np.where(denom == 0, 1.0, denom)
    return adiff / denom


def multi_match(x: Scanpath, y: Scanpath, screensize: tuple[float, float]) -> dict:
    """Compute MultiMatch metrics between two scanpaths.

    Returns a dict with keys: mm_vector, mm_direction, mm_length,
    mm_position, mm_duration, mm_position_emd.
    """
    if np.any(np.diff(x["onset"].values) <= 0):
        raise ValueError("multi_match: x `onset` vector must be strictly increasing")
    if np.any(np.diff(y["onset"].values) <= 0):
        raise ValueError("multi_match: y `onset` vector must be strictly increasing")

    if len(x) < 3 or len(y) < 3:
        warnings.warn("multi_match requires 3 or more coordinates in each scanpath, returning NAs")
        return {
            "mm_vector": np.nan, "mm_direction": np.nan,
            "mm_length": np.nan, "mm_position": np.nan,
            "mm_duration": np.nan,
        }

    sacx = x.iloc[:-1].reset_index(drop=True)
    sacy = y.iloc[:-1].reset_index(drop=True)

    gout = _create_graph(sacx, sacy)
    path = np.array(gout["vpath"])
    nc = gout["M"].shape[1]

    rnum = path // nc
    cnum = path % nc
    cds = np.column_stack([rnum, cnum])

    diag = np.sqrt(screensize[0] ** 2 + screensize[1] ** 2)

    vector_d = _vector_diff_2d(sacx, sacy, "lenx", "leny", cds)
    vector_sim = 1 - np.median(vector_d) / (2 * diag)

    direction_d = _angle_diff_1d(sacx["theta"].values, sacy["theta"].values, cds)
    direction_sim = 1 - np.median(direction_d) / np.pi

    duration_d = _duration_diff_1d(sacx["duration"].values, sacy["duration"].values, cds)
    duration_sim = 1 - np.median(duration_d)

    length_d = np.abs(_vector_diff_1d(sacx, sacy, "rho", cds))
    length_sim = 1 - np.median(length_d) / diag

    position_d = _vector_diff_2d(sacx, sacy, "x", "y", cds)
    position_sim = 1 - np.median(position_d) / diag

    emd_sim = _emd_position_similarity(sacx, sacy, screensize)

    return {
        "mm_vector": float(vector_sim),
        "mm_direction": float(direction_sim),
        "mm_length": float(length_sim),
        "mm_position": float(position_sim),
        "mm_duration": float(duration_sim),
        "mm_position_emd": float(emd_sim),
    }
