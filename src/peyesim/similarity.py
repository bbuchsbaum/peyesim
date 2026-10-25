"""Similarity metrics and template-based analysis."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import lru_cache
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as _cosine_dist
from scipy.stats import spearmanr

from peyesim._utils import emdw as _emdw, match_keys, filter_unmatched, sinkhorn_distance
from peyesim.density import EyeDensity, EyeDensityMultiscale
from peyesim.fixations import FixationGroup
from peyesim.saccades import Scanpath


# ---------------------------------------------------------------------------
# Core compute_similarity
# ---------------------------------------------------------------------------

_DENSITY_METHODS = ("pearson", "spearman", "fisherz", "cosine", "l1", "jaccard", "dcov", "emd")


def _match_choice(value, choices, label: str = "arg") -> str:
    """R ``match.arg()`` for a single string: exact or unique-prefix match."""
    if value is None:
        return choices[0]
    if isinstance(value, (list, tuple)):
        if list(value) == list(choices):
            return choices[0]
        raise ValueError(f"'{label}' must be of length 1")
    if value in choices:
        return value
    hits = [c for c in choices if isinstance(value, str) and value and c.startswith(value)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"'{label}' should be one of {', '.join(repr(c) for c in choices)}")


def compute_similarity(
    x,
    y,
    method: str = "pearson",
    saliency_map=None,
) -> float:
    """Compute similarity between two vectors / density objects.

    Supported methods: pearson (default, as in R), spearman, fisherz, cosine,
    l1, jaccard, dcov, emd.
    """
    method = method.lower() if isinstance(method, str) else method
    method = _match_choice(method, _DENSITY_METHODS, "method")

    # EMD path – needs full density objects
    if method == "emd":
        if isinstance(x, EyeDensity) and isinstance(y, EyeDensity):
            coords = np.array(np.meshgrid(x.x, x.y, indexing="ij")).reshape(2, -1).T
            wx = x.z.ravel()
            wy = y.z.ravel()
            if saliency_map is not None:
                s_mat = saliency_map.z if isinstance(saliency_map, EyeDensity) else np.asarray(saliency_map)
                if np.shape(s_mat) != np.shape(x.z):
                    raise ValueError("saliency_map dimensions must match density maps")
                r1 = x.z - s_mat
                r2 = y.z - s_mat
                pos1 = np.maximum(r1, 0).ravel()
                neg1 = np.maximum(-r1, 0).ravel()
                pos2 = np.maximum(r2, 0).ravel()
                neg2 = np.maximum(-r2, 0).ravel()
                # The residual masses differ between maps, so these are
                # partial-matching EMDs on the raw masses.
                emd_pos = _emdw(coords, pos1, coords, pos2, normalize=False)
                emd_neg = _emdw(coords, neg1, coords, neg2, normalize=False)
                return -(emd_pos + emd_neg)
            else:
                emd_dist = _emdw(coords, wx, coords, wy)
                return 1.0 / (1.0 + emd_dist)
        else:
            raise ValueError("method 'emd' requires EyeDensity objects")

    # Flatten to vectors
    if isinstance(x, EyeDensity):
        x = x.z
    if isinstance(y, EyeDensity):
        y = y.z
    vx = np.asarray(x, dtype=float).ravel()
    vy = np.asarray(y, dtype=float).ravel()

    # Find common valid entries
    valid = ~(np.isnan(vx) | np.isnan(vy))
    if valid.sum() < 2:
        warnings.warn("Less than 2 common valid data points for similarity calculation.")
        return np.nan

    vx = vx[valid]
    vy = vy[valid]

    var_x = np.var(vx, ddof=1)
    var_y = np.var(vy, ddof=1)

    eps = np.finfo(float).eps

    if method in ("pearson", "spearman", "fisherz", "dcov"):
        zv_x = np.isnan(var_x) or var_x < eps
        zv_y = np.isnan(var_y) or var_y < eps
        if zv_x or zv_y:
            if zv_x and zv_y and np.all(np.abs(vx - vy) < np.sqrt(eps)):
                # Identical constant maps: perfect correlation. Fisher z uses
                # the same clamp as non-constant identical maps below.
                return float(np.arctanh(1 - eps)) if method == "fisherz" else 1.0
            warnings.warn(
                f"Method {method} requires variance in both inputs. "
                "One or both have near-zero variance."
            )
            return np.nan

    if method == "pearson":
        return float(np.corrcoef(vx, vy)[0, 1])
    elif method == "spearman":
        return float(spearmanr(vx, vy)[0])  # `.statistic` needs SciPy >= 1.9
    elif method == "fisherz":
        # Identical maps have r = 1 exactly; corrcoef can return 1 - k * eps.
        r = 1.0 if np.array_equal(vx, vy) else float(np.corrcoef(vx, vy)[0, 1])
        # r within rounding error of +/-1 (e.g. a rescaled copy) counts as +/-1,
        # so every perfect correlation maps to the same z.
        snap = 64 * eps
        if r > 1 - snap:
            r = 1.0
        if r < -1 + snap:
            r = -1.0
        r = np.clip(r, -1 + eps, 1 - eps)
        return float(np.arctanh(r))
    elif method == "cosine":
        # proxy::simil(method = "cosine"): xy / sqrt(xx * yy), with an
        # all-zero pair counting as identical (1) and one zero vector as 0.
        nx = np.linalg.norm(vx)
        ny = np.linalg.norm(vy)
        if nx == 0 or ny == 0:
            return 1.0 if nx == 0 and ny == 0 else 0.0
        d = _cosine_dist(vx, vy)
        return 1.0 - d
    elif method == "l1":
        if np.any(vx < 0) or np.any(vy < 0):
            warnings.warn("L1 distance interpretation assumes non-negative densities.")
        sx = vx.sum()
        sy = vy.sum()
        if sx <= eps or sy <= eps:
            warnings.warn("Cannot normalize for L1 distance; sum is too small.")
            return np.nan
        x1 = vx / sx
        x2 = vy / sy
        return 1.0 - 0.5 * np.abs(x1 - x2).sum()
    elif method == "jaccard":
        # Extended (Tanimoto) Jaccard, as proxy's "eJaccard" in R eyesim.
        # proxy gives 1 when the denominator vanishes (both vectors zero).
        xy = float(np.dot(vx, vy))
        den = float(np.dot(vx, vx) + np.dot(vy, vy) - xy)
        if den == 0:
            return 1.0
        return xy / den
    elif method == "dcov":
        sx = vx.sum()
        sy = vy.sum()
        if sx <= eps or sy <= eps:
            warnings.warn("Cannot normalize for dcov; sum is too small.")
            return np.nan
        return _distance_correlation(vx / sx, vy / sy)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def _pairwise_abs_row_sums(v: np.ndarray) -> np.ndarray:
    """``sum_j |v_i - v_j|`` for every ``i`` in O(n log n)."""
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    csum = np.cumsum(sv)
    total = csum[-1]
    k = np.arange(n)
    before = np.concatenate(([0.0], csum[:-1]))
    sums_sorted = sv * k - before + (total - csum) - sv * (n - 1 - k)
    out = np.empty(n)
    out[order] = sums_sorted
    return out


def _sum_abs_products(x: np.ndarray, y: np.ndarray) -> float:
    """``sum_{i,j} |x_i - x_j| |y_i - y_j|`` in O(n log n).

    With rows sorted by ``x``, ``|x_j - x_i| = x_j - x_i`` for ``i < j`` and
    ``|y_j - y_i| = (y_j - y_i) (2 [y_i < y_j] - 1)``. The unsigned part is a
    prefix-sum identity; the indicator part is accumulated with Fenwick trees
    keyed by the rank of ``y``.
    """
    n = len(x)
    order = np.lexsort((y, x))
    xs = x[order]
    ys = y[order]
    # sum_{i<j} (x_j - x_i)(y_j - y_i)
    cx = np.concatenate(([0.0], np.cumsum(xs)[:-1]))
    cy = np.concatenate(([0.0], np.cumsum(ys)[:-1]))
    cxy = np.concatenate(([0.0], np.cumsum(xs * ys)[:-1]))
    k = np.arange(n)
    plain = float(np.sum(k * xs * ys - xs * cy - ys * cx + cxy))

    # sum_{i<j, y_i<y_j} (x_j - x_i)(y_j - y_i), via Fenwick trees over y rank.
    ranks = np.searchsorted(np.unique(ys), ys) + 1
    size = int(ranks.max())
    t_n = [0.0] * (size + 1)
    t_x = [0.0] * (size + 1)
    t_y = [0.0] * (size + 1)
    t_xy = [0.0] * (size + 1)
    below = 0.0
    xs_l = xs.tolist()
    ys_l = ys.tolist()
    rk_l = ranks.tolist()
    for j in range(n):
        xj, yj, r = xs_l[j], ys_l[j], rk_l[j]
        # query strictly smaller y ranks: prefix r - 1
        cnt = sx = sy = sxy = 0.0
        q = r - 1
        while q > 0:
            cnt += t_n[q]
            sx += t_x[q]
            sy += t_y[q]
            sxy += t_xy[q]
            q -= q & -q
        below += cnt * xj * yj - xj * sy - yj * sx + sxy
        q = r
        while q <= size:
            t_n[q] += 1.0
            t_x[q] += xj
            t_y[q] += yj
            t_xy[q] += xj * yj
            q += q & -q
    # pairs with i<j: signed sum = 2*below - plain; both orders double it.
    return 2.0 * (2.0 * below - plain)


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation (V-statistic) of two univariate samples.

    Matches ``energy::dcor`` as computed by R eyesim (``sqrt`` of
    ``energy::dcor2d(type = "V")`` on standardised values) in O(n log n) time,
    so 100 x 100 maps do not need an n x n distance matrix.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sdx = np.std(x, ddof=1)
    sdy = np.std(y, ddof=1)
    if not (np.isfinite(sdx) and np.isfinite(sdy)) or sdx <= 0 or sdy <= 0:
        return np.nan
    x = (x - x.mean()) / sdx
    y = (y - y.mean()) / sdy
    n = len(x)
    ax =_pairwise_abs_row_sums(x)
    by = _pairwise_abs_row_sums(y)
    n2 = float(n) * n
    n3 = n2 * n
    n4 = n2 * n2

    def v_stat(a_rows, b_rows, s_ab):
        return s_ab / n2 - 2.0 * float(np.dot(a_rows, b_rows)) / n3 + a_rows.sum() * b_rows.sum() / n4

    v_xy = v_stat(ax, by, _sum_abs_products(x, y))
    v_xx = v_stat(ax, ax, _sum_abs_products(x, x))
    v_yy = v_stat(by, by, _sum_abs_products(y, y))
    if v_xx <= 0 or v_yy <= 0:
        return 0.0
    v = v_xy / np.sqrt(v_xx * v_yy)
    return float(np.sqrt(max(v, 0.0)))


class LatticeMismatchError(ValueError):
    """Two density maps are not on the same grid (R class ``eyesim_lattice_mismatch``).

    Density maps are compared cell by cell, so this is an input error rather
    than a numerical failure of one comparison: callers that turn per-pair
    errors into NaN (``repetitive_similarity``, multiscale aggregation) let it
    propagate.
    """


def _all_equal(target, current, tolerance: float = 1.5e-8) -> bool:
    """R ``isTRUE(all.equal(target, current))`` for numeric vectors."""
    target = np.asarray(target, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    if target.shape != current.shape:
        return False
    if target.size == 0:
        return True
    xy = float(np.mean(np.abs(target - current)))
    xn = float(np.mean(np.abs(target)))
    if np.isfinite(xn) and xn > tolerance:
        xy /= xn
    return xy <= tolerance


def _density_lattices_equal(x, y) -> bool:
    gx, gy = np.asarray(x.x, dtype=float), np.asarray(x.y, dtype=float)
    hx, hy = np.asarray(y.x, dtype=float), np.asarray(y.y, dtype=float)
    return (
        gx.size == hx.size and gy.size == hy.size
        and _all_equal(gx, hx) and _all_equal(gy, hy)
        and np.shape(x.z) == np.shape(y.z)
    )


def _check_same_density_lattice(x, y) -> None:
    if not _density_lattices_equal(x, y):
        raise LatticeMismatchError(
            "similarity(): density maps are on different lattices. Both maps must "
            "have the same x and y grid coordinates (same bounds and outdim)."
        )


# ---------------------------------------------------------------------------
# Dispatching similarity function
# ---------------------------------------------------------------------------

def similarity(x, y, method: str | None = None, **kwargs) -> float | np.ndarray:
    """Compute similarity, dispatching on type (mirrors R S3 ``similarity``).

    ``method`` defaults as in R: ``"pearson"`` for density maps (single or
    multiscale) and numeric vectors, ``"sinkhorn"`` for fixation groups and
    ``"multimatch"`` for scanpaths (which require ``screensize``).
    """
    multiscale_aggregation = kwargs.pop("multiscale_aggregation", "mean")
    saliency_map = kwargs.pop("saliency_map", None)
    window = kwargs.pop("window", None)

    # Multiscale
    if isinstance(x, EyeDensityMultiscale):
        if method is None:
            method = "pearson"
        if not isinstance(y, EyeDensityMultiscale):
            raise TypeError("y must also be EyeDensityMultiscale")
        if len(x) == 0 or len(y) == 0:
            warnings.warn("One or both multiscale objects are empty.")
            return np.nan

        sigmas_x = [s.sigma for s in x]
        sigmas_y = [s.sigma for s in y]
        # R intersect(): x's order, without duplicates
        common = list(dict.fromkeys(sg for sg in sigmas_x if sg in sigmas_y))
        if not common:
            warnings.warn("No common sigmas found between multiscale objects. Cannot compute similarity.")
            return np.nan
        if len(common) < len(sigmas_x) or len(common) < len(sigmas_y):
            warnings.warn(
                "Not all sigmas matched between multiscale objects. Comparing only common sigmas: "
                + ", ".join(str(sg) for sg in common)
            )

        sx_map = {}
        for sc in x:
            sx_map.setdefault(sc.sigma, sc)
        sy_map = {}
        for sc in y:
            sy_map.setdefault(sc.sigma, sc)
        # Mismatched lattices are an input error, not a per-scale numerical
        # failure, so check them before the per-scale handler below.
        for sig in common:
            _check_same_density_lattice(sx_map[sig], sy_map[sig])
        per_scale = []
        for sig in common:
            try:
                val = similarity(sx_map[sig], sy_map[sig], method=method,
                                 saliency_map=saliency_map, **kwargs)
            except Exception as e:
                warnings.warn(f"Error computing similarity for sigma={sig}: {e}")
                val = np.nan
            per_scale.append(val)

        per_scale = np.array(per_scale, dtype=float)
        if np.all(np.isnan(per_scale)):
            return np.nan

        agg = _match_choice(str(multiscale_aggregation).lower(), ("mean", "none"),
                            "multiscale_aggregation")
        if agg == "mean":
            return float(np.nanmean(per_scale))
        return per_scale

    # EyeDensity
    if isinstance(x, EyeDensity):
        if method is None:
            method = "pearson"
        if isinstance(y, EyeDensity):
            _check_same_density_lattice(x, y)
        elif y is not None and np.size(y) != np.size(x.z):
            raise ValueError(
                f"similarity(): `y` has {np.size(y)} values but the density map `x` "
                f"has {np.size(x.z)} grid cells."
            )
        return compute_similarity(x, y, method=method, saliency_map=saliency_map)

    # Scanpath (must check before FixationGroup since Scanpath inherits from it)
    if isinstance(x, Scanpath) and isinstance(y, Scanpath):
        if window is not None:
            x = _filter_fixation_window(x, window)
            y = _filter_fixation_window(y, window)
            if len(x) == 0:
                warnings.warn("no observations in 'x'")
                return np.nan
            if len(y) == 0:
                warnings.warn("no observations in 'y'")
                return np.nan
        if method is None:
            method = "multimatch"
        if method == "multimatch":
            from peyesim.multimatch import multi_match
            if len(x) == 0:
                warnings.warn("no observations in 'x'")
                return np.nan
            if len(y) == 0:
                warnings.warn("no observations in 'y'")
                return np.nan
            screensize = kwargs.pop("screensize", None)
            if screensize is None:
                raise ValueError(
                    "method `multi_match` requires a `screensize` argument (e.g. (1000, 1000))"
                )
            return multi_match(x, y, screensize=screensize)
        # Fall through to fixation-level methods

    # FixationGroup
    if isinstance(x, FixationGroup) and isinstance(y, FixationGroup):
        if method is None:
            method = "sinkhorn"
        if window is not None:
            x = _filter_fixation_window(x, window)
            y = _filter_fixation_window(y, window)
        if len(x) == 0:
            warnings.warn("no observations in 'x'")
            return np.nan
        if len(y) == 0:
            warnings.warn("no observations in 'y'")
            return np.nan
        if method == "overlap":
            from peyesim.overlap import fixation_overlap
            time_samples = kwargs.get("time_samples")
            if time_samples is None:
                raise ValueError("method `overlap` requires a vector of `time_samples`")
            result = fixation_overlap(x, y, dthresh=kwargs.get("dthresh", 60),
                                      time_samples=time_samples,
                                      dist_method=kwargs.get("dist_method", "euclidean"))
            return result["perc"]
        elif method == "sinkhorn":
            # As R eyesim: an entropic OT distance between the fixations in
            # (x / xdenom, y / ydenom, onset / tdenom * tweight) space, with
            # duration weights.
            xdenom = kwargs.get("xdenom", 1000)
            ydenom = kwargs.get("ydenom", 1000)
            tdenom = kwargs.get("tdenom", 3000)
            tweight = kwargs.get("tweight", 0.8)
            lam = kwargs.get("lambda_", kwargs.get("lambda", 0.1))

            def _xyt(fg):
                return np.column_stack([
                    np.asarray(fg["x"], dtype=float) / xdenom,
                    np.asarray(fg["y"], dtype=float) / ydenom,
                    np.asarray(fg["onset"], dtype=float) / tdenom * tweight,
                ])

            from scipy.spatial.distance import cdist

            d = cdist(_xyt(x), _xyt(y))
            xdur = np.asarray(x["duration"], dtype=float)
            ydur = np.asarray(y["duration"], dtype=float)
            dist = sinkhorn_distance(d, xdur / xdur.sum(), ydur / ydur.sum(), lam=lam)
            return 1.0 / (1.0 + dist)
        else:
            raise ValueError("method for fixation groups must be one of {'sinkhorn', 'overlap'}")

    # Default: numeric vectors
    return compute_similarity(x, y, method="pearson" if method is None else method,
                              saliency_map=saliency_map)


def _filter_fixation_window(x: FixationGroup, window) -> FixationGroup:
    if len(window) != 2 or window[1] <= window[0]:
        raise ValueError("window must be a length-2 increasing interval.")
    frame = x.to_pandas(copy=True)
    frame = frame.loc[(frame["onset"] >= window[0]) & (frame["onset"] < window[1])].reset_index(drop=True)
    return x.__class__(frame)


# ---------------------------------------------------------------------------
# sample_density
# ---------------------------------------------------------------------------

def sample_density(
    dens: EyeDensity,
    fix: FixationGroup,
    times: np.ndarray | None = None,
    normalize: str = "none",
) -> pd.DataFrame:
    """Sample a density map at fixation locations.

    Parameters
    ----------
    normalize : str
        One of 'none', 'max', 'sum', 'zscore'.
    """
    valid_norms = ("none", "max", "sum", "zscore")
    if normalize not in valid_norms:
        raise ValueError(f"'arg' should be one of {valid_norms}")

    zmat = _normalize_density_values(dens.z, normalize)
    if times is None:
        z_vals = _lookup_density(zmat, dens.x, dens.y, fix["x"], fix["y"])
        return pd.DataFrame({"z": z_vals, "time": np.asarray(fix["onset"])})
    times = np.asarray(times, dtype=float)
    fg_sampled = fix.sample_fixations(times)
    z_vals = _lookup_density(zmat, dens.x, dens.y, fg_sampled["x"], fg_sampled["y"])
    return pd.DataFrame({"z": z_vals, "time": times})


def _normalize_density_values(zmat, normalize: str) -> np.ndarray:
    """Normalise a density matrix as described in :func:`sample_density`."""
    zmat = np.array(zmat, dtype=float, copy=True)
    if normalize == "max":
        mx = np.nanmax(zmat)
        if mx > 0:
            zmat = zmat / mx
    elif normalize == "sum":
        total = np.nansum(zmat)
        if total > 0:
            zmat = zmat / total
    elif normalize == "zscore":
        mu = np.nanmean(zmat)
        sd = np.nanstd(zmat.ravel(), ddof=1)
        if sd > 0:
            zmat = (zmat - mu) / sd
    return zmat


def _nearest_grid_index(coord, grid) -> np.ndarray:
    """Nearest lattice index (0-based), clamped to the grid; -1 for missing.

    Like R's ``round(approx(grid, seq_along(grid), coord, rule = 2)$y)``: an
    exact midpoint rounds half to even on the 1-based index scale (on the grid
    0, 25, 50, 75, 100 both 12.5 and 37.5 map to 25).
    """
    coord = np.asarray(coord, dtype=float)
    grid = np.asarray(grid, dtype=float)
    idx = np.full(coord.shape, -1, dtype=int)
    ok = np.isfinite(coord)
    if ok.any():
        # Round on R's 1-based index scale: half-to-even depends on the base
        # (1-based 2.5 -> 2, whereas 0-based 1.5 -> 2 would pick the next cell).
        pos = np.interp(coord[ok], grid, np.arange(1, len(grid) + 1, dtype=float))
        idx[ok] = np.clip(np.round(pos).astype(int), 1, len(grid)) - 1
    return idx


def _lookup_density(zmat, grid_x, grid_y, xs, ys) -> np.ndarray:
    ix = _nearest_grid_index(xs, grid_x)
    iy = _nearest_grid_index(ys, grid_y)
    out = np.full(ix.shape, np.nan)
    ok = (ix >= 0) & (iy >= 0)
    out[ok] = zmat[ix[ok], iy[ok]]
    return out


# ---------------------------------------------------------------------------
# sample_density_time
# ---------------------------------------------------------------------------

def sample_density_time(
    template_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str,
    times: np.ndarray = None,
    time_bins: np.ndarray | None = None,
    template_var: str = "density",
    source_var: str = "fixgroup",
    permutations: int = 0,
    permute_on: str | None = None,
    aggregate_fun: Callable = None,
    normalize: str = "none",
) -> pd.DataFrame:
    """Sample template density maps at source fixation locations over time."""
    if aggregate_fun is None:
        aggregate_fun = lambda vals, **kw: np.nanmean(vals)

    if times is None:
        times = np.arange(0, 3001, 50)
    times = np.asarray(times, dtype=float)

    # Validate
    if match_on not in template_tab.columns:
        raise ValueError(f"match_on column {match_on} not found in template_tab")
    if match_on not in source_tab.columns:
        raise ValueError(f"match_on column {match_on} not found in source_tab")
    if template_var not in template_tab.columns:
        raise ValueError(f"template_var column {template_var} not found in template_tab")
    if source_var not in source_tab.columns:
        raise ValueError(f"source_var column {source_var} not found in source_tab")

    if time_bins is not None:
        time_bins = np.asarray(time_bins, dtype=float)
        if len(time_bins) < 2:
            raise ValueError("time_bins must have at least 2 values to define bin boundaries")
        if not np.all(np.diff(time_bins) > 0):
            raise ValueError("time_bins must be monotonically increasing")

    source_tab = source_tab.copy().reset_index(drop=True)

    # Match
    matchind = match_keys(source_tab[match_on].values, template_tab[match_on].values)

    # Remove unmatched
    source_tab, matchind = filter_unmatched(
        source_tab, matchind,
        warn_msg=f"Did not find matching template for some source rows. Removing non-matching elements.",
    )

    # Setup permutation splits
    match_split = None
    if permutations > 0 and permute_on is not None:
        if permute_on not in source_tab.columns:
            raise ValueError(f"permute_on column {permute_on} not found in source_tab")
        if permute_on not in template_tab.columns:
            raise ValueError(f"permute_on column {permute_on} not found in template_tab")
        perm_groups = source_tab[permute_on].values
        match_split = {}
        for i, pg in enumerate(perm_groups):
            pg_str = str(pg)
            if pg_str not in match_split:
                match_split[pg_str] = []
            match_split[pg_str].append(matchind[i])
        # Deduplicate, keeping first-appearance order
        for k in match_split:
            match_split[k] = list(dict.fromkeys(match_split[k]))

    # Helper: bin aggregation
    def _aggregate_bins(sampled_df, time_bins, agg_fun):
        if sampled_df is None or len(sampled_df) == 0:
            return [np.nan] * (len(time_bins) - 1)
        # Every bin is half-open, [lower, upper), including the last one, so a
        # time equal to the final break falls outside all bins.
        bin_labels = np.digitize(sampled_df["time"].values, time_bins, right=False)
        results = []
        for b in range(1, len(time_bins)):
            vals = sampled_df["z"].values[bin_labels == b]
            vals = vals[~np.isnan(vals)] if len(vals) > 0 else vals
            if len(vals) == 0:
                results.append(np.nan)
            else:
                results.append(float(agg_fun(vals)))
        return results

    template_data = template_tab[template_var].values
    source_data = source_tab[source_var].values

    all_results = []
    for i, mi in enumerate(matchind):
        template_dens = template_data[mi]
        source_fix = source_data[i]

        result = {}

        if template_dens is None or source_fix is None:
            sampled = pd.DataFrame({"z": np.full(len(times), np.nan), "time": times})
        else:
            try:
                sampled = sample_density(template_dens, source_fix, times=times,
                                         normalize=normalize)
            except Exception as e:
                warnings.warn(f"Error sampling density for row {i}: {e}")
                sampled = pd.DataFrame({"z": np.full(len(times), np.nan), "time": times})

        result["sampled"] = sampled

        # Bins
        if time_bins is not None:
            bin_vals = _aggregate_bins(sampled, time_bins, aggregate_fun)
            for b, val in enumerate(bin_vals, 1):
                result[f"bin_{b}"] = val

        # Permutations
        if permutations > 0:
            if permute_on is not None:
                perm_key = str(source_tab[permute_on].iloc[i])
                mind = list(match_split.get(perm_key, []))
            else:
                mind = list(matchind)

            # Remove every copy of the current match and keep each other
            # template once, as in template_similarity()
            current_match = mi
            mind = list(dict.fromkeys(m for m in mind if m != current_match))

            if len(mind) == 0:
                perm_sampled = pd.DataFrame({"z": np.full(len(times), np.nan), "time": times})
                if time_bins is not None:
                    for b in range(1, len(time_bins)):
                        result[f"perm_bin_{b}"] = np.nan
            else:
                if permutations < len(mind):
                    mind = list(np.random.choice(mind, permutations, replace=False))

                perm_zs = []
                for j in mind:
                    perm_dens = template_data[j]
                    if perm_dens is None or source_fix is None:
                        perm_zs.append(np.full(len(times), np.nan))
                    else:
                        try:
                            ps = sample_density(perm_dens, source_fix, times=times,
                                                normalize=normalize)
                            perm_zs.append(ps["z"].values)
                        except Exception:
                            perm_zs.append(np.full(len(times), np.nan))

                perm_matrix = np.array(perm_zs)
                perm_mean = np.nanmean(perm_matrix, axis=0)
                perm_sampled = pd.DataFrame({"z": perm_mean, "time": times})

                if time_bins is not None:
                    perm_bin_vals = _aggregate_bins(perm_sampled, time_bins, aggregate_fun)
                    for b, val in enumerate(perm_bin_vals, 1):
                        result[f"perm_bin_{b}"] = val

            result["perm_sampled"] = perm_sampled

        all_results.append(result)

    # Assemble output
    out = source_tab.copy()
    out["sampled"] = [r["sampled"] for r in all_results]

    if time_bins is not None:
        n_bins = len(time_bins) - 1
        for b in range(1, n_bins + 1):
            out[f"bin_{b}"] = [r.get(f"bin_{b}", np.nan) for r in all_results]

    if permutations > 0:
        out["perm_sampled"] = [r.get("perm_sampled") for r in all_results]
        if time_bins is not None:
            n_bins = len(time_bins) - 1
            for b in range(1, n_bins + 1):
                out[f"perm_bin_{b}"] = [r.get(f"perm_bin_{b}", np.nan) for r in all_results]
            for b in range(1, n_bins + 1):
                out[f"diff_bin_{b}"] = out[f"bin_{b}"] - out[f"perm_bin_{b}"]

    return out


def _fast_cosine_vector(obj):
    if isinstance(obj, EyeDensity):
        return np.asarray(obj.z, dtype=float).ravel(order="F")
    if obj is None or isinstance(obj, (EyeDensityMultiscale, FixationGroup, str, bytes, dict)):
        return None
    try:
        arr = np.asarray(obj)
    except Exception:
        return None
    if arr.dtype.kind not in "biuf":
        return None
    return arr.astype(float).ravel(order="F")


def _fast_cosine_eligible(objs) -> bool:
    """When R ``template_similarity(method = "cosine")`` takes its vectorised path.

    R uses it when every reference and source map is a density or numeric
    vector of one length with no missing cell and all densities share a
    lattice; that path scores every pair, even a single-cell one.
    """
    vecs = [_fast_cosine_vector(o) for o in objs]
    if any(v is None for v in vecs) or len({v.size for v in vecs}) != 1:
        return False
    if any(np.isnan(v).any() for v in vecs):
        return False
    dens = [o for o in objs if isinstance(o, EyeDensity)]
    return all(_density_lattices_equal(dens[0], d) for d in dens[1:])


def _fast_cosine(src, ref) -> float:
    """R ``cosine_similarity_matrix()`` for one pair (source row, reference row).

    A product of norms at or below machine epsilon scores 0, and two zero
    vectors score 1. (R intends 1 for every pair of zero maps but, because
    ``outer()`` of two logical vectors is numeric, only sets element [1, 1].)
    """
    x = _fast_cosine_vector(src)
    y = _fast_cosine_vector(ref)
    eps = np.finfo(float).eps
    nx, ny = np.sqrt(np.sum(x ** 2)), np.sqrt(np.sum(y ** 2))
    if nx <= eps and ny <= eps:
        return 1.0
    denom = nx * ny
    if denom <= eps:
        return 0.0
    return float(np.dot(x, y) / denom)


def _stratum_key(value):
    """Key of a ``permute_on`` stratum; ``None`` for a missing value (R ``split()`` drops NA)."""
    try:
        if value is None or (np.ndim(value) == 0 and pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _is_valid_similarity_obj(obj) -> bool:
    """R ``is_valid_similarity_obj()``: a density, fixation group or numeric input."""
    if obj is None:
        return False
    if isinstance(obj, (EyeDensity, EyeDensityMultiscale, FixationGroup)):
        return True
    if isinstance(obj, (str, bytes, dict)):
        return False
    if np.ndim(obj) == 0:
        try:
            return not pd.isna(obj) and np.isreal(obj)
        except (TypeError, ValueError):
            return False
    try:
        return np.asarray(obj).dtype.kind in "biuf"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# run_similarity_analysis (internal)
# ---------------------------------------------------------------------------

def _run_similarity_analysis(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str,
    permutations: int,
    permute_on: str | None,
    method: str,
    refvar: str,
    sourcevar: str,
    window=None,
    multiscale_aggregation: str = "mean",
    rng: np.random.Generator | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Core similarity analysis loop (mirrors R ``run_similarity_analysis``).

    Permutation candidates for a row are the distinct reference rows matched by
    at least one source row (within the row's ``permute_on`` stratum, if
    given), with every copy of the row's own match removed before sampling.
    ``rng`` draws the subsets; ``None`` uses NumPy's global random state.
    """

    def mean_or_nan(values) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.nan
        return float(np.nanmean(arr))

    source_tab = source_tab.copy().reset_index(drop=True)
    matchind = match_keys(source_tab[match_on].values, ref_tab[match_on].values)
    source_tab, matchind = filter_unmatched(
        source_tab, matchind,
        warn_msg="did not find matching template map for all source maps. Removing non-matching elements.",
    )

    # Setup permutation splits. As R's split(), rows whose stratum is missing
    # belong to no stratum and so have no permutation candidates.
    match_split = None
    if permute_on is not None:
        if permute_on not in source_tab.columns or permute_on not in ref_tab.columns:
            raise ValueError(f"permute_on column {permute_on} must exist in ref_tab and source_tab")
    if permute_on is not None and permutations > 0:
        match_split = {}
        for i, mi in enumerate(matchind):
            key = _stratum_key(source_tab[permute_on].iloc[i])
            if key is None:
                continue
            match_split.setdefault(key, []).append(mi)

    eye_sim_list = []
    perm_sim_list = []
    diff_list = []
    n_perm_list = []

    fast_cosine = (
        method == "cosine" and window is None and not kwargs
        and multiscale_aggregation == "mean" and len(source_tab) > 0
        and _fast_cosine_eligible(list(ref_tab[refvar].values) + list(source_tab[sourcevar].values))
    )

    def _sample(candidates, size):
        if rng is None:
            return list(np.random.choice(candidates, size, replace=False))
        return list(rng.choice(candidates, size, replace=False))

    ref_data = ref_tab[refvar].values
    src_data = source_tab[sourcevar].values

    # Detect if method returns dict (e.g. multimatch)
    is_dict_method = method == "multimatch"

    def _common_sigmas(x, y):
        """The sigmas similarity(x, y) reports per scale, in its order (or None)."""
        if not (isinstance(x, EyeDensityMultiscale) and isinstance(y, EyeDensityMultiscale)):
            return None
        sigmas_y = {sc.sigma for sc in y}
        return list(dict.fromkeys(sc.sigma for sc in x if sc.sigma in sigmas_y))

    def _align_to_sigmas(ps, ps_sigmas, sigmas):
        """Reorder one permutation's per-scale scores to ``sigmas``; NaN where absent.

        R averages per-scale permutation scores by position, which mixes scales
        when templates list their sigmas in different orders.
        """
        arr = np.atleast_1d(np.asarray(ps, dtype=float)).ravel()
        if not ps_sigmas or arr.size != len(ps_sigmas):
            return np.full(len(sigmas), np.nan)
        by_sigma = dict(zip(ps_sigmas, arr))
        return np.array([by_sigma.get(sg, np.nan) for sg in sigmas], dtype=float)

    def _perm_mean_vector(sim, psims):
        """R colMeans(rbind(psims), na.rm = TRUE) for per-scale vectors."""
        width = np.size(sim)
        rows = []
        for ps in psims:
            arr = np.atleast_1d(np.asarray(ps, dtype=float)).ravel()
            if arr.size == 1 and width != 1:
                arr = np.repeat(arr, width)
            rows.append(arr)
        mat = np.vstack(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(mat, axis=0)

    for i, mi in enumerate(matchind):
        d1 = ref_data[mi]
        d2 = src_data[i]

        if not _is_valid_similarity_obj(d1) or not _is_valid_similarity_obj(d2):
            warnings.warn(
                "Invalid or NULL similarity input encountered in run_similarity_analysis(). "
                "Returning NA for this comparison."
            )
            eye_sim_list.append(np.nan if not is_dict_method else None)
            if permutations > 0:
                perm_sim_list.append(np.nan if not is_dict_method else None)
                diff_list.append(np.nan if not is_dict_method else None)
                n_perm_list.append(0)
            continue

        if fast_cosine:
            sim = _fast_cosine(d2, d1)
        else:
            sim = similarity(d1, d2, method=method,
                             window=window,
                             multiscale_aggregation=multiscale_aggregation, **kwargs)
        is_vector = (not is_dict_method) and np.ndim(sim) > 0
        true_sigmas = _common_sigmas(d1, d2) if is_vector else None
        if true_sigmas is not None and len(true_sigmas) != np.size(sim):
            true_sigmas = None

        if permutations > 0:
            if permute_on is not None:
                pkey = _stratum_key(source_tab[permute_on].iloc[i])
                mind = list(match_split.get(pkey, [])) if pkey is not None else []
            else:
                mind = list(matchind)

            # Remove every copy of the true match first (several source rows
            # can share a key), keep each other template once, then sample
            # from the distinct non-matching candidates only.
            mind = list(dict.fromkeys(m for m in mind if m != mi))
            if permutations < len(mind):
                mind = _sample(mind, permutations)

            if len(mind) == 0:
                warnings.warn("no matching candidate indices for permutation test. Skipping.")
                eye_sim_list.append(sim)
                if is_vector:
                    nan_vec = np.full(np.shape(sim), np.nan)
                    perm_sim_list.append(nan_vec)
                    diff_list.append(nan_vec.copy())
                else:
                    perm_sim_list.append(np.nan if not is_dict_method else None)
                    diff_list.append(np.nan if not is_dict_method else None)
                n_perm_list.append(0)
                continue

            psims = []
            for j in mind:
                d1p = ref_data[j]
                if fast_cosine:
                    psims.append(_fast_cosine(d2, d1p))
                elif not _is_valid_similarity_obj(d1p):
                    psims.append(np.nan if not is_dict_method else None)
                else:
                    ps = similarity(d1p, d2, method=method,
                                    window=window,
                                    multiscale_aggregation=multiscale_aggregation, **kwargs)
                    if true_sigmas is not None:
                        ps = _align_to_sigmas(ps, _common_sigmas(d1p, d2), true_sigmas)
                    psims.append(ps)

            if is_dict_method:
                # Average each metric across permutations
                valid_psims = [p for p in psims if isinstance(p, dict)]
                if valid_psims:
                    perm_mean = {k: mean_or_nan([p[k] for p in valid_psims]) for k in valid_psims[0]}
                else:
                    perm_mean = None
            elif is_vector:
                perm_mean = _perm_mean_vector(sim, psims)
            else:
                perm_mean = mean_or_nan([np.nanmean(p) if np.ndim(p) > 0 else p for p in psims])

            eye_sim_list.append(sim)
            perm_sim_list.append(perm_mean)
            n_perm_list.append(len(mind))

            if is_dict_method:
                if isinstance(sim, dict) and isinstance(perm_mean, dict):
                    diff_list.append({k: sim[k] - perm_mean[k] for k in sim})
                else:
                    diff_list.append(None)
            else:
                diff_list.append(np.asarray(sim, dtype=float) - perm_mean if is_vector
                                 else sim - perm_mean)
        else:
            eye_sim_list.append(sim)

    result = source_tab.copy()

    if is_dict_method:
        # Expand dict results into separate columns (e.g. mm_vector, mm_direction, ...)
        mm_keys = None
        for s in eye_sim_list:
            if isinstance(s, dict):
                mm_keys = list(s.keys())
                break
        if mm_keys:
            for k in mm_keys:
                result[k] = [s[k] if isinstance(s, dict) else np.nan for s in eye_sim_list]
            if permutations > 0:
                for k in mm_keys:
                    result[f"{k}_perm"] = [
                        s[k] if isinstance(s, dict) else np.nan for s in perm_sim_list
                    ]
                    result[f"{k}_diff"] = [
                        s[k] if isinstance(s, dict) else np.nan for s in diff_list
                    ]
                result["n_perm"] = n_perm_list
    else:
        result["eye_sim"] = eye_sim_list
        if permutations > 0:
            result["perm_sim"] = perm_sim_list
            result["eye_sim_diff"] = diff_list
            result["n_perm"] = n_perm_list

    return result


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

def template_similarity(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str,
    permute_on: str | None = None,
    refvar: str = "density",
    sourcevar: str = "density",
    method: str = "spearman",
    permutations: int = 10,
    multiscale_aggregation: str = "mean",
    similarity_transform=None,
    similarity_transform_args: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Template-based similarity analysis (mirrors R ``template_similarity``)."""
    method = _match_choice(method, _DENSITY_METHODS, "method")
    if similarity_transform is not None:
        from peyesim.latent_transforms import _resolve_similarity_transform
        similarity_transform = _resolve_similarity_transform(similarity_transform)
        if similarity_transform_args is None:
            similarity_transform_args = {}
        transform_res = similarity_transform(
            ref_tab=ref_tab, source_tab=source_tab, match_on=match_on,
            refvar=refvar, sourcevar=sourcevar, **similarity_transform_args
        )
        if transform_res.get("ref_tab") is not None:
            ref_tab = transform_res["ref_tab"]
        if transform_res.get("source_tab") is not None:
            source_tab = transform_res["source_tab"]
        if transform_res.get("refvar") is not None:
            refvar = transform_res["refvar"]
        if transform_res.get("sourcevar") is not None:
            sourcevar = transform_res["sourcevar"]

    return _run_similarity_analysis(
        ref_tab, source_tab, match_on, permutations, permute_on,
        method, refvar, sourcevar,
        multiscale_aggregation=multiscale_aggregation, **kwargs
    )


def _check_window(window) -> None:
    if window is not None and not (len(window) >= 2 and window[1] > window[0]):
        raise ValueError("window[1] must be greater than window[0]")


def fixation_similarity(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str,
    permutations: int = 0,
    permute_on: str | None = None,
    method: str = "sinkhorn",
    refvar: str = "fixgroup",
    sourcevar: str = "fixgroup",
    window=None,
    **kwargs,
) -> pd.DataFrame:
    """Compute fixation-level similarity (mirrors R ``fixation_similarity``)."""
    _check_window(window)
    method = _match_choice(method, ("sinkhorn", "overlap"), "method")
    return _run_similarity_analysis(
        ref_tab, source_tab, match_on, permutations, permute_on,
        method, refvar, sourcevar, window=window, **kwargs
    )


def scanpath_similarity(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str,
    permutations: int = 0,
    permute_on: str | None = None,
    method: str = "multimatch",
    refvar: str = "scanpath",
    sourcevar: str = "scanpath",
    window=None,
    **kwargs,
) -> pd.DataFrame:
    """Compute scanpath similarity (mirrors R ``scanpath_similarity``).

    ``screensize`` (width, height) must be passed through ``**kwargs``, as in R.
    """
    _check_window(window)
    return _run_similarity_analysis(
        ref_tab, source_tab, match_on, permutations, permute_on,
        method, refvar, sourcevar, window=window, **kwargs
    )


# ---------------------------------------------------------------------------
# Cross-validated template similarity
# ---------------------------------------------------------------------------

def _format_split_value(value) -> str:
    """Format split keys like R's interaction/as.character for common scalars."""
    if pd.isna(value):
        return "NA"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


@lru_cache(maxsize=128)
def _r_sample_order(n: int, seed: int) -> tuple[int, ...] | None:
    """Return zero-based ``sample(seq_len(n), n)`` order from R when available."""
    if n < 1 or shutil.which("Rscript") is None:
        return None
    code = (
        "args <- commandArgs(TRUE); "
        "n <- as.integer(args[[1]]); seed <- as.integer(args[[2]]); "
        "set.seed(seed); cat(sample(seq_len(n), n), sep='\\n')"
    )
    proc = subprocess.run(
        ["Rscript", "-e", code, str(int(n)), str(int(seed))],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return None
    try:
        values = tuple(int(line) - 1 for line in proc.stdout.splitlines() if line.strip())
    except ValueError:
        return None
    if len(values) != n or sorted(values) != list(range(n)):
        return None
    return values


def _make_cv_folds(
    source_tab: pd.DataFrame,
    split_on: str | list[str],
    n_folds: int | None = None,
    seed: int = 1,
) -> dict:
    """Assign CV fold IDs based on unique groups in *split_on* columns.

    Returns
    -------
    dict
        ``fold_ids`` – integer array (same length as *source_tab*) with fold
        assignments (1-based, matching R), and ``n_folds``.
    """
    if isinstance(split_on, str):
        split_on = [split_on]

    # Build a group key per row. R uses interaction(..., lex.order=TRUE,
    # sep="::"), coerces to character, sorts levels, then samples groups.
    if len(split_on) == 0 or not all(col in source_tab.columns for col in split_on):
        raise ValueError("split_on must name one or more columns in source_tab.")
    # A missing value in any split column gives R an NA key: such rows join no
    # group and no fold, so they are neither scored nor used for fitting.
    missing = source_tab[split_on].isna().any(axis=1).to_numpy()
    group_keys = source_tab[split_on].apply(
        lambda row: "::".join(_format_split_value(value) for value in row),
        axis=1,
    )
    unique_groups = sorted(group_keys[~missing].unique())

    if len(unique_groups) < 2:
        raise ValueError(
            f"Need at least 2 unique groups in split_on columns to create "
            f"CV folds, got {len(unique_groups)}."
        )

    if n_folds is None:
        n_folds = min(5, len(unique_groups))
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    n_folds = min(int(n_folds), len(unique_groups))

    # Prefer R's exact sample() order when Rscript is installed; fall back to a
    # deterministic NumPy shuffle so the Python package remains standalone.
    order = _r_sample_order(len(unique_groups), int(seed))
    if order is None:
        rng = np.random.default_rng(seed)
        order = tuple(int(i) for i in rng.permutation(len(unique_groups)))
    group_to_fold = {}
    for rank, idx in enumerate(order):
        group_to_fold[unique_groups[idx]] = (rank % n_folds) + 1

    fold_ids = np.array([0 if miss else group_to_fold[g] for g, miss in zip(group_keys, missing)],
                        dtype=int)
    return {"fold_ids": fold_ids, "n_folds": n_folds}


def template_similarity_cv(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str,
    permute_on: str | None = None,
    refvar: str = "density",
    sourcevar: str = "density",
    method: str = "spearman",
    permutations: int = 10,
    multiscale_aggregation: str = "mean",
    similarity_transform=None,
    similarity_transform_args: dict | None = None,
    split_on: str | list[str] | None = None,
    n_folds: int | None = None,
    seed: int = 1,
    fit_source_filter=None,
    eval_source_filter=None,
    **kwargs,
) -> pd.DataFrame:
    """Cross-validated template similarity (mirrors R ``template_similarity_cv``).

    When a *similarity_transform* is used (e.g. CORAL, CCA, PCA), the
    transform should not be fit and evaluated on the same data.  This
    function splits *source_tab* into folds, fits the transform on the
    training folds, and evaluates similarity on the held-out fold.

    Parameters
    ----------
    ref_tab, source_tab, match_on, permute_on, refvar, sourcevar, method,
    permutations, multiscale_aggregation, similarity_transform,
    similarity_transform_args
        Same as :func:`template_similarity`.
    split_on : str or list[str], optional
        Column(s) used to define CV groups.  Defaults to *match_on*.
    n_folds : int, optional
        Number of CV folds (default ``min(5, n_groups)``).
    seed : int
        Random seed for fold assignment and for the permutation draws that
        follow it, so results are reproducible from ``seed`` alone. NumPy's
        global random state is neither used nor changed.
    fit_source_filter : callable or array-like of bool, optional
        Boolean mask or callable ``(source_tab) -> bool array`` selecting
        rows eligible for the **training** set of the transform fit.
    eval_source_filter : callable or array-like of bool, optional
        Boolean mask or callable ``(source_tab) -> bool array`` selecting
        rows eligible for **evaluation**.

    Returns
    -------
    pd.DataFrame
        Concatenated per-fold results with an extra ``".cv_fold"`` column.

    Notes
    -----
    The transform is fit on training folds only and applied to held-out
    data, preventing data leakage. Supported transforms:
    ``latent_pca_transform``, ``coral_transform``, ``cca_transform``,
    ``contract_transform``, ``affine_transform``.
    """
    if similarity_transform_args is None:
        similarity_transform_args = {}
    method = _match_choice(method, _DENSITY_METHODS, "method")
    if similarity_transform is not None:
        from peyesim.latent_transforms import _resolve_similarity_transform
        similarity_transform = _resolve_similarity_transform(similarity_transform)

    if split_on is None:
        split_on = match_on

    # Add row IDs for reassembly
    source_tab = source_tab.copy()
    source_tab["__cv_row_id__"] = np.arange(len(source_tab))

    # Resolve filters
    if fit_source_filter is not None:
        if callable(fit_source_filter):
            fit_mask = np.asarray(fit_source_filter(source_tab), dtype=bool)
        else:
            fit_mask = np.asarray(fit_source_filter, dtype=bool)
    else:
        fit_mask = np.ones(len(source_tab), dtype=bool)

    if eval_source_filter is not None:
        if callable(eval_source_filter):
            eval_mask = np.asarray(eval_source_filter(source_tab), dtype=bool)
        else:
            eval_mask = np.asarray(eval_source_filter, dtype=bool)
    else:
        eval_mask = np.ones(len(source_tab), dtype=bool)
    for label, mask in (("fit_source_filter", fit_mask), ("eval_source_filter", eval_mask)):
        if mask.shape != (len(source_tab),):
            raise ValueError(
                f"{label} must be None, a boolean vector of length len(source_tab), "
                "or a function returning one."
            )
    if not eval_mask.any():
        raise ValueError("eval_source_filter did not select any source rows.")

    # Build folds
    cv = _make_cv_folds(source_tab, split_on, n_folds=n_folds, seed=seed)
    fold_ids = cv["fold_ids"]
    n_folds_actual = cv["n_folds"]

    # Permutation controls are drawn only from the reference rows matched by
    # the same held-out fold, with a generator seeded from ``seed``.
    perm_rng = np.random.default_rng(seed)
    fold_results = []

    for fold in range(1, n_folds_actual + 1):
        in_fold = fold_ids == fold

        # Eval rows: in this fold AND pass eval_mask
        eval_rows = in_fold & eval_mask
        if not eval_rows.any():
            continue

        # Training candidate rows: NOT in this fold AND pass fit_mask
        train_rows = (~in_fold) & (fold_ids > 0) & fit_mask

        # Remove from training any rows whose match_on key appears in eval
        # set to prevent data leakage
        if isinstance(match_on, str):
            eval_keys = set(source_tab.loc[eval_rows, match_on].unique())
            leaky = source_tab[match_on].isin(eval_keys)
        else:
            eval_key_tuples = set(
                source_tab.loc[eval_rows, match_on]
                .apply(tuple, axis=1)
                .unique()
            )
            leaky = (
                source_tab[match_on]
                .apply(tuple, axis=1)
                .isin(eval_key_tuples)
            )
        train_rows = train_rows & ~leaky

        eval_source = source_tab.loc[eval_rows].reset_index(drop=True)
        train_source = source_tab.loc[train_rows].reset_index(drop=True)

        # Determine ref subsets
        if isinstance(match_on, str):
            eval_keys = set(source_tab.loc[eval_rows, match_on].unique())
            train_keys = set(source_tab.loc[train_rows, match_on].unique())
        else:
            eval_keys = set(
                source_tab.loc[eval_rows, match_on].apply(tuple, axis=1).unique()
            )
            train_keys = set(
                source_tab.loc[train_rows, match_on].apply(tuple, axis=1).unique()
            )

        ref_eval = ref_tab[ref_tab[match_on].isin(eval_keys)].reset_index(drop=True)

        # Apply transform if provided
        fold_refvar = refvar
        fold_sourcevar = sourcevar
        if similarity_transform is not None:
            from peyesim.latent_transforms import _fit_transform, _apply_transform
            ref_train = ref_tab[ref_tab[match_on].isin(train_keys)].reset_index(drop=True)
            if len(train_source) == 0 or len(ref_train) == 0:
                raise ValueError(
                    f"Fold {fold} has no training rows for transform fitting. "
                    "Reduce n_folds or relax the fit filter."
                )
            model = _fit_transform(
                similarity_transform, ref_train, train_source,
                match_on, refvar=refvar, sourcevar=sourcevar,
                **similarity_transform_args,
            )
            transformed = _apply_transform(
                model, ref_eval, eval_source,
                refvar=refvar, sourcevar=sourcevar,
            )
            ref_eval = transformed["ref_tab"]
            eval_source = transformed["source_tab"]
            fold_refvar = transformed.get("refvar", refvar)
            fold_sourcevar = transformed.get("sourcevar", sourcevar)

        fold_res = _run_similarity_analysis(
            ref_eval,
            eval_source,
            match_on,
            permutations,
            permute_on,
            method,
            fold_refvar,
            fold_sourcevar,
            multiscale_aggregation=multiscale_aggregation,
            rng=perm_rng,
            **kwargs,
        )
        fold_res[".cv_fold"] = fold
        fold_results.append(fold_res)

    if not fold_results:
        raise ValueError("No folds produced any evaluation rows.")

    result = pd.concat(fold_results, ignore_index=True)

    # Sort by original row order and drop helper column
    result = result.sort_values("__cv_row_id__").reset_index(drop=True)
    result = result.drop(columns=["__cv_row_id__"])

    return result
