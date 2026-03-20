"""Similarity metrics and template-based analysis."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as _cosine_dist
from scipy.stats import spearmanr

from peyesim._utils import emdw as _emdw, match_keys, filter_unmatched
from peyesim.density import EyeDensity, EyeDensityMultiscale
from peyesim.fixations import FixationGroup


# ---------------------------------------------------------------------------
# Core compute_similarity
# ---------------------------------------------------------------------------

def compute_similarity(
    x,
    y,
    method: str = "spearman",
    saliency_map=None,
) -> float:
    """Compute similarity between two vectors / density objects.

    Supported methods: pearson, spearman, fisherz, cosine, l1, jaccard, dcov, emd.
    """
    method = method.lower()

    # EMD path – needs full density objects
    if method == "emd":
        if isinstance(x, EyeDensity) and isinstance(y, EyeDensity):
            coords = np.array(np.meshgrid(x.x, x.y, indexing="ij")).reshape(2, -1).T
            wx = x.z.ravel()
            wy = y.z.ravel()
            if saliency_map is not None:
                s_mat = saliency_map.z if isinstance(saliency_map, EyeDensity) else np.asarray(saliency_map)
                r1 = x.z - s_mat
                r2 = y.z - s_mat
                pos1 = np.maximum(r1, 0).ravel()
                neg1 = np.maximum(-r1, 0).ravel()
                pos2 = np.maximum(r2, 0).ravel()
                neg2 = np.maximum(-r2, 0).ravel()
                emd_pos = _emdw(coords, pos1, coords, pos2)
                emd_neg = _emdw(coords, neg1, coords, neg2)
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
            if zv_x and zv_y and np.allclose(vx, vy):
                return 1.0
            warnings.warn(
                f"Method {method} requires variance in both inputs. "
                "One or both have near-zero variance."
            )
            return np.nan

    if method == "pearson":
        return float(np.corrcoef(vx, vy)[0, 1])
    elif method == "spearman":
        return float(spearmanr(vx, vy).statistic)
    elif method == "fisherz":
        r = float(np.corrcoef(vx, vy)[0, 1])
        r = np.clip(r, -1 + eps, 1 - eps)
        return float(np.arctanh(r))
    elif method == "cosine":
        d = _cosine_dist(vx, vy)
        return 1.0 - d
    elif method == "l1":
        sx = vx.sum()
        sy = vy.sum()
        if sx <= eps or sy <= eps:
            warnings.warn("Cannot normalize for L1 distance; sum is too small.")
            return np.nan
        x1 = vx / sx
        x2 = vy / sy
        return 1.0 - 0.5 * np.abs(x1 - x2).sum()
    elif method == "jaccard":
        num = np.minimum(vx, vy).sum()
        den = np.maximum(vx, vy).sum()
        if den <= eps:
            return np.nan
        return num / den
    elif method == "dcov":
        from scipy.spatial.distance import pdist, squareform
        sx = vx.sum()
        sy = vy.sum()
        if sx <= eps or sy <= eps:
            warnings.warn("Cannot normalize for dcov; sum is too small.")
            return np.nan
        x1 = vx / sx
        x2 = vy / sy
        # Simple distance correlation implementation
        n = len(x1)
        a = squareform(pdist(x1.reshape(-1, 1)))
        b = squareform(pdist(x2.reshape(-1, 1)))
        A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
        B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
        dcov_xy = np.sqrt(np.maximum((A * B).mean(), 0))
        dcov_xx = np.sqrt(np.maximum((A * A).mean(), 0))
        dcov_yy = np.sqrt(np.maximum((B * B).mean(), 0))
        if dcov_xx * dcov_yy <= 0:
            return 0.0
        return dcov_xy / np.sqrt(dcov_xx * dcov_yy)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


# ---------------------------------------------------------------------------
# Dispatching similarity function
# ---------------------------------------------------------------------------

def similarity(x, y, method: str = "spearman", **kwargs) -> float | np.ndarray:
    """Compute similarity, dispatching on type (mirrors R S3 ``similarity``)."""
    multiscale_aggregation = kwargs.pop("multiscale_aggregation", "mean")
    saliency_map = kwargs.pop("saliency_map", None)

    # Multiscale
    if isinstance(x, EyeDensityMultiscale):
        if not isinstance(y, EyeDensityMultiscale):
            raise TypeError("y must also be EyeDensityMultiscale")
        if len(x) == 0 or len(y) == 0:
            warnings.warn("One or both multiscale objects are empty.")
            return np.nan

        sigmas_x = [s.sigma for s in x]
        sigmas_y = [s.sigma for s in y]
        common = sorted(set(sigmas_x) & set(sigmas_y))
        if not common:
            warnings.warn("No common sigmas found between multiscale objects.")
            return np.nan

        sx_map = {s.sigma: s for s in x}
        sy_map = {s.sigma: s for s in y}
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

        if multiscale_aggregation == "mean":
            return float(np.nanmean(per_scale))
        else:
            return per_scale

    # EyeDensity
    if isinstance(x, EyeDensity):
        return compute_similarity(x, y, method=method, saliency_map=saliency_map)

    # fixation_group → not supported as direct similarity here (use fixation_similarity)
    # Default: numeric vectors
    return compute_similarity(x, y, method=method, saliency_map=saliency_map)


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

    def nearest_index(coord, grid):
        idx = np.interp(coord, grid, np.arange(len(grid)))
        idx = np.clip(np.round(idx).astype(int), 0, len(grid) - 1)
        return idx

    zmat = dens.z.copy()
    if normalize == "max":
        mx = zmat.max()
        if mx > 0:
            zmat = zmat / mx
    elif normalize == "sum":
        s = zmat.sum()
        if s > 0:
            zmat = zmat / s
    elif normalize == "zscore":
        mu = zmat.mean()
        sd = zmat.std(ddof=0)
        if sd > 0:
            zmat = (zmat - mu) / sd

    if times is None:
        xs = fix["x"].values
        ys = fix["y"].values
        ix = nearest_index(xs, dens.x)
        iy = nearest_index(ys, dens.y)
        z_vals = zmat[ix, iy]
        return pd.DataFrame({"z": z_vals, "time": fix["onset"].values})
    else:
        times = np.asarray(times, dtype=float)
        fg_sampled = fix.sample_fixations(times)
        xs = fg_sampled["x"].values
        ys = fg_sampled["y"].values
        ix = nearest_index(xs, dens.x)
        iy = nearest_index(ys, dens.y)
        z_vals = zmat[ix, iy]
        return pd.DataFrame({"z": z_vals, "time": times})


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
        # Deduplicate
        for k in match_split:
            match_split[k] = list(set(match_split[k]))

    # Helper: bin aggregation
    def _aggregate_bins(sampled_df, time_bins, agg_fun):
        if sampled_df is None or len(sampled_df) == 0:
            return [np.nan] * (len(time_bins) - 1)
        bin_labels = np.digitize(sampled_df["time"].values, time_bins, right=False)
        # digitize: bin 1 means [time_bins[0], time_bins[1]), etc.
        results = []
        for b in range(1, len(time_bins)):
            vals = sampled_df["z"].values[bin_labels == b]
            # Also include values at the last boundary if include.lowest
            if b == len(time_bins) - 1:
                vals_last = sampled_df["z"].values[
                    sampled_df["time"].values == time_bins[-1]
                ]
                vals = np.concatenate([vals, vals_last])
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
                mind = list(set(matchind))

            current_match = mi
            mind = [m for m in mind if m != current_match]

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
    **kwargs,
) -> pd.DataFrame:
    """Core similarity analysis loop (mirrors R ``run_similarity_analysis``)."""
    source_tab = source_tab.copy().reset_index(drop=True)
    matchind = match_keys(source_tab[match_on].values, ref_tab[match_on].values)
    source_tab, matchind = filter_unmatched(
        source_tab, matchind,
        warn_msg="did not find matching template map for all source maps. Removing non-matching elements.",
    )

    # Setup permutation splits
    match_split = None
    if permute_on is not None and permutations > 0:
        match_split = {}
        for i, mi in enumerate(matchind):
            key = str(source_tab[permute_on].iloc[i])
            if key not in match_split:
                match_split[key] = []
            match_split[key].append(mi)

    eye_sim_list = []
    perm_sim_list = []
    diff_list = []

    ref_data = ref_tab[refvar].values
    src_data = source_tab[sourcevar].values

    for i, mi in enumerate(matchind):
        d1 = ref_data[mi]
        d2 = src_data[i]

        if d1 is None or d2 is None:
            eye_sim_list.append(np.nan)
            if permutations > 0:
                perm_sim_list.append(np.nan)
                diff_list.append(np.nan)
            continue

        sim = similarity(d1, d2, method=method,
                         multiscale_aggregation=multiscale_aggregation, **kwargs)

        if permutations > 0:
            if permute_on is not None:
                pkey = str(source_tab[permute_on].iloc[i])
                mind = list(match_split.get(pkey, []))
            else:
                mind = list(set(matchind))

            # Remove current element
            mind = [m for m in mind if m != mi]

            if permutations < len(mind):
                mind = list(np.random.choice(mind, permutations, replace=False))

            if len(mind) == 0:
                warnings.warn("no matching candidate indices for permutation test. Skipping.")
                eye_sim_list.append(sim if np.isscalar(sim) else sim)
                perm_sim_list.append(np.nan)
                diff_list.append(np.nan)
                continue

            psims = []
            for j in mind:
                d1p = ref_data[j]
                if d1p is None:
                    psims.append(np.nan)
                else:
                    ps = similarity(d1p, d2, method=method,
                                    multiscale_aggregation=multiscale_aggregation, **kwargs)
                    psims.append(ps)

            perm_mean = float(np.nanmean(psims))

            if np.isscalar(sim):
                eye_sim_list.append(sim)
                perm_sim_list.append(perm_mean)
                diff_list.append(sim - perm_mean)
            else:
                # Vector result (multiscale with aggregation='none')
                eye_sim_list.append(sim)
                perm_sim_list.append(perm_mean)
                diff_list.append(np.nanmean(sim) - perm_mean if not np.isnan(perm_mean) else np.nan)
        else:
            if np.isscalar(sim):
                eye_sim_list.append(sim)
            else:
                eye_sim_list.append(sim)

    result = source_tab.copy()
    result["eye_sim"] = eye_sim_list
    if permutations > 0:
        result["perm_sim"] = perm_sim_list
        result["eye_sim_diff"] = diff_list

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
    if similarity_transform is not None:
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
    """Compute scanpath similarity (mirrors R ``scanpath_similarity``)."""
    return _run_similarity_analysis(
        ref_tab, source_tab, match_on, permutations, permute_on,
        method, refvar, sourcevar, window=window, **kwargs
    )
