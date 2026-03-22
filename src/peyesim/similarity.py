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
from peyesim.saccades import Scanpath


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

    # Scanpath (must check before FixationGroup since Scanpath inherits from it)
    if isinstance(x, Scanpath) and isinstance(y, Scanpath):
        if method == "multimatch":
            from peyesim.multimatch import multi_match
            screensize = kwargs.pop("screensize", (1024, 768))
            return multi_match(x, y, screensize=screensize)
        # Fall through to fixation-level methods

    # FixationGroup
    if isinstance(x, FixationGroup) and isinstance(y, FixationGroup):
        if method == "overlap":
            from peyesim.overlap import fixation_overlap
            time_samples = kwargs.pop("time_samples", None)
            result = fixation_overlap(x, y, time_samples=time_samples, **kwargs)
            return result["perc"]
        elif method == "sinkhorn":
            coords_x = x[["x", "y"]].values
            coords_y = y[["x", "y"]].values
            wx = np.ones(len(x)) / len(x)
            wy = np.ones(len(y)) / len(y)
            dist = _emdw(coords_x, wx, coords_y, wy)
            return 1.0 / (1.0 + dist)

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

    # Detect if method returns dict (e.g. multimatch)
    is_dict_method = method == "multimatch"

    for i, mi in enumerate(matchind):
        d1 = ref_data[mi]
        d2 = src_data[i]

        if d1 is None or d2 is None:
            eye_sim_list.append(np.nan if not is_dict_method else None)
            if permutations > 0:
                perm_sim_list.append(np.nan if not is_dict_method else None)
                diff_list.append(np.nan if not is_dict_method else None)
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
                eye_sim_list.append(sim)
                perm_sim_list.append(np.nan if not is_dict_method else None)
                diff_list.append(np.nan if not is_dict_method else None)
                continue

            psims = []
            for j in mind:
                d1p = ref_data[j]
                if d1p is None:
                    psims.append(np.nan if not is_dict_method else None)
                else:
                    ps = similarity(d1p, d2, method=method,
                                    multiscale_aggregation=multiscale_aggregation, **kwargs)
                    psims.append(ps)

            if is_dict_method:
                # Average each metric across permutations
                valid_psims = [p for p in psims if isinstance(p, dict)]
                if valid_psims:
                    perm_mean = {k: float(np.nanmean([p[k] for p in valid_psims]))
                                 for k in valid_psims[0]}
                else:
                    perm_mean = None
            else:
                perm_mean = float(np.nanmean(psims))

            eye_sim_list.append(sim)
            perm_sim_list.append(perm_mean)

            if is_dict_method:
                if isinstance(sim, dict) and isinstance(perm_mean, dict):
                    diff_list.append({k: sim[k] - perm_mean[k] for k in sim})
                else:
                    diff_list.append(None)
            else:
                if np.isscalar(sim):
                    diff_list.append(sim - perm_mean)
                else:
                    diff_list.append(np.nanmean(sim) - perm_mean if not np.isnan(perm_mean) else np.nan)
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
    else:
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


# ---------------------------------------------------------------------------
# Cross-validated template similarity
# ---------------------------------------------------------------------------

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
        assignments (0-based), and ``n_folds``.
    """
    if isinstance(split_on, str):
        split_on = [split_on]

    # Build a group key per row
    group_keys = source_tab[split_on].apply(
        lambda row: tuple(row), axis=1
    )
    unique_groups = list(group_keys.unique())

    if len(unique_groups) < 2:
        raise ValueError(
            f"Need at least 2 unique groups in split_on columns to create "
            f"CV folds, got {len(unique_groups)}."
        )

    if n_folds is None:
        n_folds = min(5, len(unique_groups))
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if n_folds > len(unique_groups):
        raise ValueError(
            f"n_folds ({n_folds}) exceeds unique groups ({len(unique_groups)})."
        )

    # Shuffle unique groups deterministically and assign round-robin
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_groups))
    group_to_fold = {}
    for rank, idx in enumerate(order):
        group_to_fold[unique_groups[idx]] = rank % n_folds

    fold_ids = np.array([group_to_fold[g] for g in group_keys], dtype=int)
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
        Random seed for fold assignment.
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

    # Build folds
    cv = _make_cv_folds(source_tab, split_on, n_folds=n_folds, seed=seed)
    fold_ids = cv["fold_ids"]
    n_folds_actual = cv["n_folds"]

    fold_results = []

    for fold in range(n_folds_actual):
        in_fold = fold_ids == fold

        # Eval rows: in this fold AND pass eval_mask
        eval_rows = in_fold & eval_mask
        if not eval_rows.any():
            continue

        # Training candidate rows: NOT in this fold AND pass fit_mask
        train_rows = (~in_fold) & fit_mask

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
