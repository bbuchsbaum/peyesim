"""Latent-space transforms for template-based similarity (PCA, CORAL, CCA, geometric)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA

from peyesim.density import EyeDensity, EyeDensityMultiscale


def _vectorize_density(obj) -> np.ndarray:
    """Flatten a density object to a 1-D vector."""
    if isinstance(obj, EyeDensityMultiscale):
        lens = [s.z.size for s in obj]
        if len(set(lens)) != 1:
            raise ValueError(
                "All scales in EyeDensityMultiscale must have the same "
                "grid dimensions for latent transforms."
            )
        obj = obj[0]
    if isinstance(obj, EyeDensity):
        return obj.z.ravel()
    arr = np.asarray(obj)
    return arr.ravel()


def _split_rows(mat: np.ndarray) -> list[np.ndarray]:
    """Split a 2-D array into a list of row vectors."""
    return [mat[i] for i in range(mat.shape[0])]


def _latent_pca_projection(ref_tab, source_tab, refvar, sourcevar,
                           comps, center=True, scale=False):
    """Shared PCA projection used by all three transforms."""
    ref_vecs = [_vectorize_density(d) for d in ref_tab[refvar]]
    src_vecs = [_vectorize_density(d) for d in source_tab[sourcevar]]

    ref_lens = [len(v) for v in ref_vecs]
    src_lens = [len(v) for v in src_vecs]
    if len(set(ref_lens + src_lens)) != 1:
        raise ValueError("All density vectors must share the same length for latent transforms.")

    ref_mat = np.vstack(ref_vecs)
    src_mat = np.vstack(src_vecs)
    combined = np.vstack([ref_mat, src_mat])

    n_components = min(comps, combined.shape[0], combined.shape[1])

    pca = PCA(n_components=n_components)
    # Center (and optionally scale) before PCA
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_mean=center, with_std=True)
        combined = scaler.fit_transform(combined)
    elif center:
        combined = combined - combined.mean(axis=0)

    scores = pca.fit_transform(combined)

    k = scores.shape[1]
    n_ref = ref_mat.shape[0]

    return {
        "ref_scores": scores[:n_ref],
        "source_scores": scores[n_ref:],
        "k": k,
        "basis": pca,
    }


def latent_pca_transform(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str | None = None,
    refvar: str = "density",
    sourcevar: str = "density",
    comps: int = 30,
    center: bool = True,
    scale: bool = False,
    **kwargs,
) -> dict:
    """Project densities into PCA space before similarity computation."""
    proj = _latent_pca_projection(ref_tab, source_tab, refvar, sourcevar,
                                  comps=comps, center=center, scale=scale)

    ref_tab = ref_tab.copy()
    source_tab = source_tab.copy()
    ref_tab[refvar] = _split_rows(proj["ref_scores"])
    source_tab[sourcevar] = _split_rows(proj["source_scores"])

    return {
        "ref_tab": ref_tab,
        "source_tab": source_tab,
        "refvar": refvar,
        "sourcevar": sourcevar,
        "info": {"transform": "latent_pca", "comps": proj["k"],
                 "center": center, "scale": scale},
    }


def coral_transform(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str | None = None,
    refvar: str = "density",
    sourcevar: str = "density",
    comps: int = 30,
    center: bool = True,
    scale: bool = False,
    shrink: float = 1e-3,
    **kwargs,
) -> dict:
    """CORAL domain adaptation: align source covariance to reference."""
    proj = _latent_pca_projection(ref_tab, source_tab, refvar, sourcevar,
                                  comps=comps, center=center, scale=scale)

    k = proj["k"]
    cov_ref = np.cov(proj["ref_scores"], rowvar=False) + np.eye(k) * shrink
    cov_src = np.cov(proj["source_scores"], rowvar=False) + np.eye(k) * shrink

    def _mat_sqrt(m):
        vals, vecs = np.linalg.eigh(m)
        return vecs @ np.diag(np.sqrt(np.maximum(vals, 0))) @ vecs.T

    def _mat_inv_sqrt(m):
        vals, vecs = np.linalg.eigh(m)
        return vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, shrink))) @ vecs.T

    adapt = _mat_inv_sqrt(cov_src) @ _mat_sqrt(cov_ref)
    adapted_src = (adapt @ proj["source_scores"].T).T

    ref_tab = ref_tab.copy()
    source_tab = source_tab.copy()
    ref_tab[refvar] = _split_rows(proj["ref_scores"])
    source_tab[sourcevar] = _split_rows(adapted_src)

    return {
        "ref_tab": ref_tab,
        "source_tab": source_tab,
        "refvar": refvar,
        "sourcevar": sourcevar,
        "info": {"transform": "coral", "comps": k, "center": center,
                 "scale": scale, "shrink": shrink},
    }


def cca_transform(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str | None = None,
    refvar: str = "density",
    sourcevar: str = "density",
    comps: int = 10,
    center: bool = True,
    scale: bool = False,
    shrink: float = 1e-3,
    **kwargs,
) -> dict:
    """CCA-based alignment between reference and source domains."""
    proj = _latent_pca_projection(ref_tab, source_tab, refvar, sourcevar,
                                  comps=comps, center=center, scale=scale)

    k_use = min(proj["k"], comps,
                proj["ref_scores"].shape[0] - 1,
                proj["source_scores"].shape[0] - 1)

    ref_tab_out = ref_tab.copy()
    source_tab_out = source_tab.copy()

    if k_use < 1:
        warnings.warn("Not enough observations for CCA; returning PCA-transformed scores.")
        ref_tab_out[refvar] = _split_rows(proj["ref_scores"])
        source_tab_out[sourcevar] = _split_rows(proj["source_scores"])
        return {
            "ref_tab": ref_tab_out, "source_tab": source_tab_out,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": {"transform": "cca", "comps": 0, "note": "insufficient observations"},
        }

    def _scale_with_shrink(mat, k):
        mat_k = mat[:, :k]
        v = np.var(mat_k, axis=0, ddof=1)
        sds = np.sqrt(np.maximum(v, 0) + shrink)
        return (mat_k - mat_k.mean(axis=0)) / sds

    X = _scale_with_shrink(proj["ref_scores"], k_use)
    Y = _scale_with_shrink(proj["source_scores"], k_use)

    # CCA via SVD of cross-covariance
    try:
        from sklearn.cross_decomposition import CCA
        cca = CCA(n_components=k_use, max_iter=1000)
        X_c, Y_c = cca.fit_transform(X, Y)
    except Exception:
        warnings.warn("CCA failed; returning PCA-transformed scores.")
        ref_tab_out[refvar] = _split_rows(proj["ref_scores"])
        source_tab_out[sourcevar] = _split_rows(proj["source_scores"])
        return {
            "ref_tab": ref_tab_out, "source_tab": source_tab_out,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": {"transform": "cca", "comps": 0, "note": "CCA failed"},
        }

    ref_tab_out[refvar] = _split_rows(X_c)
    source_tab_out[sourcevar] = _split_rows(Y_c)

    return {
        "ref_tab": ref_tab_out,
        "source_tab": source_tab_out,
        "refvar": refvar,
        "sourcevar": sourcevar,
        "info": {"transform": "cca", "comps": k_use, "center": center,
                 "scale": scale, "shrink": shrink},
    }


# ---------------------------------------------------------------------------
# Geometric density transforms (operate in 2-D coordinate space)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fit / Apply separation for cross-validated transforms
# ---------------------------------------------------------------------------

def _fit_transform(similarity_transform, ref_tab, source_tab, match_on,
                   refvar="density", sourcevar="density", **kwargs):
    """Fit a similarity transform on training data and return a model dict.

    The model dict contains enough information to project new data via
    ``_apply_transform``.
    """
    # Identify which transform we're dealing with
    _name = getattr(similarity_transform, "__name__", "")

    if _name == "latent_pca_transform":
        return _fit_pca_model(ref_tab, source_tab, refvar, sourcevar, **kwargs)
    elif _name == "coral_transform":
        return _fit_coral_model(ref_tab, source_tab, refvar, sourcevar, **kwargs)
    elif _name == "cca_transform":
        return _fit_cca_model(ref_tab, source_tab, match_on, refvar, sourcevar, **kwargs)
    elif _name in ("contract_transform", "affine_transform"):
        return _fit_geometric_model(
            similarity_transform, ref_tab, source_tab, match_on,
            refvar, sourcevar, **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported transform for fit/apply: {_name}. "
            "Supported: latent_pca_transform, coral_transform, cca_transform, "
            "contract_transform, affine_transform."
        )


def _apply_transform(model, ref_tab, source_tab, refvar="density",
                     sourcevar="density"):
    """Apply a fitted transform model to new (ref_tab, source_tab) data."""
    kind = model["transform"]

    if kind == "latent_pca":
        return _apply_pca_model(model, ref_tab, source_tab, refvar, sourcevar)
    elif kind == "coral":
        return _apply_coral_model(model, ref_tab, source_tab, refvar, sourcevar)
    elif kind == "cca":
        return _apply_cca_model(model, ref_tab, source_tab, refvar, sourcevar)
    elif kind in ("contract", "affine"):
        return _apply_geometric_model(model, ref_tab, source_tab, refvar, sourcevar)
    else:
        raise ValueError(f"Unknown transform kind: {kind}")


# --- PCA fit/apply ---

def _fit_pca_model(ref_tab, source_tab, refvar, sourcevar,
                   comps=30, center=True, scale=False, **_):
    ref_vecs = [_vectorize_density(d) for d in ref_tab[refvar]]
    src_vecs = [_vectorize_density(d) for d in source_tab[sourcevar]]
    combined = np.vstack(ref_vecs + src_vecs)
    n_components = min(comps, combined.shape[0], combined.shape[1])
    mean = combined.mean(axis=0) if center else np.zeros(combined.shape[1])
    pca = PCA(n_components=n_components)
    centered = combined - mean if center else combined
    pca.fit(centered)
    return {"transform": "latent_pca", "pca": pca, "mean": mean,
            "center": center, "scale": scale}


def _apply_pca_model(model, ref_tab, source_tab, refvar, sourcevar):
    pca = model["pca"]
    mean = model["mean"]
    ref_vecs = np.vstack([_vectorize_density(d) for d in ref_tab[refvar]])
    src_vecs = np.vstack([_vectorize_density(d) for d in source_tab[sourcevar]])
    ref_scores = pca.transform(ref_vecs - mean)
    src_scores = pca.transform(src_vecs - mean)
    ref_tab = ref_tab.copy()
    source_tab = source_tab.copy()
    ref_tab[refvar] = _split_rows(ref_scores)
    source_tab[sourcevar] = _split_rows(src_scores)
    return {"ref_tab": ref_tab, "source_tab": source_tab,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": {"transform": "latent_pca"}}


# --- CORAL fit/apply ---

def _fit_coral_model(ref_tab, source_tab, refvar, sourcevar,
                     comps=30, center=True, scale=False, shrink=1e-3, **_):
    pca_model = _fit_pca_model(ref_tab, source_tab, refvar, sourcevar,
                               comps=comps, center=center, scale=scale)
    pca = pca_model["pca"]
    mean = pca_model["mean"]
    ref_vecs = np.vstack([_vectorize_density(d) for d in ref_tab[refvar]])
    src_vecs = np.vstack([_vectorize_density(d) for d in source_tab[sourcevar]])
    ref_scores = pca.transform(ref_vecs - mean)
    src_scores = pca.transform(src_vecs - mean)
    k = ref_scores.shape[1]
    cov_ref = np.cov(ref_scores, rowvar=False) + np.eye(k) * shrink
    cov_src = np.cov(src_scores, rowvar=False) + np.eye(k) * shrink

    vals_src, vecs_src = np.linalg.eigh(cov_src)
    inv_sqrt_src = vecs_src @ np.diag(1.0 / np.sqrt(np.maximum(vals_src, shrink))) @ vecs_src.T
    vals_ref, vecs_ref = np.linalg.eigh(cov_ref)
    sqrt_ref = vecs_ref @ np.diag(np.sqrt(np.maximum(vals_ref, 0))) @ vecs_ref.T
    adapt = inv_sqrt_src @ sqrt_ref

    return {"transform": "coral", "pca": pca, "mean": mean,
            "adapt": adapt, "shrink": shrink, "center": center}


def _apply_coral_model(model, ref_tab, source_tab, refvar, sourcevar):
    pca = model["pca"]
    mean = model["mean"]
    adapt = model["adapt"]
    ref_vecs = np.vstack([_vectorize_density(d) for d in ref_tab[refvar]])
    src_vecs = np.vstack([_vectorize_density(d) for d in source_tab[sourcevar]])
    ref_scores = pca.transform(ref_vecs - mean)
    src_scores = pca.transform(src_vecs - mean)
    adapted_src = (adapt @ src_scores.T).T
    ref_tab = ref_tab.copy()
    source_tab = source_tab.copy()
    ref_tab[refvar] = _split_rows(ref_scores)
    source_tab[sourcevar] = _split_rows(adapted_src)
    return {"ref_tab": ref_tab, "source_tab": source_tab,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": {"transform": "coral"}}


# --- CCA fit/apply ---

def _fit_cca_model(ref_tab, source_tab, match_on, refvar, sourcevar,
                   comps=10, center=True, scale=False, shrink=1e-3, **_):
    pca_model = _fit_pca_model(ref_tab, source_tab, refvar, sourcevar,
                               comps=comps, center=center, scale=scale)
    pca = pca_model["pca"]
    mean = pca_model["mean"]
    ref_vecs = np.vstack([_vectorize_density(d) for d in ref_tab[refvar]])
    src_vecs = np.vstack([_vectorize_density(d) for d in source_tab[sourcevar]])
    ref_scores = pca.transform(ref_vecs - mean)
    src_scores = pca.transform(src_vecs - mean)
    k_use = min(ref_scores.shape[1], comps,
                ref_scores.shape[0] - 1, src_scores.shape[0] - 1)
    if k_use < 1:
        return {"transform": "cca", "pca": pca, "mean": mean,
                "cca_model": None, "k_use": 0, "shrink": shrink}

    from sklearn.cross_decomposition import CCA
    cca = CCA(n_components=k_use, max_iter=1000)

    def _prep(mat, k):
        mat_k = mat[:, :k]
        v = np.var(mat_k, axis=0, ddof=1)
        sds = np.sqrt(np.maximum(v, 0) + shrink)
        m = mat_k.mean(axis=0)
        return (mat_k - m) / sds, m, sds

    X, x_mean, x_sd = _prep(ref_scores, k_use)
    Y, y_mean, y_sd = _prep(src_scores, k_use)
    cca.fit(X, Y)
    return {"transform": "cca", "pca": pca, "mean": mean,
            "cca_model": cca, "k_use": k_use, "shrink": shrink,
            "x_mean": x_mean, "x_sd": x_sd, "y_mean": y_mean, "y_sd": y_sd}


def _apply_cca_model(model, ref_tab, source_tab, refvar, sourcevar):
    pca = model["pca"]
    mean = model["mean"]
    ref_vecs = np.vstack([_vectorize_density(d) for d in ref_tab[refvar]])
    src_vecs = np.vstack([_vectorize_density(d) for d in source_tab[sourcevar]])
    ref_scores = pca.transform(ref_vecs - mean)
    src_scores = pca.transform(src_vecs - mean)

    ref_tab = ref_tab.copy()
    source_tab = source_tab.copy()

    if model["cca_model"] is None:
        ref_tab[refvar] = _split_rows(ref_scores)
        source_tab[sourcevar] = _split_rows(src_scores)
    else:
        k = model["k_use"]
        X = (ref_scores[:, :k] - model["x_mean"]) / model["x_sd"]
        Y = (src_scores[:, :k] - model["y_mean"]) / model["y_sd"]
        X_c, Y_c = model["cca_model"].transform(X, Y)
        ref_tab[refvar] = _split_rows(X_c)
        source_tab[sourcevar] = _split_rows(Y_c)

    return {"ref_tab": ref_tab, "source_tab": source_tab,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": {"transform": "cca"}}


# --- Geometric (contract/affine) fit/apply ---

def _fit_geometric_model(transform_fn, ref_tab, source_tab, match_on,
                         refvar, sourcevar, shrink=1e-6, **_):
    ref_densities, src_densities = _match_pairs(
        ref_tab, source_tab, match_on, refvar, sourcevar)
    ref_mean, ref_cov = _aggregate_density_moments(ref_densities)
    src_mean, src_cov = _aggregate_density_moments(src_densities)
    name = getattr(transform_fn, "__name__", "")
    if name == "contract_transform":
        s = np.sqrt((np.trace(ref_cov) + shrink) / (np.trace(src_cov) + shrink))
        A = s * np.eye(2)
        kind = "contract"
    else:
        A = _mat_sqrt_2d(ref_cov + shrink * np.eye(2)) @ _mat_inv_sqrt_2d(
            src_cov + shrink * np.eye(2), shrink=shrink)
        kind = "affine"
    t = ref_mean - A @ src_mean
    return {"transform": kind, "A": A, "t": t, "shrink": shrink}


def _apply_geometric_model(model, ref_tab, source_tab, refvar, sourcevar):
    A = model["A"]
    t = model["t"]
    source_tab = source_tab.copy()
    new_densities = []
    for d in source_tab[sourcevar]:
        if isinstance(d, EyeDensityMultiscale):
            new_densities.append(EyeDensityMultiscale(
                scales=[_warp_density(s, A, t) for s in d]))
        else:
            new_densities.append(_warp_density(d, A, t))
    source_tab[sourcevar] = new_densities
    return {"ref_tab": ref_tab, "source_tab": source_tab,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": {"transform": model["transform"]}}


# ---------------------------------------------------------------------------
# Geometric density transforms (operate in 2-D coordinate space)
# ---------------------------------------------------------------------------

def _density_moments(dens: EyeDensity):
    """Compute weighted mean and covariance of an EyeDensity in (x, y) space.

    Returns (mean_xy [2], cov_xy [2,2]).
    """
    xx, yy = np.meshgrid(dens.x, dens.y, indexing="ij")
    w = np.maximum(dens.z, 0).ravel()
    total = w.sum()
    if total < 1e-30:
        # Uniform fallback
        mx = dens.x.mean()
        my = dens.y.mean()
        return np.array([mx, my]), np.eye(2)
    xf = xx.ravel()
    yf = yy.ravel()
    mx = np.dot(w, xf) / total
    my = np.dot(w, yf) / total
    dx = xf - mx
    dy = yf - my
    cxx = np.dot(w, dx * dx) / total
    cxy = np.dot(w, dx * dy) / total
    cyy = np.dot(w, dy * dy) / total
    return np.array([mx, my]), np.array([[cxx, cxy], [cxy, cyy]])


def _aggregate_density_moments(density_list):
    """Aggregate weighted mean and covariance across a list of EyeDensity objects.

    Returns (mean_xy [2], cov_xy [2,2]).
    """
    means = []
    covs = []
    weights = []
    for d in density_list:
        if isinstance(d, EyeDensityMultiscale):
            d = d[0]
        m, c = _density_moments(d)
        w = np.maximum(d.z, 0).sum()
        means.append(m)
        covs.append(c)
        weights.append(w)
    means = np.array(means)
    weights = np.array(weights)
    total = weights.sum()
    if total < 1e-30:
        return means.mean(axis=0), np.eye(2)
    # Weighted aggregate mean
    agg_mean = (weights[:, None] * means).sum(axis=0) / total
    # Weighted aggregate covariance (within + between components)
    agg_cov = np.zeros((2, 2))
    for m, c, w in zip(means, covs, weights):
        d = m - agg_mean
        agg_cov += w * (c + np.outer(d, d))
    agg_cov /= total
    return agg_mean, agg_cov


def _mat_sqrt_2d(m):
    """Matrix square root of a 2x2 symmetric positive-semi-definite matrix."""
    vals, vecs = np.linalg.eigh(m)
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T


def _mat_inv_sqrt_2d(m, shrink=1e-6):
    """Inverse matrix square root of a 2x2 symmetric matrix with shrinkage."""
    vals, vecs = np.linalg.eigh(m)
    return vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, shrink))) @ vecs.T


def _warp_density(dens: EyeDensity, A: np.ndarray, t: np.ndarray) -> EyeDensity:
    """Apply affine transform (A, t) to an EyeDensity's coordinate grid.

    The new density is obtained by interpolating the original density onto
    the transformed grid: new_coords = A @ old_coords + t.
    """
    # Transform the 1-D grid vectors
    # For a regular grid we transform each grid point and interpolate back
    xx, yy = np.meshgrid(dens.x, dens.y, indexing="ij")
    new_xx = A[0, 0] * xx + A[0, 1] * yy + t[0]
    new_yy = A[1, 0] * xx + A[1, 1] * yy + t[1]

    # Build interpolator on original grid
    interp = RegularGridInterpolator(
        (dens.x, dens.y), dens.z,
        method="linear", bounds_error=False, fill_value=0.0,
    )

    # We want the density at the NEW grid positions, but the density values
    # "move" with the coordinates. The correct approach: the transformed
    # source density at position p is the original density at A^{-1}(p - t).
    # But since we are warping source to match reference, we evaluate original
    # density at positions that map TO the new grid.
    #
    # Simpler: create a new EyeDensity whose grid spans the warped extent
    # and sample the original density at the inverse-mapped positions.
    #
    # For template_similarity the grids need to match ref grids, so we keep
    # the same grid vectors as the original and pull back:
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return dens  # degenerate; return unchanged

    # For each point on the ORIGINAL grid, find where it came from in source
    # source_pos = A_inv @ (grid_pos - t)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    src_pts = (A_inv @ (pts - t).T).T
    new_z = interp(src_pts).reshape(dens.z.shape)

    return EyeDensity(
        x=dens.x.copy(), y=dens.y.copy(), z=new_z,
        sigma=dens.sigma, fixgroup=dens.fixgroup,
    )


def _match_pairs(ref_tab, source_tab, match_on, refvar, sourcevar):
    """Return matched lists of (ref_density, source_density) based on match_on."""
    if match_on is None:
        # Positional matching
        ref_densities = list(ref_tab[refvar])
        src_densities = list(source_tab[sourcevar])
        n = min(len(ref_densities), len(src_densities))
        return ref_densities[:n], src_densities[:n]
    merged = pd.merge(
        ref_tab[[match_on, refvar]].rename(columns={refvar: "_ref_d"}),
        source_tab[[match_on, sourcevar]].rename(columns={sourcevar: "_src_d"}),
        on=match_on,
    )
    return list(merged["_ref_d"]), list(merged["_src_d"])


def contract_transform(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str | None = None,
    refvar: str = "density",
    sourcevar: str = "density",
    shrink: float = 1e-6,
    **kwargs,
) -> dict:
    """Uniform-scaling geometric transform matching spatial spread of source to reference.

    Fits a scalar scaling factor so that the trace of the source covariance
    matches the trace of the reference covariance, then applies the
    corresponding affine warp to the source densities.
    """
    ref_densities, src_densities = _match_pairs(
        ref_tab, source_tab, match_on, refvar, sourcevar,
    )

    ref_mean, ref_cov = _aggregate_density_moments(ref_densities)
    src_mean, src_cov = _aggregate_density_moments(src_densities)

    scale = np.sqrt(
        (np.trace(ref_cov) + shrink) / (np.trace(src_cov) + shrink)
    )
    A = scale * np.eye(2)
    t = ref_mean - A @ src_mean

    # Warp source densities
    source_tab = source_tab.copy()
    new_densities = []
    for d in source_tab[sourcevar]:
        if isinstance(d, EyeDensityMultiscale):
            new_scales = [_warp_density(s, A, t) for s in d]
            new_densities.append(EyeDensityMultiscale(scales=new_scales))
        else:
            new_densities.append(_warp_density(d, A, t))
    source_tab[sourcevar] = new_densities

    return {
        "ref_tab": ref_tab,
        "source_tab": source_tab,
        "refvar": refvar,
        "sourcevar": sourcevar,
        "info": {"transform": "contract", "scale": scale,
                 "A": A, "t": t, "shrink": shrink},
    }


def affine_transform(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str | None = None,
    refvar: str = "density",
    sourcevar: str = "density",
    shrink: float = 1e-6,
    **kwargs,
) -> dict:
    """Full affine (rotation+scale+shear) geometric transform.

    Fits ``A = sqrtm(ref_cov + shrink*I) @ inv_sqrtm(src_cov + shrink*I)``
    and ``t = ref_mean - A @ src_mean``, then warps the source densities.
    """
    ref_densities, src_densities = _match_pairs(
        ref_tab, source_tab, match_on, refvar, sourcevar,
    )

    ref_mean, ref_cov = _aggregate_density_moments(ref_densities)
    src_mean, src_cov = _aggregate_density_moments(src_densities)

    A = _mat_sqrt_2d(ref_cov + shrink * np.eye(2)) @ _mat_inv_sqrt_2d(
        src_cov + shrink * np.eye(2), shrink=shrink,
    )
    t = ref_mean - A @ src_mean

    # Warp source densities
    source_tab = source_tab.copy()
    new_densities = []
    for d in source_tab[sourcevar]:
        if isinstance(d, EyeDensityMultiscale):
            new_scales = [_warp_density(s, A, t) for s in d]
            new_densities.append(EyeDensityMultiscale(scales=new_scales))
        else:
            new_densities.append(_warp_density(d, A, t))
    source_tab[sourcevar] = new_densities

    return {
        "ref_tab": ref_tab,
        "source_tab": source_tab,
        "refvar": refvar,
        "sourcevar": sourcevar,
        "info": {"transform": "affine", "A": A, "t": t, "shrink": shrink},
    }
