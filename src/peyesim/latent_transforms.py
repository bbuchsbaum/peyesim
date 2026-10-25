"""Latent-space transforms for template-based similarity (PCA, CORAL, CCA, geometric)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from peyesim.density import EyeDensity, EyeDensityMultiscale, _zapsmall


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
        return np.asarray(obj.z, dtype=float).ravel()
    if obj is not None and not isinstance(obj, (str, bytes, dict)):
        arr = np.asarray(obj)
        if arr.dtype.kind in "biuf":
            return arr.astype(float).ravel()
    raise ValueError("Unsupported object passed to latent transform; expected density or numeric.")


def _vectorize_density_tab(tab, var, expected_len=None) -> np.ndarray:
    """R ``vectorize_density_tab()``: one row per density, equal lengths."""
    vecs = [_vectorize_density(d) for d in tab[var]]
    if not vecs:
        return np.zeros((0, 0 if expected_len is None else expected_len))
    lens = {len(v) for v in vecs}
    if len(lens) != 1:
        raise ValueError("All density vectors must share the same length for latent transforms.")
    if expected_len is not None and lens.pop() != expected_len:
        raise ValueError("Density vectors must match the transform training dimensionality.")
    return np.vstack(vecs)


class _Prcomp:
    """``stats::prcomp(x, center, scale., rank.)``: centring/scaling and rotation."""

    def __init__(self, combined: np.ndarray, comps: int, center: bool, scale: bool):
        x = np.asarray(combined, dtype=float)
        n, p = x.shape
        self.center = x.mean(axis=0) if center else None
        xc = x - self.center if center else x
        if scale:
            # base::scale(): sd when centred, root mean square / (n - 1) otherwise
            sc = np.sqrt(np.sum(xc ** 2, axis=0) / max(n - 1, 1))
            if np.any(~np.isfinite(sc)) or np.any(sc == 0):
                raise ValueError("cannot rescale a constant/zero column to unit variance")
            self.scale = sc
            xc = xc / sc
        else:
            self.scale = None
        if not np.all(np.isfinite(xc)):
            raise ValueError("infinite or missing values in 'x'")
        rank = min(comps, n, p)
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        self.rotation = vt[:rank].T
        self.k = min(comps, self.rotation.shape[1])
        self.input_dim = p

    def transform(self, mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=float)
        if mat.shape[0] == 0:
            return np.zeros((0, self.k))
        if self.center is not None:
            mat = mat - self.center
        if self.scale is not None:
            mat = mat / self.scale
        return mat @ self.rotation[:, :self.k]


def _split_rows(mat: np.ndarray) -> list[np.ndarray]:
    """Split a 2-D array into a list of row vectors."""
    return [mat[i] for i in range(mat.shape[0])]


def _cov_with_shrink(mat: np.ndarray, shrink: float) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    k = mat.shape[1]
    if mat.shape[0] > 1:
        centered = mat - mat.mean(axis=0)
        cov = centered.T @ centered / (mat.shape[0] - 1)
    else:
        cov = np.zeros((k, k), dtype=float)
    return cov + np.eye(k) * shrink


def _mat_sqrt(m: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(m)
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0))) @ vecs.T


def _mat_inv_sqrt(m: np.ndarray, shrink: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(m)
    return vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, shrink))) @ vecs.T


def _transform_group_keys(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    fit_by: str | list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if fit_by is None:
        return (
            np.repeat("__all__", len(ref_tab)),
            np.repeat("__all__", len(source_tab)),
        )
    if isinstance(fit_by, str):
        fit_by = [fit_by]
    missing_ref = [col for col in fit_by if col not in ref_tab.columns]
    missing_src = [col for col in fit_by if col not in source_tab.columns]
    if missing_ref or missing_src:
        raise ValueError("fit_by columns must exist in both ref_tab and source_tab for grouped latent transforms.")

    def make_keys(tab: pd.DataFrame) -> np.ndarray:
        sub = tab.loc[:, fit_by]
        keys = np.array(["::".join(str(v) for v in row) for row in sub.itertuples(index=False)],
                        dtype=object)
        # As R interaction(): a missing value gives an NA key, which matches
        # no stratum, so those rows are never fitted (and keep their input).
        missing = tab.loc[:, fit_by].isna().any(axis=1).to_numpy()
        keys[missing] = _NA_GROUP
        return keys

    return make_keys(ref_tab), make_keys(source_tab)


class _NAGroup:
    """Missing ``fit_by`` key: compares unequal to everything, itself included."""

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __hash__(self):
        return 0

    def __repr__(self):
        return "NA"


_NA_GROUP = _NAGroup()


def _resolve_similarity_transform(similarity_transform):
    """R ``resolve_similarity_transform_function()``: a function or its name."""
    if callable(similarity_transform):
        return similarity_transform
    if isinstance(similarity_transform, str):
        import peyesim
        import peyesim.latent_transforms as lt

        fn = getattr(lt, similarity_transform, None) or getattr(peyesim, similarity_transform, None)
        if callable(fn):
            return fn
        raise ValueError(f"object '{similarity_transform}' of mode 'function' was not found")
    raise ValueError("similarity_transform must be a function or a single function name.")


def _latent_pca_projection(ref_tab, source_tab, refvar, sourcevar,
                           comps, center=True, scale=False):
    """Shared PCA projection used by all three transforms (R ``latent_pca_projection``)."""
    model = _fit_pca_model(ref_tab, source_tab, refvar, sourcevar,
                           comps=comps, center=center, scale=scale)
    return {
        "ref_scores": _project_scores(model, ref_tab, refvar),
        "source_scores": _project_scores(model, source_tab, sourcevar),
        "k": model["k"],
        "basis": model["basis"],
    }


def _resolve_scale_alias(scale: bool, kwargs: dict) -> bool:
    if "scale." not in kwargs:
        return scale
    scale_dot = bool(kwargs.pop("scale."))
    if bool(scale) != scale_dot and bool(scale):
        raise ValueError("Use either scale or scale., not both.")
    return scale_dot


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
    scale = _resolve_scale_alias(scale, kwargs)
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
    fit_by: str | list[str] | None = None,
    **kwargs,
) -> dict:
    """CORAL domain adaptation: align source covariance to reference."""
    scale = _resolve_scale_alias(scale, kwargs)
    proj = _latent_pca_projection(ref_tab, source_tab, refvar, sourcevar,
                                  comps=comps, center=center, scale=scale)

    k = proj["k"]
    ref_keys, src_keys = _transform_group_keys(ref_tab, source_tab, fit_by)
    group_levels = list(dict.fromkeys([*ref_keys.tolist(), *src_keys.tolist()]))
    adapted_src = proj["source_scores"].copy()
    group_info = []

    for group_key in group_levels:
        ref_rows = np.flatnonzero(ref_keys == group_key)
        src_rows = np.flatnonzero(src_keys == group_key)
        group_label = "all" if group_key == "__all__" else group_key
        if len(ref_rows) == 0 or len(src_rows) == 0:
            group_info.append({
                "group": group_label,
                "ref_n": int(len(ref_rows)),
                "source_n": int(len(src_rows)),
                "note": "missing group rows",
            })
            continue
        cov_ref = _cov_with_shrink(proj["ref_scores"][ref_rows, :], shrink)
        cov_src = _cov_with_shrink(proj["source_scores"][src_rows, :], shrink)
        adapt = _mat_inv_sqrt(cov_src, shrink) @ _mat_sqrt(cov_ref)
        # Scores are rows, so CORAL maps X -> X Cs^{-1/2} Ct^{1/2}, which gives
        # the adapted scores the reference covariance.
        adapted_src[src_rows, :k] = proj["source_scores"][src_rows, :k] @ adapt
        group_info.append({
            "group": group_label,
            "ref_n": int(len(ref_rows)),
            "source_n": int(len(src_rows)),
            "note": "ok",
        })

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
                 "scale": scale, "shrink": shrink, "fit_by": fit_by,
                 "groups": group_info},
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
    fit_by: str | list[str] | None = None,
    unique_match_only: bool = False,
    **kwargs,
) -> dict:
    """CCA-based alignment between reference and source domains."""
    scale = _resolve_scale_alias(scale, kwargs)
    model = _fit_cca_model(
        ref_tab,
        source_tab,
        match_on,
        refvar,
        sourcevar,
        comps=comps,
        center=center,
        scale=scale,
        shrink=shrink,
        fit_by=fit_by,
        unique_match_only=unique_match_only,
    )
    return _apply_cca_model(model, ref_tab, source_tab, refvar, sourcevar)


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
    similarity_transform = _resolve_similarity_transform(similarity_transform)
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
                   comps=30, center=True, scale=False, **kwargs):
    scale = _resolve_scale_alias(scale, kwargs)
    ref_mat = _vectorize_density_tab(ref_tab, refvar)
    src_mat = _vectorize_density_tab(source_tab, sourcevar, expected_len=ref_mat.shape[1])
    combined = np.vstack([ref_mat, src_mat])
    if combined.shape[0] == 0 or combined.shape[1] == 0:
        raise ValueError("Cannot fit latent transform on empty density matrices.")
    basis = _Prcomp(combined, comps=comps, center=center, scale=scale)
    return {"transform": "latent_pca", "basis": basis, "pca": basis, "k": basis.k,
            "input_dim": basis.input_dim, "center": center, "scale": scale,
            "refvar": refvar, "sourcevar": sourcevar}


def _project_scores(model, tab, var) -> np.ndarray:
    """R ``project_transform_scores()``."""
    basis = model["basis"]
    if len(tab) == 0:
        return np.zeros((0, basis.k))
    return basis.transform(_vectorize_density_tab(tab, var, expected_len=basis.input_dim))


def _apply_pca_model(model, ref_tab, source_tab, refvar, sourcevar):
    ref_scores = _project_scores(model, ref_tab, refvar)
    src_scores = _project_scores(model, source_tab, sourcevar)
    ref_tab = ref_tab.copy()
    source_tab = source_tab.copy()
    ref_tab[refvar] = _split_rows(ref_scores)
    source_tab[sourcevar] = _split_rows(src_scores)
    return {"ref_tab": ref_tab, "source_tab": source_tab,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": {"transform": "latent_pca", "comps": model["k"],
                     "center": model["center"], "scale": model["scale"]}}


# --- CORAL fit/apply ---

def _fit_coral_model(ref_tab, source_tab, refvar, sourcevar,
                     comps=30, center=True, scale=False, shrink=1e-3,
                     fit_by=None, **kwargs):
    scale = _resolve_scale_alias(scale, kwargs)
    pca_model = _fit_pca_model(ref_tab, source_tab, refvar, sourcevar,
                               comps=comps, center=center, scale=scale)
    ref_scores = _project_scores(pca_model, ref_tab, refvar)
    src_scores = _project_scores(pca_model, source_tab, sourcevar)
    ref_keys, src_keys = _transform_group_keys(ref_tab, source_tab, fit_by)
    group_levels = list(dict.fromkeys([*ref_keys.tolist(), *src_keys.tolist()]))
    group_adapts = {}
    group_info = []
    for group_key in group_levels:
        ref_rows = np.flatnonzero(ref_keys == group_key)
        src_rows = np.flatnonzero(src_keys == group_key)
        group_label = "all" if group_key == "__all__" else group_key
        if len(ref_rows) == 0 or len(src_rows) == 0:
            group_info.append({
                "group": group_label,
                "ref_n": int(len(ref_rows)),
                "source_n": int(len(src_rows)),
                "note": "missing group rows",
            })
            continue
        cov_ref = _cov_with_shrink(ref_scores[ref_rows, :], shrink)
        cov_src = _cov_with_shrink(src_scores[src_rows, :], shrink)
        group_adapts[group_key] = _mat_inv_sqrt(cov_src, shrink) @ _mat_sqrt(cov_ref)
        group_info.append({
            "group": group_label,
            "ref_n": int(len(ref_rows)),
            "source_n": int(len(src_rows)),
            "note": "ok",
        })

    return {"transform": "coral", "pca_model": pca_model,
            "group_adapts": group_adapts, "shrink": shrink, "center": center,
            "fit_by": fit_by, "groups": group_info}


def _apply_coral_model(model, ref_tab, source_tab, refvar, sourcevar):
    ref_scores = _project_scores(model["pca_model"], ref_tab, refvar)
    src_scores = _project_scores(model["pca_model"], source_tab, sourcevar)
    adapted_src = src_scores.copy()
    _, src_keys = _transform_group_keys(ref_tab, source_tab, model.get("fit_by"))
    for group_key in dict.fromkeys(src_keys.tolist()):
        adapt = model["group_adapts"].get(group_key)
        if adapt is None:
            continue
        src_rows = np.flatnonzero(src_keys == group_key)
        adapted_src[src_rows, :] = src_scores[src_rows, :] @ adapt
    ref_tab = ref_tab.copy()
    source_tab = source_tab.copy()
    ref_tab[refvar] = _split_rows(ref_scores)
    source_tab[sourcevar] = _split_rows(adapted_src)
    return {"ref_tab": ref_tab, "source_tab": source_tab,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": {"transform": "coral", "fit_by": model.get("fit_by"),
                     "groups": model.get("groups", [])}}


# --- CCA fit/apply ---

def _fit_scale_with_shrink(mat: np.ndarray, shrink: float) -> dict:
    center_vec = mat.mean(axis=0)
    centered = mat - center_vec
    if centered.shape[0] > 1:
        vars_ = np.sum(centered ** 2, axis=0) / (centered.shape[0] - 1)
    else:
        vars_ = np.zeros(centered.shape[1], dtype=float)
    scale_vec = np.sqrt(np.maximum(vars_, 0) + shrink)
    scale_vec[scale_vec == 0] = 1
    return {
        "scores": centered / scale_vec,
        "center": center_vec,
        "scale": scale_vec,
    }


def _apply_scale_with_shrink(mat: np.ndarray, fit: dict) -> np.ndarray:
    return (mat - fit["center"]) / fit["scale"]


def _fit_cancor(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit a deterministic CCA model matching R stats::cancor's SVD route."""
    from scipy.linalg import qr, solve_triangular

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("unequal number of rows in 'x' or 'y'")
    if x.shape[0] == 0 or x.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError("dimension 0 in 'x' or 'y'")

    x_center = x.mean(axis=0)
    y_center = y.mean(axis=0)
    x = x - x_center
    y = y - y_center

    qx, rx, px = qr(x, mode="economic", pivoting=True)
    qy, ry, py = qr(y, mode="economic", pivoting=True)
    dx = int(np.linalg.matrix_rank(rx))
    dy = int(np.linalg.matrix_rank(ry))
    if dx == 0:
        raise ValueError("'x' has rank 0")
    if dy == 0:
        raise ValueError("'y' has rank 0")

    z = qx[:, :dx].T @ qy[:, :dy]
    u, s, vh = np.linalg.svd(z, full_matrices=False)
    v = vh.T
    xcoef_pivot = solve_triangular(rx[:dx, :dx], u[:, :dx], lower=False)
    ycoef_pivot = solve_triangular(ry[:dy, :dy], v[:, :dy], lower=False)

    xcoef = np.zeros((x.shape[1], xcoef_pivot.shape[1]), dtype=float)
    ycoef = np.zeros((y.shape[1], ycoef_pivot.shape[1]), dtype=float)
    xcoef[np.asarray(px[:dx]), :] = xcoef_pivot
    ycoef[np.asarray(py[:dy]), :] = ycoef_pivot

    return {
        "cor": s,
        "xcoef": xcoef,
        "ycoef": ycoef,
        "xcenter": x_center,
        "ycenter": y_center,
    }


def _fit_cca_model(ref_tab, source_tab, match_on, refvar, sourcevar,
                   comps=10, center=True, scale=False, shrink=1e-3,
                   fit_by=None, unique_match_only=False, **kwargs):
    scale = _resolve_scale_alias(scale, kwargs)
    pca_model = _fit_pca_model(ref_tab, source_tab, refvar, sourcevar,
                               comps=comps, center=center, scale=scale)
    ref_scores = _project_scores(pca_model, ref_tab, refvar)
    src_scores = _project_scores(pca_model, source_tab, sourcevar)

    k_target = pca_model["k"]
    ref_keys, src_keys = _transform_group_keys(ref_tab, source_tab, fit_by)
    group_levels = list(dict.fromkeys([*ref_keys.tolist(), *src_keys.tolist()]))
    group_models = {}
    group_info = []
    total_pairs = 0

    for group_key in group_levels:
        ref_rows = np.flatnonzero(ref_keys == group_key)
        src_rows = np.flatnonzero(src_keys == group_key)
        group_label = "all" if group_key == "__all__" else group_key

        if len(ref_rows) == 0 or len(src_rows) == 0:
            group_info.append({
                "group": group_label,
                "matched_pairs": 0,
                "comps": 0,
                "note": "missing group rows",
            })
            continue

        paired_ref, paired_src = _matched_transform_indices(
            ref_tab.iloc[ref_rows],
            source_tab.iloc[src_rows],
            match_on,
            unique_match_only=unique_match_only,
        )
        n_pairs = len(paired_src)
        total_pairs += n_pairs
        if n_pairs < 2:
            group_info.append({
                "group": group_label,
                "matched_pairs": n_pairs,
                "comps": 0,
                "note": "insufficient matched observations",
            })
            continue

        k_use = min(k_target, n_pairs - 1)
        ref_fit = _fit_scale_with_shrink(ref_scores[ref_rows[paired_ref], :k_use], shrink)
        src_fit = _fit_scale_with_shrink(src_scores[src_rows[paired_src], :k_use], shrink)
        try:
            cca = _fit_cancor(ref_fit["scores"], src_fit["scores"])
        except Exception:
            group_info.append({
                "group": group_label,
                "matched_pairs": n_pairs,
                "comps": 0,
                "note": "cancor failed",
            })
            continue

        k_group = min(k_use, len(cca["cor"]))
        if k_group < 1:
            group_info.append({
                "group": group_label,
                "matched_pairs": n_pairs,
                "comps": 0,
                "note": "no canonical components",
            })
            continue

        group_models[group_key] = {
            "k": k_group,
            "k_in": k_use,
            "ref_fit": ref_fit,
            "src_fit": src_fit,
            "xcoef": cca["xcoef"][:, :k_group],
            "ycoef": cca["ycoef"][:, :k_group],
        }
        group_info.append({
            "group": group_label,
            "matched_pairs": n_pairs,
            "comps": k_group,
            "cor": cca["cor"][:k_group],
            "note": "ok",
        })

    return {"transform": "cca", "pca_model": pca_model,
            "group_models": group_models, "k": k_target,
            "shrink": shrink, "center": center, "scale": scale,
            "fit_by": fit_by, "unique_match_only": unique_match_only,
            "matched_pairs": total_pairs, "groups": group_info}


def _apply_cca_model(model, ref_tab, source_tab, refvar, sourcevar):
    ref_scores = _project_scores(model["pca_model"], ref_tab, refvar)
    src_scores = _project_scores(model["pca_model"], source_tab, sourcevar)

    ref_tab = ref_tab.copy()
    source_tab = source_tab.copy()

    ref_latent = ref_scores.copy()
    src_latent = src_scores.copy()
    ref_keys, src_keys = _transform_group_keys(ref_tab, source_tab, model.get("fit_by"))
    group_levels = list(dict.fromkeys([*ref_keys.tolist(), *src_keys.tolist()]))
    # Fitted rows hold their group's canonical variates followed by zeros, so
    # the output keeps the PCA width but similarity is never computed over a
    # mix of canonical and PCA coordinates.
    unfitted = []
    for group_key in group_levels:
        group_model = model.get("group_models", {}).get(group_key)
        if group_model is None:
            if any(k is group_key or k == group_key for k in src_keys):
                unfitted.append("all" if group_key == "__all__" else str(group_key))
            continue
        k = group_model["k"]
        k_in = group_model.get("k_in", group_model["xcoef"].shape[0])
        ref_rows = np.flatnonzero(ref_keys == group_key)
        src_rows = np.flatnonzero(src_keys == group_key)
        if len(ref_rows) > 0:
            X = _apply_scale_with_shrink(ref_scores[ref_rows, :k_in], group_model["ref_fit"])
            ref_latent[ref_rows, :] = 0.0
            ref_latent[ref_rows, :k] = X @ group_model["xcoef"]
        if len(src_rows) > 0:
            Y = _apply_scale_with_shrink(src_scores[src_rows, :k_in], group_model["src_fit"])
            src_latent[src_rows, :] = 0.0
            src_latent[src_rows, :k] = Y @ group_model["ycoef"]

    if unfitted:
        warnings.warn(
            "cca_transform: no CCA model for group(s) "
            + ", ".join(dict.fromkeys(unfitted))
            + "; their rows keep PCA scores.",
            stacklevel=2,
        )

    ref_tab[refvar] = _split_rows(ref_latent)
    source_tab[sourcevar] = _split_rows(src_latent)

    return {"ref_tab": ref_tab, "source_tab": source_tab,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": {"transform": "cca", "comps": model.get("k", 0),
                     "center": model.get("center"),
                     "scale": model.get("scale"),
                     "shrink": model.get("shrink"),
                     "fit_by": model.get("fit_by"),
                     "unique_match_only": model.get("unique_match_only", False),
                     "matched_pairs": model.get("matched_pairs", 0),
                     "groups": model.get("groups", [])}}


# --- Geometric (contract/affine) fit/apply ---

def _fit_geometric_model(transform_fn, ref_tab, source_tab, match_on,
                         refvar, sourcevar, shrink=1e-6,
                         fit_by=None, unique_match_only=False, **_):
    name = getattr(transform_fn, "__name__", "")
    kind = "contract" if name == "contract_transform" else "affine"
    ref_keys, src_keys = _transform_group_keys(ref_tab, source_tab, fit_by)
    group_levels = list(dict.fromkeys([*ref_keys.tolist(), *src_keys.tolist()]))
    group_models = {}
    group_info = []

    for group_key in group_levels:
        ref_rows = np.flatnonzero(ref_keys == group_key)
        src_rows = np.flatnonzero(src_keys == group_key)
        group_label = "all" if group_key == "__all__" else group_key

        if len(ref_rows) == 0 or len(src_rows) == 0:
            group_info.append({
                "group": group_label,
                "matched_pairs": 0,
                "note": "missing group rows",
            })
            continue

        paired_ref, paired_src = _matched_transform_indices(
            ref_tab.iloc[ref_rows],
            source_tab.iloc[src_rows],
            match_on,
            unique_match_only=unique_match_only,
        )
        n_pairs = len(paired_src)
        if n_pairs < 1:
            group_info.append({
                "group": group_label,
                "matched_pairs": 0,
                "note": "no matched observations",
            })
            continue

        ref_densities = list(ref_tab.iloc[ref_rows[paired_ref]][refvar])
        src_densities = list(source_tab.iloc[src_rows[paired_src]][sourcevar])
        ref_mean, ref_cov = _aggregate_density_moments(ref_densities)
        src_mean, src_cov = _aggregate_density_moments(src_densities)

        if kind == "contract":
            s = np.sqrt((np.trace(ref_cov) + shrink) / (np.trace(src_cov) + shrink))
            A = s * np.eye(2)
            extra = {"scale": s}
        else:
            A = _mat_sqrt_2d(ref_cov + shrink * np.eye(2)) @ _mat_inv_sqrt_2d(
                src_cov + shrink * np.eye(2), shrink=shrink)
            extra = {}
        t = ref_mean - A @ src_mean
        group_models[group_key] = {
            "A": A,
            "t": t,
            "matched_pairs": n_pairs,
            **extra,
        }
        group_info.append({
            "group": group_label,
            "matched_pairs": n_pairs,
            "note": "ok",
            **extra,
        })

    model = {
        "transform": kind,
        "group_models": group_models,
        "fit_by": fit_by,
        "shrink": shrink,
        "unique_match_only": unique_match_only,
        "groups": group_info,
    }
    if "__all__" in group_models:
        model.update(group_models["__all__"])
    return model


def _apply_geometric_model(model, ref_tab, source_tab, refvar, sourcevar):
    _, src_keys = _transform_group_keys(ref_tab, source_tab, model.get("fit_by"))
    source_tab = source_tab.copy()
    new_densities = []
    for key, d in zip(src_keys, source_tab[sourcevar]):
        group_model = model.get("group_models", {}).get(key)
        if group_model is None and "A" in model and model.get("fit_by") is None:
            group_model = model
        if group_model is None:
            new_densities.append(d)
            continue
        A = group_model["A"]
        t = group_model["t"]
        if isinstance(d, EyeDensityMultiscale):
            new_densities.append(EyeDensityMultiscale(
                scales=[_warp_density(s, A, t) for s in d]))
        else:
            new_densities.append(_warp_density(d, A, t))
    source_tab[sourcevar] = new_densities
    info = {
        "transform": model["transform"],
        "shrink": model.get("shrink"),
        "fit_by": model.get("fit_by"),
        "unique_match_only": model.get("unique_match_only", False),
        "groups": model.get("groups", []),
    }
    if "A" in model:
        info["A"] = model["A"]
        info["t"] = model["t"]
    if "scale" in model:
        info["scale"] = model["scale"]
    return {"ref_tab": ref_tab, "source_tab": source_tab,
            "refvar": refvar, "sourcevar": sourcevar,
            "info": info}


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
        mx = dens.x.mean()
        my = dens.y.mean()
        return np.array([mx, my]), np.zeros((2, 2), dtype=float)
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
    """Aggregate per-object density moments.

    Returns (mean_xy [2], cov_xy [2,2]).
    """
    means = []
    covs = []
    for d in density_list:
        if isinstance(d, EyeDensityMultiscale):
            d = d[0]
        m, c = _density_moments(d)
        means.append(m)
        covs.append(c)
    means = np.array(means)
    agg_mean = means.mean(axis=0)
    agg_cov = np.zeros((2, 2), dtype=float)
    for m, c in zip(means, covs):
        d = m - agg_mean
        agg_cov += c + np.outer(d, d)
    agg_cov /= len(means)
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
    total = new_z.sum()
    if np.isfinite(total) and total > np.finfo(float).eps:
        new_z = new_z / total
    new_z = _zapsmall(new_z, digits=7)

    return EyeDensity(
        x=dens.x.copy(), y=dens.y.copy(), z=new_z,
        sigma=dens.sigma, fixgroup=dens.fixgroup,
    )


def _match_key(row, match_on):
    def normalize_value(value):
        return ("__NA__",) if pd.isna(value) else value

    if isinstance(match_on, str):
        return normalize_value(row[match_on])
    return tuple(normalize_value(row[col]) for col in match_on)


def _matched_transform_indices(ref_tab, source_tab, match_on, unique_match_only=False):
    """Return R-style matched row indices for transform fitting."""
    if match_on is None:
        n = min(len(ref_tab), len(source_tab))
        return np.arange(n, dtype=int), np.arange(n, dtype=int)
    match_cols = [match_on] if isinstance(match_on, str) else list(match_on)
    missing_ref = [col for col in match_cols if col not in ref_tab.columns]
    missing_src = [col for col in match_cols if col not in source_tab.columns]
    if missing_ref or missing_src:
        raise ValueError("match_on column must exist in both ref_tab and source_tab for latent transforms.")

    first_ref = {}
    for i, (_, row) in enumerate(ref_tab.iterrows()):
        key = _match_key(row, match_on)
        if key not in first_ref:
            first_ref[key] = i

    ref_idx = []
    src_idx = []
    seen = set()
    for j, (_, row) in enumerate(source_tab.iterrows()):
        key = _match_key(row, match_on)
        if key not in first_ref:
            continue
        if unique_match_only and key in seen:
            continue
        seen.add(key)
        ref_idx.append(first_ref[key])
        src_idx.append(j)
    return np.asarray(ref_idx, dtype=int), np.asarray(src_idx, dtype=int)


def _match_pairs(ref_tab, source_tab, match_on, refvar, sourcevar, unique_match_only=False):
    """Return matched lists of (ref_density, source_density) based on match_on."""
    ref_idx, src_idx = _matched_transform_indices(
        ref_tab, source_tab, match_on, unique_match_only=unique_match_only)
    return list(ref_tab.iloc[ref_idx][refvar]), list(source_tab.iloc[src_idx][sourcevar])


def contract_transform(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str | None = None,
    refvar: str = "density",
    sourcevar: str = "density",
    shrink: float = 1e-6,
    fit_by: str | list[str] | None = None,
    unique_match_only: bool = False,
    **kwargs,
) -> dict:
    """Uniform-scaling geometric transform matching spatial spread of source to reference.

    Fits a scalar scaling factor so that the trace of the source covariance
    matches the trace of the reference covariance, then applies the
    corresponding affine warp to the source densities.
    """
    model = _fit_geometric_model(
        contract_transform,
        ref_tab,
        source_tab,
        match_on,
        refvar,
        sourcevar,
        shrink=shrink,
        fit_by=fit_by,
        unique_match_only=unique_match_only,
    )
    return _apply_geometric_model(model, ref_tab, source_tab, refvar, sourcevar)


def affine_transform(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str | None = None,
    refvar: str = "density",
    sourcevar: str = "density",
    shrink: float = 1e-6,
    fit_by: str | list[str] | None = None,
    unique_match_only: bool = False,
    **kwargs,
) -> dict:
    """Full affine (rotation+scale+shear) geometric transform.

    Fits ``A = sqrtm(ref_cov + shrink*I) @ inv_sqrtm(src_cov + shrink*I)``
    and ``t = ref_mean - A @ src_mean``, then warps the source densities.
    """
    model = _fit_geometric_model(
        affine_transform,
        ref_tab,
        source_tab,
        match_on,
        refvar,
        sourcevar,
        shrink=shrink,
        fit_by=fit_by,
        unique_match_only=unique_match_only,
    )
    return _apply_geometric_model(model, ref_tab, source_tab, refvar, sourcevar)
