"""Latent-space transforms for template-based similarity (PCA, CORAL, CCA)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
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
