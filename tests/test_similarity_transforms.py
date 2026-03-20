"""Port of test_similarity_transforms.R"""

import numpy as np
import pandas as pd
import pytest
from peyesim import template_similarity
from peyesim.latent_transforms import (
    latent_pca_transform, coral_transform, cca_transform,
)
from peyesim.density import EyeDensity, EyeDensityMultiscale


def _stub(val):
    z = np.full((2, 2), val)
    return EyeDensity(x=np.array([1, 2], dtype=float),
                      y=np.array([1, 2], dtype=float), z=z, sigma=50)


def _vec_dens(vec):
    z = np.array(vec, dtype=float).reshape(2, 2)
    return EyeDensity(x=np.array([1, 2], dtype=float),
                      y=np.array([1, 2], dtype=float), z=z, sigma=50)


def test_latent_pca_produces_numeric_vectors():
    ref = pd.DataFrame({"id": [1, 2], "density": [_stub(1), _stub(2)]})
    src = pd.DataFrame({"id": [1, 2], "density": [_stub(1.1), _stub(1.9)]})
    res = latent_pca_transform(ref, src, match_on="id", comps=2)
    assert isinstance(res["ref_tab"]["density"].iloc[0], np.ndarray)
    assert len(res["ref_tab"]["density"].iloc[0]) == 2
    assert len(res["source_tab"]["density"].iloc[0]) == 2


def test_coral_transform_adapts():
    ref = pd.DataFrame({"id": [1, 2], "density": [_stub(1), _stub(2)]})
    src = pd.DataFrame({"id": [1, 2], "density": [_stub(1.1), _stub(1.9)]})
    res = coral_transform(ref, src, match_on="id", comps=2, shrink=1e-2)
    assert len(res["ref_tab"]["density"].iloc[0]) == 2
    assert len(res["source_tab"]["density"].iloc[0]) == 2


def test_template_similarity_accepts_transform_hook():
    ref = pd.DataFrame({"id": [1, 2], "density": [_stub(1), _stub(2)]})
    src = pd.DataFrame({"id": [1, 2], "density": [_stub(1.1), _stub(1.9)]})
    sim = template_similarity(
        ref, src, match_on="id", permutations=0, method="cosine",
        similarity_transform=latent_pca_transform,
        similarity_transform_args={"comps": 2},
    )
    assert "eye_sim" in sim.columns
    assert len(sim) == 2


def test_latent_transforms_reject_differing_multiscale():
    def _mk_ms(val, sizes):
        scales = []
        for s, n in zip([10, 20], sizes):
            z = np.full((n, n), val)
            scales.append(EyeDensity(x=np.arange(n, dtype=float),
                                     y=np.arange(n, dtype=float),
                                     z=z, sigma=s))
        return EyeDensityMultiscale(scales=scales)

    ms1 = _mk_ms(1, [2, 3])
    ms2 = _mk_ms(2, [2, 3])
    ref = pd.DataFrame({"id": [1], "density": [ms1]})
    src = pd.DataFrame({"id": [1], "density": [ms2]})
    with pytest.raises(ValueError, match="grid dimensions"):
        latent_pca_transform(ref, src, match_on="id")


def test_coral_improves_similarity_under_scaling():
    np.random.seed(123)
    n = 50
    base_vecs = [np.random.randn(4) for _ in range(n)]
    scale_mat = np.diag([2, 0.6, 1.7, 0.8])
    source_vecs = [scale_mat @ v for v in base_vecs]

    ref = pd.DataFrame({
        "id": np.arange(n),
        "density": [_vec_dens(v) for v in base_vecs],
    })
    src = pd.DataFrame({
        "id": np.arange(n),
        "density": [_vec_dens(v) for v in source_vecs],
    })

    transformed = coral_transform(ref, src, match_on="id", comps=4, shrink=1e-6)

    ref_mat = np.vstack(base_vecs)
    raw_src_mat = np.vstack(source_vecs)
    adapted_mat = np.vstack(transformed["source_tab"]["density"].values)

    cov_ref = np.cov(ref_mat, rowvar=False)
    cov_raw = np.cov(raw_src_mat, rowvar=False)
    cov_adapt = np.cov(adapted_mat, rowvar=False)

    frob = lambda m: np.sqrt((m ** 2).sum())
    assert frob(cov_adapt - cov_ref) < frob(cov_raw - cov_ref)


def test_cca_recovers_linear_mixing():
    np.random.seed(456)
    n = 50
    base_vecs = [np.random.uniform(0.5, 1.5, 4) for _ in range(n)]
    # Use a permutation matrix to scramble dimensions aggressively
    mix_mat = np.array([
        [0.1, 0.8, 0.0, 0.1],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.0, 0.1, 0.8],
        [0.1, 0.1, 0.8, 0.0],
    ])
    source_vecs = [mix_mat @ v for v in base_vecs]

    ref = pd.DataFrame({
        "id": np.arange(n),
        "density": [_vec_dens(v) for v in base_vecs],
    })
    src = pd.DataFrame({
        "id": np.arange(n),
        "density": [_vec_dens(v) for v in source_vecs],
    })

    raw = template_similarity(ref, src, match_on="id", permutations=0,
                              method="cosine")
    cca_res = template_similarity(
        ref, src, match_on="id", permutations=0, method="cosine",
        similarity_transform=cca_transform,
        similarity_transform_args={"comps": 4, "shrink": 1e-6},
    )

    assert cca_res["eye_sim"].mean() > raw["eye_sim"].mean()
