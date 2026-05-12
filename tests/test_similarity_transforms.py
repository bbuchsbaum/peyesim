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


def test_latent_transform_accepts_r_scale_dot_alias():
    ref = pd.DataFrame({
        "id": [1, 2, 3],
        "density": [
            _vec_dens([1, 2, 3, 4]),
            _vec_dens([2, 3, 4, 5]),
            _vec_dens([3, 4, 5, 6]),
        ],
    })
    src = pd.DataFrame({
        "id": [1, 2, 3],
        "density": [
            _vec_dens([1, 3, 5, 7]),
            _vec_dens([2, 4, 6, 8]),
            _vec_dens([3, 5, 7, 9]),
        ],
    })

    res = latent_pca_transform(ref, src, match_on="id", comps=2, **{"scale.": True})

    assert res["info"]["scale"] is True
    with pytest.raises(ValueError, match="scale"):
        latent_pca_transform(ref, src, match_on="id", comps=2, scale=True, **{"scale.": False})


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


def test_coral_transform_single_fit_by_group_matches_pooled():
    np.random.seed(111)
    n = 20
    base_vecs = [np.random.randn(4) for _ in range(n)]
    scale_mat = np.diag([1.7, 0.9, 1.2, 0.8])
    source_vecs = [scale_mat @ v for v in base_vecs]
    ref = pd.DataFrame({
        "id": np.arange(n),
        "pid": "p1",
        "density": [_vec_dens(v) for v in base_vecs],
    })
    src = pd.DataFrame({
        "id": np.arange(n),
        "pid": "p1",
        "density": [_vec_dens(v) for v in source_vecs],
    })

    pooled = coral_transform(ref, src, match_on="id", comps=4, shrink=1e-6)
    grouped = coral_transform(ref, src, match_on="id", comps=4, shrink=1e-6, fit_by="pid")

    np.testing.assert_allclose(
        np.vstack(grouped["ref_tab"]["density"]),
        np.vstack(pooled["ref_tab"]["density"]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.vstack(grouped["source_tab"]["density"]),
        np.vstack(pooled["source_tab"]["density"]),
        atol=1e-8,
    )


def test_coral_transform_grouped_fit_by_improves_group_covariance_alignment():
    np.random.seed(222)
    n_per_group = 30
    base_a = [np.random.randn(4) for _ in range(n_per_group)]
    base_b = [np.random.randn(4) for _ in range(n_per_group)]
    scale_a = np.diag([2.2, 0.7, 1.0, 1.3])
    scale_b = np.diag([0.8, 1.9, 1.4, 0.6])
    ref_vecs = base_a + base_b
    src_vecs = [scale_a @ v for v in base_a] + [scale_b @ v for v in base_b]
    pids = np.repeat(["p1", "p2"], n_per_group)
    ref = pd.DataFrame({
        "id": np.arange(2 * n_per_group),
        "pid": pids,
        "density": [_vec_dens(v) for v in ref_vecs],
    })
    src = pd.DataFrame({
        "id": np.arange(2 * n_per_group),
        "pid": pids,
        "density": [_vec_dens(v) for v in src_vecs],
    })

    pooled = coral_transform(ref, src, match_on="id", comps=4, shrink=1e-6)
    grouped = coral_transform(ref, src, match_on="id", comps=4, shrink=1e-6, fit_by="pid")

    def cov_gap(transformed_source):
        total = 0.0
        for pid in ("p1", "p2"):
            ref_mat = np.vstack(ref.loc[ref["pid"] == pid, "density"].map(lambda d: d.z.ravel()))
            src_mat = np.vstack(transformed_source.loc[transformed_source["pid"] == pid, "density"])
            total += np.sqrt(((np.cov(src_mat, rowvar=False) - np.cov(ref_mat, rowvar=False)) ** 2).sum())
        return total

    assert cov_gap(grouped["source_tab"]) < cov_gap(pooled["source_tab"])
    notes = {entry["group"]: entry["note"] for entry in grouped["info"]["groups"]}
    assert notes["p1"] == "ok"
    assert notes["p2"] == "ok"


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


def test_cca_transform_single_fit_by_group_matches_pooled():
    np.random.seed(333)
    n = 30
    base_vecs = [np.random.uniform(0.5, 1.5, 4) for _ in range(n)]
    mix_mat = np.array([
        [0.2, 0.7, 0.0, 0.1],
        [0.6, 0.2, 0.1, 0.1],
        [0.1, 0.0, 0.2, 0.7],
        [0.1, 0.1, 0.7, 0.1],
    ])
    source_vecs = [mix_mat @ v for v in base_vecs]
    ref = pd.DataFrame({
        "id": np.arange(n),
        "pid": "p1",
        "density": [_vec_dens(v) for v in base_vecs],
    })
    src = pd.DataFrame({
        "id": np.arange(n),
        "pid": "p1",
        "density": [_vec_dens(v) for v in source_vecs],
    })

    pooled = cca_transform(ref, src, match_on="id", comps=4, shrink=1e-6)
    grouped = cca_transform(ref, src, match_on="id", comps=4, shrink=1e-6, fit_by="pid")

    np.testing.assert_allclose(
        np.vstack(grouped["ref_tab"]["density"]),
        np.vstack(pooled["ref_tab"]["density"]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.vstack(grouped["source_tab"]["density"]),
        np.vstack(pooled["source_tab"]["density"]),
        atol=1e-8,
    )


def test_cca_transform_missing_fit_by_group_keeps_pca_projection():
    ref = pd.DataFrame({
        "id": [1, 2],
        "pid": ["p1", "p1"],
        "density": [_vec_dens([1, 2, 3, 4]), _vec_dens([2, 3, 4, 5])],
    })
    src = pd.DataFrame({
        "id": [1, 2, 3],
        "pid": ["p1", "p1", "p2"],
        "density": [
            _vec_dens([1.1, 2.2, 3.1, 4.2]),
            _vec_dens([2.1, 3.2, 4.1, 5.2]),
            _vec_dens([9, 8, 7, 6]),
        ],
    })

    pca = latent_pca_transform(ref, src, match_on="id", comps=3)
    grouped = cca_transform(ref, src, match_on="id", comps=3, fit_by="pid")

    np.testing.assert_allclose(
        grouped["source_tab"]["density"].iloc[2],
        pca["source_tab"]["density"].iloc[2],
        atol=1e-8,
    )
    notes = {entry["group"]: entry["note"] for entry in grouped["info"]["groups"]}
    assert notes["p2"] == "missing group rows"
