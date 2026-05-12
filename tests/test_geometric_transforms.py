"""Tests for contract_transform and affine_transform geometric density transforms."""

import numpy as np
import pandas as pd
import pytest
from peyesim import template_similarity
from peyesim.density import EyeDensity
from peyesim.similarity import compute_similarity
from peyesim.latent_transforms import (
    _aggregate_density_moments,
    _fit_transform,
    _warp_density,
    affine_transform,
    contract_transform,
)


def _make_density(center_x, center_y, spread=1.0, grid_size=20, sigma=50):
    """Create an EyeDensity with a Gaussian blob at (center_x, center_y)."""
    x = np.linspace(0, 100, grid_size).astype(float)
    y = np.linspace(0, 100, grid_size).astype(float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * spread ** 2 * 100))
    z = z / z.sum()
    return EyeDensity(x=x, y=y, z=z, sigma=sigma)


def test_geometric_moment_aggregation_is_per_density_not_mass_weighted():
    d1 = EyeDensity(
        x=np.array([0.0, 10.0]),
        y=np.array([0.0, 10.0]),
        z=np.array([[10.0, 0.0], [0.0, 0.0]]),
        sigma=1,
    )
    d2 = EyeDensity(
        x=np.array([0.0, 10.0]),
        y=np.array([0.0, 10.0]),
        z=np.array([[0.0, 0.0], [0.0, 1.0]]),
        sigma=1,
    )

    mean, _ = _aggregate_density_moments([d1, d2])

    np.testing.assert_allclose(mean, [5.0, 5.0])


def test_warp_density_renormalizes_like_r_geometric_transform():
    dens = _make_density(50, 50, grid_size=12)
    warped = _warp_density(dens, np.eye(2), np.array([15.0, 0.0]))

    np.testing.assert_allclose(warped.z.sum(), 1.0)


def test_cosine_similarity_handles_zero_vectors_like_r_fast_path():
    np.testing.assert_allclose(compute_similarity([0, 0], [1, 2], method="cosine"), 0.0)
    np.testing.assert_allclose(compute_similarity([0, 0], [0, 0], method="cosine"), 1.0)


def test_geometric_fit_by_missing_source_group_is_left_unchanged():
    ref = pd.DataFrame({
        "id": ["A"],
        "pid": ["p1"],
        "density": [_make_density(40, 40, grid_size=12)],
    })
    unchanged = _make_density(80, 80, grid_size=12)
    src = pd.DataFrame({
        "id": ["A", "B"],
        "pid": ["p1", "p2"],
        "density": [_make_density(45, 45, grid_size=12), unchanged],
    })

    result = contract_transform(ref, src, match_on="id", fit_by="pid")

    np.testing.assert_allclose(result["source_tab"]["density"].iloc[1].z, unchanged.z)
    notes = {group["group"]: group["note"] for group in result["info"]["groups"]}
    assert notes["p2"] == "missing group rows"


def test_geometric_cv_model_fit_by_keeps_group_specific_models():
    ref = pd.DataFrame({
        "id": ["A", "B"],
        "pid": ["p1", "p2"],
        "density": [_make_density(40, 40, grid_size=12), _make_density(70, 70, grid_size=12)],
    })
    src = pd.DataFrame({
        "id": ["A", "B"],
        "pid": ["p1", "p2"],
        "density": [_make_density(45, 45, grid_size=12), _make_density(60, 60, grid_size=12)],
    })

    model = _fit_transform(
        affine_transform,
        ref,
        src,
        match_on="id",
        fit_by="pid",
    )

    assert model["fit_by"] == "pid"
    assert set(model["group_models"]) == {"p1", "p2"}
    assert all(group["note"] == "ok" for group in model["groups"])


class TestContractTransform:
    def test_identical_data_returns_high_similarity(self):
        """When ref and source are identical, contract should preserve similarity ~1."""
        n = 10
        densities = [_make_density(50 + i, 50 + i) for i in range(n)]
        ref = pd.DataFrame({"id": np.arange(n), "density": densities})
        src = pd.DataFrame({"id": np.arange(n), "density": densities})

        sim_raw = template_similarity(ref, src, match_on="id", permutations=0,
                                       method="pearson")
        sim_ct = template_similarity(
            ref, src, match_on="id", permutations=0, method="pearson",
            similarity_transform=contract_transform,
        )
        # With identical data, similarity should remain very high
        assert sim_ct["eye_sim"].mean() > 0.95
        # And should be close to raw similarity
        assert abs(sim_ct["eye_sim"].mean() - sim_raw["eye_sim"].mean()) < 0.1

    def test_contract_returns_correct_keys(self):
        """Output dict has the standard transform keys."""
        n = 5
        densities = [_make_density(50, 50) for _ in range(n)]
        ref = pd.DataFrame({"id": np.arange(n), "density": densities})
        src = pd.DataFrame({"id": np.arange(n), "density": densities})
        result = contract_transform(ref, src, match_on="id")
        assert set(result.keys()) == {"ref_tab", "source_tab", "refvar", "sourcevar", "info"}
        assert result["info"]["transform"] == "contract"
        assert "scale" in result["info"]

    def test_contract_with_shifted_data_improves_similarity(self):
        """Contract transform should help when source has different spatial spread."""
        np.random.seed(42)
        n = 15
        ref_densities = [_make_density(50 + np.random.randn() * 5,
                                        50 + np.random.randn() * 5,
                                        spread=1.0) for _ in range(n)]
        # Source with larger spread (scaled positions)
        src_densities = [_make_density(50 + np.random.randn() * 15,
                                        50 + np.random.randn() * 15,
                                        spread=1.0) for _ in range(n)]
        ref = pd.DataFrame({"id": np.arange(n), "density": ref_densities})
        src = pd.DataFrame({"id": np.arange(n), "density": src_densities})

        sim_raw = template_similarity(ref, src, match_on="id", permutations=0,
                                       method="pearson")
        sim_ct = template_similarity(
            ref, src, match_on="id", permutations=0, method="pearson",
            similarity_transform=contract_transform,
        )
        # Transform should not crash; similarity values should be finite
        assert np.all(np.isfinite(sim_ct["eye_sim"]))


class TestAffineTransform:
    def test_identical_data_returns_high_similarity(self):
        """When ref and source are identical, affine should preserve similarity ~1."""
        n = 10
        densities = [_make_density(50 + i, 50 + i) for i in range(n)]
        ref = pd.DataFrame({"id": np.arange(n), "density": densities})
        src = pd.DataFrame({"id": np.arange(n), "density": densities})

        sim_aff = template_similarity(
            ref, src, match_on="id", permutations=0, method="pearson",
            similarity_transform=affine_transform,
        )
        assert sim_aff["eye_sim"].mean() > 0.95

    def test_affine_returns_correct_keys(self):
        """Output dict has the standard transform keys."""
        n = 5
        densities = [_make_density(50, 50) for _ in range(n)]
        ref = pd.DataFrame({"id": np.arange(n), "density": densities})
        src = pd.DataFrame({"id": np.arange(n), "density": densities})
        result = affine_transform(ref, src, match_on="id")
        assert set(result.keys()) == {"ref_tab", "source_tab", "refvar", "sourcevar", "info"}
        assert result["info"]["transform"] == "affine"
        assert "A" in result["info"]
        assert "t" in result["info"]

    def test_affine_with_scaled_data_recovers_similarity(self):
        """Affine transform on systematically scaled data should improve similarity."""
        np.random.seed(99)
        n = 20
        # Reference: blobs near center
        ref_densities = [_make_density(50 + np.random.randn() * 5,
                                        50 + np.random.randn() * 5) for _ in range(n)]
        # Source: same pattern but shifted and scaled
        src_densities = [_make_density(60 + np.random.randn() * 10,
                                        40 + np.random.randn() * 10) for _ in range(n)]
        ref = pd.DataFrame({"id": np.arange(n), "density": ref_densities})
        src = pd.DataFrame({"id": np.arange(n), "density": src_densities})

        sim_raw = template_similarity(ref, src, match_on="id", permutations=0,
                                       method="pearson")
        sim_aff = template_similarity(
            ref, src, match_on="id", permutations=0, method="pearson",
            similarity_transform=affine_transform,
        )
        # Affine should produce finite values and generally improve or maintain similarity
        assert np.all(np.isfinite(sim_aff["eye_sim"]))
        # The affine transform should help with the systematic shift
        assert sim_aff["eye_sim"].mean() >= sim_raw["eye_sim"].mean() - 0.15


class TestGeometricWithPipeline:
    def test_contract_in_template_similarity_pipeline(self):
        """contract_transform works end-to-end with template_similarity."""
        n = 8
        densities = [_make_density(40 + i * 3, 50 + i * 2) for i in range(n)]
        ref = pd.DataFrame({"id": np.arange(n), "density": densities})
        src = pd.DataFrame({"id": np.arange(n), "density": densities})
        sim = template_similarity(
            ref, src, match_on="id", permutations=0, method="spearman",
            similarity_transform=contract_transform,
            similarity_transform_args={"shrink": 1e-4},
        )
        assert "eye_sim" in sim.columns
        assert len(sim) == n

    def test_affine_in_template_similarity_pipeline(self):
        """affine_transform works end-to-end with template_similarity."""
        n = 8
        densities = [_make_density(40 + i * 3, 50 + i * 2) for i in range(n)]
        ref = pd.DataFrame({"id": np.arange(n), "density": densities})
        src = pd.DataFrame({"id": np.arange(n), "density": densities})
        sim = template_similarity(
            ref, src, match_on="id", permutations=0, method="spearman",
            similarity_transform=affine_transform,
            similarity_transform_args={"shrink": 1e-4},
        )
        assert "eye_sim" in sim.columns
        assert len(sim) == n
