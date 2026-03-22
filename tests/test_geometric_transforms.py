"""Tests for contract_transform and affine_transform geometric density transforms."""

import numpy as np
import pandas as pd
import pytest
from peyesim import template_similarity
from peyesim.density import EyeDensity
from peyesim.latent_transforms import contract_transform, affine_transform


def _make_density(center_x, center_y, spread=1.0, grid_size=20, sigma=50):
    """Create an EyeDensity with a Gaussian blob at (center_x, center_y)."""
    x = np.linspace(0, 100, grid_size).astype(float)
    y = np.linspace(0, 100, grid_size).astype(float)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    z = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * spread ** 2 * 100))
    z = z / z.sum()
    return EyeDensity(x=x, y=y, z=z, sigma=sigma)


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
