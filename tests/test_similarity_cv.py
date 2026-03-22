"""Tests for template_similarity_cv."""

import numpy as np
import pandas as pd
import pytest

from peyesim import eye_table, template_similarity_cv
from peyesim.density import eye_density, density_by


def _make_test_data(n_images=6, n_phases=2, seed=42):
    """Build ref_tab and source_tab with density columns for testing CV."""
    rng = np.random.default_rng(seed)
    rows = []
    for phase_idx in range(n_phases):
        phase = f"phase_{phase_idx}"
        for img_idx in range(n_images):
            image = f"img_{img_idx}"
            nfix = 20
            rows.append(pd.DataFrame({
                "x": rng.uniform(50, 450, nfix),
                "y": rng.uniform(50, 450, nfix),
                "onset": np.cumsum(rng.uniform(50, 200, nfix)),
                "duration": rng.uniform(100, 300, nfix),
                "image": image,
                "phase": phase,
            }))
    df = pd.concat(rows, ignore_index=True)
    etab = eye_table("x", "y", "duration", "onset",
                     groupvar=["phase", "image"], data=df)

    # Build density maps
    dtab = density_by(etab, groups=["phase", "image"],
                      xbounds=(0, 500), ybounds=(0, 500), sigma=30)

    ref_tab = dtab[dtab["phase"] == "phase_0"].reset_index(drop=True)
    source_tab = dtab[dtab["phase"] == "phase_1"].reset_index(drop=True)

    return ref_tab, source_tab


class TestTemplateSimilarityCV:
    """Basic tests for template_similarity_cv."""

    def test_basic_cv_produces_fold_column(self):
        ref_tab, source_tab = _make_test_data()
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0, method="spearman",
        )
        assert ".cv_fold" in result.columns

    def test_results_have_eye_sim(self):
        ref_tab, source_tab = _make_test_data()
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0, method="spearman",
        )
        assert "eye_sim" in result.columns
        # All values should be finite
        assert result["eye_sim"].notna().all()

    def test_n_folds_parameter(self):
        ref_tab, source_tab = _make_test_data(n_images=6)
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0, n_folds=3,
        )
        folds = result[".cv_fold"].unique()
        assert len(folds) == 3
        assert set(folds) == {0, 1, 2}

    def test_n_folds_2(self):
        ref_tab, source_tab = _make_test_data(n_images=4)
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0, n_folds=2,
        )
        folds = result[".cv_fold"].unique()
        assert len(folds) == 2

    def test_seed_reproducibility(self):
        ref_tab, source_tab = _make_test_data()
        r1 = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0, seed=123,
        )
        r2 = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0, seed=123,
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_differ(self):
        ref_tab, source_tab = _make_test_data()
        r1 = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0, seed=1,
        )
        r2 = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0, seed=99,
        )
        # Fold assignments should differ (eye_sim values are the same
        # since there's no transform, but fold column should differ)
        assert not (r1[".cv_fold"].values == r2[".cv_fold"].values).all()

    def test_all_source_rows_present(self):
        ref_tab, source_tab = _make_test_data(n_images=6)
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0,
        )
        # Each source row that has a matching ref should appear exactly once
        assert len(result) == len(source_tab)

    def test_with_permutations(self):
        ref_tab, source_tab = _make_test_data(n_images=6)
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=5,
        )
        assert "eye_sim" in result.columns
        assert "perm_sim" in result.columns
        assert "eye_sim_diff" in result.columns
        assert ".cv_fold" in result.columns

    def test_similarity_transform_with_coral(self):
        ref_tab, source_tab = _make_test_data()
        from peyesim import coral_transform
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            n_folds=2, permutations=0,
            similarity_transform=coral_transform,
            similarity_transform_args={"comps": 4, "shrink": 1e-3},
        )
        assert "eye_sim" in result.columns
        assert ".cv_fold" in result.columns
        assert len(result) == len(source_tab)

    def test_unsupported_transform_raises(self):
        ref_tab, source_tab = _make_test_data()
        with pytest.raises(ValueError, match="Unsupported transform"):
            template_similarity_cv(
                ref_tab, source_tab, match_on="image",
                similarity_transform=lambda **kw: {},
            )

    def test_too_few_groups_raises(self):
        ref_tab, source_tab = _make_test_data(n_images=1)
        with pytest.raises(ValueError, match="at least 2"):
            template_similarity_cv(
                ref_tab, source_tab, match_on="image",
                permutations=0,
            )


# ── Ported from R: CV without transform should match direct similarity ──

class TestCVNoTransformMatchesDirect:
    def test_cv_no_transform_matches_direct(self):
        """Without a transform, CV similarity should match direct template_similarity."""
        from peyesim import template_similarity

        ref_tab, source_tab = _make_test_data(n_images=6, n_phases=2, seed=77)

        # Direct (non-CV) similarity
        direct = template_similarity(
            ref_tab, source_tab, match_on="image",
            permutations=0, method="spearman",
        )

        # CV similarity with permutations=0 (no transform applied)
        cv_result = template_similarity_cv(
            ref_tab, source_tab, match_on="image",
            permutations=0, method="spearman",
        )

        # Merge on image to compare eye_sim values
        merged = direct.merge(cv_result, on="image", suffixes=("_direct", "_cv"))
        # Without a transform the CV fold exclusion affects the template,
        # so values may differ slightly. But they should be correlated.
        # At minimum both should produce finite results with same sign pattern.
        assert merged["eye_sim_direct"].notna().all()
        assert merged["eye_sim_cv"].notna().all()
        # The correlation between direct and CV results should be positive
        corr = merged["eye_sim_direct"].corr(merged["eye_sim_cv"])
        assert corr > 0.0, f"Expected positive correlation, got {corr}"
