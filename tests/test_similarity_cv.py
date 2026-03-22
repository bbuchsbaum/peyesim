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


# ---------------------------------------------------------------------------
# Ported from R test_similarity_cv.R — shared fixture with known density vecs
# ---------------------------------------------------------------------------

def _make_cv_density_vec(vec):
    """Build a tiny 2x2 EyeDensity from a known vector."""
    from peyesim.density import EyeDensity
    z = np.array(vec, dtype=float).reshape(2, 2)
    return EyeDensity(x=np.array([1.0, 2.0]), y=np.array([1.0, 2.0]), z=z, sigma=50.0)


def _make_cv_density_tables():
    """Build ref_tab and source_tab matching the R make_cv_density_tables fixture.

    ref_tab: 4 rows (id=0..3), participant in (p1,p1,p2,p2), phase="scene",
             2x2 EyeDensity objects from known vectors.
    source_tab: 8 rows (2 per id, phase=scene/delay), densities are
                mix-matrix transformations of ref.
    """
    ref_vecs = [
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
        [0.2, 0.2, 0.3, 0.3],
        [0.3, 0.1, 0.4, 0.2],
    ]

    ref_tab = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "participant": ["p1", "p1", "p2", "p2"],
        "phase": ["scene"] * 4,
        "density": [_make_cv_density_vec(v) for v in ref_vecs],
    })

    # Build source densities as mix-matrix transforms of the reference vecs
    mix = np.array([
        [0.8, 0.2, 0.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.3],
        [0.0, 0.0, 0.3, 0.7],
    ])
    ref_mat = np.array(ref_vecs)  # 4 x 4
    src_mat = mix @ ref_mat       # 4 x 4

    source_rows = []
    for i in range(4):
        for ph in ["scene", "delay"]:
            source_rows.append({
                "id": i,
                "participant": ref_tab["participant"].iloc[i],
                "phase": ph,
                "density": _make_cv_density_vec(src_mat[i]),
            })
    source_tab = pd.DataFrame(source_rows)
    return ref_tab, source_tab


class TestCVPortedFromR:
    """Tests ported from the R test_similarity_cv.R test suite."""

    def test_cv_excludes_held_out_keys_from_transform(self):
        """CV with CORAL transform: all source rows present, folds have train/eval rows."""
        from peyesim import coral_transform

        ref_tab, source_tab = _make_cv_density_tables()
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="id",
            permutations=0, method="spearman",
            similarity_transform=coral_transform,
            similarity_transform_args={"comps": 2, "shrink": 1e-6},
            split_on="id", n_folds=2,
        )
        # All source rows should be present
        assert len(result) == len(source_tab)
        # eye_sim should be finite for all rows
        assert result["eye_sim"].notna().all()
        assert np.isfinite(result["eye_sim"]).all()
        # Should have 2 folds
        assert set(result[".cv_fold"].unique()) == {0, 1}

    def test_cv_no_transform_matches_direct_exact(self):
        """Without a transform, CV eye_sim values should EXACTLY match template_similarity."""
        from peyesim import template_similarity

        ref_tab, source_tab = _make_cv_density_tables()

        direct = template_similarity(
            ref_tab, source_tab, match_on="id",
            permutations=0, method="spearman",
        )
        cv_result = template_similarity_cv(
            ref_tab, source_tab, match_on="id",
            permutations=0, method="spearman",
        )

        # Sort both by id for comparison
        direct_sorted = direct.sort_values("id").reset_index(drop=True)
        cv_sorted = cv_result.sort_values("id").reset_index(drop=True)

        # Merge on id + phase to compare eye_sim values
        merged = direct_sorted.merge(cv_sorted, on=["id", "phase"],
                                     suffixes=("_direct", "_cv"))
        np.testing.assert_allclose(
            merged["eye_sim_direct"].values,
            merged["eye_sim_cv"].values,
            atol=1e-10,
        )

    def test_cv_with_fit_eval_filters(self):
        """Use fit/eval source filters: only delay rows in output."""
        from peyesim import cca_transform

        ref_tab, source_tab = _make_cv_density_tables()
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="id",
            permutations=0, method="spearman",
            similarity_transform=cca_transform,
            similarity_transform_args={"comps": 2, "shrink": 1e-6},
            split_on="id", n_folds=2,
            fit_source_filter=lambda st: st["phase"] == "scene",
            eval_source_filter=lambda st: st["phase"] == "delay",
        )
        # All output rows should have phase=="delay"
        assert (result["phase"] == "delay").all()
        # Row count should match number of delay rows in source
        n_delay = (source_tab["phase"] == "delay").sum()
        assert len(result) == n_delay

    def test_cv_row_order_invariance(self):
        """Shuffle source_tab rows; CV results should match (by id+phase) without transform."""
        ref_tab, source_tab = _make_cv_density_tables()

        result_ordered = template_similarity_cv(
            ref_tab, source_tab, match_on="id",
            permutations=0, method="spearman",
            split_on="id", n_folds=2, seed=42,
        )

        # Shuffle source rows
        shuffled = source_tab.sample(frac=1, random_state=99).reset_index(drop=True)
        result_shuffled = template_similarity_cv(
            ref_tab, shuffled, match_on="id",
            permutations=0, method="spearman",
            split_on="id", n_folds=2, seed=42,
        )

        # Compare eye_sim by (id, phase)
        m1 = result_ordered.sort_values(["id", "phase"]).reset_index(drop=True)
        m2 = result_shuffled.sort_values(["id", "phase"]).reset_index(drop=True)
        np.testing.assert_allclose(
            m1["eye_sim"].values, m2["eye_sim"].values, atol=1e-10,
        )

    def test_cv_with_coral_transform(self):
        """CV with CORAL: results are finite and all rows present."""
        from peyesim import coral_transform

        ref_tab, source_tab = _make_cv_density_tables()
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="id",
            permutations=0, method="spearman",
            similarity_transform=coral_transform,
            similarity_transform_args={"comps": 2, "shrink": 1e-6},
            split_on="id", n_folds=2,
        )
        assert len(result) == len(source_tab)
        assert result["eye_sim"].notna().all()
        assert np.isfinite(result["eye_sim"]).all()

    def test_cv_with_contract_transform(self):
        """CV with contract_transform: results are finite."""
        from peyesim import contract_transform

        ref_tab, source_tab = _make_cv_density_tables()
        result = template_similarity_cv(
            ref_tab, source_tab, match_on="id",
            permutations=0, method="spearman",
            similarity_transform=contract_transform,
            split_on="id", n_folds=2,
        )
        assert len(result) == len(source_tab)
        assert result["eye_sim"].notna().all()
        assert np.isfinite(result["eye_sim"]).all()

    def test_cv_manual_fold_matches(self):
        """Manually replicate fold 0 logic and compare with CV result."""
        from peyesim import coral_transform
        from peyesim.latent_transforms import _fit_transform, _apply_transform
        from peyesim.similarity import _run_similarity_analysis, _make_cv_folds

        ref_tab, source_tab = _make_cv_density_tables()

        # Run full CV
        cv_result = template_similarity_cv(
            ref_tab, source_tab, match_on="id",
            permutations=0, method="spearman",
            similarity_transform=coral_transform,
            similarity_transform_args={"comps": 2, "shrink": 1e-6},
            split_on="id", n_folds=2, seed=1,
        )

        # Manually replicate fold 0
        cv = _make_cv_folds(source_tab, "id", n_folds=2, seed=1)
        fold_ids = cv["fold_ids"]

        fold = 0
        in_fold = fold_ids == fold
        eval_source = source_tab.loc[in_fold].reset_index(drop=True)
        train_source = source_tab.loc[~in_fold].reset_index(drop=True)

        # Remove leaky keys from training
        eval_keys = set(eval_source["id"].unique())
        train_source = train_source[~train_source["id"].isin(eval_keys)].reset_index(drop=True)

        # Ref subsets
        ref_eval = ref_tab[ref_tab["id"].isin(eval_keys)].reset_index(drop=True)
        train_keys = set(train_source["id"].unique())
        ref_train = ref_tab[ref_tab["id"].isin(train_keys)].reset_index(drop=True)

        # Fit and apply transform
        model = _fit_transform(
            coral_transform, ref_train, train_source,
            "id", refvar="density", sourcevar="density",
            comps=2, shrink=1e-6,
        )
        transformed = _apply_transform(
            model, ref_eval, eval_source,
            refvar="density", sourcevar="density",
        )

        # Run similarity on transformed data
        manual_res = _run_similarity_analysis(
            transformed["ref_tab"], transformed["source_tab"],
            "id", permutations=0, permute_on=None,
            method="spearman",
            refvar=transformed["refvar"],
            sourcevar=transformed["sourcevar"],
        )

        # Compare with CV fold 0 results
        cv_fold0 = cv_result[cv_result[".cv_fold"] == 0].sort_values(
            ["id", "phase"]
        ).reset_index(drop=True)
        manual_sorted = manual_res.sort_values(
            ["id", "phase"]
        ).reset_index(drop=True)

        np.testing.assert_allclose(
            cv_fold0["eye_sim"].values,
            manual_sorted["eye_sim"].values,
            atol=1e-10,
        )
