"""Regression tests for the R-parity fixes found by randomized differential testing.

Each test pins one behaviour of R eyesim 0.3.0 that peyesim previously got
wrong; the R value is quoted where it is a number. None of these needs R at
test time (``tests/test_parity_random_similarity.py`` re-checks them live).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from peyesim import (
    cca_transform,
    compute_similarity,
    contract_transform,
    fixation_group,
    fixation_similarity,
    latent_pca_transform,
    repetitive_similarity,
    scanpath,
    scanpath_similarity,
    similarity,
    template_multireg,
    template_regression,
    template_similarity,
    template_similarity_cv,
)
from peyesim._utils import match_keys
from peyesim.density import EyeDensityMultiscale, gen_density

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _d(v, nx=3, ny=2, sigma=None):
    d = gen_density(np.arange(1, nx + 1, dtype=float), np.arange(1, ny + 1, dtype=float),
                    np.asarray(v, dtype=float).reshape(ny, nx).T)
    d.sigma = sigma
    return d


D1 = _d([1, 5, 2, 8, 3, 3])
D2 = _d([2, 1, 4, 3, 9, 0])


# --- compute_similarity / similarity ---------------------------------------

def test_default_method_is_pearson_like_r():
    # R: similarity(d1, d2) dispatches to similarity.density(method = "pearson")
    assert similarity(D1, D2) == pytest.approx(-0.0918742562830411, abs=1e-14)
    assert compute_similarity(D1.z, D2.z) == pytest.approx(-0.0918742562830411, abs=1e-14)
    assert similarity([1, 2, 3, 5], [2, 1, 4, 4]) == pytest.approx(
        compute_similarity([1, 2, 3, 5], [2, 1, 4, 4], method="pearson"))


def test_jaccard_follows_proxy_ejaccard():
    # proxy::simil(method = "eJaccard") gives 1 for two zero vectors ...
    assert compute_similarity([0, 0, 0], [0, 0, 0], method="jaccard") == 1.0
    # ... and does not treat small magnitudes as zero (R: 0.875)
    assert compute_similarity([1e-9, 2e-9], [1e-9, 3e-9], method="jaccard") == pytest.approx(0.875)


def test_cosine_zero_rule_is_exact_zero_like_proxy():
    a, b = [1e-17, 0.0], [0.0, 1e-17]
    assert compute_similarity(a, b, method="cosine") == pytest.approx(0.0)
    assert compute_similarity(a, [1.0, 0.0], method="cosine") == pytest.approx(1.0)
    assert compute_similarity([0, 0], [0, 0], method="cosine") == 1.0


def test_multiscale_keeps_x_sigma_order_and_validates_aggregation():
    a = EyeDensityMultiscale(scales=[_d([1, 5, 2, 8, 3, 3], sigma=20.0), _d([2, 2, 3, 1, 0, 4], sigma=10.0)])
    b = EyeDensityMultiscale(scales=[_d([2, 1, 4, 3, 9, 0], sigma=10.0), _d([1, 3, 3, 2, 2, 5], sigma=20.0)])
    # R: c(sigma_20 = 0.0194920293896363, sigma_10 = -0.754005696373664)
    np.testing.assert_allclose(similarity(a, b, multiscale_aggregation="none"),
                               [0.0194920293896363, -0.754005696373664], atol=1e-14)
    np.testing.assert_allclose(similarity(a, b, multiscale_aggregation="NONE"),
                               similarity(a, b, multiscale_aggregation="none"))
    with pytest.raises(ValueError):
        similarity(a, b, multiscale_aggregation="max")


def test_scanpath_similarity_requires_screensize_like_r():
    fg = fixation_group(x=[1, 50, 90, 20], y=[5, 40, 10, 70], onset=[0, 100, 250, 400],
                        duration=[90, 140, 100, 80])
    sp = scanpath(fg)
    with pytest.raises(ValueError, match="screensize"):
        similarity(sp, sp)
    assert similarity(sp, sp, screensize=(100, 100))["mm_vector"] == pytest.approx(1.0)


# --- template_similarity -----------------------------------------------------

def test_multiscale_none_baseline_is_per_scale():
    ms = lambda v1, v2: EyeDensityMultiscale(scales=[_d(v1, sigma=20.0), _d(v2, sigma=10.0)])
    a = ms([1, 5, 2, 8, 3, 3], [2, 2, 3, 1, 0, 4])
    b = ms([2, 1, 4, 3, 9, 0], [1, 3, 3, 2, 2, 5])
    c = ms([3, 1, 4, 1, 5, 9], [2, 6, 5, 3, 5, 8])
    ref = pd.DataFrame({"k": [1, 2, 3], "density": [a, b, c]})
    src = pd.DataFrame({"k": [1, 2, 3], "density": [b, c, a]})
    res = template_similarity(ref, src, match_on="k", method="pearson", permutations=5,
                              multiscale_aggregation="none")
    # R: perm_sim = colMeans over the candidate templates, one value per scale
    # (it used to be a single scalar averaged over scales and candidates).
    sim = lambda r, s_: similarity(r, s_, multiscale_aggregation="none")
    np.testing.assert_allclose(res["eye_sim"][0], sim(a, b))
    np.testing.assert_allclose(res["perm_sim"][0], (sim(b, b) + sim(c, b)) / 2)
    np.testing.assert_allclose(res["eye_sim_diff"][0], sim(a, b) - (sim(b, b) + sim(c, b)) / 2)
    assert np.shape(res["perm_sim"][0]) == (2,)
    assert list(res["n_perm"]) == [2, 2, 2]


def test_missing_match_keys_match_each_other_like_r_match():
    assert match_keys(np.array([np.nan, 1.0, 2.0]), np.array([1.0, np.nan])) == [1, 0, None]
    ref = pd.DataFrame({"k": [1.0, np.nan], "density": [_d([1, 5, 2, 8, 3, 3]), _d([2, 1, 4, 3, 9, 0])]})
    src = pd.DataFrame({"k": [np.nan, 1.0], "density": [_d([2, 1, 4, 3, 9, 1]), _d([1, 5, 2, 8, 3, 4])]})
    res = template_similarity(ref, src, match_on="k", method="pearson", permutations=0)
    # R: c(0.992959788589177, 0.986627086327737)
    np.testing.assert_allclose(res["eye_sim"], [0.992959788589177, 0.986627086327737], atol=1e-14)


def test_missing_permute_on_stratum_has_no_candidates():
    maps = [_d([1, 5, 2, 8, 3, 3]), _d([2, 1, 4, 3, 9, 0]), _d([3, 1, 4, 1, 5, 9]), _d([2, 6, 5, 3, 5, 8])]
    ref = pd.DataFrame({"k": [1, 2, 3, 4], "p": ["a", None, "a", np.nan], "density": maps})
    src = ref.copy()
    src["density"] = maps[::-1]
    res = template_similarity(ref, src, match_on="k", permute_on="p", method="pearson", permutations=5)
    # R split() drops NA, so the two missing-stratum rows (None and NaN) get
    # no baseline instead of being pooled into one "nan" stratum.
    assert list(res["n_perm"]) == [1, 0, 1, 0]
    assert np.isnan(res["perm_sim"][1]) and np.isnan(res["perm_sim"][3])


def test_similarity_transform_may_be_named():
    rng = np.random.default_rng(3)
    maps = [_d(rng.random(6)) for _ in range(6)]
    ref = pd.DataFrame({"k": range(6), "density": maps})
    src = pd.DataFrame({"k": range(6), "density": [_d(m.z.T.ravel() + rng.random(6) * 0.1) for m in maps]})
    by_name = template_similarity(ref, src, match_on="k", method="cosine", permutations=0,
                                  similarity_transform="latent_pca_transform",
                                  similarity_transform_args={"comps": 2})
    by_fn = template_similarity(ref, src, match_on="k", method="cosine", permutations=0,
                                similarity_transform=latent_pca_transform,
                                similarity_transform_args={"comps": 2})
    np.testing.assert_allclose(by_name["eye_sim"], by_fn["eye_sim"])


def test_cosine_template_similarity_scores_single_component_vectors():
    # R's vectorised cosine path scores 1-D latent vectors as +/-1; the general
    # compute_similarity() rule (fewer than 2 points -> NA) does not apply.
    ref = pd.DataFrame({"k": [1, 2, 3], "density": [np.array([2.0]), np.array([-1.0]), np.array([0.5])]})
    src = pd.DataFrame({"k": [1, 2, 3], "density": [np.array([3.0]), np.array([1.0]), np.array([0.0])]})
    res = template_similarity(ref, src, match_on="k", method="cosine", permutations=10)
    np.testing.assert_allclose(res["eye_sim"], [1.0, -1.0, 0.0])
    np.testing.assert_allclose(res["perm_sim"], [0.0, 1.0, 0.0])
    zero = pd.DataFrame({"k": [1, 2], "density": [_d([0] * 6), _d([0] * 6)]})
    assert list(template_similarity(zero, zero, match_on="k", method="cosine",
                                    permutations=0)["eye_sim"]) == [1.0, 1.0]


def test_template_similarity_validates_method_like_match_arg():
    ref = pd.DataFrame({"k": [1, 2], "density": [D1, D2]})
    with pytest.raises(ValueError):
        template_similarity(ref, ref, match_on="k", method="multimatch", permutations=0)
    with pytest.raises(ValueError):
        fixation_similarity(ref, ref, match_on="k", method="pearson")


# --- latent transforms -------------------------------------------------------

def _latent_tables(seed=1):
    rng = np.random.default_rng(seed)
    z = rng.random((8, 12))
    z2 = z + 0.3 * rng.random((8, 12))
    mk = lambda v: gen_density(np.arange(1, 5.0), np.arange(1, 4.0), v.reshape(3, 4).T)
    ref = pd.DataFrame({"k": range(1, 9), "density": [mk(z[i]) for i in range(8)]})
    src = pd.DataFrame({"k": range(1, 9), "density": [mk(z2[i]) for i in range(8)]})
    return ref, src


def _cos(transform, **args):
    ref, src = _latent_tables()
    res = template_similarity(ref, src, match_on="k", method="cosine", permutations=0,
                              similarity_transform=transform, similarity_transform_args=args)
    return res["eye_sim"].to_numpy()


def test_pca_honours_center_false_and_scale_like_prcomp():
    # R values from template_similarity(..., latent_pca_transform / cca_transform)
    np.testing.assert_allclose(
        _cos(latent_pca_transform, comps=3, center=False),
        [0.996079, 0.992105, 0.999949, 0.997772, 0.998712, 0.993102, 0.996987, 0.986672], atol=1e-6)
    np.testing.assert_allclose(
        _cos(latent_pca_transform, comps=3, center=False, **{"scale.": True}),
        [0.994857, 0.99359, 0.999553, 0.998372, 0.997558, 0.995327, 0.997375, 0.988202], atol=1e-6)
    np.testing.assert_allclose(
        _cos(cca_transform, comps=3, **{"scale.": True}),
        [0.958095, 0.984757, 0.956427, 0.933618, 0.993187, 0.998633, 0.998624, 0.998999], atol=1e-6)


def test_pca_scale_refuses_constant_columns_like_prcomp():
    ref, src = _latent_tables()
    for tab in (ref, src):
        tab["density"] = [gen_density(d.x, d.y, np.where(np.arange(3) == 0, 0.0, d.z)) for d in tab["density"]]
    with pytest.raises(ValueError, match="cannot rescale"):
        latent_pca_transform(ref, src, comps=2, **{"scale.": True})


def test_missing_fit_by_stratum_is_not_fitted():
    ref, src = _latent_tables()
    ref["g"] = ["a", "b"] * 4
    src["g"] = ["a", "b"] * 4
    src.loc[0, "g"] = None
    out = contract_transform(ref, src, match_on="k", fit_by="g")
    # R: an NA stratum never gets a model, so its row passes through unwarped
    np.testing.assert_array_equal(out["source_tab"]["density"][0].z, src["density"][0].z)
    assert not np.array_equal(out["source_tab"]["density"][1].z, src["density"][1].z)


def test_cv_rows_with_missing_split_key_are_not_scored():
    ref, src = _latent_tables()
    src["grp"] = ["s0", "s1", None, "s2", "s0", "s1", "s2", "s3"]
    res = template_similarity_cv(ref, src, match_on="k", split_on="grp", method="pearson",
                                 permutations=0, n_folds=2, seed=3)
    assert 3 not in set(res["k"]) and len(res) == 7


# --- regression --------------------------------------------------------------

def _reg_tables(base_z, src_z):
    g = np.arange(4.0), np.arange(3.0)
    ref = pd.DataFrame({"k": [1], "density": [gen_density(*g, np.linspace(0, 1, 12).reshape(4, 3) ** 2)]})
    src = pd.DataFrame({"k": [1], "b": ["x"], "density": [gen_density(*g, src_z)]})
    base = pd.DataFrame({"b": ["x"], "density": [gen_density(*g, base_z)]})
    return ref, src, base


def test_lm_reports_aliased_baseline_as_nan_like_r():
    rng = np.random.default_rng(4)
    ref, src, base = _reg_tables(np.full((4, 3), 0.5), rng.random((4, 3)))
    res = template_regression(ref, src, "k", base, "b", method="lm")
    assert np.isnan(res["beta_baseline"][0]) and np.isfinite(res["beta_source"][0])
    with pytest.raises(ValueError, match="singular"):
        template_regression(ref, src, "k", base, "b", method="rlm")
    # ppcor::pcor(): the constant baseline's partial correlation is NA (not an error)
    rank = template_regression(ref, src, "k", base, "b", method="rank")
    assert np.isnan(rank["beta_baseline"][0]) and np.isfinite(rank["beta_source"][0])


def test_lm_drops_missing_cells_like_na_omit():
    rng = np.random.default_rng(5)
    base_z, src_z = rng.random((4, 3)), rng.random((4, 3))
    ref, src, base = _reg_tables(base_z, src_z)
    with_nan = src_z.copy()
    with_nan[1, 2] = np.nan
    _, src_nan, _ = _reg_tables(base_z, with_nan)
    res = template_regression(ref, src_nan, "k", base, "b", method="lm")
    keep = ~np.isnan(with_nan.ravel())
    X = np.column_stack([np.ones(keep.sum()), base_z.ravel()[keep], ref["density"][0].z.ravel()[keep]])
    coef = np.linalg.lstsq(X, with_nan.ravel()[keep], rcond=None)[0]
    np.testing.assert_allclose([res["beta_baseline"][0], res["beta_source"][0]], coef[1:])


def test_multireg_aliased_covariate_and_tidy_columns_like_r():
    rng = np.random.default_rng(2)
    z = rng.random((4, 12))
    z[3] = 0.5
    mk = lambda v: gen_density(np.arange(1, 5.0), np.arange(1, 4.0), v.reshape(3, 4).T)
    tab = pd.DataFrame({"id": [1], "resp": [mk(z[0])], "c1": [mk(z[2])], "c2": [mk(z[3])]})
    fit = template_multireg(tab, "resp", ["c1", "c2"], method="lm")["multireg"][0]
    # R: (Intercept) 0.112046702292744, c1 -0.344560427512928, c2 NA
    np.testing.assert_allclose(fit["estimate"][:2], [0.112046702292744, -0.344560427512928], atol=1e-13)
    assert np.isnan(fit["estimate"][2])
    logit = template_multireg(tab, "resp", ["c1", "c2"], method="logistic")["multireg"][0]
    assert np.isnan(logit["estimate"][2])
    tab["c1"] = [mk(z[1])]
    tab["c2"] = [mk(z[2])]
    rlm = template_multireg(tab, "resp", ["c1", "c2"], method="rlm")["multireg"][0]
    # broom::tidy(rlm) has no p.value; R estimates (maxit = 20 IRLS)
    assert list(rlm.columns) == ["term", "estimate", "std.error", "statistic"]
    np.testing.assert_allclose(rlm["estimate"], [0.0504763187707245, 0.585131080821409, -0.254593693753098],
                               atol=1e-12)


# --- repetitive_similarity -------------------------------------------------

def test_missing_condition_joins_neither_set():
    maps = [_d([1, 5, 2, 8, 3, 3]), _d([2, 1, 4, 3, 9, 0]), _d([3, 1, 4, 1, 5, 9]), _d([2, 6, 5, 3, 5, 8])]
    tab = pd.DataFrame({"cond": ["a", "a", None, "b"], "density": maps})
    res = repetitive_similarity(tab, condition_var="cond", method="pearson")
    assert np.isnan(res["repsim"][2]) and np.isnan(res["othersim"][2])
    # row 0's "other" set is row 3 only (R: NA != "a" is NA, not TRUE)
    assert res["othersim"][0] == pytest.approx(similarity(maps[0], maps[3]))


def test_scanpath_similarity_table_needs_screensize():
    fg = scanpath(fixation_group(x=[1, 50, 90, 20], y=[5, 40, 10, 70], onset=[0, 100, 250, 400],
                                 duration=[90, 140, 100, 80]))
    tab = pd.DataFrame({"trial": [1], "scanpath": [fg]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="screensize"):
            scanpath_similarity(tab, tab, match_on="trial")
