"""Regressions for the eyesim 0.3.0 pre-CRAN fixes in latent transforms,
MultiMatch and template regression."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from peyesim import (
    cca_transform,
    coral_transform,
    fixation_group,
    multi_match,
    scanpath,
    template_multireg,
    template_regression,
    template_sample,
)
from peyesim.density import gen_density
from peyesim.latent_transforms import _warp_density, _zapsmall


def _map(k, shift=0.0, n=4):
    i = np.arange(1, n + 1)[:, None]
    j = np.arange(1, n + 1)[None, :]
    grid = np.arange(n, dtype=float)
    return gen_density(grid, grid, 1.5 + np.sin(0.7 * k + 1.3 * i + 0.4 * j * k + shift))


def _tables(n=8):
    ids = [f"i{k}" for k in range(1, n + 1)]
    ref = pd.DataFrame({"id": ids, "pid": "p1", "density": [_map(k) for k in range(1, n + 1)]})
    src = pd.DataFrame({"id": ids, "pid": "p1", "density": [_map(k, 0.35) for k in range(1, n + 1)]})
    return ref, src


def test_coral_gives_adapted_source_the_reference_covariance():
    ref, src = _tables()
    out = coral_transform(ref, src, comps=3, shrink=1e-8)
    ref_scores = np.vstack(out["ref_tab"]["density"].to_list())
    src_scores = np.vstack(out["source_tab"]["density"].to_list())
    np.testing.assert_allclose(np.cov(src_scores, rowvar=False), np.cov(ref_scores, rowvar=False),
                               rtol=1e-5, atol=1e-8)


def test_cca_rows_are_canonical_variates_then_zeros():
    ref, src = _tables()
    out = cca_transform(ref, src, match_on="id", comps=5, shrink=1e-3)
    pca_width = 5
    for col in ("density",):
        scores = np.vstack(out["source_tab"][col].to_list())
        assert scores.shape[1] == pca_width
        k = out["info"]["groups"][0]["comps"] if "info" in out else None
        nonzero = np.flatnonzero(np.abs(scores).sum(axis=0) > 0)
        assert nonzero.max() < pca_width
        assert np.all(scores[:, nonzero.max() + 1:] == 0)
        if k is not None:
            assert nonzero.size == k


def test_cca_single_component_model():
    ref, src = _tables()
    out = cca_transform(ref, src, match_on="id", comps=1, shrink=1e-3)
    scores = np.vstack(out["source_tab"]["density"].to_list())
    assert scores.shape == (8, 1)
    assert np.all(np.isfinite(scores))


def test_coral_single_component_model():
    ref, src = _tables()
    out = coral_transform(ref, src, comps=1, shrink=1e-3)
    scores = np.vstack(out["source_tab"]["density"].to_list())
    assert scores.shape == (8, 1)
    assert np.all(np.isfinite(scores))


def test_cca_warns_for_source_group_without_model():
    ref, src = _tables()
    ref["pid"] = ["p1"] * 4 + ["p2"] * 4
    src["pid"] = ["p1"] * 4 + ["p2"] * 2 + ["p3"] * 2
    with pytest.warns(UserWarning, match=r"no CCA model for group\(s\) p3; their rows keep PCA scores"):
        cca_transform(ref, src, match_on="id", comps=3, fit_by="pid")


def test_zapsmall_matches_r_fixed_precision():
    z = np.array([1e-4, 3e-12, 2.123456789e-5])
    # 7 - log10(1e-4) = 11 decimal places
    np.testing.assert_array_equal(_zapsmall(z), np.round(z, 11))
    assert _zapsmall(np.zeros(3)).tolist() == [0, 0, 0]


def test_warp_density_zaps_negligible_mass():
    dens = _map(1)
    out = _warp_density(dens, np.eye(2) * 1.01, np.array([0.003, 0.0]))
    nz = out.z[out.z != 0]
    assert np.all(np.abs(nz) >= np.abs(out.z).max() * 1e-8)


def test_mm_position_emd_uses_all_fixations():
    fg1 = fixation_group(x=[100, 200, 300], y=[100, 100, 100], duration=[1, 1, 1], onset=[0, 1, 2])
    # Identical except for the last fixation, which saccade rows would drop.
    fg2 = fixation_group(x=[100, 200, 400], y=[100, 100, 100], duration=[1, 1, 1], onset=[0, 1, 2])
    res = multi_match(scanpath(fg1), scanpath(fg2), screensize=(1000, 1000))
    diag = np.sqrt(2) * 1000
    assert res["mm_position_emd"] == pytest.approx(1 - (100 / 3) / diag, abs=1e-9)


def test_template_multireg_accepts_r_style_method_vector():
    tab = pd.DataFrame({"resp": [_map(1, 0.3)], "c1": [_map(1)], "c2": [_map(2)]})
    a = template_multireg(tab, "resp", ["c1", "c2"])
    b = template_multireg(tab, "resp", ["c1", "c2"], method=["lm", "rlm", "nnls", "logistic"])
    pd.testing.assert_frame_equal(a["multireg"].iloc[0], b["multireg"].iloc[0])
    with pytest.raises(ValueError, match="Unknown regression method"):
        template_multireg(tab, "resp", ["c1", "c2"], method="nope")


def _reg_tables():
    ref = pd.DataFrame({"id": ["a", "b"], "map": [_map(1), _map(2)]})
    src = pd.DataFrame({"id": ["a", "b"], "base": ["x", "y"], "map": [_map(1, 0.3), _map(2, 0.3)]})
    base = pd.DataFrame({"base": ["x", "y"], "map": [_map(5), _map(6)]})
    return ref, src, base


def test_template_regression_uses_density_var():
    ref, src, base = _reg_tables()
    res = template_regression(ref, src, "id", base, "base", density_var="map")
    assert np.all(np.isfinite(res[["beta_baseline", "beta_source"]].to_numpy()))


def test_template_regression_refuses_duplicated_baseline_keys():
    ref, src, base = _reg_tables()
    dup = pd.concat([base, base.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="more than one row for base = y"):
        template_regression(ref, src, "id", dup, "base", density_var="map")


def test_template_regression_duplicates_of_unused_keys_are_allowed():
    ref, src, base = _reg_tables()
    extra = pd.concat([base, pd.DataFrame({"base": ["z", "z"], "map": [_map(7), _map(8)]})],
                      ignore_index=True)
    template_regression(ref, src, "id", extra, "base", density_var="map")


def test_template_regression_refuses_missing_baseline_keys():
    ref, src, base = _reg_tables()
    with pytest.raises(ValueError, match="no row for base = y"):
        template_regression(ref, src, "id", base.iloc[[0]], "base", density_var="map")


def test_template_sample_drops_rows_with_missing_inputs():
    fg = fixation_group(x=[0.5, 2.5], y=[1.0, 2.0], duration=[1, 1], onset=[0, 1])
    tab = pd.DataFrame({"id": [1, 2, 3], "tmpl": [_map(1), None, _map(3)], "fixgroup": [fg, fg, None]})
    with pytest.warns(UserWarning, match=r"removing 2 row\(s\) with a NULL template or fixation group"):
        res = template_sample(tab, "tmpl")
    assert res["id"].tolist() == [1]
    assert len(res["sample_out"].iloc[0]) == 2


def test_template_sample_without_missing_inputs_does_not_warn():
    fg = fixation_group(x=[0.5, 2.5], y=[1.0, 2.0], duration=[1, 1], onset=[0, 1])
    tab = pd.DataFrame({"tmpl": [_map(1)], "fixgroup": [fg]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        template_sample(tab, "tmpl")
