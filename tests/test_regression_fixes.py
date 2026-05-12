"""Port of test_regression_fixes.R"""

import numpy as np
import pandas as pd
import pytest
from peyesim import (
    fixation_group, eye_table, simulate_eye_table, eye_density, template_multireg,
    template_regression,
)
from peyesim.density import EyeDensity
from peyesim.regression import _partial_spearman_two_predictors


def test_eye_table_constructs_correctly_with_groupvar_and_vars():
    np.random.seed(42)
    n = 20
    df = pd.DataFrame({
        "xpos": np.random.uniform(0, 500, n),
        "ypos": np.random.uniform(0, 500, n),
        "dur": np.abs(np.random.normal(200, 30, n)),
        "ons": np.tile(np.cumsum(np.abs(np.random.normal(300, 50, 10))), 2),
        "trial": np.repeat(["A", "B"], 10),
        "cond": np.repeat(["old", "new"], 10),
    })

    et = eye_table("xpos", "ypos", "dur", "ons",
                   groupvar="trial", extra_vars=["cond"], data=df,
                   clip_bounds=(0, 500, 0, 500))

    from peyesim.eye_table import EyeTable
    assert isinstance(et, EyeTable)
    assert len(et) == 2
    assert "fixgroup" in et.columns
    assert "cond" in et.columns
    from peyesim.fixations import FixationGroup
    assert isinstance(et["fixgroup"].iloc[0], FixationGroup)
    assert isinstance(et["fixgroup"].iloc[1], FixationGroup)

    et_r_alias = eye_table("xpos", "ypos", "dur", "ons",
                           groupvar="trial", vars=["cond"], data=df,
                           clip_bounds=(0, 500, 0, 500))
    assert "cond" in et_r_alias.columns
    assert list(et_r_alias["cond"]) == ["old", "new"]

    with pytest.raises(ValueError, match="either extra_vars or vars"):
        eye_table("xpos", "ypos", "dur", "ons",
                  groupvar="trial", extra_vars=["cond"], vars=["cond"],
                  data=df, clip_bounds=(0, 500, 0, 500))


def test_simulate_eye_table_generates_per_group_onsets():
    np.random.seed(1)
    et = simulate_eye_table(n_fixations=40, n_groups=4,
                            clip_bounds=(0, 500, 0, 500))

    from peyesim.eye_table import EyeTable
    assert isinstance(et, EyeTable)
    assert len(et) == 4

    for i in range(len(et)):
        fg = et["fixgroup"].iloc[i]
        assert fg["onset"].iloc[0] < 1000, \
            f"Group {i} onset starts at {fg['onset'].iloc[0]} - should be near 0"


def test_normalize_fixation_group_scales_to_unit():
    fg = fixation_group(
        x=[100, 500, 900],
        y=[200, 600, 1000],
        onset=[0, 200, 400],
        duration=[200, 200, 200],
    )
    normed = fg.normalize(xbounds=(100, 900), ybounds=(200, 1000))
    assert normed["x"].min() == pytest.approx(0)
    assert normed["x"].max() == pytest.approx(1)
    assert normed["y"].min() == pytest.approx(0)
    assert normed["y"].max() == pytest.approx(1)
    assert normed["x"].iloc[1] == pytest.approx(0.5)
    assert normed["y"].iloc[1] == pytest.approx(0.5)


def test_eye_density_rejects_non_positive_sigma():
    fg = fixation_group(
        x=[100, 200, 300], y=[100, 150, 200],
        onset=[0, 200, 400], duration=[200, 200, 200],
    )
    with pytest.raises(ValueError, match="sigma must be a positive"):
        eye_density(fg, sigma=0, xbounds=(0, 400), ybounds=(0, 300))
    with pytest.raises(ValueError, match="sigma must be a positive"):
        eye_density(fg, sigma=-10, xbounds=(0, 400), ybounds=(0, 300))
    d = eye_density(fg, sigma=50, xbounds=(0, 400), ybounds=(0, 300))
    assert isinstance(d, EyeDensity)


def test_eye_density_accepts_r_kde_pkg_alias():
    fg = fixation_group(
        x=[100, 200, 300], y=[100, 150, 200],
        onset=[0, 200, 400], duration=[200, 200, 200],
    )

    d = eye_density(fg, sigma=50, xbounds=(0, 400), ybounds=(0, 300), kde_pkg="ks")

    assert isinstance(d, EyeDensity)
    with pytest.raises(ValueError, match="kde_pkg"):
        eye_density(fg, sigma=50, xbounds=(0, 400), ybounds=(0, 300), kde_pkg="other")


def test_as_dataframe_eye_density_returns_correct_grid():
    fg = fixation_group(
        x=[100, 200, 300], y=[100, 150, 200],
        onset=[0, 200, 400], duration=[200, 200, 200],
    )
    ed = eye_density(fg, sigma=50, xbounds=(0, 400), ybounds=(0, 300),
                     outdim=(10, 10))
    df = ed.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100  # 10 x 10
    assert set(["x", "y", "z"]).issubset(df.columns)
    assert df["x"].nunique() == 10
    assert df["y"].nunique() == 10


def test_template_multireg_lm_returns_tidy_tables():
    response = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 2], [3, 4]], dtype=float),
        sigma=1,
    )
    cov1 = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 0], [0, 1]], dtype=float),
        sigma=1,
    )
    cov2 = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[0, 1], [1, 0]], dtype=float),
        sigma=1,
    )

    res = template_multireg(
        pd.DataFrame({"response": [response], "cov1": [cov1], "cov2": [cov2]}),
        response="response",
        covars=["cov1", "cov2"],
        method="lm",
    )

    tidy = res["multireg"].iloc[0]
    assert list(tidy["term"]) == ["(Intercept)", "cov1", "cov2"]
    assert "estimate" in tidy.columns


def test_template_multireg_logistic_returns_tidy_tables():
    response = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 2], [3, 4]], dtype=float),
        sigma=1,
    )
    cov1 = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 0], [0, 1]], dtype=float),
        sigma=1,
    )
    cov2 = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[0, 1], [0, 1]], dtype=float),
        sigma=1,
    )

    res = template_multireg(
        pd.DataFrame({"response": [response], "cov1": [cov1], "cov2": [cov2]}),
        response="response",
        covars=["cov1", "cov2"],
        method="logistic",
    )

    tidy = res["multireg"].iloc[0]
    assert list(tidy["term"]) == ["(Intercept)", "cov1", "cov2"]
    assert np.isfinite(tidy["estimate"]).all()


def test_template_multireg_nnls_returns_r_shaped_table():
    response = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 2], [3, 4]], dtype=float),
        sigma=1,
    )
    cov1 = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 0], [0, 1]], dtype=float),
        sigma=1,
    )
    cov2 = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[0, 1], [0, 1]], dtype=float),
        sigma=1,
    )

    res = template_multireg(
        pd.DataFrame({"response": [response], "cov1": [cov1], "cov2": [cov2]}),
        response="response",
        covars=["cov1", "cov2"],
        method="nnls",
    )

    tidy = res["multireg"].iloc[0]
    assert list(tidy.columns) == ["term", "estimate"]
    assert list(tidy["term"]) == ["cov1", "cov2"]
    assert (tidy["estimate"] >= 0).all()


def test_template_multireg_rlm_returns_tidy_tables():
    response = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 2], [30, 4]], dtype=float),
        sigma=1,
    )
    cov1 = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 0], [0, 1]], dtype=float),
        sigma=1,
    )
    cov2 = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[0, 1], [0, 1]], dtype=float),
        sigma=1,
    )

    res = template_multireg(
        pd.DataFrame({"response": [response], "cov1": [cov1], "cov2": [cov2]}),
        response="response",
        covars=["cov1", "cov2"],
        method="rlm",
    )

    tidy = res["multireg"].iloc[0]
    assert list(tidy["term"]) == ["(Intercept)", "cov1", "cov2"]
    assert np.isfinite(tidy["estimate"]).all()


def test_template_regression_rlm_returns_coefficients():
    baseline = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 0], [0, 1]], dtype=float),
        sigma=1,
    )
    ref = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[0, 1], [2, 3]], dtype=float),
        sigma=1,
    )
    source = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 2], [3, 5]], dtype=float),
        sigma=1,
    )

    res = template_regression(
        pd.DataFrame({"id": ["A"], "density": [ref]}),
        pd.DataFrame({"id": ["A"], "scene": ["S1"], "density": [source]}),
        match_on="id",
        baseline_tab=pd.DataFrame({"scene": ["S1"], "density": [baseline]}),
        baseline_key="scene",
        method="rlm",
    )

    assert np.isfinite(
        res.loc[0, ["beta_baseline", "beta_source"]].to_numpy(dtype=float)
    ).all()


def test_template_regression_rank_uses_partial_spearman():
    baseline = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[1, 4], [2, 8]], dtype=float),
        sigma=1,
    )
    ref = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[4, 1], [7, 3]], dtype=float),
        sigma=1,
    )
    source = EyeDensity(
        x=np.array([0, 1], dtype=float),
        y=np.array([0, 1], dtype=float),
        z=np.array([[2, 5], [5, 9]], dtype=float),
        sigma=1,
    )

    res = template_regression(
        pd.DataFrame({"id": ["A"], "density": [ref]}),
        pd.DataFrame({"id": ["A"], "scene": ["S1"], "density": [source]}),
        match_on="id",
        baseline_tab=pd.DataFrame({"scene": ["S1"], "density": [baseline]}),
        baseline_key="scene",
        method="rank",
    )

    expected = _partial_spearman_two_predictors(
        source.z.ravel(), baseline.z.ravel(), ref.z.ravel()
    )
    np.testing.assert_allclose(
        res.loc[0, ["beta_baseline", "beta_source"]].to_numpy(dtype=float),
        expected,
    )
