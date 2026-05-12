import numpy as np
import pandas as pd
import pytest

from peyesim import estimate_scale, fixation_group, match_scale


def test_estimate_scale_filters_source_fixations_by_window():
    ref = fixation_group(
        x=[10, 20, 30],
        y=[10, 20, 30],
        onset=[0, 100, 200],
        duration=[100, 100, 100],
    )
    source = fixation_group(
        x=[5, 10, 15, 1000],
        y=[5, 10, 15, 1000],
        onset=[0, 100, 200, 300],
        duration=[100, 100, 100, 100],
    )

    result = estimate_scale(ref, source, window=(0, 250))

    assert result["n_ref"] == 3
    assert result["n_source"] == 3
    np.testing.assert_allclose(result["par"], [2.0, 2.0], atol=1e-4)


def test_estimate_scale_returns_identity_for_empty_source_after_window():
    ref = fixation_group(x=[10], y=[10], onset=[0], duration=[100])
    source = fixation_group(x=[5], y=[5], onset=[500], duration=[100])

    result = estimate_scale(ref, source, window=(0, 250))

    np.testing.assert_allclose(result["par"], [1.0, 1.0])
    assert result["success"] is True


def test_match_scale_adds_scale_columns_and_drops_unmatched_sources():
    ref_fix = fixation_group(x=[10, 20, 30], y=[10, 20, 30], onset=[0, 1, 2], duration=[1, 1, 1])
    src_fix = fixation_group(x=[5, 10, 15], y=[5, 10, 15], onset=[0, 1, 2], duration=[1, 1, 1])
    ref_tab = pd.DataFrame({"trial": ["A"], "fixgroup": [ref_fix]})
    source_tab = pd.DataFrame(
        {
            "trial": ["A", "B"],
            "fixgroup": [src_fix, src_fix],
        }
    )

    with pytest.warns(UserWarning, match="did not find matching template"):
        result = match_scale(ref_tab, source_tab, match_on="trial")

    assert list(result["trial"]) == ["A"]
    assert "scale_x" in result.columns
    assert "scale_y" in result.columns
    assert result["scale_x"].iloc[0] == pytest.approx(2.0, abs=1e-4)
    assert result["scale_y"].iloc[0] == pytest.approx(2.0, abs=1e-4)


def test_estimate_scale_rejects_invalid_window():
    ref = fixation_group(x=[10], y=[10], onset=[0], duration=[100])
    source = fixation_group(x=[5], y=[5], onset=[0], duration=[100])

    with pytest.raises(ValueError, match="upper bound"):
        estimate_scale(ref, source, window=(250, 0))
