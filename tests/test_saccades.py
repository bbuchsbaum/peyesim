"""Tests for scanpath construction (ported from R test_saccades.R)."""
import numpy as np
import pandas as pd
import pytest
from peyesim import fixation_group, add_scanpath, as_eye_table, calcangle, cart2pol
from peyesim.eye_table import EyeTable
from peyesim.saccades import Scanpath


def test_cart2pol_and_calcangle_match_r_contracts():
    polar = cart2pol([3, 0], [4, 1])

    np.testing.assert_allclose(polar[:, 0], [5.0, 1.0])
    np.testing.assert_allclose(polar[:, 1], [np.arctan2(4, 3), np.pi / 2])

    assert calcangle([1, 0], [0, 1]) == pytest.approx(90.0)
    assert calcangle([1, 1], [1, 1]) == pytest.approx(0.0)


def test_add_scanpath_builds_per_row():
    """add_scanpath should create distinct scanpaths for each fixgroup row."""
    fg1 = fixation_group(
        x=[1, 2, 3], y=[1, 2, 3], onset=[0, 100, 200], duration=[100, 100, 100]
    )
    fg2 = fixation_group(
        x=[5, 6, 7], y=[5, 4, 3], onset=[0, 100, 200], duration=[100, 100, 100]
    )

    df = pd.DataFrame({"id": [1, 2], "fixgroup": [fg1, fg2]})
    out = add_scanpath(df)

    # Each row should have a Scanpath
    assert isinstance(out["scanpath"].iloc[0], Scanpath)
    assert isinstance(out["scanpath"].iloc[1], Scanpath)

    # x values should match original fixgroups
    np.testing.assert_array_equal(
        out["scanpath"].iloc[0]["x"].values, fg1["x"].values
    )
    np.testing.assert_array_equal(
        out["scanpath"].iloc[1]["x"].values, fg2["x"].values
    )

    # theta should differ between the two scanpaths
    assert not np.array_equal(
        out["scanpath"].iloc[0]["theta"].values,
        out["scanpath"].iloc[1]["theta"].values,
    )

    eye_tab = as_eye_table(df)
    out_eye_tab = add_scanpath(eye_tab)

    assert isinstance(out_eye_tab, EyeTable)
    assert isinstance(out_eye_tab["scanpath"].iloc[0], Scanpath)
    assert isinstance(out_eye_tab["scanpath"].iloc[1], Scanpath)
    np.testing.assert_array_equal(
        out_eye_tab["scanpath"].iloc[0]["x"].values, fg1["x"].values
    )
    np.testing.assert_array_equal(
        out_eye_tab["scanpath"].iloc[1]["x"].values, fg2["x"].values
    )
