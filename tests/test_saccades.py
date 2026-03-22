"""Tests for scanpath construction (ported from R test_saccades.R)."""
import numpy as np
import pandas as pd
from peyesim import fixation_group, scanpath, add_scanpath
from peyesim.saccades import Scanpath


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
