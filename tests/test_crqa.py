import numpy as np
import pandas as pd
import pytest

from peyesim import crqa, fixation_group


def test_crqa_matches_radius_recurrence_for_r_wrapper_fixture():
    fg1 = pd.DataFrame({"x": np.arange(1, 6), "y": np.arange(1, 6)})
    fg2 = pd.DataFrame({"x": np.arange(2, 7), "y": np.arange(3, 8)})

    result = crqa(fg1, fg2, radius=60)

    assert result["n_points"] == 5
    assert result["recurrent_points"] == 25
    assert result["recurrence_rate"] == pytest.approx(1.0)
    assert result["determinism"] == pytest.approx(23 / 25)
    assert result["laminarity"] == pytest.approx(1.0)
    assert result["max_line"] == 5
    assert result["RR"] == pytest.approx(100.0)
    assert result["DET"] == pytest.approx(92.0)
    assert result["NRLINE"] == 7
    assert result["maxL"] == 5
    assert result["L"] == pytest.approx(23 / 7)
    assert result["LAM"] == pytest.approx(100.0)
    assert result["TT"] == pytest.approx(5.0)
    assert result["max_vertlength"] == 5
    np.testing.assert_array_equal(result["RP"], result["recurrence_matrix"])
    assert result["recurrence_matrix"].shape == (5, 5)
    assert result["recurrence_matrix"].dtype == bool


def test_crqa_truncates_to_shared_length_and_uses_xy_columns():
    fg1 = fixation_group(x=[0, 10, 20, 30], y=[0, 0, 0, 0], onset=[0, 1, 2, 3], duration=[1, 1, 1, 1])
    fg2 = fixation_group(x=[0, 10], y=[0, 0], onset=[0, 1], duration=[1, 1])

    result = crqa(fg1, fg2, radius=0)

    assert result["n_points"] == 2
    assert result["recurrent_points"] == 2
    assert result["recurrence_rate"] == pytest.approx(0.5)
    np.testing.assert_array_equal(result["recurrence_matrix"], [[True, False], [False, True]])


def test_crqa_supports_manhattan_metric_and_embedding():
    coords1 = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
    coords2 = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])

    result = crqa(coords1, coords2, radius=1, metric="manhattan", embed=2, delay=1)

    assert result["n_points"] == 3
    assert result["recurrence_matrix"].shape == (3, 3)
    assert result["recurrence_rate"] > 0


def test_crqa_rejects_invalid_coordinate_inputs():
    with pytest.raises(ValueError, match="at least two coordinate columns"):
        crqa(np.array([1, 2, 3]), np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="leave no observations"):
        crqa(np.ones((2, 2)), np.ones((2, 2)), embed=3, delay=1)
