import numpy as np
import pytest

from peyesim import fixation_group, multi_match, scanpath
from peyesim.multimatch import _create_graph


def _simple_multimatch_fixture():
    fg1 = fixation_group(
        x=[100, 200, 260, 300],
        y=[100, 140, 120, 180],
        duration=[0.2, 0.25, 0.2, 0.3],
        onset=[0.1, 0.3, 0.6, 1.0],
    )
    fg2 = fixation_group(
        x=[120, 210, 255, 310],
        y=[110, 150, 130, 175],
        duration=[0.22, 0.24, 0.21, 0.28],
        onset=[0.05, 0.35, 0.7, 1.2],
    )
    return fg1, fg2


def test_multi_match_matches_r_reference_on_simple_fixture():
    fg1, fg2 = _simple_multimatch_fixture()

    result = multi_match(scanpath(fg1), scanpath(fg2), screensize=(500, 500))

    expected = {
        "mm_vector": 0.989393398282,
        "mm_direction": 0.969291443602,
        "mm_length": 0.986968420655,
        "mm_position": 0.98,
        "mm_duration": 0.952380952381,
        "mm_position_emd": 0.976803995716,
    }
    for key, value in expected.items():
        tol = 1e-8 if key == "mm_position_emd" else 1e-12
        assert result[key] == pytest.approx(value, abs=tol)


def test_create_graph_path_and_cost_match_r_reference():
    fg1, fg2 = _simple_multimatch_fixture()
    sp1 = scanpath(fg1)
    sp2 = scanpath(fg2)

    gout = _create_graph(sp1.iloc[:-1].reset_index(drop=True), sp2.iloc[:-1].reset_index(drop=True))

    expected_m = np.array(
        [
            [10.0, 81.3941029805, 45.2769256907],
            [67.082039325, 15.0, 65.192024052],
            [53.8516480713, 80.1560977094, 21.2132034356],
        ]
    )

    np.testing.assert_allclose(gout["M"], expected_m, atol=1e-12)
    assert gout["vpath"] == [0, 4, 8]

    cost = 0.0
    for start, end in zip(gout["vpath"][:-1], gout["vpath"][1:]):
        cost += gout["g"][start][end]["weight"]
    assert cost == pytest.approx(36.2132034356, abs=1e-9)


def test_multi_match_translation_invariance_matches_r_reference():
    x = np.array([348.890929786, 338.665475114, 300.23525299, 257.210120861, 231.58505992, 260.786102503])
    y = np.array([184.006235283, 231.212485349, 197.798033804, 193.708400382, 160.883208062, 134.030918241])
    dur = np.array([0.24320298644, 0.0740753854159, 0.163361942524, 0.0711751782335, 0.190166466765, 0.0521761499811])
    ons = np.array([0.395007981744, 0.555812663177, 0.829619792849, 0.982947925234, 1.38179421809, 1.74890168018])

    fg1 = fixation_group(x=x, y=y, duration=dur, onset=ons)
    fg2 = fixation_group(x=x + 37, y=y - 21, duration=dur, onset=ons)

    result = multi_match(scanpath(fg1), scanpath(fg2), screensize=(640, 480))

    assert result["mm_vector"] > 0.999
    assert result["mm_direction"] > 0.999
    assert result["mm_length"] > 0.999
    assert result["mm_duration"] > 0.999
    assert result["mm_position"] < 1.0

    expected = {
        "mm_vector": 1.0,
        "mm_direction": 1.0,
        "mm_length": 1.0,
        "mm_position": 0.946819881535,
        "mm_duration": 1.0,
        "mm_position_emd": 0.946819882393,
    }
    for key, value in expected.items():
        tol = 1e-8 if key == "mm_position_emd" else 1e-9
        assert result[key] == pytest.approx(value, abs=tol)


def test_multi_match_direction_scale_invariance_matches_r_reference():
    x = np.array([197.30361931, 173.929484673, 197.902121283, 210.051826704, 195.772561021, 213.286761343])
    y = np.array([283.269870952, 317.851456013, 339.383212365, 350.942503307, 347.506094389, 314.650175236])
    dur = np.array([0.15809784336, 0.186240528896, 0.0845560865244, 0.281953063048, 0.0503254303592, 0.116114715883])
    ons = np.array([0.146786286437, 0.379173751047, 0.50746487428, 0.700238537148, 0.965445049456, 1.09068586556])

    fg1 = fixation_group(x=x, y=y, duration=dur, onset=ons)
    scale = 1.8
    fg2 = fixation_group(
        x=x[0] + scale * (x - x[0]),
        y=y[0] + scale * (y - y[0]),
        duration=dur,
        onset=ons,
    )

    result = multi_match(scanpath(fg1), scanpath(fg2), screensize=(800, 600))

    assert result["mm_direction"] > 0.999

    expected = {
        "mm_vector": 0.987110911047,
        "mm_direction": 1.0,
        "mm_length": 0.974221822095,
        "mm_position": 0.955106773506,
        "mm_duration": 1.0,
        "mm_position_emd": 0.963037643433,
    }
    for key, value in expected.items():
        tol = 1e-8 if key == "mm_position_emd" else 1e-9
        assert result[key] == pytest.approx(value, abs=tol)
