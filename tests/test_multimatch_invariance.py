"""MultiMatch invariance properties (ported from R test_multimatch_invariance.R)."""
import numpy as np
from peyesim import fixation_group, scanpath, multi_match


def test_translation_invariance():
    """Non-position metrics should be ~1 for pure translation."""
    rng = np.random.default_rng(7)
    n = 6
    x = np.cumsum(rng.uniform(-50, 50, n)) + 300
    y = np.cumsum(rng.uniform(-50, 50, n)) + 200
    dur = rng.uniform(50, 300, n)
    ons = np.cumsum(rng.uniform(50, 400, n))

    fg1 = fixation_group(x, y, dur, ons)
    dx, dy = 37, -21
    fg2 = fixation_group(x + dx, y + dy, dur, ons)

    sp1, sp2 = scanpath(fg1), scanpath(fg2)
    result = multi_match(sp1, sp2, screensize=(640, 480))

    assert result["mm_vector"] > 0.999
    assert result["mm_direction"] > 0.999
    assert result["mm_length"] > 0.999
    assert result["mm_duration"] > 0.999
    assert result["mm_position"] < 1.0  # position should drop


def test_direction_scale_invariance():
    """Direction should be ~1 for uniform scaling around first fixation."""
    rng = np.random.default_rng(8)
    n = 6
    x = np.cumsum(rng.uniform(-40, 40, n)) + 200
    y = np.cumsum(rng.uniform(-40, 40, n)) + 300
    dur = rng.uniform(50, 300, n)
    ons = np.cumsum(rng.uniform(50, 400, n))

    fg1 = fixation_group(x, y, dur, ons)
    s = 1.8
    fg2 = fixation_group(x[0] + s * (x - x[0]), y[0] + s * (y - y[0]), dur, ons)

    sp1, sp2 = scanpath(fg1), scanpath(fg2)
    result = multi_match(sp1, sp2, screensize=(800, 600))

    assert result["mm_direction"] > 0.999
