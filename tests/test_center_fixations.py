"""Port of test_center_fixations.R"""

import numpy as np
from peyesim import fixation_group


def test_can_center_a_fixation_group():
    np.random.seed(42)
    x = np.random.uniform(size=10)
    y = np.random.uniform(size=10)
    onset = np.arange(1, len(x) * 50 + 1, 50, dtype=float)
    duration = np.ones(len(x))
    fg = fixation_group(x, y, onset=onset, duration=duration)
    cfix = fg.center()
    assert abs(cfix["x"].mean()) < 1e-10
    assert abs(cfix["y"].mean()) < 1e-10
