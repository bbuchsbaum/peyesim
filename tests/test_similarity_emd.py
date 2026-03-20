"""Port of test_similarity_emd.R"""

import numpy as np
import pytest
from peyesim import similarity
from peyesim.density import EyeDensity


def test_emd_similarity_equals_1_for_identical():
    z = np.array([[0.25, 0.25], [0.25, 0.25]])
    d = EyeDensity(x=np.array([1, 2], dtype=float),
                   y=np.array([1, 2], dtype=float), z=z, sigma=1)
    assert similarity(d, d, method="emd") == pytest.approx(1.0)


def test_signed_emd_returns_0_for_identical_residuals():
    z = np.array([[0.3, 0.2], [0.2, 0.3]])
    d = EyeDensity(x=np.array([1, 2], dtype=float),
                   y=np.array([1, 2], dtype=float), z=z, sigma=1)
    sal_z = np.full((2, 2), 0.25)
    sal = EyeDensity(x=np.array([1, 2], dtype=float),
                     y=np.array([1, 2], dtype=float), z=sal_z, sigma=1)
    val = similarity(d, d, method="emd", saliency_map=sal)
    assert val == pytest.approx(0.0, abs=1e-6)
