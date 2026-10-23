import numpy as np
import pytest

from peyesim.visual_angle import px_to_visual_angle, visual_angle_to_px
from tests.r_oracle import as_float, r_unavailable_reason, requires_r, run_r  # noqa: F401


def test_visual_angle_and_pixels_convert_exactly_and_invert():
    assert visual_angle_to_px(2, 60, 52, 1920) == pytest.approx(2 * 60 * np.tan(np.pi / 180) * 1920 / 52)
    px = visual_angle_to_px([0, 1, 2, 10], distance=57, screen_size=40, screen_res=1024)
    assert px[0] == 0
    np.testing.assert_allclose(
        px_to_visual_angle(px, distance=57, screen_size=40, screen_res=1024), [0, 1, 2, 10]
    )
    assert px_to_visual_angle(1024 / 40, 57.2958, 40, 1024) == pytest.approx(1, abs=1e-4)


def test_scalar_in_scalar_out():
    assert isinstance(visual_angle_to_px(2, 60, 52, 1920), float)
    assert isinstance(px_to_visual_angle(77, 60, 52, 1920), float)


def test_invalid_display_geometry_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        visual_angle_to_px(2, distance=0, screen_size=40, screen_res=1024)
    with pytest.raises(ValueError, match="single"):
        visual_angle_to_px(2, distance=60, screen_size=[40, 30], screen_res=1024)
    with pytest.raises(ValueError, match="deg"):
        visual_angle_to_px(-1, 60, 40, 1024)
    with pytest.raises(ValueError, match="deg"):
        visual_angle_to_px(180, 60, 40, 1024)
    with pytest.raises(ValueError, match="px"):
        px_to_visual_angle(-5, 60, 40, 1024)


R_CODE = r"""
out <- list(
  to_px = visual_angle_to_px(c(0, 0.5, 1, 2, 10, 45, 179), distance = 60, screen_size = 52, screen_res = 1920),
  to_deg = px_to_visual_angle(c(0, 1, 25, 77, 500, 1920), distance = 57, screen_size = 40, screen_res = 1024)
)
"""


@pytest.fixture(scope="module")
def r():
    return run_r(R_CODE)


@requires_r
def test_visual_angle_matches_r(r):
    np.testing.assert_allclose(
        visual_angle_to_px([0, 0.5, 1, 2, 10, 45, 179], distance=60, screen_size=52, screen_res=1920),
        as_float(r["to_px"]), rtol=1e-12,
    )
    np.testing.assert_allclose(
        px_to_visual_angle([0, 1, 25, 77, 500, 1920], distance=57, screen_size=40, screen_res=1024),
        as_float(r["to_deg"]), rtol=1e-12,
    )
