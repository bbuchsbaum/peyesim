"""Conversion between degrees of visual angle and screen pixels."""

from __future__ import annotations

import numpy as np


def _check_display_geometry(distance, screen_size, screen_res) -> None:
    def ok(v) -> bool:
        arr = np.asarray(v)
        if arr.ndim != 0 or arr.dtype.kind not in "iuf":
            return False
        return bool(np.isfinite(arr) and arr > 0)

    if not (ok(distance) and ok(screen_size) and ok(screen_res)):
        raise ValueError("distance, screen_size and screen_res must each be a single positive number.")


def _scalar_or_array(values: np.ndarray, original):
    return float(values) if np.ndim(original) == 0 else values


def visual_angle_to_px(deg, distance: float, screen_size: float, screen_res: float):
    """Convert a visual angle in degrees to an on-screen extent in pixels.

    The extent is centred on the line of sight: an angle ``theta`` spans
    ``2 * distance * tan(theta / 2)`` on the screen. ``screen_size`` is the
    physical width (or height) of the screen in the units of ``distance`` and
    ``screen_res`` the number of pixels along that dimension. Use it to give
    distance thresholds, such as the ``radius`` of
    :func:`~peyesim.revisits.fixation_revisits`, in degrees.

    Examples
    --------
    >>> round(visual_angle_to_px(2, distance=60, screen_size=52, screen_res=1920), 3)
    77.333
    """
    _check_display_geometry(distance, screen_size, screen_res)
    arr = np.asarray(deg, dtype=float)
    if np.any(~np.isfinite(arr) | (arr < 0) | (arr >= 180)):
        raise ValueError("deg must be finite and in [0, 180).")
    out = 2 * distance * np.tan(arr * np.pi / 360) * screen_res / screen_size
    return _scalar_or_array(out, deg)


def px_to_visual_angle(px, distance: float, screen_size: float, screen_res: float):
    """Convert an on-screen extent in pixels to degrees of visual angle.

    Inverse of :func:`visual_angle_to_px`.
    """
    _check_display_geometry(distance, screen_size, screen_res)
    arr = np.asarray(px, dtype=float)
    if np.any(~np.isfinite(arr) | (arr < 0)):
        raise ValueError("px must be finite and non-negative.")
    out = 2 * np.arctan(arr * screen_size / screen_res / (2 * distance)) * 180 / np.pi
    return _scalar_or_array(out, px)
