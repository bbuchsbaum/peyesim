import builtins

import numpy as np
import pytest

from peyesim import fixation_group, gen_density
from peyesim.visualization import (
    _fixation_point_sizes,
    _transform_values,
    anim_scanpath,
    plot_eye_density,
    plot_fixation_group,
)


def test_transform_values_matches_r_transform_names():
    z = np.array([0.0, 1.0, 8.0])

    np.testing.assert_allclose(_transform_values(z, "identity"), z)
    np.testing.assert_allclose(_transform_values(z, "sqroot"), [0.0, 1.0, np.sqrt(8.0)])
    np.testing.assert_allclose(_transform_values(z, "curoot"), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(_transform_values(z, "rank"), [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="transform must be one of"):
        _transform_values(z, "unknown")


def test_visualization_helpers_report_optional_matplotlib_dependency(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    dens = gen_density(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.ones((2, 2)))

    with pytest.raises(ImportError, match="peyesim\\[viz\\]"):
        plot_eye_density(dens)


def test_plot_eye_density_returns_axes_when_matplotlib_available():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dens = gen_density(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([[0.0, 1.0], [2.0, 3.0]]))

    ax = plot_eye_density(dens, transform="rank")

    assert ax.get_xlim() == pytest.approx((0.0, 1.0))
    assert ax.get_ylim() == pytest.approx((0.0, 1.0))
    assert len(ax.images) == 1
    np.testing.assert_allclose(ax.images[0].get_array(), np.array([[1.0, 3.0], [2.0, 4.0]]))
    assert not any(spine.get_visible() for spine in ax.spines.values())

    ax.figure.canvas.draw()
    pixels = np.asarray(ax.figure.canvas.buffer_rgba())
    assert pixels[..., :3].std() > 0
    plt.close(ax.figure)


def test_fixation_point_sizes_match_r_aesthetic():
    np.testing.assert_allclose(
        _fixation_point_sizes([10.0, 20.0, 40.0], for_matplotlib=False),
        [1.0, 1.6666666667, 3.0],
    )
    np.testing.assert_allclose(
        _fixation_point_sizes([10.0, 20.0, 40.0], for_matplotlib=True),
        [1.0, 2.7777777778, 9.0],
    )
    np.testing.assert_allclose(
        _fixation_point_sizes([10.0, 10.0], for_matplotlib=False),
        [1.0, 1.0],
    )


def test_plot_fixation_group_supports_windowed_points_and_raster():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PathCollection

    fg = fixation_group(
        x=[0.0, 1.0, 2.0, 3.0],
        y=[0.0, 1.0, 0.0, 1.0],
        onset=[0.0, 100.0, 200.0, 300.0],
        duration=[10.0, 20.0, 30.0, 40.0],
    )

    ax_points = plot_fixation_group(fg, window=(0.0, 250.0), type="points")
    path_collection = next(c for c in ax_points.collections if isinstance(c, LineCollection))
    scatter_collection = next(c for c in ax_points.collections if isinstance(c, PathCollection))
    assert len(path_collection.get_segments()) == 2
    np.testing.assert_allclose(scatter_collection.get_offsets(), [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    np.testing.assert_allclose(scatter_collection.get_sizes(), [1.0, 4.0, 9.0])
    assert ax_points.get_xlim() == pytest.approx((-0.2, 2.2))
    assert ax_points.get_ylim() == pytest.approx((-0.1, 1.1))
    assert not any(spine.get_visible() for spine in ax_points.spines.values())

    ax_raster = plot_fixation_group(fg, type="raster", bins=4, transform="sqroot")
    assert len(ax_raster.images) == 1
    assert np.asarray(ax_raster.images[0].get_array()).std() > 0
    ax_raster.figure.canvas.draw()
    pixels = np.asarray(ax_raster.figure.canvas.buffer_rgba())
    assert pixels[..., :3].std() > 0
    plt.close(ax_points.figure)
    plt.close(ax_raster.figure)


def test_plot_fixation_group_rejects_empty_window():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    fg = fixation_group(x=[0.0], y=[0.0], onset=[0.0], duration=[10.0])

    with pytest.raises(ValueError, match="No fixations remain"):
        plot_fixation_group(fg, window=(10.0, 20.0))


def test_anim_scanpath_supports_r_time_bin_behavior():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fg = fixation_group(
        x=[0.0, 1.0, 2.0],
        y=[0.0, 1.0, 0.0],
        onset=[0.0, 50.0, 100.0],
        duration=[10.0, 10.0, 10.0],
    )

    anim = anim_scanpath(fg, time_bin=50, type="points")

    assert isinstance(anim, FuncAnimation)
    assert anim._fig.axes[0].get_xlim() == pytest.approx((-0.2, 2.2))
    assert anim._fig.axes[0].get_ylim() == pytest.approx((-0.1, 1.1))
    first_frame = next(anim.new_frame_seq())
    artists = anim._func(first_frame)
    scatter = artists[0]
    np.testing.assert_allclose(scatter.get_offsets(), [[0.0, 0.0]])
    anim._fig.canvas.draw()
    pixels = np.asarray(anim._fig.canvas.buffer_rgba())
    assert pixels[..., :3].std() > 0
    anim._draw_was_started = True
    plt.close(anim._fig)


def test_anim_scanpath_rejects_non_positive_time_bin():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    fg = fixation_group(x=[0.0], y=[0.0], onset=[0.0], duration=[10.0])

    with pytest.raises(ValueError, match="time_bin must be positive"):
        anim_scanpath(fg, time_bin=0)
