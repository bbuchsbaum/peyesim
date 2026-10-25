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
    assert ax.get_xticks().size == 0

    ax.figure.canvas.draw()
    pixels = np.asarray(ax.figure.canvas.buffer_rgba())
    assert pixels[..., :3].std() > 0
    plt.close(ax.figure)


def test_fixation_point_area_is_proportional_to_duration():
    # R: scale_size_area(max_size = 5), so size = 5 * sqrt(duration / max)
    np.testing.assert_allclose(
        _fixation_point_sizes([10.0, 20.0, 40.0], for_matplotlib=False),
        5 * np.sqrt([0.25, 0.5, 1.0]),
    )
    area = _fixation_point_sizes([10.0, 20.0, 40.0], for_matplotlib=True)
    np.testing.assert_allclose(area / area[-1], [0.25, 0.5, 1.0])
    np.testing.assert_allclose(_fixation_point_sizes([10.0, 10.0], for_matplotlib=False), [5.0, 5.0])


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
    sizes = scatter_collection.get_sizes()
    np.testing.assert_allclose(sizes / sizes[-1], [1 / 3, 2 / 3, 1.0])
    # range of the windowed fixations plus ggplot2's 5% expansion
    assert ax_points.get_xlim() == pytest.approx((-0.1, 2.1))
    assert ax_points.get_ylim() == pytest.approx((-0.05, 1.05))
    assert ax_points.get_aspect() == 1.0

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
    assert anim._fig.axes[0].get_xlim() == pytest.approx((-0.1, 2.1))
    assert anim._fig.axes[0].get_ylim() == pytest.approx((-0.05, 1.05))
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


# --- eyesim 0.2.0 visual system -------------------------------------------

from peyesim.visualization import (  # noqa: E402
    EYESIM_DENSITY_COLOURS,
    EYESIM_TIME_COLOURS,
    _kde2d,
    _kde_peak,
    density_opacity,
    eyesim_colours,
    eyesim_density_cmap,
    format_density,
    format_limit,
    theme_eyesim,
    theme_eyesim_spatial,
)
from tests.r_oracle import as_float, r_unavailable_reason, requires_r, run_r  # noqa: E402,F401


def _mpl():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _viz_fixations():
    rng = np.random.default_rng(11)
    centres = np.array([[256, 476], [696, 422], [890, 225]])
    who = rng.integers(0, 3, 24)
    return fixation_group(
        x=centres[who, 0] + rng.normal(0, 35, 24),
        y=centres[who, 1] + rng.normal(0, 35, 24),
        duration=np.round(rng.uniform(100, 450, 24)),
        onset=np.cumsum(rng.uniform(150, 400, 24)),
    )


def test_eyesim_colours_roles_and_unknown_role():
    cols = eyesim_colours()
    assert list(cols)[:2] == ["reference", "source"]
    assert eyesim_colours(["source", "reference"]) == {"source": "#C23B4B", "reference": "#1B2A41"}
    assert eyesim_colours("ink") == {"ink": "#1F2328"}
    with pytest.raises(ValueError, match="Unknown"):
        eyesim_colours("nope")


def test_density_cmap_is_transparent_at_zero_and_clamps_above():
    pytest.importorskip("matplotlib")
    cmap = eyesim_density_cmap()
    assert cmap(0.0)[3] == 0
    assert cmap(1.0)[3] == pytest.approx(0.9)
    assert cmap(2.0) == cmap(1.0)  # above the scale takes the top colour
    assert cmap(np.nan)[3] == 0


def test_all_plot_types_build_with_equal_aspect():
    plt = _mpl()
    fg = _viz_fixations()
    for type_ in ("points", "contour", "filled_contour", "density", "raster"):
        ax = plot_fixation_group(fg, type=type_)
        assert ax.get_aspect() == 1.0
        ax.figure.canvas.draw()
        plt.close(ax.figure)
    ax = plot_fixation_group(fg, aspect="free")
    assert ax.get_aspect() == "auto"
    plt.close(ax.figure)


def test_fixations_numbered_and_joined_in_onset_order():
    plt = _mpl()
    from matplotlib.collections import LineCollection

    fg = fixation_group(x=[10, 50, 90], y=[10, 60, 20], duration=[200, 200, 200], onset=[400, 0, 200])
    ax = plot_fixation_group(fg)
    path = next(c for c in ax.collections if isinstance(c, LineCollection))
    segs = path.get_segments()
    np.testing.assert_allclose([s[0][0] for s in segs] + [segs[-1][1][0]], [50, 90, 10])
    labels = {t.get_text(): t.xy for t in ax.texts}
    assert labels["1"][0] == 50 and labels["2"][0] == 90 and labels["3"][0] == 10
    plt.close(ax.figure)


def test_xlim_does_not_drop_fixations():
    plt = _mpl()
    from matplotlib.collections import PathCollection

    fg = fixation_group(x=[0, 50, 100], y=[0, 50, 100], duration=[1, 1, 1], onset=[0, 1, 2])
    ax = plot_fixation_group(fg, xlim=(0, 40), ylim=(0, 40))
    pts = next(c for c in ax.collections if isinstance(c, PathCollection))
    assert len(pts.get_offsets()) == 3
    plt.close(ax.figure)


def test_density_bands_zero_band_empty_and_top_colour_above_limits():
    plt = _mpl()
    fg = _viz_fixations()
    for type_ in ("density", "filled_contour"):
        ax = plot_fixation_group(fg, type=type_)
        cs = ax.collections[0]
        fc = np.asarray(cs.get_facecolor())
        assert fc[0, 3] == 0
        assert fc[1:-1, 3].min() > 0.1
        plt.close(ax.figure)

        peak = _kde_peak(np.asarray(fg["x"]), np.asarray(fg["y"]), 60.0, (0, 1100), (0, 700))
        ax = plot_fixation_group(fg, type=type_, limits=(0, peak / 2))
        cs = ax.collections[0]
        assert np.isinf(cs.levels[-1])
        assert np.asarray(cs.get_facecolor())[-1, 3] > 0.5
        plt.close(ax.figure)

    with pytest.warns(UserWarning, match="ignored"):
        ax = plot_fixation_group(fg, type="density", transform="rank", limits=(0, 1e-4))
    plt.close(ax.figure)


def test_eye_density_shared_limits_and_transform():
    plt = _mpl()
    z = np.outer(np.hanning(20), np.hanning(20))
    dens = gen_density(np.linspace(0, 1, 20), np.linspace(0, 1, 20), z)
    ax = plot_eye_density(dens, limits=(0, z.max() / 2))
    im = ax.images[-1]
    assert im.get_clim() == pytest.approx((0, z.max() / 2))
    rgba = im.to_rgba(im.get_array())
    assert np.allclose(rgba[z.T > z.max() / 2], rgba[np.unravel_index(np.argmax(z.T), z.shape)])
    plt.close(ax.figure)
    a = plot_eye_density(dens).images[-1]
    b = plot_eye_density(dens, transform="sqroot").images[-1]
    q = np.unravel_index(np.argmin(np.abs(z.T - z.max() / 4)), z.shape)
    assert not np.allclose(a.to_rgba(a.get_array())[q], b.to_rgba(b.get_array())[q])
    plt.close("all")


def test_bg_image_keeps_stored_colours(tmp_path):
    plt = _mpl()
    img = np.zeros((4, 4, 3))
    img[..., 0], img[..., 1], img[..., 2] = 0xD9 / 255, 0xD4 / 255, 0xC7 / 255
    path = tmp_path / "bg.png"
    plt.imsave(path, img)
    dens = gen_density(np.array([0.0, 4.0]), np.array([0.0, 4.0]), np.zeros((2, 2)))
    ax = plot_eye_density(dens, bg_image=str(path), legend=False)
    bg = ax.images[0]
    stored = np.asarray(bg.get_array())[..., :3]
    np.testing.assert_allclose(stored, img, atol=1 / 255)
    plt.close(ax.figure)


def test_theme_helpers_style_axes():
    plt = _mpl()
    fig, ax = plt.subplots()
    theme_eyesim(ax)
    assert not ax.spines["top"].get_visible()
    theme_eyesim_spatial(ax)
    assert ax.get_xticks().size == 0 and ax.spines["top"].get_visible()
    plt.close(fig)


def test_anim_scanpath_titles_frames_and_keeps_faded_marks():
    plt = _mpl()
    fg = fixation_group(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 0.0], onset=[0.0, 50.0, 100.0], duration=[10.0] * 3)
    anim = anim_scanpath(fg)
    frames = list(anim.new_frame_seq())
    current, shadow = anim._func(frames[-1])
    assert anim._fig.axes[0].get_title() == "Fixation 3"
    np.testing.assert_allclose(current.get_offsets(), [[2.0, 0.0]])
    np.testing.assert_allclose(shadow.get_offsets(), [[0.0, 0.0], [1.0, 1.0]])
    assert shadow.get_alpha() == pytest.approx(0.35)
    anim._draw_was_started = True
    plt.close(anim._fig)
    anim = anim_scanpath(fg, anim_over="onset")
    anim._func(50.0)
    assert anim._fig.axes[0].get_title() == "Onset 50"
    anim._draw_was_started = True
    plt.close(anim._fig)


R_VIZ = r"""
fx <- c(10, 40, 45, 80, 30)
fy <- c(20, 25, 60, 70, 40)
k <- MASS::kde2d(fx, fy, h = 30, n = 25, lims = c(-10, 100, 0, 90))
dens_alpha <- scales::alpha(eyesim:::eyesim_density_colours(), seq(0, 1, length.out = 9)^0.5 * 0.9)
out <- list(
  roles = as.list(eyesim_colours()),
  time = toupper(substr(eyesim:::eyesim_time_colours(), 1, 7)),
  density = toupper(substr(eyesim:::eyesim_density_colours(transparent = FALSE), 1, 7)),
  density_alpha = grDevices::col2rgb(dens_alpha, alpha = TRUE)[4, ] / 255,
  kde_z = k$z,
  kde_peak = eyesim:::kde_peak(fx, fy, c(30, 30), c(0, 90), c(10, 80)),
  fmt_small = eyesim:::format_density(c(0, 1.234e-4, 5e-5)),
  fmt_big = eyesim:::format_density(c(0, 0.25, 1.5)),
  fmt_limit = c(eyesim:::format_limit(1.2345e-4), eyesim:::format_limit(0.123456)),
  order_index = rank(c(400, 0, 200), ties.method = "first")
)
"""


@pytest.fixture(scope="module")
def r_viz():
    return run_r(R_VIZ)


@requires_r
def test_r_parity_colour_roles_and_ramps(r_viz):
    assert eyesim_colours() == r_viz["roles"]
    assert list(EYESIM_TIME_COLOURS) == r_viz["time"]
    assert list(EYESIM_DENSITY_COLOURS) == r_viz["density"]
    np.testing.assert_allclose(
        density_opacity(np.linspace(0, 1, 9)), as_float(r_viz["density_alpha"]), atol=1 / 255
    )


@requires_r
def test_r_parity_kde_grid_and_peak(r_viz):
    fx = np.array([10, 40, 45, 80, 30.0])
    fy = np.array([20, 25, 60, 70, 40.0])
    _, _, z = _kde2d(fx, fy, 30, n=25, lims=(-10, 100, 0, 90))
    np.testing.assert_allclose(z, as_float(r_viz["kde_z"]), rtol=1e-10)
    assert _kde_peak(fx, fy, 30.0, (0, 90), (10, 80)) == pytest.approx(r_viz["kde_peak"], rel=1e-10)


@requires_r
def test_r_parity_legend_number_formats(r_viz):
    assert format_density([0, 1.234e-4, 5e-5]) == r_viz["fmt_small"]
    assert format_density([0, 0.25, 1.5]) == r_viz["fmt_big"]
    assert format_limit([1.2345e-4]) + format_limit([0.123456]) == r_viz["fmt_limit"]


def test_plot_fixation_group_skips_missing_onsets():
    matplotlib = pytest.importorskip("matplotlib")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from peyesim import fixation_group, plot_fixation_group

    fg = fixation_group(x=[10, 20, 30], y=[10, 25, 15], onset=[0, np.nan, 200], duration=[100, 100, 100])
    ax = plot_fixation_group(fg)
    assert ax is not None
    ax = plot_fixation_group(fg, window=(0, 150))
    assert ax is not None
    plt.close("all")
