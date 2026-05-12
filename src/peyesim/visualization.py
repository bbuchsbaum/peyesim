"""Optional visualization helpers."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from peyesim.density import EyeDensity
from peyesim.fixations import FixationGroup


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization helpers. "
            "Install it with `pip install peyesim[viz]`."
        ) from exc
    return plt, imread


def _transform_values(values, transform: str = "identity"):
    transform = transform.lower()
    arr = np.asarray(values, dtype=float)
    if transform == "identity":
        return arr
    if transform == "sqroot":
        return np.sqrt(np.maximum(arr, 0))
    if transform == "curoot":
        return np.cbrt(arr)
    if transform == "rank":
        flat = arr.ravel()
        valid = ~np.isnan(flat)
        ranks = np.full(flat.shape, np.nan, dtype=float)
        if valid.any():
            order = np.argsort(flat[valid], kind="mergesort")
            valid_ranks = np.empty(order.shape, dtype=float)
            valid_ranks[order] = np.arange(1, len(order) + 1)
            ranks[valid] = valid_ranks
        return ranks.reshape(arr.shape)
    raise ValueError("transform must be one of {'identity', 'sqroot', 'curoot', 'rank'}")


def _expanded_limits(values, *, proportion: float = 0.10) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = hi - lo
    if span <= np.finfo(float).eps:
        return (lo - 0.5, hi + 0.5)
    pad = span * proportion
    return (lo - pad, hi + pad)


def _fixation_point_sizes(duration, *, for_matplotlib: bool = True) -> np.ndarray:
    """R-compatible fixation-size aesthetic from duration values.

    The R plot method computes ``ps = (duration - min(duration)) / range`` and
    then maps size to ``ps * 2 + 1``. Matplotlib scatter interprets ``s`` as
    marker area, so the default squares those aesthetic sizes for visual scale.
    """
    duration = np.asarray(duration, dtype=float)
    if duration.size == 0:
        return np.array([], dtype=float)
    dmin = np.nanmin(duration)
    span = np.nanmax(duration) - dmin
    aesthetic = np.ones(duration.shape, dtype=float)
    if span > np.finfo(float).eps:
        aesthetic = ((duration - dmin) / span) * 2.0 + 1.0
    return aesthetic ** 2 if for_matplotlib else aesthetic


def _blank_plot_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_eye_density(
    x: EyeDensity,
    alpha: float = 0.8,
    bg_image: str | None = None,
    transform: str = "identity",
    ax=None,
    cmap: str = "Spectral_r",
):
    """Plot an :class:`~peyesim.density.EyeDensity` raster.

    Returns the matplotlib ``Axes`` so callers can keep composing the figure.
    """
    plt, imread = _require_matplotlib()
    if not isinstance(x, EyeDensity):
        raise TypeError("plot_eye_density() expects an EyeDensity object.")
    if ax is None:
        _, ax = plt.subplots()

    xlim = (float(np.min(x.x)), float(np.max(x.x)))
    ylim = (float(np.min(x.y)), float(np.max(x.y)))

    if bg_image is not None:
        image = imread(bg_image)
        ax.imshow(image, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), origin="lower")

    z = _transform_values(x.z, transform=transform)
    ax.imshow(
        z.T,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        aspect="auto",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _blank_plot_axes(ax)
    return ax


def plot_fixation_group(
    x: FixationGroup,
    type: str = "points",
    bandwidth: float = 60,
    xlim=None,
    ylim=None,
    size_points: bool = True,
    show_points: bool = True,
    show_path: bool = True,
    bins: int | None = None,
    bg_image: str | None = None,
    alpha_range: tuple[float, float] = (0.5, 1.0),
    alpha: float = 0.8,
    window: tuple[float, float] | None = None,
    transform: str = "identity",
    ax=None,
    cmap: str = "Spectral_r",
):
    """Plot fixation points, paths, or density summaries.

    Supported ``type`` values mirror the R package: ``points``, ``contour``,
    ``filled_contour``, ``density``, and ``raster``.
    """
    plt, imread = _require_matplotlib()
    if not isinstance(x, FixationGroup):
        raise TypeError("plot_fixation_group() expects a FixationGroup object.")
    if type not in {"points", "contour", "filled_contour", "density", "raster"}:
        raise ValueError("type must be one of {'points', 'contour', 'filled_contour', 'density', 'raster'}")
    _transform_values([0.0, 1.0], transform=transform)

    df = x.to_pandas(copy=True)
    if window is not None:
        if len(window) != 2 or window[1] <= window[0]:
            raise ValueError("window must be a length-2 increasing interval.")
        df = df.loc[(df["onset"] >= window[0]) & (df["onset"] < window[1])]
    if df.empty:
        raise ValueError("No fixations remain after filtering.")

    if ax is None:
        _, ax = plt.subplots()
    if xlim is None:
        xlim = _expanded_limits(df["x"])
    if ylim is None:
        ylim = _expanded_limits(df["y"])
    if bins is None:
        bins = max(int(len(df) / 10), 4)

    if bg_image is not None:
        image = imread(bg_image)
        ax.imshow(image, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), origin="lower")

    if type in {"density", "raster", "contour", "filled_contour"}:
        hist, xedges, yedges = np.histogram2d(
            df["x"],
            df["y"],
            bins=bins,
            range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
        )
        z = gaussian_filter(hist.T, sigma=max(float(bandwidth) / 60.0, 0.1))
        z = _transform_values(z, transform=transform)

        if type == "raster":
            ax.imshow(z, extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                      origin="lower", cmap=cmap, alpha=alpha, aspect="auto")
        else:
            xcent = (xedges[:-1] + xedges[1:]) / 2
            ycent = (yedges[:-1] + yedges[1:]) / 2
            xx, yy = np.meshgrid(xcent, ycent)
            if type == "contour":
                ax.contour(xx, yy, z, cmap=cmap, alpha=alpha)
            else:
                ax.contourf(xx, yy, z, cmap=cmap, alpha=alpha)

    if type == "points" and show_path and len(df) > 1:
        from matplotlib.collections import LineCollection

        coords = df[["x", "y"]].to_numpy(dtype=float)
        segments = np.stack([coords[:-1], coords[1:]], axis=1)
        path = LineCollection(segments, cmap=cmap, alpha=alpha, linewidth=1.0)
        path.set_array(df["onset"].to_numpy(dtype=float)[1:])
        ax.add_collection(path)

    if show_points:
        if size_points and "duration" in df:
            sizes = _fixation_point_sizes(df["duration"].to_numpy(dtype=float), for_matplotlib=True)
        else:
            sizes = np.ones(len(df), dtype=float)
        colours = df["onset"].to_numpy(dtype=float) if "onset" in df else None
        ax.scatter(df["x"], df["y"], s=sizes, c=colours, cmap=cmap, alpha=alpha,
                   edgecolors="none")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _blank_plot_axes(ax)
    return ax


def anim_scanpath(
    x: FixationGroup,
    bg_image: str | None = None,
    xlim=None,
    ylim=None,
    alpha: float = 1.0,
    anim_over: str = "index",
    type: str = "points",
    time_bin: float = 1,
):
    """Animate a fixation scanpath with matplotlib.

    Returns a :class:`matplotlib.animation.FuncAnimation` object.
    """
    plt, imread = _require_matplotlib()
    from matplotlib.animation import FuncAnimation

    if anim_over not in {"index", "onset"}:
        raise ValueError("anim_over must be one of {'index', 'onset'}")
    if type not in {"points", "raster"}:
        raise ValueError("type must be one of {'points', 'raster'}")
    if time_bin <= 0:
        raise ValueError("time_bin must be positive")

    df = x.copy()
    if xlim is None:
        xlim = _expanded_limits(df["x"])
    if ylim is None:
        ylim = _expanded_limits(df["y"])

    if time_bin > 1:
        df["frame"] = np.round(df["onset"] / time_bin).astype(int)
        anim_over = "time_bin"
    elif anim_over == "index":
        df["frame"] = np.arange(len(df))
    else:
        df["frame"] = df["onset"].to_numpy()

    frames = np.sort(df["frame"].unique())

    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    _blank_plot_axes(ax)

    if bg_image is not None:
        image = imread(bg_image)
        ax.imshow(image, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), origin="lower")

    if type == "points":
        scatter = ax.scatter([], [], c=[], cmap="Spectral_r", alpha=alpha)

        def update(frame):
            current = df.loc[df["frame"] <= frame]
            offsets = current[["x", "y"]].to_numpy()
            scatter.set_offsets(offsets if len(offsets) else np.empty((0, 2)))
            scatter.set_array(current["onset"].to_numpy(dtype=float))
            return (scatter,)

    else:
        image = ax.imshow(
            np.zeros((50, 50)),
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            cmap="Spectral_r",
            alpha=alpha,
            aspect="auto",
        )

        def update(frame):
            current = df.loc[df["frame"] <= frame]
            hist, _, _ = np.histogram2d(
                current["x"],
                current["y"],
                bins=50,
                range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
            )
            image.set_data(gaussian_filter(hist.T, sigma=1.5))
            return (image,)

    return FuncAnimation(fig, update, frames=frames, blit=False)
