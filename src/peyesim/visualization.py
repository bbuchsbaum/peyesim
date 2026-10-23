"""Optional visualization helpers.

These follow the shared visual system of R eyesim (0.2.0 onwards):

* one set of colour roles (:func:`eyesim_colours`), one time ramp for fixation
  order/onset and one density ramp whose opacity is anchored at zero, so zero
  density is transparent and a background stimulus shows through;
* stimulus-space plots keep a 1:1 aspect ratio (``aspect="free"`` opts out),
  ``xlim``/``ylim`` set the visible extent without dropping fixations, and a
  ``bg_image`` is drawn with its stored colours (top row at ``ylim[1]``);
* fixations are numbered and joined in onset order, whatever the row order;
  point area is proportional to duration; a ring marks the first fixation;
* density maps share colours across plots through ``limits``; density above
  the upper limit takes the top colour; ``transform`` changes the colour scale;
* ``type="density"``/``"filled_contour"`` draw bands between evenly spaced
  edges from zero, with the band containing zero transparent.

Kernel densities for :func:`plot_fixation_group` use the bandwidth convention
of ``MASS::kde2d`` (kernel standard deviation ``bandwidth / 4``), as
``ggplot2::stat_density_2d`` does in R.

Approximations relative to the ggplot2 rendering: the house theme is a
matplotlib axes styler rather than a theme object; title wrapping
(``element_text_wrap``) is not ported; fixation numbers are drawn next to each
fixation with a fixed offset instead of R's collision-free placement solver
with leaders and grouped labels; colour bars are matplotlib colour bars (the
banded types use the band edges as ticks); the duration size legend is
omitted. Output is not pixel-compatible with ggplot2.
"""

from __future__ import annotations

import warnings

import numpy as np

from peyesim.density import EyeDensity
from peyesim.fixations import FixationGroup

_COLOUR_ROLES = {
    "reference": "#1B2A41",
    "source": "#C23B4B",
    "raw": "#8A8F98",
    "correspondence": "#5B4FCF",
    "diagnostic": "#3F4A5A",
    "ink": "#1F2328",
    "muted": "#5A6270",
    "rule": "#D5D9DE",
    "grid": "#ECEEF1",
}

# viridisLite::viridis(7, begin = 0.05, end = 0.82)
EYESIM_TIME_COLOURS = (
    "#481467", "#433E84", "#34618D", "#26818E", "#1FA088", "#3FBC73", "#86D549",
)

# viridisLite::rocket(9, begin = 0.12, end = 0.97, direction = -1)
EYESIM_DENSITY_COLOURS = (
    "#F9DFCC", "#F6B792", "#F58A61", "#EF5840", "#D92847",
    "#B21758", "#841E5A", "#581E4F", "#2E1739",
)

_TRANSFORMS = ("identity", "sqroot", "curoot", "rank")

# ggplot2 size units are millimetres of diameter; matplotlib marker area is pt^2.
_PT_PER_MM = 72.27 / 25.4


def eyesim_colours(role=None) -> dict[str, str]:
    """Named colour roles used by every eyesim plot.

    Parameters
    ----------
    role : str or sequence of str, optional
        Role names to return. ``None`` returns all roles.

    Returns
    -------
    dict
        Role name to hex colour, in the requested order.
    """
    if role is None:
        return dict(_COLOUR_ROLES)
    roles = [role] if isinstance(role, str) else list(role)
    unknown = [r for r in roles if r not in _COLOUR_ROLES]
    if unknown:
        raise ValueError(f"Unknown eyesim colour role(s): {', '.join(unknown)}")
    return {r: _COLOUR_ROLES[r] for r in roles}


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


def _check_transform(transform: str) -> str:
    transform = transform.lower()
    if transform not in _TRANSFORMS:
        raise ValueError("transform must be one of {'identity', 'sqroot', 'curoot', 'rank'}")
    return transform


def _transform_values(values, transform: str = "identity"):
    transform = _check_transform(transform)
    arr = np.asarray(values, dtype=float)
    if transform == "identity":
        return arr
    if transform == "sqroot":
        return np.sqrt(np.maximum(arr, 0))
    if transform == "curoot":
        return np.cbrt(arr)
    flat = arr.ravel()
    valid = ~np.isnan(flat)
    ranks = np.full(flat.shape, np.nan, dtype=float)
    if valid.any():
        order = np.argsort(flat[valid], kind="mergesort")
        valid_ranks = np.empty(order.shape, dtype=float)
        valid_ranks[order] = np.arange(1, len(order) + 1)
        ranks[valid] = valid_ranks
    return ranks.reshape(arr.shape)


def _ramp_rgba(colours, n: int) -> np.ndarray:
    from matplotlib.colors import LinearSegmentedColormap, to_rgba_array

    base = to_rgba_array(list(colours))
    n = max(int(n), len(base))
    if len(base) == 1:
        return np.repeat(base, n, axis=0)
    return LinearSegmentedColormap.from_list("ramp", base, N=n)(np.linspace(0, 1, n))


def density_opacity(position, alpha_range=(0.0, 0.9)) -> np.ndarray:
    """Opacity of the density ramp at ``position`` in ``[0, 1]`` (square-root ramp)."""
    position = np.clip(np.asarray(position, dtype=float), 0.0, 1.0)
    return alpha_range[0] + (alpha_range[1] - alpha_range[0]) * position ** 0.5


def eyesim_density_cmap(colours=None, alpha_range=(0.0, 0.9), transparent: bool = True, n: int = 256):
    """Matplotlib colormap for the eyesim density scale.

    Opacity rises from ``alpha_range[0]`` at zero density to ``alpha_range[1]``
    at the top of the scale. Values above the scale take the top colour and
    missing values are transparent. Equivalent of R
    ``scale_fill_eyesim_density()``.
    """
    from matplotlib.colors import ListedColormap

    base = EYESIM_DENSITY_COLOURS if colours is None else colours
    rgba = _ramp_rgba(base, n)
    if transparent:
        rgba[:, 3] = density_opacity(np.linspace(0, 1, len(rgba)), alpha_range)
    return ListedColormap(rgba, name="eyesim_density").with_extremes(
        over=rgba[-1], bad=(0, 0, 0, 0), under=(0, 0, 0, 0)
    )


def eyesim_time_cmap(colours=None, n: int = 256):
    """Matplotlib colormap for fixation order/onset (R ``scale_colour_eyesim_time()``)."""
    from matplotlib.colors import ListedColormap

    base = EYESIM_TIME_COLOURS if colours is None else colours
    return ListedColormap(_ramp_rgba(base, n), name="eyesim_time")


def theme_eyesim(ax):
    """Apply the eyesim house style to a matplotlib ``Axes`` and return it."""
    col = _COLOUR_ROLES
    ax.set_facecolor("none")
    ax.title.set_color(col["ink"])
    ax.title.set_fontweight("bold")
    ax.title.set_horizontalalignment("left")
    ax.title.set_x(0.0)
    ax.tick_params(colors=col["muted"], labelsize="small")
    ax.xaxis.label.set_color(col["muted"])
    ax.yaxis.label.set_color(col["muted"])
    ax.grid(True, color=col["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name in ("left", "bottom"))
        spine.set_color(col["rule"])
    return ax


def theme_eyesim_spatial(ax):
    """Stimulus-space variant: no axes or grid, a thin frame at the stimulus extent."""
    theme_eyesim(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(_COLOUR_ROLES["rule"])
        spine.set_linewidth(0.6)
    return ax


def format_density(x) -> list[str]:
    """Colour-bar labels for density values: one notation per scale."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    finite = x[np.isfinite(x) & (x != 0)]
    small = finite.size > 0 and np.max(np.abs(finite)) < 1e-2
    out = [_format_e(v, 1) if small else _format_g(v, 2) for v in x]
    return ["0" if (np.isfinite(v) and v == 0) else s for v, s in zip(x, out)]


def format_limit(x) -> list[str]:
    """A limit to three significant digits, so ``>= limit`` is not rounded across it."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.size == 0:
        return []
    if abs(x[0]) < 1e-2:
        return [_format_e(v, 2) for v in x]
    return [_format_g(v, 3) for v in x]


def _format_e(v: float, digits: int) -> str:
    # C printf, as R's formatC(format = "e"); exponent has at least two digits
    return f"{v:.{digits}e}"


def _format_g(v: float, digits: int) -> str:
    return f"{v:.{digits}g}"


def density_legend_title(transform: str = "identity", base: str = "Fixation\ndensity") -> str:
    suffix = {"rank": "\n(rank, low to high)", "sqroot": "\n(square root)", "curoot": "\n(cube root)"}
    return base + suffix.get(transform, "")


def _eye_density_quantity(z) -> str:
    total = np.nansum(z)
    return "Fixation\nproportion\nper cell" if abs(total - 1) < 1e-6 else "Fixation\ndensity"


def _expanded_limits(values, *, proportion: float = 0.05) -> tuple[float, float]:
    """Range of ``values`` expanded like ggplot2's default continuous scale (5%)."""
    arr = np.asarray(values, dtype=float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = hi - lo
    if span <= np.finfo(float).eps:
        return (lo - 0.5, hi + 0.5)
    pad = span * proportion
    return (lo - pad, hi + pad)


def _fixation_point_sizes(duration, *, for_matplotlib: bool = True, max_size: float = 5.0) -> np.ndarray:
    """Point sizes with area proportional to duration (ggplot2 ``scale_size_area``).

    The aesthetic size is ``max_size * sqrt(duration / max(duration))`` (a ggplot2
    size, roughly a diameter in mm). For matplotlib the diameter is converted to
    points and squared, since ``scatter(s=)`` is a marker area.
    """
    duration = np.asarray(duration, dtype=float)
    if duration.size == 0:
        return np.array([], dtype=float)
    top = np.nanmax(duration)
    if not np.isfinite(top) or top <= 0:
        aesthetic = np.full(duration.shape, max_size)
    else:
        aesthetic = max_size * np.sqrt(np.clip(duration, 0, None) / top)
    return (aesthetic * _PT_PER_MM) ** 2 if for_matplotlib else aesthetic


def _kde2d(x, y, h, n=100, lims=None):
    """``MASS::kde2d``: Gaussian KDE on an ``n x n`` grid, kernel sd ``h / 4``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = np.broadcast_to(np.asarray(n, dtype=int), (2,))
    h = np.broadcast_to(np.asarray(h, dtype=float), (2,)) / 4.0
    if lims is None:
        lims = (x.min(), x.max(), y.min(), y.max())
    gx = np.linspace(lims[0], lims[1], n[0])
    gy = np.linspace(lims[2], lims[3], n[1])
    kx = np.exp(-0.5 * ((gx[:, None] - x[None, :]) / h[0]) ** 2) / np.sqrt(2 * np.pi)
    ky = np.exp(-0.5 * ((gy[:, None] - y[None, :]) / h[1]) ** 2) / np.sqrt(2 * np.pi)
    z = kx @ ky.T / (len(x) * h[0] * h[1])
    return gx, gy, z


def _density_extent(x, y, xlim, ylim, bandwidth):
    """Limits of the KDE grid: data plus the visible extent padded by ``bandwidth``."""
    xr = (min(np.min(x), xlim[0] - bandwidth), max(np.max(x), xlim[1] + bandwidth))
    yr = (min(np.min(y), ylim[0] - bandwidth), max(np.max(y), ylim[1] + bandwidth))
    return xr + yr


def _kde_peak(x, y, h, xlim, ylim, n=100):
    """Peak of the KDE over ``range(c(x, xlim))`` x ``range(c(y, ylim))`` (R ``kde_peak``)."""
    lims = (min(np.min(x), xlim[0]), max(np.max(x), xlim[1]),
            min(np.min(y), ylim[0]), max(np.max(y), ylim[1]))
    return float(np.max(_kde2d(x, y, h, n=n, lims=lims)[2]))


def _resolve_limits(z, limits, transform):
    """Colour-scale limits on the transformed scale, zero-anchored unless rank."""
    if transform == "rank":
        if limits is not None:
            warnings.warn('`limits` is ignored for transform = "rank".', stacklevel=3)
        return float(np.nanmin(z)), float(np.nanmax(z)), False
    lo, hi = (0.0, np.nan) if limits is None else (float(limits[0]), float(limits[1]))
    overflow = limits is not None and np.isfinite(hi)
    lo = 0.0 if not np.isfinite(lo) else lo
    if not np.isfinite(hi):
        hi = float(np.nanmax(z))
    tlo, thi = _transform_values([lo, hi], transform)
    if not thi > tlo:
        thi = tlo + 1.0
    return float(tlo), float(thi), overflow


def _load_bg_image(bg_image, imread):
    img = imread(bg_image) if isinstance(bg_image, str) else np.asarray(bg_image)
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating) and np.nanmax(img) > 1:
        img = img / np.nanmax(img)
    return img


def _draw_bg(ax, bg_image, xlim, ylim, imread):
    """Draw the stimulus with its stored colours, stretched to the extent, top row at ylim[1]."""
    img = _load_bg_image(bg_image, imread)
    kwargs = {}
    if img.ndim == 2:
        kwargs = {"cmap": "gray", "vmin": 0.0, "vmax": 255.0 if img.dtype == np.uint8 else 1.0}
    ax.imshow(img, extent=(xlim[0], xlim[1], ylim[0], ylim[1]), origin="upper",
              aspect="auto", interpolation="nearest", zorder=0, **kwargs)


def _colorbar(ax, mappable, label, *, ticks=None, ticklabels=None, rank=False):
    cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.7, pad=0.03)
    cbar.set_label(label, color=_COLOUR_ROLES["muted"], fontsize="small")
    cbar.ax.tick_params(colors=_COLOUR_ROLES["muted"], labelsize="x-small")
    cbar.outline.set_visible(False)
    if rank:
        cbar.set_ticks([])
    elif ticks is not None:
        cbar.set_ticks(ticks)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
    return cbar


def _density_ticks(lo, hi, overflow, raw_hi, transform):
    """Colour-bar ticks: a few round values plus ``>= limit`` at the top when clamped."""
    from matplotlib.ticker import MaxNLocator

    if transform == "rank":
        return None, None
    ticks = [t for t in MaxNLocator(5).tick_values(lo, hi) if lo - 1e-12 <= t <= hi + 1e-12]
    if not overflow:
        return ticks, None
    raw = [float(np.sign(t) * abs(t) ** {"identity": 1, "sqroot": 2, "curoot": 3}[transform]) for t in ticks]
    keep = [i for i, r in enumerate(raw) if r < raw_hi * 0.85]
    ticks = [ticks[i] for i in keep] + [hi]
    labels = format_density([raw[i] for i in keep]) + [">= " + format_limit([raw_hi])[0]]
    return ticks, labels


def plot_eye_density(
    x: EyeDensity,
    alpha: float = 0.8,
    bg_image=None,
    transform: str = "identity",
    ax=None,
    cmap=None,
    colours=None,
    legend: bool = True,
    limits=None,
    aspect: str = "equal",
):
    """Plot an :class:`~peyesim.density.EyeDensity` on the eyesim density scale.

    Zero density is transparent and opacity rises to ``alpha`` at the top of
    the scale. ``limits`` (default ``(0, max)``) put several maps on one scale;
    density above the upper limit takes the top colour. ``colours`` (or a
    matplotlib ``cmap``) replaces the density ramp. Over a ``bg_image``, thin
    white contours at five evenly spaced levels keep density edges visible.

    Returns the matplotlib ``Axes`` so callers can keep composing the figure.
    """
    plt, imread = _require_matplotlib()
    if not isinstance(x, EyeDensity):
        raise TypeError("plot_eye_density() expects an EyeDensity object.")
    transform = _check_transform(transform)
    if aspect not in ("equal", "free"):
        raise ValueError("aspect must be 'equal' or 'free'")
    if ax is None:
        _, ax = plt.subplots()

    xlim = (float(np.min(x.x)), float(np.max(x.x)))
    ylim = (float(np.min(x.y)), float(np.max(x.y)))

    if bg_image is not None:
        _draw_bg(ax, bg_image, xlim, ylim, imread)

    raw = np.asarray(x.z, dtype=float)
    z = _transform_values(raw, transform=transform)
    lo, hi, overflow = _resolve_limits(raw if transform != "rank" else z, limits, transform)
    if colours is None and cmap is not None:
        import matplotlib

        colours = matplotlib.colormaps[cmap](np.linspace(0, 1, 9)) if isinstance(cmap, str) else cmap(np.linspace(0, 1, 9))
    dcmap = eyesim_density_cmap(colours, alpha_range=(0.0, alpha))
    im = ax.imshow(
        z.T,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        cmap=dcmap,
        vmin=lo,
        vmax=hi,
        aspect="equal" if aspect == "equal" else "auto",
        zorder=1,
    )
    if bg_image is not None:
        top = float(limits[1]) if limits is not None and np.isfinite(limits[1]) else float(np.nanmax(raw))
        levels = np.linspace(0, top, 6)[1:]
        if top > 0:
            xx, yy = np.meshgrid(np.asarray(x.x, float), np.asarray(x.y, float))
            ax.contour(xx, yy, raw.T, levels=levels, colors="white", linewidths=0.3, alpha=0.8, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    theme_eyesim_spatial(ax)
    if legend:
        ticks, labels = _density_ticks(lo, hi, overflow, None if limits is None else float(limits[1]), transform)
        _colorbar(ax, im, density_legend_title(transform, _eye_density_quantity(raw)),
                  ticks=ticks, ticklabels=labels, rank=transform == "rank")
    return ax


def _onset_order(df, window):
    """Number fixations by onset over the whole sequence, then window and sort."""
    df = df.copy()
    df["index"] = df["onset"].rank(method="first").astype(int)
    if window is not None:
        if len(window) != 2 or window[1] <= window[0]:
            raise ValueError("window must be a length-2 increasing interval.")
        df = df.loc[(df["onset"] >= window[0]) & (df["onset"] < window[1])]
    return df.sort_values("onset", kind="mergesort")


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
    bg_image=None,
    alpha_range: tuple[float, float] = (0.0, 0.9),
    alpha: float = 0.8,
    window: tuple[float, float] | None = None,
    transform: str = "identity",
    ax=None,
    cmap=None,
    colours=None,
    legend: bool = True,
    limits=None,
    aspect: str = "equal",
    label_fixations: bool = True,
):
    """Plot a fixation group in stimulus coordinates with the eyesim style.

    ``type="points"`` draws the scanpath: fixations coloured by onset, sized by
    duration (area proportional to duration), numbered and joined in onset
    order, with a ring on the first fixation. The density types (``contour``,
    ``filled_contour``, ``density``, ``raster``) draw a kernel density
    (``MASS::kde2d`` bandwidth ``bandwidth``) on the eyesim density scale with
    the fixations overlaid in ink. ``density``/``filled_contour`` use bands
    between evenly spaced edges from zero to the upper ``limits`` (or the
    peak), plus an open top band; the band containing zero is transparent.

    ``xlim``/``ylim`` set the visible extent (and the extent of ``bg_image``)
    without dropping fixations; ``aspect="equal"`` (default) keeps a 1:1
    aspect ratio, ``"free"`` fills the panel. ``colours`` (or a matplotlib
    ``cmap``) replaces the time ramp for points and the density ramp for the
    density types. Fixations are labelled when ``label_fixations`` and there
    are fewer than 50.
    """
    plt, imread = _require_matplotlib()
    if not isinstance(x, FixationGroup):
        raise TypeError("plot_fixation_group() expects a FixationGroup object.")
    if type not in {"points", "contour", "filled_contour", "density", "raster"}:
        raise ValueError("type must be one of {'points', 'contour', 'filled_contour', 'density', 'raster'}")
    if aspect not in ("equal", "free"):
        raise ValueError("aspect must be 'equal' or 'free'")
    transform = _check_transform(transform)
    density_type = type != "points"
    col = _COLOUR_ROLES

    df = _onset_order(x.to_pandas(copy=True), window)
    if df.empty:
        raise ValueError("No fixations remain after filtering.")
    if colours is None and cmap is not None:
        import matplotlib

        colours = (matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap)(np.linspace(0, 1, 9))

    if ax is None:
        _, ax = plt.subplots()
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    # Defaults follow R: the data range (padded by the bandwidth for density
    # types without a stimulus, so contours close), then ggplot2's 5% expansion
    # unless a stimulus fixes the extent.
    expand = bg_image is None
    if xlim is None:
        xlim = (float(np.min(xs)), float(np.max(xs)))
        if density_type and bg_image is None:
            xlim = (xlim[0] - bandwidth, xlim[1] + bandwidth)
    if ylim is None:
        ylim = (float(np.min(ys)), float(np.max(ys)))
        if density_type and bg_image is None:
            ylim = (ylim[0] - bandwidth, ylim[1] + bandwidth)
    xlim = tuple(float(v) for v in xlim)
    ylim = tuple(float(v) for v in ylim)
    shown_xlim = _expanded_limits(xlim) if expand else xlim
    shown_ylim = _expanded_limits(ylim) if expand else ylim
    if bins is None:
        bins = max(int(len(x) / 10), 4)

    if bg_image is not None:
        _draw_bg(ax, bg_image, xlim, ylim, imread)

    if density_type:
        h = float(bandwidth)
        gx, gy, z = _kde2d(xs, ys, h, n=100, lims=_density_extent(xs, ys, xlim, ylim, h))
        _draw_density(ax, type, gx, gy, z, xs, ys, xlim, ylim, h, transform, limits, colours,
                      alpha_range, bins, bg_image is not None, legend)
    elif show_path and len(df) > 1:
        from matplotlib.collections import LineCollection

        coords = np.column_stack([xs, ys])
        segments = np.stack([coords[:-1], coords[1:]], axis=1)
        path = LineCollection(segments, cmap=eyesim_time_cmap(colours), alpha=0.75, linewidth=1.0, zorder=2)
        onset = df["onset"].to_numpy(dtype=float)
        path.set_array(onset[:-1])
        path.set_clim(np.min(onset), np.max(onset))
        ax.add_collection(path)

    if show_points:
        sizes = (_fixation_point_sizes(df["duration"].to_numpy(dtype=float))
                 if size_points and "duration" in df else np.full(len(df), (1.5 * _PT_PER_MM) ** 2))
        if density_type:
            ax.scatter(xs, ys, s=sizes, color=col["ink"], edgecolors="white", linewidths=0.3,
                       alpha=alpha * 0.8, zorder=4)
        else:
            onset = df["onset"].to_numpy(dtype=float)
            pts = ax.scatter(xs, ys, s=sizes, c=onset, cmap=eyesim_time_cmap(colours), alpha=alpha,
                             edgecolors="white", linewidths=0.4, zorder=3)
            if legend:
                _colorbar(ax, pts, "Onset")
        # Ring marking the start of the scanpath.
        ring = (7 * _PT_PER_MM) ** 2
        ax.scatter(xs[:1], ys[:1], s=ring, facecolors="none", edgecolors="white", linewidths=1.8, zorder=5)
        ax.scatter(xs[:1], ys[:1], s=ring, facecolors="none", edgecolors=col["ink"], linewidths=0.7, zorder=5)
        if label_fixations and len(df) < 50:
            for xi, yi, idx in zip(xs, ys, df["index"].to_numpy()):
                ax.annotate(str(int(idx)), (xi, yi), xytext=(5, 5), textcoords="offset points",
                            fontsize="x-small", color=col["ink"], zorder=6)

    ax.set_xlim(*shown_xlim)
    ax.set_ylim(*shown_ylim)
    ax.set_aspect("equal" if aspect == "equal" else "auto", adjustable="box")
    theme_eyesim_spatial(ax)
    return ax


def _band_colours(edges, cmap_colours, alpha_range, transform):
    """RGBA for each band, coloured at its midpoint; the zero band is transparent."""
    from matplotlib.colors import to_rgba_array

    base = EYESIM_DENSITY_COLOURS if cmap_colours is None else cmap_colours
    ramp = _ramp_rgba(base, 256)
    finite = np.asarray(edges[:-1], dtype=float)
    step = finite[1] - finite[0] if len(finite) > 1 else 1.0
    lows = np.asarray(edges[:-1], dtype=float)
    highs = np.asarray(edges[1:], dtype=float)
    mids = np.where(np.isfinite(highs), (lows + highs) / 2, lows + step / 2)
    span = (finite[-1] + step) if not np.isfinite(edges[-1]) else finite[-1]
    t = _transform_values(np.concatenate([[finite[0]], mids, [span]]), transform)
    pos = np.clip((t[1:-1] - t[0]) / (t[-1] - t[0]), 0, 1)
    rgba = ramp[np.round(pos * (len(ramp) - 1)).astype(int)].copy()
    rgba[:, 3] = density_opacity(pos, alpha_range)
    rgba[lows <= 0] = (0, 0, 0, 0)
    return to_rgba_array(rgba)


def _draw_density(ax, type, gx, gy, z, xs, ys, xlim, ylim, h, transform, limits, colours,
                  alpha_range, bins, has_bg, legend):
    xx, yy = np.meshgrid(gx, gy)
    if type in ("density", "filled_contour"):
        n_bands = bins if type == "density" else 10
        if transform == "rank" and limits is not None:
            warnings.warn('`limits` is ignored for transform = "rank".', stacklevel=3)
            limits = None
        if limits is not None and np.isfinite(limits[0]) and limits[0] != 0:
            warnings.warn(f'Density bands start at zero; the lower limit is ignored for type = "{type}".',
                          stacklevel=3)
        overflow = limits is not None and np.isfinite(limits[1])
        if transform == "rank":
            zt = _transform_values(z, "rank")
            edges = np.linspace(np.nanmin(zt) - 1, np.nanmax(zt), n_bands + 1)
            rgba = _band_colours(np.append(edges, np.inf), colours,
                                 (max(alpha_range[0], 0.3), alpha_range[1]), "identity")[:-1]
            rgba[0] = (0, 0, 0, 0)
            cs = ax.contourf(xx, yy, zt.T, levels=edges, colors=rgba, zorder=1)
        else:
            top = float(limits[1]) if overflow else _kde_peak(xs, ys, h, (xlim[0] - h, xlim[1] + h),
                                                              (ylim[0] - h, ylim[1] + h))
            edges = np.linspace(0.0, top, n_bands + 1)
            all_edges = np.append(edges, np.inf)
            rgba = _band_colours(all_edges, colours, alpha_range, transform)
            cs = ax.contourf(xx, yy, z.T, levels=all_edges, colors=rgba, zorder=1)
            if has_bg:
                ax.contour(xx, yy, z.T, levels=edges[1:], colors="white", linewidths=0.3, alpha=0.8, zorder=2)
        if legend:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import BoundaryNorm, ListedColormap

            shown = edges if transform == "rank" or not overflow else np.append(edges, edges[-1] + edges[1])
            shown_rgba = rgba[: len(shown) - 1]
            sm = ScalarMappable(norm=BoundaryNorm(shown, len(shown) - 1), cmap=ListedColormap(shown_rgba))
            labels = None if transform == "rank" else format_density(shown)
            if labels is not None and overflow:
                labels[-2] = ">= " + format_limit([edges[-1]])[0]
                labels[-1] = ""
            _colorbar(ax, sm, density_legend_title(transform), ticks=list(shown), ticklabels=labels,
                      rank=transform == "rank")
        return cs

    zt = _transform_values(z, transform)
    lo, hi, overflow = _resolve_limits(z if transform != "rank" else zt, limits, transform)
    if type == "contour":
        cmap = eyesim_density_cmap(colours, alpha_range=(0.5, 1.0))
        art = ax.contour(xx, yy, zt.T, cmap=cmap, vmin=lo, vmax=hi, linewidths=0.8, zorder=2)
    else:
        cmap = eyesim_density_cmap(colours, alpha_range=alpha_range)
        art = ax.imshow(zt.T, extent=(gx[0], gx[-1], gy[0], gy[-1]), origin="lower", cmap=cmap,
                        vmin=lo, vmax=hi, aspect="auto", interpolation="bilinear", zorder=1)
        if has_bg:
            top = float(limits[1]) if overflow else _kde_peak(xs, ys, h, (xlim[0] - h, xlim[1] + h),
                                                              (ylim[0] - h, ylim[1] + h))
            ax.contour(xx, yy, z.T, levels=np.linspace(0, top, 6)[1:], colors="white",
                       linewidths=0.3, alpha=0.8, zorder=2)
    if legend:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm = ScalarMappable(norm=Normalize(lo, hi), cmap=cmap)
        ticks, labels = _density_ticks(lo, hi, overflow, None if limits is None else float(limits[1]), transform)
        _colorbar(ax, sm, density_legend_title(transform), ticks=ticks, ticklabels=labels,
                  rank=transform == "rank")
    return art


def anim_scanpath(
    x: FixationGroup,
    bg_image=None,
    xlim=None,
    ylim=None,
    alpha: float = 1.0,
    anim_over: str = "index",
    type: str = "points",
    time_bin: float = 1,
):
    """Animate a fixation scanpath with matplotlib.

    Fixations appear one at a time (by order or by onset); earlier fixations
    stay visible as faded marks, and colour encodes onset on the eyesim time
    scale. Frames are titled ``"Fixation N"``, ``"Onset N"`` or
    ``"Time bin N"``. ``type="raster"`` accumulates a kernel density
    (bandwidth 100) on the density scale.

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
    expand = bg_image is None
    if xlim is None:
        xlim = (float(df["x"].min()), float(df["x"].max()))
    if ylim is None:
        ylim = (float(df["y"].min()), float(df["y"].max()))
    shown_xlim = _expanded_limits(xlim) if expand else tuple(xlim)
    shown_ylim = _expanded_limits(ylim) if expand else tuple(ylim)

    if time_bin > 1:
        df["frame"] = np.round(df["onset"] / time_bin).astype(int)
        anim_over = "time_bin"
    elif anim_over == "index":
        df["frame"] = df["index"].to_numpy() if "index" in df else np.arange(1, len(df) + 1)
    else:
        df["frame"] = df["onset"].to_numpy()
    title = {"index": "Fixation {}", "onset": "Onset {}", "time_bin": "Time bin {}"}[anim_over]

    frames = np.sort(df["frame"].unique())

    fig, ax = plt.subplots()
    if bg_image is not None:
        _draw_bg(ax, bg_image, xlim, ylim, imread)
    ax.set_xlim(*shown_xlim)
    ax.set_ylim(*shown_ylim)
    ax.set_aspect("equal", adjustable="box")
    theme_eyesim_spatial(ax)
    onset = df["onset"].to_numpy(dtype=float)
    clim = (float(np.min(onset)), float(np.max(onset))) if len(onset) else (0.0, 1.0)
    point_area = (4.5 * _PT_PER_MM) ** 2
    shadow_area = (2.2 * _PT_PER_MM) ** 2

    if type == "points":
        tcmap = eyesim_time_cmap()
        current = ax.scatter([], [], c=[], cmap=tcmap, s=point_area, alpha=alpha,
                             edgecolors="white", linewidths=0.5, zorder=3)
        shadow = ax.scatter([], [], c=[], cmap=tcmap, s=shadow_area, alpha=0.35,
                            edgecolors="none", zorder=2)
        current.set_clim(*clim)
        shadow.set_clim(*clim)

        def update(frame):
            now = df.loc[df["frame"] == frame]
            past = df.loc[df["frame"] < frame]
            for artist, sel in ((current, now), (shadow, past)):
                offsets = sel[["x", "y"]].to_numpy(dtype=float)
                artist.set_offsets(offsets if len(offsets) else np.empty((0, 2)))
                artist.set_array(sel["onset"].to_numpy(dtype=float))
            ax.set_title(title.format(int(round(float(frame)))))
            return (current, shadow)

    else:
        lims = (shown_xlim[0], shown_xlim[1], shown_ylim[0], shown_ylim[1])
        image = ax.imshow(
            np.zeros((100, 100)),
            extent=lims,
            origin="lower",
            cmap=eyesim_density_cmap(alpha_range=(0.0, 0.9)),
            alpha=alpha,
            aspect="auto",
        )

        def update(frame):
            sel = df.loc[df["frame"] <= frame]
            z = _kde2d(sel["x"].to_numpy(float), sel["y"].to_numpy(float), 100.0, n=100, lims=lims)[2]
            image.set_data(z.T)
            image.set_clim(0.0, float(z.max()) if z.max() > 0 else 1.0)
            ax.set_title(title.format(int(round(float(frame)))))
            return (image,)

    return FuncAnimation(fig, update, frames=frames, blit=False)
