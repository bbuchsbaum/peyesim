"""Optional visualization helpers."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from peyesim.fixations import FixationGroup


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
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.image import imread
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for anim_scanpath(). Install it with `pip install matplotlib`."
        ) from exc

    if anim_over not in {"index", "onset"}:
        raise ValueError("anim_over must be one of {'index', 'onset'}")
    if type not in {"points", "raster"}:
        raise ValueError("type must be one of {'points', 'raster'}")

    df = x.copy()
    if xlim is None:
        xlim = (float(df["x"].min()), float(df["x"].max()))
    if ylim is None:
        ylim = (float(df["y"].min()), float(df["y"].max()))

    if time_bin > 1:
        df["frame"] = np.round(df["onset"] / time_bin).astype(int)
    elif anim_over == "index":
        df["frame"] = np.arange(len(df))
    else:
        df["frame"] = df["onset"].to_numpy()

    frames = np.sort(df["frame"].unique())

    fig, ax = plt.subplots()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("Scanpath animation")

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
            ax.set_xlabel(f"frame={frame}")
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
            ax.set_xlabel(f"frame={frame}")
            return (image,)

    return FuncAnimation(fig, update, frames=frames, blit=False)
