"""Kernel density estimation for fixation data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import iqr as scipy_iqr
from scipy.ndimage import gaussian_filter

from peyesim.fixations import FixationGroup


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EyeDensity:
    """A 2-D fixation density map (mirrors R ``eye_density``)."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sigma: float
    fixgroup: FixationGroup | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "sigma": self.sigma,
            "xlim": (self.x.min(), self.x.max()),
            "ylim": (self.y.min(), self.y.max()),
            "grid_dim": self.z.shape,
            "z_range": (self.z.min(), self.z.max()),
            "z_mean": self.z.mean(),
            "z_sum": self.z.sum(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Expand to long-form DataFrame with x, y, z columns."""
        xx, yy = np.meshgrid(self.x, self.y, indexing="ij")
        return pd.DataFrame({"x": xx.ravel(), "y": yy.ravel(), "z": self.z.ravel()})

    # arithmetic mirrors R Ops.eye_density
    def __sub__(self, other: "EyeDensity") -> "EyeDensity":
        assert np.allclose(self.x, other.x) and np.allclose(self.y, other.y)
        return EyeDensity(self.x, self.y, self.z - other.z, self.sigma)

    def __add__(self, other: "EyeDensity") -> "EyeDensity":
        assert np.allclose(self.x, other.x) and np.allclose(self.y, other.y)
        return EyeDensity(self.x, self.y, (self.z + other.z) / 2, self.sigma)

    def __truediv__(self, other: "EyeDensity") -> "EyeDensity":
        assert np.allclose(self.x, other.x) and np.allclose(self.y, other.y)
        with np.errstate(divide="ignore", invalid="ignore"):
            div = np.log(self.z / other.z)
        return EyeDensity(self.x, self.y, div, self.sigma)

    def __repr__(self):
        return (
            f"EyeDensity(sigma={self.sigma}, "
            f"xlim=[{self.x.min():.1f}, {self.x.max():.1f}], "
            f"ylim=[{self.y.min():.1f}, {self.y.max():.1f}], "
            f"z_range=[{self.z.min():.4g}, {self.z.max():.4g}])"
        )


@dataclass
class EyeDensityMultiscale:
    """A list of :class:`EyeDensity` objects at different bandwidths."""
    scales: list[EyeDensity]

    @property
    def sigmas(self) -> list[float]:
        return [s.sigma for s in self.scales]

    def __len__(self):
        return len(self.scales)

    def __getitem__(self, idx):
        return self.scales[idx]

    def __iter__(self):
        return iter(self.scales)

    def __repr__(self):
        return (
            f"EyeDensityMultiscale(n_scales={len(self.scales)}, "
            f"sigmas={self.sigmas})"
        )


# ---------------------------------------------------------------------------
# Core KDE
# ---------------------------------------------------------------------------

def _kde2d_weighted(
    x_data: np.ndarray,
    y_data: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    xbounds: tuple[float, float],
    ybounds: tuple[float, float],
    outdim: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted 2-D KDE using binning + Gaussian filter (fast, matches ks::kde behavior)."""
    gx = np.linspace(xbounds[0], xbounds[1], outdim[0])
    gy = np.linspace(ybounds[0], ybounds[1], outdim[1])

    dx = gx[1] - gx[0] if len(gx) > 1 else 1.0
    dy = gy[1] - gy[0] if len(gy) > 1 else 1.0

    # Bin the data
    z = np.zeros(outdim, dtype=float)
    xi = np.clip(np.round((x_data - xbounds[0]) / dx).astype(int), 0, outdim[0] - 1)
    yi = np.clip(np.round((y_data - ybounds[0]) / dy).astype(int), 0, outdim[1] - 1)

    np.add.at(z, (xi, yi), weights)

    # Smooth with Gaussian kernel
    sigma_pixels_x = sigma / dx if dx > 0 else 0
    sigma_pixels_y = sigma / dy if dy > 0 else 0
    z = gaussian_filter(z, sigma=[sigma_pixels_x, sigma_pixels_y], mode="constant")

    return gx, gy, z


def _compute_single_eye_density(
    fg: FixationGroup,
    sigma_val: float,
    xbounds: tuple[float, float],
    ybounds: tuple[float, float],
    outdim: tuple[int, int],
    normalize: bool,
    duration_weighted: bool,
    weights: np.ndarray,
) -> EyeDensity | None:
    """Compute a single-scale density map."""
    x_data = fg["x"].to_numpy()
    y_data = fg["y"].to_numpy()

    gx, gy, z = _kde2d_weighted(x_data, y_data, weights, sigma_val,
                                 xbounds, ybounds, outdim)

    if normalize:
        s = z.sum()
        if s > np.finfo(float).eps:
            z = z / s
        else:
            warnings.warn(f"Sum of density matrix is near zero, cannot normalize. Sigma: {sigma_val}")

    z = np.around(z, decimals=15)  # zapsmall equivalent

    return EyeDensity(x=gx, y=gy, z=z, sigma=sigma_val, fixgroup=fg)


def eye_density(
    fg: FixationGroup,
    sigma: float | Sequence[float] = 50,
    xbounds: tuple[float, float] | None = None,
    ybounds: tuple[float, float] | None = None,
    outdim: tuple[int, int] = (100, 100),
    normalize: bool = True,
    duration_weighted: bool = False,
    window: tuple[float, float] | None = None,
    min_fixations: int = 2,
    origin: tuple[float, float] = (0, 0),
    weights: np.ndarray | None = None,
) -> EyeDensity | EyeDensityMultiscale | None:
    """Compute a density map for a :class:`FixationGroup`.

    Parameters mirror the R ``eye_density.fixation_group`` method.
    """
    sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=float))
    if np.any(sigma_arr <= 0):
        raise ValueError("sigma must be a positive numeric value or vector")

    if xbounds is None:
        xbounds = (fg["x"].min(), fg["x"].max())
    if ybounds is None:
        ybounds = (fg["y"].min(), fg["y"].max())

    # Window filtering
    fg_filtered = fg
    if window is not None:
        if len(window) != 2:
            raise ValueError("Window must be length 2")
        if window[1] <= window[0]:
            raise ValueError("window[1] must be > window[0]")
        fg_filtered = fg[
            (fg["onset"] >= window[0]) & (fg["onset"] < window[1])
        ].reset_index(drop=True)
        fg_filtered = FixationGroup(fg_filtered)
        if len(fg_filtered) == 0:
            warnings.warn("No fixations remain after applying the window filter. Returning None.")
            return None

    if len(fg_filtered) < min_fixations:
        warnings.warn(
            f"Not enough fixations (need >= {min_fixations}) to compute density. "
            f"Returning None. Provided: {len(fg_filtered)}"
        )
        return None

    # Weights
    if weights is not None:
        w = np.asarray(weights, dtype=float)
    elif duration_weighted:
        w = fg_filtered["duration"].to_numpy(dtype=float)
    else:
        w = np.ones(len(fg_filtered))

    if len(sigma_arr) > 1:
        # Multiscale
        densities = []
        for s in sigma_arr:
            d = _compute_single_eye_density(fg_filtered, s, xbounds, ybounds,
                                            outdim, normalize, duration_weighted, w)
            if d is not None:
                densities.append(d)
        if len(densities) == 0:
            warnings.warn("All multiscale density computations failed.")
            return None
        return EyeDensityMultiscale(scales=densities)
    else:
        return _compute_single_eye_density(
            fg_filtered, sigma_arr[0], xbounds, ybounds,
            outdim, normalize, duration_weighted, w
        )


def gen_density(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> EyeDensity:
    """Create an :class:`EyeDensity` from pre-computed grids."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if z.shape != (len(x), len(y)):
        raise ValueError("length of x and y must equal nrow(z) and ncol(z)")
    return EyeDensity(x=x, y=y, z=z, sigma=0.0)


def get_density(x: EyeDensity | EyeDensityMultiscale):
    """Return the raw density matrix or matrices from a density object."""
    if isinstance(x, EyeDensity):
        return x.z
    if isinstance(x, EyeDensityMultiscale):
        return [scale.z for scale in x]
    raise TypeError("get_density() expects an EyeDensity or EyeDensityMultiscale object.")


def density_matrix(x, groups=None, **kwargs):
    """Extract density matrices or compute grouped densities from a table-like object."""
    if isinstance(x, (EyeDensity, EyeDensityMultiscale)):
        return get_density(x)
    if isinstance(x, pd.DataFrame):
        result_name = kwargs.get("result_name", "density")
        dens = density_by(x, groups=groups, **kwargs)
        if result_name not in dens.columns:
            raise KeyError(f"Result column '{result_name}' not found in density_by() output.")
        matrices = [get_density(item) if item is not None else None for item in dens[result_name]]
        return matrices[0] if len(matrices) == 1 else matrices
    raise TypeError("density_matrix() expects a density object or pandas DataFrame.")


# ---------------------------------------------------------------------------
# density_by
# ---------------------------------------------------------------------------

def density_by(
    table: pd.DataFrame,
    groups: str | list[str] | None = None,
    sigma: float | Sequence[float] = 50,
    xbounds: tuple[float, float] = (0, 1000),
    ybounds: tuple[float, float] = (0, 1000),
    outdim: tuple[int, int] = (100, 100),
    duration_weighted: bool = True,
    window: tuple[float, float] | None = None,
    min_fixations: int = 2,
    keep_vars: list[str] | None = None,
    fixvar: str = "fixgroup",
    result_name: str = "density",
    origin: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Compute density maps grouped by variables (mirrors R ``density_by``)."""
    if origin is None:
        origin = getattr(table, "_origin", None) or (0, 0)

    if groups is not None:
        if isinstance(groups, str):
            groups = [groups]

        grouped = table.groupby(groups, sort=True)
        rows = []
        for keys, grp in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            # Concatenate all fixation groups in this group
            all_fg = pd.concat(
                [fg.to_pandas(copy=False) if isinstance(fg, FixationGroup) else pd.DataFrame(fg) for fg in grp[fixvar]],
                ignore_index=True,
            )
            all_fg = FixationGroup(all_fg)
            d = eye_density(
                all_fg, sigma, xbounds=xbounds, ybounds=ybounds,
                outdim=outdim, duration_weighted=duration_weighted,
                window=window, min_fixations=min_fixations, origin=origin,
            )
            row = dict(zip(groups, keys))
            row[fixvar] = all_fg
            row[result_name] = d
            if keep_vars:
                for v in keep_vars:
                    row[v] = grp[v].iloc[0]
            rows.append(row)

        result = pd.DataFrame(rows)
    else:
        all_fg = pd.concat(
            [fg.to_pandas(copy=False) if isinstance(fg, FixationGroup) else pd.DataFrame(fg) for fg in table[fixvar]],
            ignore_index=True,
        )
        all_fg = FixationGroup(all_fg)
        d = eye_density(
            all_fg, sigma, xbounds=xbounds, ybounds=ybounds,
            outdim=outdim, duration_weighted=duration_weighted,
            window=window, min_fixations=min_fixations, origin=origin,
        )
        result = pd.DataFrame([{fixvar: all_fg, result_name: d}])

    # Remove rows with None density
    null_mask = result[result_name].isna() | result[result_name].apply(lambda x: x is None)
    if null_mask.any():
        warnings.warn("Removing rows with NULL density results in density_by().")
        result = result[~null_mask].reset_index(drop=True)

    return result


def suggest_sigma(
    x,
    y=None,
    xbounds: tuple[float, float] | None = None,
    ybounds: tuple[float, float] | None = None,
) -> float:
    """Suggest a kernel bandwidth using a 2-D Silverman rule."""
    if isinstance(x, FixationGroup):
        xvals = x["x"].to_numpy()
        yvals = x["y"].to_numpy()
    else:
        xvals = np.asarray(x, dtype=float)
        if y is None:
            raise ValueError("'y' must be provided when 'x' is not a FixationGroup.")
        yvals = np.asarray(y, dtype=float)

    n = len(xvals)
    if n < 2:
        return np.nan

    iqr_x = scipy_iqr(xvals)
    iqr_y = scipy_iqr(yvals)

    spread = np.sqrt((iqr_x ** 2 + iqr_y ** 2) / 2) / 1.349
    sigma = spread * n ** (-1.0 / 6)

    if xbounds is not None and ybounds is not None:
        display_scale = np.mean([
            np.ptp(xbounds),
            np.ptp(ybounds),
        ])
        sigma = max(sigma, display_scale * 0.01)
        sigma = min(sigma, display_scale * 0.15)

    return sigma
