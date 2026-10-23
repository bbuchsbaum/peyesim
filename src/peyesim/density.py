"""Kernel density estimation for fixation data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import iqr as scipy_iqr

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

def _ks_support_mask(points: np.ndarray, grid: np.ndarray, half_width: float) -> np.ndarray:
    """Grid points that ``ks::kde()`` evaluates for each data point (one axis).

    ks (``kde.grid.2d``) only adds a point's kernel on the grid points from the
    last one at or below ``point - half_width`` (at least the first) up to the
    last one at or below ``point + half_width`` (at least the first), where
    ``half_width = supp * sigma``. Returns a ``(len(grid), len(points))`` mask.
    """
    lo = np.sum((points - half_width)[None, :] >= grid[:, None], axis=0)
    hi = np.sum((points + half_width)[None, :] >= grid[:, None], axis=0)
    lo = np.maximum(lo, 1)
    hi = np.maximum(hi, 1)
    idx = np.arange(1, len(grid) + 1)[:, None]
    return (idx >= lo[None, :]) & (idx <= hi[None, :])


def _kde2d_weighted(
    x_data: np.ndarray,
    y_data: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    xbounds: tuple[float, float],
    ybounds: tuple[float, float],
    outdim: tuple[int, int],
    supp: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted 2-D Gaussian KDE on a regular grid.

    ``sigma`` is the Gaussian kernel standard deviation in coordinate units.
    With ``supp=None`` the full kernel is evaluated (``MASS::kde2d``); with a
    number, each kernel is truncated to ``supp`` standard deviations the way
    the exact (unbinned) ``ks::kde()`` grid estimator does.
    """
    gx = np.linspace(xbounds[0], xbounds[1], outdim[0])
    gy = np.linspace(ybounds[0], ybounds[1], outdim[1])

    hx = float(sigma)
    hy = float(sigma)
    if hx <= 0 or hy <= 0:
        raise ValueError("sigma must be positive")

    weights = np.asarray(weights, dtype=float)
    if len(weights) == 1:
        weights = np.repeat(weights, len(x_data))
    if len(weights) != len(x_data):
        raise ValueError("weights must have length 1 or match the number of fixations")
    total_weight = weights.sum()
    if total_weight <= np.finfo(float).eps:
        warnings.warn("Sum of KDE weights is near zero. Returning an all-zero density grid.")
        return gx, gy, np.zeros((len(gx), len(gy)), dtype=float)

    norm = 1.0 / np.sqrt(2.0 * np.pi)
    ax = (gx[:, None] - x_data[None, :]) / hx
    ay = (gy[:, None] - y_data[None, :]) / hy
    kx = norm * np.exp(-0.5 * ax ** 2)
    ky = norm * np.exp(-0.5 * ay ** 2)
    if supp is not None:
        kx = np.where(_ks_support_mask(x_data, gx, supp * hx), kx, 0.0)
        ky = np.where(_ks_support_mask(y_data, gy, supp * hy), ky, 0.0)
    # Weights apply per fixation (columns of the kernel matrices).
    z = (kx * weights[None, :]) @ ky.T / (total_weight * hx * hy)

    return gx, gy, z


def _zapsmall(z: np.ndarray, digits: int = 7) -> np.ndarray:
    """R's ``zapsmall(z, digits)`` as implemented in R 4.5.

    R rounds to ``max(0, digits - log10(max|z|))`` decimal places, and
    ``round()`` itself rounds a fractional ``digits`` to the nearest integer
    (``floor(d + 0.5)``). Values below about ``max|z| * 5e-8`` become zero.
    """
    z = np.asarray(z, dtype=float)
    finite = ~np.isnan(z)
    if not finite.any():
        return z
    mx = float(np.max(np.abs(z[finite])))
    if mx > 0:
        dig = max(0.0, digits - np.log10(mx))
    else:
        dig = float(digits)
    return np.round(z, int(np.floor(dig + 0.5)))


def _compute_single_eye_density(
    fg: FixationGroup,
    sigma_val: float,
    xbounds: tuple[float, float],
    ybounds: tuple[float, float],
    outdim: tuple[int, int],
    normalize: bool,
    weighted: bool,
    weights: np.ndarray,
    kde_pkg: str = "ks",
    supp: float = 3.7,
) -> EyeDensity | None:
    """Compute a single-scale density map."""
    x_data = fg["x"].to_numpy(dtype=float)
    y_data = fg["y"].to_numpy(dtype=float)
    final_weights = weights if weighted else np.ones(len(x_data))

    # ks::kde uses sigma as the kernel SD; MASS::kde2d treats it as a bandwidth
    # h whose kernel SD is h / 4.
    kernel_sd = sigma_val if kde_pkg == "ks" else sigma_val / 4.0
    gx, gy, z = _kde2d_weighted(x_data, y_data, final_weights, kernel_sd,
                                 xbounds, ybounds, outdim,
                                 supp=supp if kde_pkg == "ks" else None)

    if normalize:
        s = z.sum()
        if s > np.finfo(float).eps:
            z = z / s
        else:
            warnings.warn(f"Sum of density matrix is near zero, cannot normalize. Sigma: {sigma_val}")

    # Fixed precision so results match R's zapsmall(z, digits = 7).
    z = _zapsmall(z, digits=7)

    return EyeDensity(x=gx, y=gy, z=z, sigma=sigma_val, fixgroup=fg)


# Arguments of ks::kde() that eye_density() sets itself.
_KS_MANAGED_ARGS = {"x", "H", "h", "gridsize", "xmin", "xmax", "w", "eval.points"}
# Other ks::kde() arguments.
_KS_OTHER_ARGS = {
    "gridtype", "supp", "binned", "bgridsize", "positive", "adj.positive",
    "compute.cont", "approx.cont", "unit.interval", "density", "verbose",
}
# ks::kde() arguments that do not change the estimate on eye_density()'s grid
# (contour computation, progress output, binning-only settings).
_KS_NO_EFFECT_ARGS = {"compute.cont", "approx.cont", "verbose", "bgridsize"}


def _check_kde_args(kde_args: dict, kde_pkg: str) -> None:
    """Validate extra ``ks::kde()`` arguments the way R's eye_density() does."""
    if not kde_args:
        return
    if kde_pkg == "MASS":
        raise ValueError(
            "Additional arguments are passed to ks::kde() and are not supported "
            'when kde_pkg = "MASS".'
        )
    names = {k.replace("_", ".") for k in kde_args}
    bad = sorted(n for n in names if n in _KS_MANAGED_ARGS or n not in _KS_OTHER_ARGS)
    if bad:
        raise ValueError(
            f"Unsupported ks::kde() argument(s): {', '.join(bad)}. eye_density() sets "
            "x, H, gridsize, xmin, xmax, w, and eval.points itself."
        )
    for key, value in kde_args.items():
        name = key.replace("_", ".")
        if name == "supp":
            if not (np.isscalar(value) and np.isfinite(value) and value > 0):
                raise ValueError("supp must be a single positive number.")
            continue
        if name in _KS_NO_EFFECT_ARGS:
            continue
        if name == "binned" and not value:
            continue
        if name == "positive" and not value:
            continue
        if name in {"unit.interval", "density"} and not value:
            continue
        raise NotImplementedError(
            f"ks::kde() argument {name}={value!r} is not implemented by peyesim's native "
            "KDE, which evaluates the exact (unbinned) Gaussian kernel estimate."
        )


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
    kde_pkg: str | None = "ks",
    **kde_args,
) -> EyeDensity | EyeDensityMultiscale | None:
    """Compute a density map for a :class:`FixationGroup`.

    Parameters mirror the R ``eye_density.fixation_group`` method.

    ``sigma`` is the standard deviation of the isotropic Gaussian kernel under
    ``kde_pkg="ks"`` (default). Under ``kde_pkg="MASS"`` it is the bandwidth
    ``h`` of ``MASS::kde2d``, whose kernel standard deviation is ``h / 4``; use
    ``4 * sigma`` under MASS to approximate the ks map.

    ``weights`` are optional non-negative per-fixation weights, one per row of
    ``fg`` (before ``window`` filtering); they take precedence over
    ``duration_weighted``. If the weights of the fixations used sum to zero, a
    warning is raised and ``None`` returned.

    Extra keyword arguments are ``ks::kde()`` arguments (R-style dotted names
    may be passed as ``**{"compute.cont": False}`` or with underscores). The
    arguments eye_density() sets itself (``x``, ``H``, ``h``, ``gridsize``,
    ``xmin``, ``xmax``, ``w``, ``eval.points``) and unknown names are errors,
    and any extra argument is an error under ``kde_pkg="MASS"``. peyesim
    evaluates the exact grid estimate of ``ks::kde()`` with ``binned=FALSE``
    (the ks default for up to 500 fixations), including its truncation of each
    kernel at ``supp`` (default 3.7) standard deviations; ``supp`` may be set.
    ``binned=True`` (the ks default above 500 fixations) and other settings that
    change the estimate raise ``NotImplementedError``. ``kde_pkg="MASS"``
    evaluates the untruncated kernel, as ``MASS::kde2d`` does.

    After optional normalization, each map is rounded like R's
    ``zapsmall(z, digits = 7)``: to 7 significant digits relative to the map
    maximum, so values below roughly ``max(z) * 5e-8`` become zero.
    """
    if kde_pkg is None:
        kde_pkg = "ks"
    elif kde_pkg == "scipy":
        warnings.warn('kde_pkg="scipy" is deprecated; use "ks".', DeprecationWarning, stacklevel=2)
        kde_pkg = "ks"
    if not isinstance(kde_pkg, str) or kde_pkg not in ("ks", "MASS"):
        raise ValueError('kde_pkg must be "ks" or "MASS".')
    _check_kde_args(kde_args, kde_pkg)
    supp = float(kde_args.get("supp", 3.7))

    try:
        sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=float)).ravel()
    except (TypeError, ValueError):
        sigma_arr = np.array([np.nan])
    if sigma_arr.size == 0 or not np.all(np.isfinite(sigma_arr) & (sigma_arr > 0)):
        raise ValueError("sigma must be a positive, finite numeric value or vector")

    n_rows = len(fg)
    if weights is not None:
        w_all = np.asarray(weights, dtype=float).ravel()
        if w_all.shape[0] != n_rows:
            raise ValueError("weights must be a numeric vector with one value per fixation.")
        if not np.all(np.isfinite(w_all)) or np.any(w_all < 0):
            raise ValueError("weights must be finite and non-negative.")
        if not np.any(w_all > 0):
            raise ValueError("weights must not all be zero.")
    else:
        w_all = None

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
        keep = ((fg["onset"] >= window[0]) & (fg["onset"] < window[1])).to_numpy()
        if w_all is not None:
            w_all = w_all[keep]
        fg_filtered = FixationGroup(fg.to_pandas(copy=False).loc[keep].reset_index(drop=True))
        if len(fg_filtered) == 0:
            warnings.warn("No fixations remain after applying the window filter. Returning None.")
            return None

    if len(fg_filtered) < min_fixations:
        warnings.warn(
            f"Not enough fixations (need >= {min_fixations}) to compute density. "
            f"Returning None. Provided: {len(fg_filtered)}"
        )
        return None

    # Weights: explicit weights take precedence over duration weighting.
    weighted = w_all is not None or duration_weighted
    if w_all is not None:
        w = w_all
    elif duration_weighted:
        if "duration" not in fg_filtered:
            raise ValueError("The data frame must contain a 'duration' column.")
        w = fg_filtered["duration"].to_numpy(dtype=float)
        if np.any(w < 0):
            raise ValueError("All durations must be non-negative.")
    else:
        w = np.ones(len(fg_filtered))

    if weighted and w.sum() <= 0:
        warnings.warn("Sum of weights is zero among the fixations used. Returning None.")
        return None

    if len(sigma_arr) > 1:
        # Multiscale
        densities = []
        for s in sigma_arr:
            d = _compute_single_eye_density(fg_filtered, s, xbounds, ybounds,
                                            outdim, normalize, weighted, w, kde_pkg, supp)
            if d is not None:
                densities.append(d)
        if len(densities) == 0:
            warnings.warn("All multiscale density computations failed.")
            return None
        return EyeDensityMultiscale(scales=densities)
    else:
        return _compute_single_eye_density(
            fg_filtered, sigma_arr[0], xbounds, ybounds,
            outdim, normalize, weighted, w, kde_pkg, supp
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
    """Suggest a kernel bandwidth using a 2-D Silverman rule.

    Returns ``nan`` when there are fewer than two fixations or, without display
    bounds to clamp against, when the fixations have no spread.
    """
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

    # Without bounds to clamp against, coincident fixations give a zero spread,
    # which is not a usable bandwidth.
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan

    return float(sigma)
