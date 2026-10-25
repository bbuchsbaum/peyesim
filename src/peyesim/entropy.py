"""Entropy measures for fixation density maps and fixation groups."""

from __future__ import annotations

import warnings

import numpy as np

from peyesim.fixations import FixationGroup
from peyesim.density import (
    EyeDensity,
    EyeDensityMultiscale,
    eye_density,
    suggest_sigma,
)


def entropy_from_mass(
    mass: np.ndarray,
    normalize: bool = True,
    base: float = np.e,
) -> float:
    """Compute Shannon entropy from a probability/mass array.

    Parameters
    ----------
    mass : array-like
        Non-negative values (counts or probabilities).  Will be
        normalised to sum to 1 internally.  Any negative value is an error;
        zero total mass gives ``nan``.
    normalize : bool
        If *True* divide the raw entropy by ``log(N, base)`` so the
        result lies in [0, 1].
    base : float
        Logarithm base (default *e* for nats).

    Returns
    -------
    float
        Entropy value, or ``nan`` if the input is empty / all-zero.
    """
    arr = np.asarray(mass, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return np.nan

    # Shannon entropy needs a probability mass; a signed map (for example a
    # difference of two densities) has none, whatever its total.
    if np.any(arr < 0):
        raise ValueError(
            "fixation_entropy() requires non-negative mass, but the map has negative "
            "values (for example a difference of two densities)."
        )

    total = arr.sum()
    if not np.isfinite(total) or total <= np.finfo(float).eps:
        return np.nan

    n_cells = len(arr)
    p = arr / total
    # Only sum over positive entries (0*log(0) == 0 by convention)
    pos = p[p > 0]

    log_fn = np.log if base == np.e else lambda v: np.log(v) / np.log(base)
    ent = -np.sum(pos * log_fn(pos))

    if normalize:
        if n_cells <= 1:
            # Single cell: entropy is 0 by convention (no uncertainty)
            return 0.0
        max_ent = log_fn(n_cells)
        ent = ent / max_ent

    return float(ent)


def _entropy_eye_density(ed: EyeDensity, normalize: bool, base: float) -> float:
    return entropy_from_mass(ed.z, normalize=normalize, base=base)


def _r_as_character(value: float) -> str:
    """Format a number like R's ``as.character()`` (15 significant digits).

    R writes the shorter of the fixed and scientific forms (fixed on a tie), so
    names match R's ``paste0("sigma_", sigma)``: 30 -> "30", 12.345678901 ->
    "12.345678901", 1e5 -> "1e+05".
    """
    v = float(value)
    if not np.isfinite(v):
        return "NA" if np.isnan(v) else ("Inf" if v > 0 else "-Inf")
    if v == 0:
        return "0"
    sci = f"{v:.14e}"
    mant, exp = sci.split("e")
    mant = mant.rstrip("0").rstrip(".")
    digits = len(mant.replace("-", "").replace(".", ""))
    exp_i = int(exp)
    sci_str = f"{mant}e{'-' if exp_i < 0 else '+'}{abs(exp_i):02d}"
    fixed = f"{v:.{max(0, digits - 1 - exp_i)}f}"
    return fixed if len(fixed) <= len(sci_str) else sci_str


def _entropy_multiscale(
    edm: EyeDensityMultiscale,
    normalize: bool,
    base: float,
    aggregate: str = "mean",
) -> float | dict[str, float]:
    per_scale = {
        f"sigma_{_r_as_character(scale.sigma)}": _entropy_eye_density(scale, normalize=normalize, base=base)
        for scale in edm.scales
    }
    if aggregate == "none":
        return per_scale
    return float(np.nanmean(list(per_scale.values())))


def _resolve_fixation_entropy_bounds(
    fg: FixationGroup,
    xbounds: tuple[float, float] | None = None,
    ybounds: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    def pad_range(values) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        span = hi - lo
        if not np.isfinite(span) or span <= np.finfo(float).eps:
            span = 1.0
        return (lo - 0.05 * span, hi + 0.05 * span)

    if xbounds is None:
        xbounds = pad_range(fg["x"])
    if ybounds is None:
        ybounds = pad_range(fg["y"])
    return xbounds, ybounds


def _entropy_fixation_group_density(
    fg: FixationGroup,
    normalize: bool,
    base: float,
    sigma=None,
    outdim: tuple[int, int] = (50, 50),
    aggregate: str = "mean",
    duration_weighted: bool = False,
    **kwargs,
) -> float | dict[str, float]:
    xbounds, ybounds = _resolve_fixation_entropy_bounds(
        fg,
        xbounds=kwargs.pop("xbounds", None),
        ybounds=kwargs.pop("ybounds", None),
    )

    if sigma is None:
        sigma = suggest_sigma(fg, xbounds=xbounds, ybounds=ybounds)
        # suggest_sigma() needs at least two fixations; without a bandwidth the
        # density entropy is undefined.
        if not np.isfinite(sigma):
            return np.nan

    with warnings.catch_warnings():
        # A group below min_fixations gives no density and an NA entropy.
        warnings.filterwarnings("ignore", message="Not enough fixations")
        ed = eye_density(
            fg,
            sigma=sigma,
            xbounds=xbounds,
            ybounds=ybounds,
            outdim=outdim,
            duration_weighted=duration_weighted,
            **kwargs,
        )
    if ed is None:
        return np.nan
    if isinstance(ed, EyeDensityMultiscale):
        return _entropy_multiscale(ed, normalize=normalize, base=base, aggregate=aggregate)
    return _entropy_eye_density(ed, normalize=normalize, base=base)


def _entropy_fixation_group_grid(
    fg: FixationGroup,
    normalize: bool,
    base: float,
    grid_size: tuple[int, int] = (10, 10),
    **kwargs,
) -> float:
    if "grid" in kwargs:
        grid_size = kwargs.pop("grid")

    if len(grid_size) != 2:
        raise ValueError("grid_size/grid must be a length-2 sequence.")
    grid_size = tuple(int(v) for v in grid_size)
    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError("grid_size/grid values must be positive integers.")

    x_vals = fg["x"].to_numpy(dtype=float)
    y_vals = fg["y"].to_numpy(dtype=float)

    if len(x_vals) < 1:
        return np.nan

    xbounds = kwargs.get("xbounds", None)
    ybounds = kwargs.get("ybounds", None)
    unexpected = set(kwargs).difference({"xbounds", "ybounds"})
    if unexpected:
        unknown = ", ".join(sorted(unexpected))
        raise TypeError(f"Unexpected grid entropy argument(s): {unknown}")

    xbounds, ybounds = _resolve_fixation_entropy_bounds(fg, xbounds=xbounds, ybounds=ybounds)

    # Bin fixations into a grid like R's findInterval(all.inside = TRUE): a
    # fixation outside the bounds is counted in the nearest edge cell.
    x_edges = np.linspace(xbounds[0], xbounds[1], grid_size[0] + 1)
    y_edges = np.linspace(ybounds[0], ybounds[1], grid_size[1] + 1)

    def cell(values, edges):
        idx = np.searchsorted(edges, values, side="right") - 1
        return np.clip(idx, 0, len(edges) - 2)

    ok = ~(np.isnan(x_vals) | np.isnan(y_vals))
    counts = np.zeros(grid_size, dtype=float)
    np.add.at(counts, (cell(x_vals[ok], x_edges), cell(y_vals[ok], y_edges)), 1.0)

    return entropy_from_mass(counts, normalize=normalize, base=base)


def fixation_entropy(
    x,
    normalize: bool = True,
    base: float = np.e,
    method: str = "density",
    **kwargs,
) -> float:
    """Compute Shannon entropy of fixation patterns.

    Parameters
    ----------
    x : EyeDensity, EyeDensityMultiscale, or FixationGroup
        The fixation data to compute entropy for.
    normalize : bool
        If *True*, return normalised entropy in [0, 1].
    base : float
        Logarithm base (default *e*).
    method : str
        For ``FixationGroup`` input only: ``"density"`` (default) builds a
        KDE density map first; ``"grid"`` bins fixations into an occupancy
        grid.
    **kwargs
        Extra arguments forwarded to the density or grid computation
        (e.g. ``sigma``, ``outdim``, ``grid_size``, ``xbounds``,
        ``ybounds``, ``aggregate``).  ``aggregate`` (``"mean"`` or
        ``"none"``) applies to multiscale densities, including a fixation
        group with a vector ``sigma`` and ``method="density"``.

    Notes
    -----
    Entropy needs non-negative mass: a map with a negative value is an error,
    and zero total mass gives ``nan``. For a fixation group, ``sigma=None``
    uses :func:`suggest_sigma`; a group with fewer than two fixations has no
    suggested bandwidth, so its density entropy is ``nan`` (with an explicit
    ``sigma`` it is ``nan`` too unless ``min_fixations`` is lowered). With
    ``method="grid"`` a single fixation gives 0, and fixations outside the
    bounds are counted in the nearest edge cell.

    Returns
    -------
    float (or list[float] when ``aggregate="none"`` for multiscale)
    """
    if isinstance(x, EyeDensity):
        return _entropy_eye_density(x, normalize=normalize, base=base)

    if isinstance(x, EyeDensityMultiscale):
        aggregate = kwargs.pop("aggregate", "mean")
        return _entropy_multiscale(
            x, normalize=normalize, base=base, aggregate=aggregate,
        )

    if isinstance(x, FixationGroup):
        aggregate = kwargs.pop("aggregate", "mean")
        if aggregate not in ("mean", "none"):
            raise ValueError("aggregate should be one of 'mean', 'none'.")
        if method not in ("density", "grid"):
            raise ValueError(
                f"Unknown method '{method}'. Use 'density' or 'grid'."
            )
        if len(x) == 0:
            return np.nan

        # R's method has formals for both methods and ignores the ones the
        # chosen method does not use.
        if method == "density":
            for key in ("grid", "grid_size"):
                kwargs.pop(key, None)
            return _entropy_fixation_group_density(
                x, normalize=normalize, base=base, aggregate=aggregate, **kwargs,
            )
        else:
            for key in ("sigma", "outdim", "duration_weighted"):
                kwargs.pop(key, None)
            return _entropy_fixation_group_grid(
                x, normalize=normalize, base=base, **kwargs,
            )

    raise TypeError(
        f"fixation_entropy() does not support type {type(x).__name__}. "
        "Expected EyeDensity, EyeDensityMultiscale, or FixationGroup."
    )
