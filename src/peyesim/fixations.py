"""Core fixation data structures."""

from __future__ import annotations

import numpy as np
import pandas as pd


class _FrameBacked:
    """Small composition wrapper around a pandas DataFrame."""

    def __init__(self, data=None):
        if isinstance(data, _FrameBacked):
            frame = data.to_pandas(copy=True)
        elif isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            frame = pd.DataFrame(data).copy()
        self._frame = frame

    def __len__(self) -> int:
        return len(self._frame)

    def __contains__(self, item) -> bool:
        return item in self._frame

    def __getitem__(self, key):
        return self._frame.__getitem__(key)

    def __setitem__(self, key, value) -> None:
        if isinstance(key, str):
            # Replace the column: an in-place .loc[:, key] assignment keeps the old
            # dtype, so centring an integer x column raised TypeError.
            self._frame[key] = value
        else:
            self._frame.__setitem__(key, value)

    # Not iterable: pandas treats any iterable cell value as a sequence when it
    # prints a frame, and iterating through ``__getitem__`` raised KeyError(0),
    # so printing a table with a fixgroup column failed.
    __iter__ = None

    # pandas recognises DataFrames by their ``_typ`` attribute, which
    # ``__getattr__`` would otherwise forward from the wrapped frame; pandas
    # 3.1 then formats a fixgroup cell as a sequence and fails on iter().
    _typ = None

    def __array__(self, dtype=None):
        return np.asarray(self._frame, dtype=dtype)

    def __getattr__(self, name):
        if name == "_frame":
            raise AttributeError(name)
        return getattr(self._frame, name)

    @property
    def columns(self):
        return self._frame.columns

    @property
    def index(self):
        return self._frame.index

    @property
    def values(self):
        return self._frame.values

    @property
    def shape(self):
        return self._frame.shape

    @property
    def empty(self) -> bool:
        return self._frame.empty

    @property
    def loc(self):
        return self._frame.loc

    @property
    def iloc(self):
        return self._frame.iloc

    def copy(self):
        return self.__class__(self._frame.copy())

    def reset_index(self, *args, **kwargs):
        return self.__class__(self._frame.reset_index(*args, **kwargs))

    def to_pandas(self, copy: bool = True) -> pd.DataFrame:
        return self._frame.copy() if copy else self._frame


class FixationGroup(_FrameBacked):
    """A group of fixations with x, y, duration, onset, and metadata columns."""

    def coords(self) -> np.ndarray:
        """Return an (n, 2) array of ``[x, y]`` coordinates."""
        return self[["x", "y"]].to_numpy()

    def center(self, origin: tuple[float, float] | None = None) -> "FixationGroup":
        """Center fixations around *origin* (default: centroid).

        As in R, a missing coordinate makes the centroid (and so every centered
        value on that axis) missing.
        """
        if origin is None:
            origin = (self["x"].mean(skipna=False), self["y"].mean(skipna=False))
        out = self.copy()
        out["x"] = out["x"] - origin[0]
        out["y"] = out["y"] - origin[1]
        return out

    def normalize(
        self,
        xbounds: tuple[float, float],
        ybounds: tuple[float, float],
    ) -> "FixationGroup":
        """Normalize x/y to [0, 1] given bounds."""
        out = self.copy()
        out["x"] = (out["x"] - xbounds[0]) / (xbounds[1] - xbounds[0])
        out["y"] = (out["y"] - ybounds[0]) / (ybounds[1] - ybounds[0])
        return out

    def rescale(self, sx: float, sy: float) -> "FixationGroup":
        """Rescale spatial coordinates by ``(sx, sy)``."""
        out = self.copy()
        out["x"] = out["x"] * sx
        out["y"] = out["y"] * sy
        return out

    def rep_fixations(self, resolution: float = 100) -> "FixationGroup":
        """Replicate each fixation proportional to its duration.

        Each fixation is repeated ``floor(duration * resolution)`` times, and at
        least once. The floor tolerates floating-point error, so a duration of
        0.29 at resolution 100 gives 29 copies rather than 28. A missing or
        infinite duration is an error, as in R.
        """
        counts = self["duration"].to_numpy(dtype=float) * resolution
        if not np.all(np.isfinite(counts)):
            # R's rep() fails here too ("invalid 'times' argument").
            raise ValueError("rep_fixations() needs finite durations and resolution; "
                             "a duration is missing or infinite.")
        tol = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(counts))
        nreps = np.floor(counts + tol)
        nreps = np.where(nreps >= 1, nreps, 1).astype(int)
        out = self.to_pandas(copy=False).iloc[np.repeat(np.arange(len(self)), nreps)]
        return FixationGroup(out.reset_index(drop=True))

    def sample_fixations(self, time: np.ndarray, fast: bool = True) -> "FixationGroup":
        """Sample fixation coordinates at arbitrary *time* points.

        Each time point takes the coordinates of the most recent fixation whose
        onset is at or before it; durations are not used, and the last fixation
        is held for all later time points. Time points before the first onset
        give ``NaN``. Fixations are first ordered by onset (stable, so the last
        of tied onsets wins) and fixations with a missing onset are dropped.
        ``fast=True`` uses a vectorized lookup and ``fast=False`` evaluates each
        time point in turn; both return the same result.
        """
        time = np.atleast_1d(np.asarray(time, dtype=float))
        onsets = self["onset"].to_numpy(dtype=float)
        keep = ~np.isnan(onsets)
        order = np.argsort(onsets[keep], kind="stable")
        onsets = onsets[keep][order]
        xs = self["x"].to_numpy(dtype=float)[keep][order]
        ys = self["y"].to_numpy(dtype=float)[keep][order]

        if fast:
            # number of onsets <= t, like R's findInterval()
            idx = np.searchsorted(onsets, time, side="right")
        else:
            idx = np.array([int(np.sum(onsets <= t)) for t in time], dtype=int)
        valid = idx > 0
        pos = np.clip(idx - 1, 0, max(len(onsets) - 1, 0))
        if len(onsets):
            x_out = np.where(valid, xs[pos], np.nan)
            y_out = np.where(valid, ys[pos], np.nan)
        else:
            x_out = np.full(len(time), np.nan)
            y_out = np.full(len(time), np.nan)
        return FixationGroup(
            {
                "x": x_out,
                "y": y_out,
                "onset": time,
                "duration": np.ones(len(time)),
            }
        )

    def summary(self) -> dict:
        """Summary statistics mirroring R's ``summary.fixation_group``."""
        return {
            "cen_x": self["x"].mean(skipna=False),
            "cen_y": self["y"].mean(skipna=False),
            "sd_x": self["x"].std(ddof=1, skipna=False),
            "sd_y": self["y"].std(ddof=1, skipna=False),
            "nfix": len(self),
        }

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return "FixationGroup: 0 fixations"

        def rng(col):
            vals = self[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                return "[NA, NA]"
            return f"[{_r_round1(vals.min())}, {_r_round1(vals.max())}]"

        return (
            f"FixationGroup: {n} fixations\n"
            f"  x range: {rng('x')}\n"
            f"  y range: {rng('y')}\n"
            f"  duration range: {rng('duration')}\n"
            f"  onset range: {rng('onset')}"
        )


def _r_round1(value: float) -> str:
    """Format like R's ``round(value, 1)`` printed with ``cat()``."""
    return f"{round(float(value), 1):.15g}"


def fixation_group(
    x: np.ndarray,
    y: np.ndarray,
    duration=None,
    onset=None,
    group: int = 0,
) -> FixationGroup:
    """Construct a :class:`FixationGroup`."""
    if onset is None:
        raise ValueError("fixation_group() requires onset.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    onset = np.asarray(onset, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) != len(onset):
        raise ValueError("x and onset must have the same length")

    if duration is None:
        duration = np.append(np.diff(onset), 0.0)
    else:
        duration = np.asarray(duration, dtype=float)
    if len(x) != len(duration):
        raise ValueError("x and duration must have the same length")

    return FixationGroup(
        {
            "index": np.arange(1, len(x) + 1),
            "index_col": np.arange(1, len(x) + 1),
            "x": x,
            "y": y,
            "duration": duration,
            "onset": onset,
            "group_index": group,
        }
    )


def concat_fixation_groups(*groups: FixationGroup) -> FixationGroup:
    """Concatenate fixation groups, shifting onsets sequentially."""
    groups = [g for g in groups if g is not None]
    if len(groups) == 0:
        return None
    invalid = [type(g).__name__ for g in groups if not isinstance(g, FixationGroup)]
    if invalid:
        raise TypeError("All arguments to concat_fixation_groups() must be fixation_group objects.")
    if len(groups) == 1:
        return groups[0]

    result = groups[0].to_pandas(copy=True)
    for fg in groups[1:]:
        fg_frame = fg.to_pandas(copy=True)
        # Like R's max(), a missing onset makes the offset (and all later onsets) missing.
        offset = result["onset"].max(skipna=False) + result["duration"].iloc[-1]
        fg_frame["onset"] = fg_frame["onset"] + offset
        result = pd.concat([result, fg_frame], ignore_index=True)

    result["index"] = np.arange(1, len(result) + 1)
    result["index_col"] = result["index"]
    return FixationGroup(result)


def coords(x: FixationGroup) -> np.ndarray:
    """Return fixation coordinates as an ``(n, 2)`` array."""
    return x.coords()


def center(
    x: FixationGroup,
    origin: tuple[float, float] | None = None,
) -> FixationGroup:
    """Center a fixation group around ``origin`` or its centroid."""
    return x.center(origin=origin)


def normalize(
    x: FixationGroup,
    xbounds: tuple[float, float],
    ybounds: tuple[float, float],
) -> FixationGroup:
    """Normalize fixation coordinates to the unit square."""
    return x.normalize(xbounds=xbounds, ybounds=ybounds)


def rescale(x: FixationGroup, sx: float, sy: float) -> FixationGroup:
    """Rescale fixation coordinates along the x and y axes."""
    return x.rescale(sx=sx, sy=sy)


def rep_fixations(x: FixationGroup, resolution: float = 100) -> FixationGroup:
    """Replicate fixations in proportion to their durations."""
    return x.rep_fixations(resolution=resolution)


def sample_fixations(
    x: FixationGroup,
    time: np.ndarray,
    fast: bool = True,
) -> FixationGroup:
    """Sample a fixation group at arbitrary time points."""
    return x.sample_fixations(time=time, fast=fast)
