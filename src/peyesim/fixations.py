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

    def __iter__(self):
        return iter(self._frame)

    def __contains__(self, item) -> bool:
        return item in self._frame

    def __getitem__(self, key):
        return self._frame.__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self._frame.__setitem__(key, value)

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
        """Center fixations around *origin* (default: centroid)."""
        if origin is None:
            origin = (self["x"].mean(), self["y"].mean())
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
        """Replicate each fixation proportional to its duration."""
        nreps = np.maximum((self["duration"] / (1.0 / resolution)).astype(int), 1)
        out = self.loc[self.index.repeat(nreps)].reset_index(drop=True)
        return FixationGroup(out)

    def sample_fixations(self, time: np.ndarray, fast: bool = True) -> "FixationGroup":
        """Sample fixation coordinates at arbitrary *time* points (forward-fill)."""
        time = np.asarray(time, dtype=float)
        onsets = self["onset"].to_numpy()
        xs = self["x"].to_numpy()
        ys = self["y"].to_numpy()
        if fast:
            idx = np.searchsorted(onsets, time, side="right") - 1
            valid = idx >= 0
            idx = np.clip(idx, 0, len(onsets) - 1)
            return FixationGroup(
                {
                    "x": np.where(valid, xs[idx], np.nan),
                    "y": np.where(valid, ys[idx], np.nan),
                    "onset": time,
                    "duration": np.ones(len(time)),
                }
            )

        rows = []
        for t in time:
            if t < onsets[0]:
                rows.append({"x": np.nan, "y": np.nan, "onset": t, "duration": np.nan})
                continue
            delta = t - onsets
            valid = np.where(delta >= 0)[0]
            if len(valid) == 0:
                rows.append({"x": np.nan, "y": np.nan, "onset": t, "duration": np.nan})
            else:
                i = valid[-1]
                rows.append({"x": xs[i], "y": ys[i], "onset": t, "duration": 0})
        return FixationGroup(rows)

    def summary(self) -> dict:
        """Summary statistics mirroring R's ``summary.fixation_group``."""
        return {
            "cen_x": self["x"].mean(),
            "cen_y": self["y"].mean(),
            "sd_x": self["x"].std(ddof=1),
            "sd_y": self["y"].std(ddof=1),
            "nfix": len(self),
        }

    def __repr__(self) -> str:
        n = len(self)
        if n == 0:
            return "FixationGroup: 0 fixations"
        return (
            f"FixationGroup: {n} fixations\n"
            f"  x range: [{self['x'].min():.1f}, {self['x'].max():.1f}]\n"
            f"  y range: [{self['y'].min():.1f}, {self['y'].max():.1f}]\n"
            f"  duration range: [{self['duration'].min():.1f}, {self['duration'].max():.1f}]\n"
            f"  onset range: [{self['onset'].min():.1f}, {self['onset'].max():.1f}]"
        )


def fixation_group(
    x: np.ndarray,
    y: np.ndarray,
    duration=None,
    onset=None,
    group: int = 0,
) -> FixationGroup:
    """Construct a :class:`FixationGroup`."""
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
        offset = result["onset"].max() + result["duration"].iloc[-1]
        fg_frame["onset"] = fg_frame["onset"] + offset
        result = pd.concat([result, fg_frame], ignore_index=True)

    result["index_col"] = np.arange(1, len(result) + 1)
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
