"""Eye-movement table container and simulation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from peyesim.fixations import _FrameBacked, fixation_group


class EyeTable(_FrameBacked):
    """Container that groups fixations by key variables and preserves an origin."""

    def __init__(self, data=None, origin=None):
        super().__init__(data)
        self._origin = origin

    @property
    def origin(self):
        return self._origin

    def copy(self):
        return EyeTable(self.to_pandas(copy=True), origin=self._origin)

    def reset_index(self, *args, **kwargs):
        return EyeTable(self._frame.reset_index(*args, **kwargs), origin=self._origin)

    def __repr__(self):
        ngroups = len(self)
        nfix = sum(len(fg) for fg in self["fixgroup"]) if "fixgroup" in self.columns else 0
        s = f"EyeTable: {ngroups} groups, {nfix} total fixations\n"
        if self._origin is not None:
            s += f"  origin: ({self._origin[0]:.1f}, {self._origin[1]:.1f})\n"
        return s


def as_eye_table(x) -> EyeTable:
    """Coerce a tabular object to :class:`EyeTable` while preserving origin metadata."""
    if isinstance(x, EyeTable):
        return x
    origin = getattr(x, "_origin", None)
    if origin is None and hasattr(x, "attrs"):
        origin = x.attrs.get("_origin", x.attrs.get("origin"))
    return EyeTable(x, origin=origin)


def eye_table(
    x: str,
    y: str,
    duration: str,
    onset: str,
    groupvar: str | list[str],
    data: pd.DataFrame,
    extra_vars: list[str] | None = None,
    clip_bounds: tuple[float, float, float, float] = (0, 1280, 0, 1280),
    relative_coords: bool = True,
) -> EyeTable:
    """Construct an :class:`EyeTable` from a raw data frame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a DataFrame")

    if isinstance(groupvar, str):
        groupvar = [groupvar]

    required = [x, y, duration, onset] + groupvar
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Column(s) not found in data: {', '.join(missing)}")

    for col in [x, y, duration, onset]:
        if not np.issubdtype(data[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric, but is {data[col].dtype}")

    if (data[duration] < 0).any():
        raise ValueError(f"Column '{duration}' contains negative values. Durations must be non-negative.")
    if (data[onset] < 0).any():
        raise ValueError(f"Column '{onset}' contains negative values. Onset times must be non-negative.")

    df = data.rename(columns={x: "x", y: "y", duration: "duration", onset: "onset"}).copy()

    keep = ["x", "y", "duration", "onset"] + groupvar
    if extra_vars is not None:
        keep += extra_vars
    df = df[keep]

    xmin, xmax = sorted([clip_bounds[0], clip_bounds[1]])
    ymin, ymax = sorted([clip_bounds[2], clip_bounds[3]])
    df = df[(df["x"] >= xmin) & (df["x"] <= xmax) & (df["y"] >= ymin) & (df["y"] <= ymax)].copy()

    xdir = np.sign(clip_bounds[1] - clip_bounds[0])
    ydir = np.sign(clip_bounds[3] - clip_bounds[2])

    if relative_coords:
        df["x"] = (df["x"] - clip_bounds[0]) * xdir
        df["y"] = (df["y"] - clip_bounds[2]) * ydir

    rows = []
    for keys, grp in df.groupby(groupvar, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fg = fixation_group(
            grp["x"].to_numpy(),
            grp["y"].to_numpy(),
            grp["duration"].to_numpy(),
            grp["onset"].to_numpy(),
        )
        row = dict(zip(groupvar, keys))
        row["fixgroup"] = fg
        if extra_vars is not None:
            for v in extra_vars:
                row[v] = grp[v].iloc[0]
        rows.append(row)

    if relative_coords:
        xr1 = (xmin - clip_bounds[0]) * xdir
        xr2 = (xmax - clip_bounds[0]) * xdir
        yr1 = (ymin - clip_bounds[2]) * ydir
        yr2 = (ymax - clip_bounds[2]) * ydir
        origin = ((xr1 + xr2) / 2, (yr1 + yr2) / 2)
    else:
        origin = ((xmin + xmax) / 2, (ymin + ymax) / 2)

    return EyeTable(rows, origin=origin)


def simulate_eye_table(
    n_fixations: int,
    n_groups: int,
    clip_bounds: tuple[float, float, float, float] = (0, 1280, 0, 1280),
    relative_coords: bool = True,
) -> EyeTable:
    """Generate a simulated :class:`EyeTable`."""
    fix_per_group = n_fixations // n_groups
    rows = []
    for g in range(1, n_groups + 1):
        xvals = np.random.uniform(clip_bounds[0], clip_bounds[1], fix_per_group)
        yvals = np.random.uniform(clip_bounds[2], clip_bounds[3], fix_per_group)
        durs = np.abs(np.random.normal(300, 50, fix_per_group))
        onsets = np.cumsum(np.abs(np.random.normal(400, 100, fix_per_group)))
        for i in range(fix_per_group):
            rows.append(
                {
                    "x": xvals[i],
                    "y": yvals[i],
                    "duration": durs[i],
                    "onset": onsets[i],
                    "groupvar": str(g),
                }
            )

    return eye_table(
        "x",
        "y",
        "duration",
        "onset",
        groupvar="groupvar",
        data=pd.DataFrame(rows),
        clip_bounds=clip_bounds,
        relative_coords=relative_coords,
    )
