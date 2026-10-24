"""Revisitation of previously fixated locations within an episode."""

from __future__ import annotations

import numpy as np
import pandas as pd

from peyesim.eye_table import EyeTable
from peyesim.fixations import FixationGroup, fixation_group

REVISIT_SUMMARY_COLUMNS = [
    "n_fix",
    "opportunities",
    "revisits",
    "revisit_rate",
    "expected_revisits",
    "expected_rate",
    "excess_revisits",
    "excess_rate",
]

_REVISIT_ATTRS = ("radius", "min_lag", "max_lag", "require_departure", "null")


class RevisitTable(pd.DataFrame):
    """Per-fixation revisit scores returned by :func:`fixation_revisits`.

    A :class:`pandas.DataFrame` whose ``attrs`` record the scoring settings
    (``radius``, ``min_lag``, ``max_lag``, ``require_departure``) and whether a
    spatial null was supplied (``null``).
    """

    @property
    def _constructor(self):
        return RevisitTable

    def __repr__(self) -> str:
        a = self.attrs
        header = "Revisits: {} of {} opportunities ({} fixations; radius {:g}, lag {}-{}{})".format(
            int(self["revisit"].sum()) if "revisit" in self else 0,
            int(self["opportunity"].sum()) if "opportunity" in self else 0,
            len(self),
            a.get("radius", float("nan")),
            a.get("min_lag"),
            "Inf" if a.get("max_lag") == np.inf else a.get("max_lag"),
            ", departure required" if a.get("require_departure") else "",
        )
        return header + "\n" + pd.DataFrame(self).__repr__()


def _check_revisit_args(radius, min_lag, max_lag) -> None:
    if radius is None or np.ndim(radius) != 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a single positive number (in fixation coordinate units).")
    if np.ndim(min_lag) != 0 or not np.isfinite(min_lag) or min_lag < 1 or min_lag != round(min_lag):
        raise ValueError("min_lag must be a positive integer.")
    if np.ndim(max_lag) != 0 or max_lag is None or np.isnan(max_lag) or max_lag < min_lag:
        raise ValueError("max_lag must be >= min_lag (Inf for no limit).")


def _eligible(i: int, min_lag: int, max_lag) -> np.ndarray:
    """0-based indices of fixations ``min_lag``..``max_lag`` back from ``i`` (0-based)."""
    hi = i - int(min_lag)
    lo = max(0, i - int(max_lag)) if np.isfinite(max_lag) else 0
    return np.arange(lo, hi + 1) if hi >= lo else np.array([], dtype=int)


def _revisit_dist(a: np.ndarray, b: np.ndarray, method: str) -> np.ndarray:
    dx = a[:, 0][:, None] - b[:, 0][None, :]
    dy = a[:, 1][:, None] - b[:, 1][None, :]
    if method == "euclidean":
        return np.sqrt(dx**2 + dy**2)
    return np.abs(dx) + np.abs(dy)


def _as_reference_matrix(reference) -> np.ndarray:
    if isinstance(reference, (pd.DataFrame, FixationGroup)) and {"x", "y"} <= set(reference.columns):
        m = np.column_stack([np.asarray(reference["x"], dtype=float),
                             np.asarray(reference["y"], dtype=float)])
    else:
        m = np.asarray(reference, dtype=float)
        if m.ndim == 1 and m.size == 0:
            m = m.reshape(0, 2)
    if m.ndim != 2 or m.shape[1] != 2:
        raise ValueError("reference must have two columns (x, y).")
    return m[np.isfinite(m[:, 0]) & np.isfinite(m[:, 1])]


def _new_revisit_table(out: dict, radius, min_lag, max_lag, require_departure, has_null) -> RevisitTable:
    tab = RevisitTable(out)
    tab.attrs.update(
        radius=radius,
        min_lag=min_lag,
        max_lag=max_lag,
        require_departure=require_departure,
        null=has_null,
    )
    return tab


def _fixation_group_revisits(x: FixationGroup, radius, min_lag, max_lag, require_departure,
                             reference, dist_method) -> RevisitTable:
    frame = x.to_pandas(copy=True)
    keep = np.isfinite(frame["x"].to_numpy(dtype=float)) & np.isfinite(
        frame["y"].to_numpy(dtype=float)
    ) & ~frame["onset"].isna().to_numpy()
    frame = frame.loc[keep]
    frame = frame.iloc[np.argsort(frame["onset"].to_numpy(dtype=float), kind="stable")]
    n = len(frame)
    index = frame["index"].to_numpy() if "index" in frame else np.arange(1, n + 1)

    opportunity = np.zeros(n, dtype=bool)
    revisit = np.zeros(n, dtype=bool)
    lag = pd.array([pd.NA] * n, dtype="Int64")
    match_index = pd.array([pd.NA] * n, dtype="Int64")
    distance = np.full(n, np.nan)
    expected = np.full(n, np.nan)

    if n > 0:
        xy = frame[["x", "y"]].to_numpy(dtype=float)
        d = _revisit_dist(xy, xy, dist_method)
        ref = None if reference is None else _as_reference_matrix(reference)
        dref = None if ref is None or len(ref) == 0 else _revisit_dist(ref, xy, dist_method)

        for i in range(n):
            elig = _eligible(i, min_lag, max_lag)
            if elig.size == 0:
                continue
            departed = i == 0 or d[i, i - 1] >= radius
            if require_departure and not departed:
                continue
            opportunity[i] = True
            distance[i] = d[i, elig].min()
            near = elig[d[i, elig] < radius]
            if near.size:
                j = int(near.max())
                revisit[i] = True
                match_index[i] = int(index[j])
                lag[i] = i - j
            if dref is not None:
                ok = dref[:, i - 1] >= radius if (require_departure and i > 0) else np.ones(len(dref), bool)
                if ok.any():
                    hit = (dref[np.ix_(ok, elig)] < radius).any(axis=1)
                    expected[i] = hit.mean()

    out = {
        "index": pd.array(index, dtype="Int64"),
        "onset": frame["onset"].to_numpy(dtype=float),
        "opportunity": opportunity,
        "revisit": revisit,
        "lag": lag,
        "match_index": match_index,
        "distance": distance,
        "expected": expected,
    }
    return _new_revisit_table(out, radius, min_lag, max_lag, require_departure, reference is not None)


def _empty_fixation_group() -> FixationGroup:
    return fixation_group(np.array([]), np.array([]), np.array([]), np.array([]))


def _xy_of(fg) -> np.ndarray:
    if fg is None or len(fg) == 0:
        return np.empty((0, 2))
    return np.column_stack([np.asarray(fg["x"], dtype=float), np.asarray(fg["y"], dtype=float)])


def _eye_table_revisits(x, radius, min_lag, max_lag, require_departure, null, subject_var,
                        item_var, var, keep_fixations, dist_method):
    if null not in ("none", "subject", "item"):
        raise ValueError("null must be one of 'none', 'subject', 'item'.")
    is_eye_table = isinstance(x, EyeTable)
    frame = x.to_pandas(copy=True) if is_eye_table else pd.DataFrame(x).copy()
    if var not in frame.columns:
        raise ValueError(f"Column '{var}' not found in the eye_table.")

    def has_col(v) -> bool:
        return isinstance(v, str) and v in frame.columns

    if null == "subject" and not has_col(subject_var):
        raise ValueError('null = "subject" needs subject_var naming a column of the eye_table.')
    if null == "item" and not has_col(item_var):
        raise ValueError('null = "item" needs item_var naming a column of the eye_table.')
    if null == "item" and subject_var is not None and not has_col(subject_var):
        raise ValueError(f"subject_var '{subject_var}' is not a column of the eye_table.")

    # Columns written by an earlier call are recorded in attrs and replaced; any other
    # column with one of these names belongs to the caller and is never overwritten.
    ours = REVISIT_SUMMARY_COLUMNS + ["revisit_fixations"]
    prev = list(frame.attrs.get("revisit_columns", []))
    clash = [c for c in frame.columns if c in ours and c not in prev]
    if clash:
        raise ValueError(
            f"The eye_table already has column(s) {', '.join(clash)} that fixation_revisits() writes. "
            "Rename or drop them first."
        )

    fgs = list(frame[var])
    nrow = len(frame)
    per = []
    for k in range(nrow):
        fg = fgs[k]
        if fg is None:
            fg = _empty_fixation_group()
        ref = None
        if null != "none":
            if null == "subject":
                subj = frame[subject_var].to_numpy()
                pool = [i for i in range(nrow) if subj[i] == subj[k] and i != k]
            else:
                item = frame[item_var].to_numpy()
                same_item = [i for i in range(nrow) if item[i] == item[k]]
                if subject_var is None:
                    pool = [i for i in same_item if i != k]
                else:
                    subj = frame[subject_var].to_numpy()
                    pool = [i for i in same_item if subj[i] != subj[k]]
            parts = [_xy_of(fgs[i]) for i in pool]
            ref = np.vstack(parts) if parts else np.empty((0, 2))
        per.append(_fixation_group_revisits(fg, radius, min_lag, max_lag, require_departure,
                                            ref, dist_method))

    summ = pd.concat([revisit_summary(rv) for rv in per], ignore_index=True) if per else \
        pd.DataFrame({c: pd.Series(dtype=float) for c in REVISIT_SUMMARY_COLUMNS})
    assert list(summ.columns) == REVISIT_SUMMARY_COLUMNS

    attrs = {k: v for k, v in frame.attrs.items() if k != "revisit_columns"}
    old = [c for c in frame.columns if c in prev]
    frame = frame.drop(columns=old).reset_index(drop=True)
    out = pd.concat([frame, summ.reset_index(drop=True)], axis=1)
    if keep_fixations:
        out["revisit_fixations"] = pd.Series(per, dtype=object)
    out.attrs.update(attrs)
    out.attrs["revisit_columns"] = REVISIT_SUMMARY_COLUMNS + (["revisit_fixations"] if keep_fixations else [])
    if is_eye_table:
        return EyeTable(out, origin=x.origin)
    return out


def fixation_revisits(
    x,
    radius: float | None = None,
    min_lag: int = 2,
    max_lag: float = np.inf,
    require_departure: bool = True,
    reference=None,
    dist_method: str = "euclidean",
    *,
    null: str = "none",
    subject_var: str | None = None,
    item_var: str | None = None,
    var: str = "fixgroup",
    keep_fixations: bool = False,
):
    """Score returns to previously fixated locations ("revisitation").

    Fixation ``i`` is a *revisit* when it lies within ``radius`` of at least
    one earlier fixation ``j`` that is at least ``min_lag`` fixations back
    (and at most ``max_lag``). With ``require_departure=True`` (default) the
    preceding fixation must also lie at least ``radius`` away, so a run of
    small refixations on one spot is not a return. A fixation is an
    *opportunity* when it could have been a revisit: it has an eligible
    earlier fixation and, under ``require_departure``, arrives from at least
    ``radius`` away.

    Revisit counts rise mechanically with fixation number and spatial
    dispersion. Supplying ``reference`` locations gives a spatial null: for
    each opportunity, ``expected`` is the proportion of reference locations
    (among those that also satisfy the departure condition) that would count
    as a revisit of the same eligible earlier fixations. ``radius`` is in
    fixation coordinate units; see :func:`~peyesim.visual_angle.visual_angle_to_px`.

    Parameters
    ----------
    x
        A :class:`~peyesim.fixations.FixationGroup`, or an
        :class:`~peyesim.eye_table.EyeTable` (or data frame) with a column of
        fixation groups.
    radius
        Distance within which two fixations count as the same location.
    min_lag, max_lag
        Minimum (default 2) and maximum (default ``inf``) number of fixations
        back for an eligible earlier fixation.
    require_departure
        Require the preceding fixation to lie at least ``radius`` away.
    reference
        Fixation groups only: ``(n, 2)`` array or data frame with ``x``/``y``
        columns of null-model locations.
    dist_method
        ``"euclidean"`` (default) or ``"manhattan"``.
    null
        Eye tables only: ``"none"`` (default), ``"subject"`` (the same
        subject's fixations from the other rows, leave-episode-out) or
        ``"item"`` (fixations from other rows showing the same ``item_var``;
        with ``subject_var`` every row of the same subject is excluded).
    subject_var, item_var
        Eye tables only: columns identifying the subject and the viewed item.
    var
        Eye tables only: name of the fixation-group column.
    keep_fixations
        Eye tables only: keep the per-fixation tables in a
        ``revisit_fixations`` column.

    Returns
    -------
    RevisitTable or EyeTable
        For a fixation group, one row per fixation in onset order with
        ``index``, ``onset``, ``opportunity``, ``revisit``, ``lag``,
        ``match_index``, ``distance`` and ``expected``. For an eye table, the
        table with the :func:`revisit_summary` columns appended. Re-scoring a
        scored table replaces the columns an earlier call wrote; a caller's
        own column of the same name is an error.
    """
    if dist_method not in ("euclidean", "manhattan"):
        raise ValueError("dist_method must be one of 'euclidean', 'manhattan'.")
    _check_revisit_args(radius, min_lag, max_lag)
    if isinstance(x, FixationGroup):
        return _fixation_group_revisits(x, radius, min_lag, max_lag, require_departure,
                                        reference, dist_method)
    if isinstance(x, (EyeTable, pd.DataFrame)):
        return _eye_table_revisits(x, radius, min_lag, max_lag, require_departure, null,
                                   subject_var, item_var, var, keep_fixations, dist_method)
    raise TypeError("fixation_revisits() expects a FixationGroup or an EyeTable.")


def revisit_summary(x: RevisitTable) -> pd.DataFrame:
    """Summarise one episode's :func:`fixation_revisits` table.

    Returns a one-row data frame: ``n_fix``; ``opportunities``; ``revisits``;
    ``revisit_rate`` (revisits per opportunity); ``expected_revisits`` and
    ``expected_rate`` under the null (``NaN`` without a reference, or when any
    opportunity has no expectation); ``excess_revisits`` (observed minus
    expected) and ``excess_rate``.
    """
    if not isinstance(x, RevisitTable):
        raise TypeError("revisit_summary() needs the output of fixation_revisits() for a fixation group.")
    opp = int(np.sum(x["opportunity"]))
    rev = int(np.sum(x["revisit"]))
    has_null = bool(x.attrs.get("null", False))
    e = x["expected"].to_numpy(dtype=float)[x["opportunity"].to_numpy(dtype=bool)]
    exp_n = float(e.sum()) if has_null and opp > 0 and np.all(np.isfinite(e)) else np.nan
    rate = rev / opp if opp > 0 else np.nan
    exp_rate = exp_n / opp if has_null and opp > 0 else np.nan
    return pd.DataFrame(
        {
            "n_fix": [len(x)],
            "opportunities": [opp],
            "revisits": [rev],
            "revisit_rate": [rate],
            "expected_revisits": [exp_n],
            "expected_rate": [exp_rate],
            "excess_revisits": [rev - exp_n],
            "excess_rate": [rate - exp_rate],
        }
    )
