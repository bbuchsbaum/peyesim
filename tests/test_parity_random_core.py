"""Randomized differential tests of peyesim's core functions against R eyesim.

Cases are drawn once from a fixed numpy seed, written into a single R script
as literals (``repr(float)`` round-trips exactly, so both languages see the
same doubles) and evaluated by R eyesim through ``tests/r_oracle.py``.

R returns every number as a ``"%.17g"`` string (``"NA"``, ``"NaN"``, ``"Inf"``
kept apart), so comparisons are not limited by JSON's 15 significant digits.
Tolerances:

* **exact** (``rtol=0``): values R computes with the same IEEE operations as
  peyesim -- constructor columns, normalize/rescale, rep/sample_fixations,
  scanpath vectors, eye_table coordinates, revisit flags/lags/indices,
  CRQA counts and the recurrence plot, grid entropy counts.
* **1e-13 relative** (plus ``1e-12 * scale`` absolute for values that cancel
  to ~0): quantities that involve a sum or mean, where R accumulates in long
  double (and ``mean()`` adds a second-pass correction) and numpy uses
  pairwise summation -- center/summary, suggest_sigma, entropy, CRQA ratios,
  revisit expectations, visual-angle conversions.
* **densities: rtol 1e-9, atol 1e-12 * max(z)**. Both sides evaluate the
  exact Gaussian KDE (ks unbinned for <= 500 fixations, MASS::kde2d) and
  then round with ``zapsmall(z, digits = 7)``, i.e. to a fixed decimal
  step of roughly ``1e-7 * max(z)``. Before rounding the two maps differ by
  ~1e-15 relative (summation order), so after rounding they are identical
  except in the rare cell whose unrounded value sits within ~1e-15 of a
  rounding midpoint, where the two sides may land one step apart. Such cells
  are allowed explicitly (at most one rounding step, in at most 0.5% of the
  cells); every other cell must agree to rtol 1e-9.
* **estimate_scale: 1e-3 absolute** -- PARITY.md documents that the two
  L-BFGS-B implementations of the non-smooth Hausdorff objective agree to
  about this level; only cases with an exact, unique optimum are drawn.

Edge cases where R raises an error are compared as "both raise".
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import peyesim as pe
from peyesim.expansion import estimate_scale, match_scale
from tests.r_oracle import requires_r, run_r

# PEYESIM_PARITY_SEED re-runs the whole suite on another draw.
SEED = int(os.environ.get("PEYESIM_PARITY_SEED", "20261025"))
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# R literal helpers
# ---------------------------------------------------------------------------

def rnum(v) -> str:
    if v is None:
        return "NA"
    v = float(v)
    if math.isnan(v):
        return "NA"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return repr(v)


def rvec(a) -> str:
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        return "numeric(0)"
    return "c(" + ", ".join(rnum(v) for v in a) + ")"


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def rstr(a) -> str:
    items = ["NA" if _missing(v) else '"' + str(v) + '"' for v in a]
    return "c(" + ", ".join(items) + ")" if items else "character(0)"


def rbool(b) -> str:
    return "TRUE" if b else "FALSE"


def rmat(m) -> str:
    m = np.asarray(m, dtype=float).reshape(-1, 2)
    return f"matrix({rvec(m.ravel(order='F'))}, ncol = 2)"


def dec(v) -> np.ndarray:
    """Decode R's ``enc()`` strings into a float array."""
    if v is None:
        return np.array([])
    vals = v if isinstance(v, list) else [v]

    def one(s):
        if s is None or s == "NA":
            return np.nan
        return float(s)

    return np.array([one(s) for s in vals], dtype=float)


def decm(m) -> np.ndarray:
    dim = m["dim"] if isinstance(m["dim"], list) else [m["dim"]]
    return dec(m["v"]).reshape(dim, order="F")


def is_err(r) -> bool:
    return isinstance(r, dict) and "error" in r


def py_safe(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - "both error" comparisons
            return PyError(exc)


class PyError:
    def __init__(self, exc):
        self.exc = exc

    def __repr__(self):
        return f"PyError({type(self.exc).__name__}: {self.exc})"


def assert_both_error_or(r_val, py_val, check):
    if is_err(r_val):
        assert isinstance(py_val, PyError), f"R errored ({r_val['error']!r}) but Python returned {py_val!r}"
        return
    assert not isinstance(py_val, PyError), f"Python raised {py_val!r}; R returned a value"
    check(r_val, py_val)


def close(actual, expected_enc, rtol=0.0, atol=0.0):
    expected = dec(expected_enc) if not isinstance(expected_enc, np.ndarray) else expected_enc
    actual = np.atleast_1d(np.asarray(actual, dtype=float)).ravel()
    expected = np.atleast_1d(expected).ravel()
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    # NaN positions must agree (assert_allclose treats NaN == NaN only when both are NaN).


REL = 1e-13


def close_rel(actual, expected_enc, scale=None):
    expected = dec(expected_enc)
    if scale is None:
        finite = expected[np.isfinite(expected)]
        scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    close(actual, expected, rtol=REL, atol=1e-12 * max(scale, 1e-300))


def assert_density_close(actual, expected):
    actual = np.asarray(actual, dtype=float)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    if not finite.any():
        return
    zmax = float(np.max(np.abs(expected[finite]))) or 1.0
    # zapsmall(digits = 7) rounds to round(7 - log10(max)) decimals.
    step = 10.0 ** (-math.floor(7 - math.log10(zmax) + 0.5)) if zmax > 0 else 0.0
    diff = np.abs(actual[finite] - expected[finite])
    tol = 1e-9 * np.abs(expected[finite]) + 1e-12 * zmax
    off = diff > tol
    # A cell may land one rounding step apart (midpoint ties, see module docstring).
    assert np.all(diff[off] <= step * (1 + 1e-6)), f"max diff {diff.max()} > step {step}"
    assert off.sum() <= max(1, int(0.005 * diff.size)), f"{off.sum()} cells differ by a rounding step"


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------

def gen_fg(n, kind="sorted", dur_scale="ms", ties=False, na_onset=False, na_xy=False):
    x = rng.uniform(-60, 860, n)
    y = rng.uniform(-60, 660, n)
    if ties and n:
        x = np.round(x, -1)
        y = np.round(y, -1)
        if n > 2:
            k = rng.integers(0, n - 1)
            x[k + 1], y[k + 1] = x[k], y[k]
    else:
        x = np.round(x, 3)
        y = np.round(y, 3)
    gaps = rng.integers(40, 450, n).astype(float)
    onset = np.cumsum(gaps) - (gaps[0] if n else 0)
    if kind == "unsorted":
        onset = rng.permutation(onset)
    elif kind == "tied" and n > 1:
        for k in rng.choice(np.arange(1, n), size=max(1, n // 4), replace=False):
            onset[k] = onset[k - 1]
    if dur_scale == "ms":
        duration = rng.integers(0, 600, n).astype(float) if n else np.zeros(0)
    else:  # seconds, for rep_fixations
        duration = np.round(rng.uniform(0.001, 0.6, n), int(rng.integers(2, 4)))
    if na_onset and n > 1:
        onset[rng.integers(0, n)] = np.nan
    if na_xy and n > 1:
        k = rng.integers(0, n)
        x[k] = np.nan
        if rng.random() < 0.5:
            y[k] = np.nan
    return {"x": x, "y": y, "onset": onset, "duration": duration}


FG_CASES = []
for i, n in enumerate([0, 1, 2, 2, 3, 4, 5, 7, 8, 10, 12, 15, 20, 25, 30, 40, 60, 90, 2, 3,
                       6, 9, 14, 18, 33, 1, 5, 11, 50, 75]):
    kind = ["sorted", "unsorted", "tied"][i % 3]
    spec = gen_fg(n, kind=kind, dur_scale="s" if i % 4 == 1 else "ms", ties=(i % 5 == 0),
                  na_onset=(i % 7 == 3), na_xy=(i in (19, 27)))
    spec["omit_duration"] = (i % 6 == 4)
    spec["origin"] = (float(np.round(rng.uniform(-100, 900), 2)), float(np.round(rng.uniform(-100, 700), 2)))
    xb = np.sort(np.round(rng.uniform(-100, 900, 2), 1))
    yb = np.sort(np.round(rng.uniform(-100, 700, 2), 1))
    if xb[0] == xb[1]:
        xb[1] += 1
    if yb[0] == yb[1]:
        yb[1] += 1
    spec["xb"], spec["yb"] = tuple(xb), tuple(yb)
    spec["sx"], spec["sy"] = float(np.round(rng.uniform(-2, 3), 3)), float(np.round(rng.uniform(0.1, 3), 3))
    # durations are ms or s; keep the number of copies small
    spec["res"] = float(rng.choice([1, 10, 100, 250])) if i % 4 == 1 else float(rng.choice([0.01, 0.0333, 0.1]))
    lo = np.nanmin(spec["onset"]) if n and np.isfinite(spec["onset"]).any() else 0.0
    hi = np.nanmax(spec["onset"]) if n and np.isfinite(spec["onset"]).any() else 100.0
    times = np.sort(np.round(rng.uniform(lo - 100, hi + 200, int(rng.integers(0, 12))), 1))
    if n and rng.random() < 0.6:
        times = np.concatenate([times, spec["onset"][np.isfinite(spec["onset"])][:3]])  # exact onsets
    if i % 3 == 0:
        times = rng.permutation(times)  # findInterval() does not need sorted time points
    spec["times"] = times
    FG_CASES.append(spec)


def make_fg(spec):
    if spec.get("omit_duration"):
        return pe.fixation_group(spec["x"], spec["y"], onset=spec["onset"])
    return pe.fixation_group(spec["x"], spec["y"], duration=spec["duration"], onset=spec["onset"])


def r_fg(spec):
    if spec.get("omit_duration"):
        return f"fixation_group(x = {rvec(spec['x'])}, y = {rvec(spec['y'])}, onset = {rvec(spec['onset'])})"
    return (f"fixation_group(x = {rvec(spec['x'])}, y = {rvec(spec['y'])}, "
            f"duration = {rvec(spec['duration'])}, onset = {rvec(spec['onset'])})")


# calcangle / cart2pol vectors, including parallel, antiparallel and zero vectors
ANGLE_CASES = []
for i in range(30):
    d = int(rng.choice([2, 2, 2, 3, 5]))
    a = np.round(rng.normal(0, 10, d), int(rng.integers(0, 4)))
    kind = i % 6
    if kind == 0:
        b = a * float(rng.choice([2, 0.5, 3.7]))
    elif kind == 1:
        b = -a * float(rng.choice([1, 0.1, 7]))
    elif kind == 2:
        b = np.zeros(d)
    else:
        b = np.round(rng.normal(0, 10, d), 2)
    ANGLE_CASES.append((a, b))
POLAR_X = np.round(rng.normal(0, 50, 40), 2)
POLAR_Y = np.round(rng.normal(0, 50, 40), 2)
POLAR_X[:4] = [0, 0, -3, 5]
POLAR_Y[:4] = [0, -2, 0, 0]


# eye_density cases -----------------------------------------------------------
DENS_CASES = []
for i in range(40):
    n = int(rng.choice([0, 1, 2, 3, 5, 8, 15, 30, 60, 120, 300]))
    spec = gen_fg(n, kind=["sorted", "unsorted", "tied"][i % 3], ties=(i % 5 == 0))
    if i % 9 == 4 and n:
        spec["duration"][:] = 0  # zero durations
    args = {}
    sig = [float(np.round(rng.uniform(5, 120), 2))]
    if i % 5 == 2:
        sig = sorted({float(np.round(rng.uniform(5, 150), 1)) for _ in range(int(rng.integers(2, 4)))})
    args["sigma"] = sig
    if i % 4 != 3:
        args["xbounds"] = (0.0, 800.0) if i % 2 else (float(np.round(rng.uniform(-50, 200), 1)), float(np.round(rng.uniform(400, 900), 1)))
        args["ybounds"] = (0.0, 600.0) if i % 2 else (float(np.round(rng.uniform(-50, 200), 1)), float(np.round(rng.uniform(300, 700), 1)))
    args["outdim"] = [(2, 2), (7, 3), (13, 17), (25, 40), (50, 50), (64, 48), (100, 100), (1, 5)][i % 8]
    args["duration_weighted"] = bool(i % 3 == 1)
    if i % 6 == 5 and n:
        w = np.round(rng.uniform(0, 3, n), 2)
        if n > 2:
            w[rng.integers(0, n)] = 0.0
        args["weights"] = w
    if i % 7 == 2 and n:
        lo = float(np.nanmin(spec["onset"]))
        hi = float(np.nanmax(spec["onset"]))
        args["window"] = (float(np.round(lo + 0.2 * (hi - lo), 1)), float(np.round(lo + 0.8 * (hi - lo) + 1, 1)))
    if i % 4 == 1:
        args["kde_pkg"] = "MASS"
        args["sigma"] = [4 * s for s in args["sigma"]]
    if i % 10 == 7:
        args["normalize"] = False
    if i % 11 == 6:
        args["min_fixations"] = 1
    DENS_CASES.append((spec, args))
# explicit edge cases
DENS_CASES.append((gen_fg(4), {"sigma": [0.0]}))                     # invalid sigma -> error
DENS_CASES.append((gen_fg(4), {"sigma": [10.0], "kde_pkg": "other"}))  # invalid backend -> error
DENS_CASES.append((gen_fg(4), {"sigma": [10.0], "weights": np.array([1.0, 2.0])}))  # bad length
DENS_CASES.append((gen_fg(4), {"sigma": [10.0], "weights": np.zeros(4)}))  # all zero -> error
DENS_CASES.append((gen_fg(4), {"sigma": [10.0], "window": (5.0, 1.0)}))   # bad window
DENS_CASES.append((gen_fg(6, na_xy=True), {"sigma": [30.0], "xbounds": (0.0, 800.0), "ybounds": (0.0, 600.0), "outdim": (10, 10)}))
DENS_CASES.append((gen_fg(6, na_xy=True), {"sigma": [120.0], "xbounds": (0.0, 800.0), "ybounds": (0.0, 600.0), "outdim": (10, 10), "kde_pkg": "MASS"}))
_coinc = gen_fg(3)
_coinc["x"][:] = 400.0
DENS_CASES.append((_coinc, {"sigma": [20.0], "outdim": (5, 5)}))     # degenerate default xbounds


def r_dens_call(spec, args, fgvar="g"):
    parts = [fgvar]
    sig = args["sigma"]
    parts.append(f"sigma = {rvec(sig)}")
    if "xbounds" in args:
        parts.append(f"xbounds = {rvec(args['xbounds'])}")
        parts.append(f"ybounds = {rvec(args['ybounds'])}")
    if "outdim" in args:
        parts.append(f"outdim = {rvec(args['outdim'])}")
    if "duration_weighted" in args:
        parts.append(f"duration_weighted = {rbool(args['duration_weighted'])}")
    if "weights" in args:
        parts.append(f"weights = {rvec(args['weights'])}")
    if "window" in args:
        parts.append(f"window = {rvec(args['window'])}")
    if "kde_pkg" in args:
        parts.append(f'kde_pkg = "{args["kde_pkg"]}"')
    if "normalize" in args:
        parts.append(f"normalize = {rbool(args['normalize'])}")
    if "min_fixations" in args:
        parts.append(f"min_fixations = {args['min_fixations']}")
    return "eye_density(" + ", ".join(parts) + ")"


def py_dens_kwargs(args):
    kw = dict(args)
    kw["sigma"] = args["sigma"][0] if len(args["sigma"]) == 1 else list(args["sigma"])
    return kw


# fixation_entropy cases ----------------------------------------------------------
ENT_CASES = []
for i in range(30):
    n = int(rng.choice([0, 1, 2, 3, 6, 10, 25, 50, 100]))
    spec = gen_fg(n, ties=(i % 4 == 0))
    kw = {}
    method = "grid" if i % 3 == 2 else "density"
    kw["method"] = method
    if i % 5 == 1:
        kw["normalize"] = False
    if i % 6 == 3:
        kw["base"] = 2.0
    if method == "density":
        if i % 4 == 1:
            kw["sigma"] = [float(np.round(rng.uniform(10, 100), 1))]
        elif i % 4 == 2:
            kw["sigma"] = [20.0, 45.5, 90.0]
            kw["aggregate"] = "none" if i % 8 == 2 else "mean"
        kw["outdim"] = [(50, 50), (20, 30), (8, 8)][i % 3]
        kw["duration_weighted"] = bool(i % 2)
    else:
        kw["grid"] = [(10, 10), (4, 3), (1, 1), (7, 2)][i % 4]
    if i % 3 == 0:
        kw["xbounds"] = (0.0, 800.0)
        kw["ybounds"] = (0.0, 600.0)
    ENT_CASES.append((spec, kw))
ENT_CASES.append((gen_fg(5, na_xy=True), {"method": "grid"}))
ENT_CASES.append((gen_fg(4), {"method": "grid", "grid": (0, 3)}))  # invalid grid -> error

MASS_CASES = []
for i in range(25):
    k = int(rng.choice([0, 1, 2, 5, 10, 50]))
    m = np.round(rng.uniform(0, 5, k), 3)
    if k and i % 3 == 0:
        m[rng.random(k) < 0.4] = 0.0
    if k and i % 5 == 1:
        m[0] = np.nan
    if k and i % 7 == 2:
        m[-1] = np.inf
    if k and i % 8 == 3:
        m[0] = -0.5  # error
    MASS_CASES.append((m, bool(i % 2), [math.e, 2.0, 10.0][i % 3]))
MASS_CASES.append((np.zeros(4), True, math.e))


# eye_table / density_by cases ------------------------------------------------------
def gen_raw_table(n_rows, n_groups, na_group=False):
    subj = rng.choice([f"s{k}" for k in range(1, n_groups + 1)], n_rows).astype(object)
    img = rng.choice(["a", "B", "c"], n_rows).astype(object)
    if na_group and n_rows:
        subj[rng.integers(0, n_rows)] = None
    frame = pd.DataFrame({
        "px": np.round(rng.uniform(-80, 900, n_rows), 2),
        "py": np.round(rng.uniform(-80, 700, n_rows), 2),
        "dur": rng.integers(0, 500, n_rows).astype(float),
        "on": np.round(rng.uniform(0, 3000, n_rows), 0),
        "subject": subj,
        "image": img,
        "cond": rng.choice([1.0, 2.0, 3.0], n_rows),
    })
    return frame


ET_CASES = []
for i in range(14):
    frame = gen_raw_table(int(rng.choice([1, 5, 12, 30, 80])), int(rng.integers(1, 5)), na_group=(i % 5 == 3))
    groupvar = ["subject"] if i % 3 else ["subject", "image"]
    clip = [(0, 800, 0, 600), (0, 1280, 0, 1280), (800, 0, 600, 0), (100, 700, 50, 550)][i % 4]
    rel = bool(i % 2 == 0)
    vars_ = ["cond"] if i % 4 == 1 else None
    ET_CASES.append((frame, groupvar, clip, rel, vars_))
_neg = gen_raw_table(6, 2)
_neg.loc[2, "dur"] = -5.0
ET_CASES.append((_neg, ["subject"], (0, 800, 0, 600), True, None))  # negative duration -> error


def r_frame(frame):
    return ("data.frame(" + ", ".join([
        f"px = {rvec(frame.px)}", f"py = {rvec(frame.py)}", f"dur = {rvec(frame.dur)}",
        f"on = {rvec(frame.on)}", f"subject = {rstr(frame.subject)}", f"image = {rstr(frame.image)}",
        f"cond = {rvec(frame.cond)}"]) + ", stringsAsFactors = FALSE)")


DBY_CASES = []
for i in range(10):
    frame = gen_raw_table(int(rng.choice([4, 20, 60])), int(rng.integers(1, 4)), na_group=(i in (5, 6)))
    groups = [None, ["subject"], ["subject", "image"], ["image"]][i % 4]
    kw = {"sigma": [float(np.round(rng.uniform(20, 120), 1))], "outdim": (20, 15)}
    if i % 5 == 2:
        kw["sigma"] = [30.0, 80.0]
    if i % 3 == 1:
        kw["duration_weighted"] = False
    if i % 4 == 3:
        kw["window"] = (500.0, 2500.0)
    if i % 2:
        kw["keep_vars"] = ["cond"]
    if i % 5 == 4:
        kw["min_fixations"] = 5
    if i == 9:
        kw["kde_pkg"] = "MASS"
    DBY_CASES.append((frame, groups, kw))


# CRQA cases ----------------------------------------------------------------------
CRQA_CASES = []
for i in range(36):
    n1 = int(rng.choice([1, 2, 3, 5, 8, 12, 20, 30]))
    n2 = int(rng.choice([n1, n1, max(1, n1 - 2), n1 + 3]))
    a = gen_fg(n1, ties=(i % 3 == 0))
    b = gen_fg(n2, ties=(i % 3 == 0))
    if i % 4 == 0:  # make b a noisy copy of a so there is recurrence structure
        m = min(n1, n2)
        b["x"][:m] = np.round(a["x"][:m] + rng.normal(0, 20, m), 2)
        b["y"][:m] = np.round(a["y"][:m] + rng.normal(0, 20, m), 2)
    rescale = int(i % 5)
    radius = float(np.round(rng.uniform(20, 250), 1)) if rescale == 0 else float(np.round(rng.uniform(0.05, 1.2), 3))
    s = dict(radius=radius, delay=int(rng.choice([1, 1, 2, 3])), embed=int(rng.choice([1, 1, 2, 3])),
             rescale=rescale, metric=["euclidean", "manhattan"][i % 2])
    CRQA_CASES.append((a, b, s))
CRQA_KEYS = ["RR", "DET", "NRLINE", "maxL", "L", "ENTR", "rENTR", "LAM", "TT", "max_vertlength"]


# revisit cases ---------------------------------------------------------------------
REV_CASES = []
for i in range(36):
    n = int(rng.choice([0, 1, 2, 3, 4, 6, 10, 20, 40]))
    spec = gen_fg(n, kind=["sorted", "unsorted", "tied"][i % 3], ties=(i % 4 == 0),
                  na_onset=(i % 6 == 5), na_xy=(i % 9 == 4))
    if n > 3 and i % 2 == 0:  # plant returns
        for k in range(3, n, 3):
            spec["x"][k] = spec["x"][k - 3] + float(np.round(rng.normal(0, 10), 2))
            spec["y"][k] = spec["y"][k - 3] + float(np.round(rng.normal(0, 10), 2))
    kw = {"radius": float(np.round(rng.uniform(15, 200), 1))}
    if i % 4 == 1:
        kw["min_lag"] = int(rng.integers(1, 4))
    if i % 5 == 2:
        kw["max_lag"] = float(rng.integers(2, 6))
    if i % 3 == 1:
        kw["require_departure"] = False
    if i % 7 == 3:
        kw["dist_method"] = "manhattan"
    ref = None
    if i % 2 == 1:
        k = int(rng.choice([0, 1, 5, 30]))
        ref = np.round(np.column_stack([rng.uniform(0, 800, k), rng.uniform(0, 600, k)]), 2)
        if k > 2 and i % 4 == 3:
            ref[0, 0] = np.nan
    REV_CASES.append((spec, kw, ref))
REV_CASES.append((gen_fg(5), {"radius": 0.0}, None))                   # error
REV_CASES.append((gen_fg(5), {"radius": 50.0, "min_lag": 0}, None))    # error
REV_CASES.append((gen_fg(5), {"radius": 50.0, "min_lag": 3, "max_lag": 2.0}, None))  # error
REV_CASES.append((gen_fg(5), {"radius": 50.0, "min_lag": 1.5}, None))  # error

REV_ET_CASES = []
for i in range(10):
    n_rows = int(rng.integers(2, 9))
    fgs = [gen_fg(int(rng.choice([0, 1, 3, 6, 12])), ties=(i % 3 == 0)) for _ in range(n_rows)]
    subj = [f"s{int(k)}" for k in rng.integers(1, 3, n_rows)]
    item = [f"i{int(k)}" for k in rng.integers(1, 4, n_rows)]
    null = ["none", "subject", "item", "item"][i % 4]
    kw = {"radius": float(np.round(rng.uniform(40, 200), 1)), "null": null}
    if null == "subject":
        kw["subject_var"] = "subject"
    if null == "item":
        kw["item_var"] = "item"
        if i % 8 == 2:
            kw["subject_var"] = "subject"
    if i % 3 == 2:
        kw["require_departure"] = False
    REV_ET_CASES.append((fgs, subj, item, kw))


# visual angle --------------------------------------------------------------------
VA_CASES = []
for i in range(20):
    geom = dict(distance=float(np.round(rng.uniform(30, 120), 1)),
                screen_size=float(np.round(rng.uniform(20, 70), 1)),
                screen_res=float(rng.choice([1024, 1280, 1920, 2560])))
    deg = np.round(rng.uniform(0, 60, int(rng.integers(1, 6))), 3)
    px = np.round(rng.uniform(0, 3000, int(rng.integers(1, 6))), 2)
    if i == 17:
        deg = np.array([180.0])
    if i == 18:
        px = np.array([-1.0])
    if i == 19:
        geom["distance"] = 0.0
    VA_CASES.append((geom, deg, px))


# estimate_scale -------------------------------------------------------------------
SCALE_CASES = []
for i in range(12):
    n = int(rng.integers(3, 12))
    ref = gen_fg(n)
    ref["x"] = np.round(rng.uniform(50, 750, n), 1)
    ref["y"] = np.round(rng.uniform(50, 550, n), 1)
    sx, sy = float(np.round(rng.uniform(0.5, 2.5), 2)), float(np.round(rng.uniform(0.5, 2.5), 2))
    src = dict(ref)
    src["x"] = ref["x"] / sx
    src["y"] = ref["y"] / sy
    SCALE_CASES.append((ref, src, (sx, sy)))
SCALE_EMPTY = (gen_fg(4), gen_fg(0))
MS_SPEC = gen_fg(40)        # multiscale entropy fixture
GETDENS_SPEC = gen_fg(25)   # get_density fixture
MATCH_SRC_KEYS = ["k3", "k1", "zz", "k2"]
MATCH_SRC_ORDER = [2, 0, 3, 1]


# ---------------------------------------------------------------------------
# The R script
# ---------------------------------------------------------------------------

def build_r_code() -> str:
    L = [r"""
enc <- function(v) {
  if (is.null(v)) return(NULL)
  v <- as.numeric(v)
  s <- sprintf("%.17g", v)
  s[is.na(v) & !is.nan(v)] <- "NA"
  s[is.nan(v)] <- "NaN"
  s
}
encm <- function(m) list(dim = dim(m), v = enc(as.vector(m)))
safe <- function(expr) tryCatch(suppressWarnings(suppressMessages(expr)),
                                error = function(e) list(error = conditionMessage(e)))
fgdf <- function(g) list(n = nrow(g), index = enc(g$index), x = enc(g$x), y = enc(g$y),
                         duration = enc(g$duration), onset = enc(g$onset))
dz1 <- function(e) list(x = enc(e$x), y = enc(e$y), z = encm(e$z), sigma = enc(e$sigma))
dz <- function(d) {
  if (is.null(d)) return(list(null = TRUE))
  if (inherits(d, "eye_density_multiscale")) return(list(ms = lapply(d, dz1), sigmas = enc(attr(d, "sigmas_vector"))))
  dz1(d)
}
out <- list()
"""]
    # fixation groups
    L.append("out$fg <- list()")
    for i, s in enumerate(FG_CASES):
        L.append(f"""
g <- {r_fg(s)}
res <- list(fg = fgdf(g), coords = encm(coords(g)))
res$center <- safe(fgdf(center(g)))
res$center_o <- safe(fgdf(center(g, origin = {rvec(s['origin'])})))
res$norm <- safe(fgdf(normalize(g, {rvec(s['xb'])}, {rvec(s['yb'])})))
res$rescale <- safe(fgdf(rescale(g, {rnum(s['sx'])}, {rnum(s['sy'])})))
res$rep <- safe({{ rp <- rep_fixations(g, {rnum(s['res'])}); list(n = nrow(rp), runs = rle(rp$index)$lengths, index = enc(unique(rp$index))) }})
res$sf <- safe(fgdf(sample_fixations(g, {rvec(s['times'])})))
res$sf_slow <- safe(fgdf(sample_fixations(g, {rvec(s['times'])}, fast = FALSE)))
res$sp <- safe({{ sp <- scanpath(g); list(n = nrow(sp), lenx = enc(sp$lenx), leny = enc(sp$leny), rho = enc(sp$rho), theta = enc(sp$theta), x = enc(sp$x)) }})
res$summary <- safe(lapply(as.list(summary(g)), enc))
res$sig <- safe(enc(suggest_sigma(g)))
res$sig_xy <- safe(enc(suggest_sigma(g$x, g$y)))
res$sig_b <- safe(enc(suggest_sigma(g, xbounds = {rvec(s['xb'])}, ybounds = {rvec(s['yb'])})))
out$fg[[{i + 1}]] <- res
""")
    # concatenation of neighbouring non-empty groups
    L.append("out$concat <- list()")
    for i in range(len(FG_CASES) - 2):
        L.append(f"out$concat[[{i + 1}]] <- safe(fgdf(c({r_fg(FG_CASES[i + 1])}, {r_fg(FG_CASES[i + 2])})))")
    L.append("out$concat3 <- safe(fgdf(c(" + ", ".join(r_fg(FG_CASES[k]) for k in (3, 5, 9)) + ")))")
    # angles / polar
    L.append("out$angle <- list(" + ", ".join(
        f"safe(enc(calcangle({rvec(a)}, {rvec(b)})))" for a, b in ANGLE_CASES) + ")")
    L.append(f"out$polar <- encm(cart2pol({rvec(POLAR_X)}, {rvec(POLAR_Y)}))")
    # add_scanpath on an eye table
    L.append(f"""
et_sp <- eye_table("px", "py", "dur", "on", groupvar = "subject", data = {r_frame(ET_CASES[3][0])})
sp_tab <- add_scanpath(et_sp)
out$add_scanpath <- list(names = names(sp_tab), rho = lapply(sp_tab$scanpath, function(s) enc(s$rho)),
                         theta = lapply(sp_tab$scanpath, function(s) enc(s$theta)))
""")
    # densities
    L.append("out$dens <- list()")
    for i, (spec, args) in enumerate(DENS_CASES):
        L.append(f"g <- {r_fg(spec)}\nout$dens[[{i + 1}]] <- safe(dz({r_dens_call(spec, args)}))")
    L.append("out$getdens <- encm(get_density(eye_density(" + r_fg(GETDENS_SPEC) +
             ", sigma = 40, xbounds = c(0, 800), ybounds = c(0, 600), outdim = c(9, 7))))")
    # entropy
    L.append("out$ent <- list()")
    for i, (spec, kw) in enumerate(ENT_CASES):
        parts = [f'method = "{kw["method"]}"']
        for key in ("normalize",):
            if key in kw:
                parts.append(f"{key} = {rbool(kw[key])}")
        if "base" in kw:
            parts.append(f"base = {rnum(kw['base'])}")
        if "sigma" in kw:
            parts.append(f"sigma = {rvec(kw['sigma'])}")
        if "aggregate" in kw:
            parts.append(f'aggregate = "{kw["aggregate"]}"')
        for key in ("outdim", "grid", "xbounds", "ybounds"):
            if key in kw:
                parts.append(f"{key} = {rvec(kw[key])}")
        if "duration_weighted" in kw:
            parts.append(f"duration_weighted = {rbool(kw['duration_weighted'])}")
        L.append(f"g <- {r_fg(spec)}\nout$ent[[{i + 1}]] <- safe({{ e <- fixation_entropy(g, {', '.join(parts)}); list(v = enc(e), names = names(e)) }})")
    L.append("out$mass <- list(" + ", ".join(
        f"safe(enc(entropy_from_mass({rvec(m)}, normalize = {rbool(nz)}, base = {rnum(b)})))"
        for m, nz, b in MASS_CASES) + ")")
    # entropy of a multiscale density object directly
    L.append("""
ms_obj <- eye_density(""" + r_fg(MS_SPEC) + """, sigma = c(25, 50.5, 12.345678901), xbounds = c(0, 800), ybounds = c(0, 600), outdim = c(30, 20))
e_none <- fixation_entropy(ms_obj, aggregate = "none")
out$ent_ms <- list(mean = enc(fixation_entropy(ms_obj)), none = enc(e_none), names = names(e_none),
                   base2 = enc(fixation_entropy(ms_obj, base = 2, normalize = FALSE, aggregate = "none")))
""")
    # eye tables
    L.append("out$et <- list()")
    for i, (frame, groupvar, clip, rel, vars_) in enumerate(ET_CASES):
        vars_s = "NULL" if vars_ is None else rstr(vars_)
        L.append(f"""
out$et[[{i + 1}]] <- safe({{
  et <- eye_table("px", "py", "dur", "on", groupvar = {rstr(groupvar)}, vars = {vars_s},
                  data = {r_frame(frame)}, clip_bounds = {rvec(clip)}, relative_coords = {rbool(rel)})
  keys <- lapply(et[{rstr(groupvar)}], function(v) as.character(v))
  list(names = names(et), keys = keys, origin = enc(attr(et, "origin")), fg = lapply(et$fixgroup, fgdf),
       vars = if (is.null({vars_s})) NULL else lapply(et[{vars_s}], enc))
}})""")
    # density_by
    L.append("out$dby <- list()")
    for i, (frame, groups, kw) in enumerate(DBY_CASES):
        gvar = "subject" if groups is None else groups
        parts = [f"groups = {'NULL' if groups is None else rstr(groups)}", f"sigma = {rvec(kw['sigma'])}",
                 "xbounds = c(0, 800)", "ybounds = c(0, 600)", f"outdim = {rvec(kw['outdim'])}"]
        if "duration_weighted" in kw:
            parts.append(f"duration_weighted = {rbool(kw['duration_weighted'])}")
        if "window" in kw:
            parts.append(f"window = {rvec(kw['window'])}")
        if "keep_vars" in kw:
            parts.append(f"keep_vars = {rstr(kw['keep_vars'])}")
        if "min_fixations" in kw:
            parts.append(f"min_fixations = {kw['min_fixations']}")
        if "kde_pkg" in kw:
            parts.append(f'kde_pkg = "{kw["kde_pkg"]}"')
        gv = rstr(["subject", "image"])
        L.append(f"""
out$dby[[{i + 1}]] <- safe({{
  et <- eye_table("px", "py", "dur", "on", groupvar = {gv}, vars = "cond", data = {r_frame(frame)},
                  clip_bounds = c(0, 800, 0, 600))
  d <- density_by(et, {', '.join(parts)})
  keycols <- intersect(names(d), c("subject", "image", "cond"))
  list(names = names(d), keys = lapply(d[keycols], as.character),
       dens = lapply(d$density, dz), nfix = vapply(d$fixgroup, nrow, integer(1)))
}})""")
    # crqa
    L.append("crqa_keys <- " + rstr(CRQA_KEYS) + "\nout$crqa <- list()")
    for i, (a, b, s) in enumerate(CRQA_CASES):
        L.append(f"""
out$crqa[[{i + 1}]] <- safe({{
  r <- crqa({r_fg(a)}, {r_fg(b)}, radius = {rnum(s['radius'])}, delay = {s['delay']}, embed = {s['embed']},
            rescale = {s['rescale']}, metric = "{s['metric']}")
  vals <- lapply(crqa_keys, function(k) enc(if (is.null(r[[k]]) || !length(r[[k]])) NA else r[[k]]))
  names(vals) <- crqa_keys
  vals$RP <- if (inherits(r$RP, "Matrix") || is.matrix(r$RP)) encm(as.matrix(r$RP) * 1) else NULL
  vals
}})""")
    # revisits
    L.append("rv_cols <- c('index', 'onset', 'opportunity', 'revisit', 'lag', 'match_index', 'distance', 'expected')\nout$rev <- list()")
    for i, (spec, kw, ref) in enumerate(REV_CASES):
        parts = [f"radius = {rnum(kw['radius'])}"]
        if "min_lag" in kw:
            parts.append(f"min_lag = {rnum(kw['min_lag'])}")
        if "max_lag" in kw:
            parts.append(f"max_lag = {rnum(kw['max_lag'])}")
        if "require_departure" in kw:
            parts.append(f"require_departure = {rbool(kw['require_departure'])}")
        if "dist_method" in kw:
            parts.append(f'dist_method = "{kw["dist_method"]}"')
        if ref is not None:
            parts.append(f"reference = {rmat(ref)}")
        L.append(f"""
out$rev[[{i + 1}]] <- safe({{
  rv <- fixation_revisits({r_fg(spec)}, {', '.join(parts)})
  list(cols = lapply(as.list(rv)[rv_cols], enc), summary = lapply(as.list(revisit_summary(rv)), enc))
}})""")
    L.append("out$rev_et <- list()")
    for i, (fgs, subj, item, kw) in enumerate(REV_ET_CASES):
        parts = [f"radius = {rnum(kw['radius'])}", f'null = "{kw["null"]}"']
        for key in ("subject_var", "item_var"):
            if key in kw:
                parts.append(f'{key} = "{kw[key]}"')
        if "require_departure" in kw:
            parts.append(f"require_departure = {rbool(kw['require_departure'])}")
        fg_list = "list(" + ", ".join(r_fg(f) for f in fgs) + ")"
        L.append(f"""
out$rev_et[[{i + 1}]] <- safe({{
  et <- as_eye_table(tibble::tibble(subject = {rstr(subj)}, item = {rstr(item)}, fixgroup = {fg_list}))
  res <- fixation_revisits(et, {', '.join(parts)})
  list(names = names(res), summary = lapply(as.list(res)[revisit_summary_cols], enc))
}})""")
    # visual angle
    L.append("out$va <- list()")
    for i, (geom, deg, px) in enumerate(VA_CASES):
        g = f"distance = {rnum(geom['distance'])}, screen_size = {rnum(geom['screen_size'])}, screen_res = {rnum(geom['screen_res'])}"
        L.append(f"out$va[[{i + 1}]] <- list(px = safe(enc(visual_angle_to_px({rvec(deg)}, {g}))), "
                 f"deg = safe(enc(px_to_visual_angle({rvec(px)}, {g}))))")
    # estimate_scale / match_scale
    L.append("out$scale <- list()")
    for i, (ref, src, _) in enumerate(SCALE_CASES):
        L.append(f"out$scale[[{i + 1}]] <- enc(estimate_scale({r_fg(ref)}, {r_fg(src)})$par)")
    L.append(f"out$scale_empty <- enc(estimate_scale({r_fg(SCALE_EMPTY[0])}, {r_fg(SCALE_EMPTY[1])})$par)")
    L.append(f"out$scale_window <- enc(estimate_scale({r_fg(SCALE_CASES[0][0])}, {r_fg(SCALE_CASES[0][1])}, window = c(-1, 1e9))$par)")
    refs = "list(" + ", ".join(r_fg(c[0]) for c in SCALE_CASES[:4]) + ")"
    srcs = "list(" + ", ".join(r_fg(SCALE_CASES[k][1]) for k in MATCH_SRC_ORDER) + ")"
    L.append(f"""
ms_ref <- tibble::tibble(key = c("k1", "k2", "k3", "k4"), fixgroup = {refs})
ms_src <- tibble::tibble(key = {rstr(MATCH_SRC_KEYS)}, fixgroup = {srcs})
out$match_scale <- safe({{ ms_res <- match_scale(ms_ref, ms_src, match_on = "key")
  list(key = ms_res$key, scale_x = enc(ms_res$scale_x), scale_y = enc(ms_res$scale_y)) }})
""")
    return "\n".join(L)


R_CODE = build_r_code()


@pytest.fixture(scope="module")
def r():
    return run_r(R_CODE)


def py_fgdf(g):
    f = g.to_pandas(copy=False) if hasattr(g, "to_pandas") else g
    return {c: f[c].to_numpy(dtype=float) for c in ("index", "x", "y", "duration", "onset") if c in f}


def check_fg(r_fg_out, py_fg, rtol=0.0, atol_scale=None):
    assert r_fg_out["n"] == len(py_fg)
    got = py_fgdf(py_fg)
    for col in ("index", "x", "y", "duration", "onset"):
        if r_fg_out.get(col) is None:
            continue
        if col not in got:
            raise AssertionError(f"column {col} missing in Python")
        if atol_scale is None:
            close(got[col], r_fg_out[col], rtol=rtol)
        else:
            close_rel(got[col], r_fg_out[col], scale=atol_scale)


# ---------------------------------------------------------------------------
# Fixation groups
# ---------------------------------------------------------------------------

@requires_r
@pytest.mark.parametrize("i", range(len(FG_CASES)))
def test_fixation_group_ops_match_r(r, i):
    spec = FG_CASES[i]
    res = r["fg"][i]
    g = make_fg(spec)
    check_fg(res["fg"], g)
    np.testing.assert_array_equal(pe.coords(g), decm(res["coords"]) if len(g) else np.empty((0, 2)))
    assert set(g.to_pandas().columns) >= {"index", "x", "y", "duration", "onset", "group_index"}

    scale = max(1.0, float(np.nanmax(np.abs(spec["x"]), initial=0)), float(np.nanmax(np.abs(spec["y"]), initial=0)))
    assert_both_error_or(res["center"], py_safe(lambda: pe.center(g)),
                         lambda rv, pv: check_fg(rv, pv, atol_scale=scale))
    assert_both_error_or(res["center_o"], py_safe(lambda: pe.center(g, origin=spec["origin"])),
                         lambda rv, pv: check_fg(rv, pv))
    assert_both_error_or(res["norm"], py_safe(lambda: pe.normalize(g, spec["xb"], spec["yb"])),
                         lambda rv, pv: check_fg(rv, pv))
    assert_both_error_or(res["rescale"], py_safe(lambda: pe.rescale(g, spec["sx"], spec["sy"])),
                         lambda rv, pv: check_fg(rv, pv))

    def check_rep(rv, pv):
        assert rv["n"] == len(pv)
        idx = pv["index"].to_numpy()
        if len(idx):
            change = np.flatnonzero(np.diff(idx) != 0)
            runs = np.diff(np.concatenate([[0], change + 1, [len(idx)]]))
            close(runs, [str(v) for v in np.atleast_1d(rv["runs"])])
            close(idx[np.concatenate([[0], change + 1])], rv["index"])
    assert_both_error_or(res["rep"], py_safe(lambda: pe.rep_fixations(g, spec["res"])), check_rep)

    for key, fast in (("sf", True), ("sf_slow", False)):
        assert_both_error_or(res[key], py_safe(lambda: pe.sample_fixations(g, spec["times"], fast=fast)),
                             lambda rv, pv: check_fg(rv, pv))

    def check_sp(rv, pv):
        assert rv["n"] == len(pv)
        for col in ("lenx", "leny", "rho", "x"):
            close(pv[col].to_numpy(dtype=float), rv[col])
        # atan2 comes from different libms (R: the C library, numpy: its own),
        # which may differ in the last ulp.
        close(pv["theta"].to_numpy(dtype=float), rv["theta"], rtol=1e-15, atol=1e-15)
    assert_both_error_or(res["sp"], py_safe(lambda: pe.scanpath(g)), check_sp)

    def check_summary(rv, pv):
        for key in ("cen_x", "cen_y", "sd_x", "sd_y", "nfix"):
            close_rel([pv[key]], rv[key], scale=scale)
    assert_both_error_or(res["summary"], py_safe(g.summary), check_summary)

    for key, call in (("sig", lambda: pe.suggest_sigma(g)),
                      ("sig_xy", lambda: pe.suggest_sigma(spec["x"], spec["y"])),
                      ("sig_b", lambda: pe.suggest_sigma(g, xbounds=spec["xb"], ybounds=spec["yb"]))):
        assert_both_error_or(res[key], py_safe(call), lambda rv, pv: close_rel([pv], rv))


@requires_r
@pytest.mark.parametrize("i", range(len(FG_CASES) - 2))
def test_concat_fixation_groups_matches_r_c(r, i):
    a, b = make_fg(FG_CASES[i + 1]), make_fg(FG_CASES[i + 2])
    assert_both_error_or(r["concat"][i], py_safe(lambda: pe.concat_fixation_groups(a, b)),
                         lambda rv, pv: check_fg(rv, pv))


@requires_r
def test_concat_three_groups_matches_r(r):
    gs = [make_fg(FG_CASES[k]) for k in (3, 5, 9)]
    check_fg(r["concat3"], pe.concat_fixation_groups(*gs))


@requires_r
def test_calcangle_and_cart2pol_match_r(r):
    for (a, b), rv in zip(ANGLE_CASES, r["angle"]):
        got = py_safe(lambda: pe.calcangle(a, b))
        # arccos is ill-conditioned at +/-1: an ULP in the cosine moves the angle
        # by ~1e-6 degrees, so (anti)parallel pairs get 1e-5 degrees.
        assert_both_error_or(rv, got, lambda rv_, pv: close([pv], rv_, rtol=1e-12, atol=1e-5))
    polar = pe.cart2pol(POLAR_X, POLAR_Y)
    expected = decm(r["polar"])
    close(polar[:, 0], expected[:, 0])
    # atan2: last-ulp differences between libms (see test_fixation_group_ops_match_r)
    close(polar[:, 1], expected[:, 1], rtol=1e-15, atol=1e-15)


@requires_r
def test_add_scanpath_matches_r(r):
    frame, groupvar, clip, rel, vars_ = ET_CASES[3]
    et = pe.eye_table("px", "py", "dur", "on", groupvar="subject", data=frame)
    tab = pe.add_scanpath(et.to_pandas())
    assert list(tab.columns) == r["add_scanpath"]["names"]
    for sp, rho, theta in zip(tab["scanpath"], r["add_scanpath"]["rho"], r["add_scanpath"]["theta"]):
        close(sp["rho"].to_numpy(), rho)
        close(sp["theta"].to_numpy(), theta, rtol=1e-15, atol=1e-15)  # atan2 ulp


# ---------------------------------------------------------------------------
# Densities
# ---------------------------------------------------------------------------

def check_density(rv, pv):
    if rv.get("null"):
        assert pv is None, f"R returned NULL, Python {pv!r}"
        return
    assert pv is not None, "Python returned None, R a density"
    if "ms" in rv:
        assert isinstance(pv, pe.EyeDensityMultiscale)
        assert len(pv) == len(rv["ms"])
        close(pv.sigmas, rv["sigmas"])
        for e, d in zip(rv["ms"], pv):
            check_density(e, d)
        return
    assert isinstance(pv, pe.EyeDensity)
    close(pv.x, rv["x"])
    close(pv.y, rv["y"])
    close([pv.sigma], rv["sigma"])
    assert_density_close(pv.z, decm(rv["z"]))


@requires_r
@pytest.mark.parametrize("i", range(len(DENS_CASES)))
def test_eye_density_matches_r(r, i):
    spec, args = DENS_CASES[i]
    g = make_fg(spec)
    got = py_safe(lambda: pe.eye_density(g, **py_dens_kwargs(args)))
    assert_both_error_or(r["dens"][i], got, check_density)


@requires_r
def test_get_density_matches_r(r):
    g = make_fg(GETDENS_SPEC)
    d = pe.eye_density(g, sigma=40, xbounds=(0, 800), ybounds=(0, 600), outdim=(9, 7))
    assert_density_close(pe.get_density(d), decm(r["getdens"]))


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

@requires_r
@pytest.mark.parametrize("i", range(len(ENT_CASES)))
def test_fixation_entropy_matches_r(r, i):
    spec, kw = ENT_CASES[i]
    g = make_fg(spec)
    pkw = dict(kw)
    if "sigma" in pkw:
        pkw["sigma"] = pkw["sigma"][0] if len(pkw["sigma"]) == 1 else pkw["sigma"]
    got = py_safe(lambda: pe.fixation_entropy(g, **pkw))

    def check(rv, pv):
        if isinstance(pv, dict):
            assert list(pv.keys()) == rv["names"]
            close_rel(list(pv.values()), rv["v"])
        else:
            assert rv.get("names") is None
            close_rel([pv], rv["v"])
    assert_both_error_or(r["ent"][i], got, check)


@requires_r
@pytest.mark.parametrize("i", range(len(MASS_CASES)))
def test_entropy_from_mass_matches_r(r, i):
    m, nz, b = MASS_CASES[i]
    got = py_safe(lambda: pe.entropy_from_mass(m, normalize=nz, base=b))
    assert_both_error_or(r["mass"][i], got, lambda rv, pv: close_rel([pv], rv))


@requires_r
def test_multiscale_entropy_matches_r(r):
    g = make_fg(MS_SPEC)
    ms = pe.eye_density(g, sigma=[25, 50.5, 12.345678901], xbounds=(0, 800), ybounds=(0, 600),
                        outdim=(30, 20))
    close_rel([pe.fixation_entropy(ms)], r["ent_ms"]["mean"])
    none = pe.fixation_entropy(ms, aggregate="none")
    assert list(none.keys()) == r["ent_ms"]["names"]
    close_rel(list(none.values()), r["ent_ms"]["none"])
    raw = pe.fixation_entropy(ms, base=2, normalize=False, aggregate="none")
    close_rel(list(raw.values()), r["ent_ms"]["base2"])


# ---------------------------------------------------------------------------
# eye_table and density_by
# ---------------------------------------------------------------------------

def _keys(values):
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in values]


def _rkeys(values):
    vals = values if isinstance(values, list) else [values]
    return [None if v is None else str(v) for v in vals]


@requires_r
@pytest.mark.parametrize("i", range(len(ET_CASES)))
def test_eye_table_matches_r(r, i):
    frame, groupvar, clip, rel, vars_ = ET_CASES[i]
    got = py_safe(lambda: pe.eye_table("px", "py", "dur", "on", groupvar=groupvar, data=frame, vars=vars_,
                                       clip_bounds=clip, relative_coords=rel))

    def check(rv, et):
        assert list(et.columns) == rv["names"]
        for gv in groupvar:
            assert _keys(et[gv]) == _rkeys(rv["keys"][gv])
        close(et.origin, rv["origin"])
        assert len(et) == len(rv["fg"])
        for fg, rfg in zip(et["fixgroup"], rv["fg"]):
            check_fg(rfg, fg)
        if vars_:
            for v in vars_:
                close(et[v].to_numpy(dtype=float), rv["vars"][v])
    assert_both_error_or(r["et"][i], got, check)


@requires_r
@pytest.mark.parametrize("i", range(len(DBY_CASES)))
def test_density_by_matches_r(r, i):
    frame, groups, kw = DBY_CASES[i]
    et = pe.eye_table("px", "py", "dur", "on", groupvar=["subject", "image"], vars=["cond"], data=frame,
                      clip_bounds=(0, 800, 0, 600))
    pkw = dict(kw)
    pkw["sigma"] = kw["sigma"][0] if len(kw["sigma"]) == 1 else kw["sigma"]
    got = py_safe(lambda: pe.density_by(et, groups=groups, xbounds=(0, 800), ybounds=(0, 600), **pkw))

    def check(rv, d):
        assert list(d.columns) == rv["names"]
        for col, vals in rv["keys"].items():
            if col == "cond":
                close(d[col].to_numpy(dtype=float), [str(float(v)) for v in _rkeys(vals)])
            else:
                assert _keys(d[col]) == _rkeys(vals)
        close([len(fg) for fg in d["fixgroup"]], [str(v) for v in np.atleast_1d(rv["nfix"])])
        assert len(d) == len(rv["dens"])
        for pv, dv in zip(d["density"], rv["dens"]):
            check_density(dv, pv)
    assert_both_error_or(r["dby"][i], got, check)


# ---------------------------------------------------------------------------
# CRQA
# ---------------------------------------------------------------------------

@requires_r
@pytest.mark.parametrize("i", range(len(CRQA_CASES)))
def test_crqa_matches_r(r, i):
    a, b, s = CRQA_CASES[i]
    got = py_safe(lambda: pe.crqa(make_fg(a), make_fg(b), **s))
    n_embedded = min(len(a["x"]), len(b["x"])) - (s["embed"] - 1) * s["delay"]
    if s["embed"] > 1 and n_embedded == 1:
        # crqa::crqa() embeds with ts[1:newLength, ] (no drop = FALSE): a single
        # embedded point collapses to a vector and comes back as a 2 x 2 plot
        # of x and y "points". A crqa bug, not reproduced: peyesim scores the one
        # embedded point.
        rv = r["crqa"][i]
        assert not is_err(rv)
        assert rv.get("RP") is None or decm(rv["RP"]).shape == (2, 2)
        assert got["n_points"] == 1 and got["RP"].shape == (1, 1)
        return

    def check(rv, pv):
        for key in CRQA_KEYS:
            close_rel([pv[key]], rv[key])
        if rv.get("RP") is not None:
            np.testing.assert_array_equal(np.asarray(pv["RP"], dtype=float), decm(rv["RP"]))
    assert_both_error_or(r["crqa"][i], got, check)


# ---------------------------------------------------------------------------
# Revisits
# ---------------------------------------------------------------------------

REV_COLS = ["index", "onset", "opportunity", "revisit", "lag", "match_index", "distance", "expected"]


@requires_r
@pytest.mark.parametrize("i", range(len(REV_CASES)))
def test_fixation_revisits_matches_r(r, i):
    spec, kw, ref = REV_CASES[i]
    g = make_fg(spec)
    got = py_safe(lambda: pe.fixation_revisits(g, reference=ref, **kw))

    def check(rv, tab):
        for col in REV_COLS:
            vals = pd.array(tab[col]).to_numpy(dtype=float, na_value=np.nan)
            if col in ("distance", "expected"):
                close_rel(vals, rv["cols"][col])
            else:
                close(vals, rv["cols"][col])
        summ = pe.revisit_summary(tab)
        for col, v in rv["summary"].items():
            close_rel(summ[col].to_numpy(dtype=float), v)
    assert_both_error_or(r["rev"][i], got, check)


@requires_r
@pytest.mark.parametrize("i", range(len(REV_ET_CASES)))
def test_fixation_revisits_eye_table_matches_r(r, i):
    fgs, subj, item, kw = REV_ET_CASES[i]
    et = pe.as_eye_table(pd.DataFrame({"subject": subj, "item": item,
                                       "fixgroup": pd.Series([make_fg(f) for f in fgs], dtype=object)}))
    got = py_safe(lambda: pe.fixation_revisits(et, **kw))

    def check(rv, res):
        assert list(res.columns) == rv["names"]
        for col, v in rv["summary"].items():
            close_rel(res[col].to_numpy(dtype=float), v)
    assert_both_error_or(r["rev_et"][i], got, check)


# ---------------------------------------------------------------------------
# Visual angle and scale estimation
# ---------------------------------------------------------------------------

@requires_r
@pytest.mark.parametrize("i", range(len(VA_CASES)))
def test_visual_angle_matches_r(r, i):
    geom, deg, px = VA_CASES[i]
    rv = r["va"][i]
    assert_both_error_or(rv["px"], py_safe(lambda: pe.visual_angle_to_px(deg, **geom)),
                         lambda e, p: close_rel(p, e))
    assert_both_error_or(rv["deg"], py_safe(lambda: pe.px_to_visual_angle(px, **geom)),
                         lambda e, p: close_rel(p, e))


@requires_r
@pytest.mark.parametrize("i", range(len(SCALE_CASES)))
def test_estimate_scale_matches_r(r, i):
    ref, src, truth = SCALE_CASES[i]
    got = estimate_scale(make_fg(ref), make_fg(src))["par"]
    close(got, r["scale"][i], atol=1e-3)
    np.testing.assert_allclose(got, truth, atol=1e-3)


@requires_r
def test_estimate_scale_edge_cases_match_r(r):
    close(estimate_scale(make_fg(SCALE_EMPTY[0]), make_fg(SCALE_EMPTY[1]))["par"], r["scale_empty"])
    ref, src, _ = SCALE_CASES[0]
    close(estimate_scale(make_fg(ref), make_fg(src), window=(-1, 1e9))["par"], r["scale_window"], atol=1e-3)


@requires_r
def test_match_scale_r_is_broken_python_matches_estimate_scale(r):
    # R's match_scale() forwards each row's columns through `...` into
    # estimate_scale(), which rejects them ("unused arguments"), so it cannot be
    # used as an oracle. Check Python against its own estimate_scale() instead.
    assert is_err(r["match_scale"]) and "unused argument" in r["match_scale"]["error"]
    ref_tab = pd.DataFrame({"key": ["k1", "k2", "k3", "k4"],
                            "fixgroup": pd.Series([make_fg(c[0]) for c in SCALE_CASES[:4]], dtype=object)})
    src_tab = pd.DataFrame({"key": MATCH_SRC_KEYS,
                            "fixgroup": pd.Series([make_fg(SCALE_CASES[k][1]) for k in MATCH_SRC_ORDER],
                                                  dtype=object)})
    with pytest.warns(UserWarning, match="did not find matching"):
        res = match_scale(ref_tab, src_tab, match_on="key")
    assert res["key"].tolist() == ["k3", "k1", "k2"]
    for _, row in res.iterrows():
        k = int(row["key"][1]) - 1
        np.testing.assert_allclose([row["scale_x"], row["scale_y"]], SCALE_CASES[k][2], atol=1e-3)
