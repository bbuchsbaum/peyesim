"""Randomized differential tests of the similarity family against live R eyesim.

Every case is drawn from one fixed NumPy seed. Inputs are generated in Python
and handed to R as exact literals (``repr(float)``), so both languages see
bit-identical numbers; all R cases run in one R script. Covered:

* ``compute_similarity`` / ``similarity`` for every method on random vectors and
  density maps (negative values, ties, constant and all-zero maps, NaN cells,
  tiny magnitudes, mismatched lattices), saliency EMD and multiscale maps with
  shuffled and partly shared sigmas;
* ``template_similarity`` with duplicated and unmatched (incl. missing) keys,
  ``None`` cells, ``permute_on`` strata of size one and missing strata,
  multiscale maps, every latent/geometric transform (compared through
  sign-invariant similarities), and ``template_similarity_cv`` with seeded folds,
  filters, transforms and missing split keys;
* ``sample_density``, ``sample_density_time`` (bins, normalisation, aggregate
  functions, strata), ``fixation_overlap``, ``fixation_similarity``,
  ``multi_match`` and ``scanpath_similarity`` on random scanpaths of 2-15
  fixations, ``repetitive_similarity``, ``template_regression``,
  ``template_multireg`` and ``template_sample``.

Permutation baselines always use ``permutations`` >= the number of candidates
(exhaustive), so no random stream is involved. When R raises an error, peyesim
must raise too; when R warns and returns NA, peyesim must return NaN.

Tolerances: 1e-9 relative/1e-12 absolute for closed-form quantities (both sides
do the same floating-point work in a different order); 1e-6 absolute for EMD
(``emdist`` works in single precision); 1e-8 for the Sinkhorn fixed point and
iterative fits whose stopping rule is reproduced exactly (MASS ``rlm``); 1e-5
for logistic regression (R's IRLS and peyesim's BFGS stop at different
tolerances on the same likelihood).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from peyesim import (
    affine_transform,
    cca_transform,
    contract_transform,
    coral_transform,
    fixation_group,
    fixation_overlap,
    fixation_similarity,
    latent_pca_transform,
    multi_match,
    repetitive_similarity,
    sample_density,
    sample_density_time,
    scanpath,
    scanpath_similarity,
    similarity,
    template_multireg,
    template_regression,
    template_sample,
    template_similarity,
    template_similarity_cv,
)
from peyesim.density import EyeDensity, EyeDensityMultiscale, gen_density
from peyesim.fixations import FixationGroup
from peyesim.saccades import Scanpath
from peyesim.similarity import compute_similarity
from tests.r_oracle import as_float, requires_r, run_r

SEED = 20261025
RNG = np.random.default_rng(SEED)
METHODS = ["pearson", "spearman", "fisherz", "cosine", "l1", "jaccard", "dcov"]
TRANSFORMS = {
    "latent_pca_transform": latent_pca_transform,
    "coral_transform": coral_transform,
    "cca_transform": cca_transform,
    "contract_transform": contract_transform,
    "affine_transform": affine_transform,
}


# ---------------------------------------------------------------------------
# Literal rendering (Python object -> R source)
# ---------------------------------------------------------------------------

def _num(v) -> str:
    v = float(v)
    if np.isnan(v):
        return "NA"
    if np.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    return repr(v)


def rnum(values) -> str:
    arr = np.asarray(values, dtype=float).ravel(order="F")
    return "c(" + ", ".join(_num(t) for t in arr) + ")" if arr.size else "numeric(0)"


def rchr(values) -> str:
    return "c(" + ", ".join("NA" if v is None else f'"{v}"' for v in values) + ")"


def rlgl(values) -> str:
    return "c(" + ", ".join("TRUE" if v else "FALSE" for v in values) + ")"


def robj(obj) -> str:
    if obj is None:
        return "NULL"
    if isinstance(obj, EyeDensityMultiscale):
        return ('structure(list(' + ", ".join(robj(s) for s in obj)
                + '), class = c("eye_density_multiscale", "list"))')
    if isinstance(obj, EyeDensity):
        nx, ny = obj.z.shape
        core = f"gen_density({rnum(obj.x)}, {rnum(obj.y)}, matrix({rnum(obj.z)}, {nx}, {ny}))"
        if obj.sigma is None:
            return core
        return f"{{ d <- {core}; d$sigma <- {_num(obj.sigma)}; d }}"
    if isinstance(obj, FixationGroup):
        core = (f"fixation_group(x = {rnum(obj['x'])}, y = {rnum(obj['y'])}, "
                f"onset = {rnum(obj['onset'])}, duration = {rnum(obj['duration'])})")
        return f"scanpath({core})" if isinstance(obj, Scanpath) else core
    if isinstance(obj, (list, tuple, np.ndarray)):
        return rnum(obj)
    if isinstance(obj, bool):
        return "TRUE" if obj else "FALSE"
    if isinstance(obj, str):
        return f'"{obj}"'
    return _num(obj)


def rlist(objs) -> str:
    return "list(" + ", ".join(robj(o) for o in objs) + ")"


def rargs(args: dict) -> str:
    parts = []
    for k, v in args.items():
        name = f"`{k}`" if "." in k else k
        parts.append(f"{name} = {robj(v)}")
    return "list(" + ", ".join(parts) + ")"


def rtable(cols: dict) -> str:
    """``cols``: name -> ("num"|"chr"|"list", values)."""
    parts = []
    for name, (kind, values) in cols.items():
        if kind == "num":
            parts.append(f"{name} = {rnum(values)}")
        elif kind == "chr":
            parts.append(f"{name} = {rchr(values)}")
        else:
            parts.append(f"{name} = {rlist(values)}")
    return "tibble::tibble(" + ", ".join(parts) + ")"


def pytable(cols: dict) -> pd.DataFrame:
    data = {}
    for name, (kind, values) in cols.items():
        if kind == "num":
            data[name] = np.asarray(values, dtype=float)
        else:
            data[name] = pd.Series(list(values), dtype=object)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Random generators
# ---------------------------------------------------------------------------

def rand_grid(n, lo=0.0, hi=100.0):
    return np.round(np.sort(RNG.choice(np.arange(lo, hi, 0.5), size=n, replace=False)), 3)


def rand_z(nx, ny, kind):
    if kind == "smooth":
        xx, yy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
        z = np.zeros((nx, ny))
        for _ in range(RNG.integers(1, 3)):
            cx, cy, s = RNG.uniform(0.1, 0.9), RNG.uniform(0.1, 0.9), RNG.uniform(0.15, 0.5)
            z += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
        return z / z.sum()
    if kind == "rand":
        return RNG.random((nx, ny))
    if kind == "ties":
        return RNG.integers(0, 4, size=(nx, ny)).astype(float)
    if kind == "zeros":
        return RNG.random((nx, ny)) * (RNG.random((nx, ny)) < 0.5)
    if kind == "const":
        return np.full((nx, ny), float(RNG.integers(1, 5)))
    if kind == "zero":
        return np.zeros((nx, ny))
    if kind == "neg":
        return RNG.normal(size=(nx, ny))
    if kind == "nan":
        z = RNG.random((nx, ny))
        z.flat[RNG.integers(z.size)] = np.nan
        return z
    if kind == "tiny":
        return RNG.random((nx, ny)) * 1e-9
    raise ValueError(kind)


def rand_fg(n, lo=0.0, hi=100.0, t0=0.0, scan=False):
    onset = t0 + np.cumsum(RNG.integers(50, 400, size=n)).astype(float) - RNG.integers(0, 50)
    fg = fixation_group(x=np.round(RNG.uniform(lo, hi, n), 2), y=np.round(RNG.uniform(lo, hi, n), 2),
                        onset=onset, duration=RNG.integers(40, 400, size=n).astype(float))
    return scanpath(fg) if scan else fg


def run_py(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn()
        except Exception as exc:  # compared against an R error
            return {"error": f"{type(exc).__name__}: {exc}"}


def is_err(v) -> bool:
    return isinstance(v, dict) and "error" in v


def same(py, r, rtol=1e-9, atol=1e-12) -> str | None:
    """Return None when ``py`` matches the R value ``r``, else a description."""
    if is_err(r) or is_err(py):
        if is_err(r) and is_err(py):
            return None
        return f"R={r!r} PY={py!r}"
    a = np.asarray(py, dtype=float).ravel()
    b = np.asarray(as_float(r), dtype=float).ravel()
    if a.shape != b.shape:
        return f"shape R={b.shape} PY={a.shape}: R={b} PY={a}"
    if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        return f"R={b} PY={a}"
    return None


def as_list(v):
    """jsonlite unboxes length-1 vectors; undo that for key/term comparisons."""
    return v if isinstance(v, list) else [v]


# R crashes (reported upstream) where peyesim returns the intended result:
# * multiscale_aggregation = "none" with an invalid (NULL) row: the NA row is a
#   scalar while the others are list cells, and dplyr::bind_rows() refuses;
# * method = "emd" with a NULL template among the permutation candidates:
#   compute_similarity(NULL, ...) stops instead of giving NA.
KNOWN_R_CRASHES = ("Can't combine", "method 'emd' requires eye_density objects")


def known_r_crash(expected, py) -> bool:
    return (is_err(expected) and not is_err(py)
            and any(k in expected["error"] for k in KNOWN_R_CRASHES))


def report(failures):
    assert not failures, f"{len(failures)} mismatches:\n" + "\n".join(failures[:40])


# ---------------------------------------------------------------------------
# A. compute_similarity on random vectors
# ---------------------------------------------------------------------------

VEC_KINDS = ["normal", "nonneg", "ties", "const", "zero", "nan", "tiny", "identical",
             "short", "scaled", "negcorr"]


def _vec_case(kind):
    n = int(RNG.integers(3, 40))
    x = RNG.random(n)
    y = RNG.random(n)
    if kind == "normal":
        x, y = RNG.normal(size=n), RNG.normal(size=n)
    elif kind == "ties":
        x, y = RNG.integers(0, 3, n).astype(float), RNG.integers(0, 3, n).astype(float)
    elif kind == "const":
        x = np.full(n, 2.0)
        if RNG.random() < 0.5:
            y = x.copy()
    elif kind == "zero":
        x = np.zeros(n)
        if RNG.random() < 0.5:
            y = np.zeros(n)
    elif kind == "nan":
        x[RNG.integers(n)] = np.nan
        y[RNG.integers(n)] = np.nan
    elif kind == "tiny":
        x, y = x * 1e-9, y * 1e-9
    elif kind == "identical":
        y = x.copy()
    elif kind == "short":
        x, y = x[:2], y[:2]
        x[0] = np.nan if RNG.random() < 0.5 else x[0]
    elif kind == "scaled":
        y = 3.7 * x
    elif kind == "negcorr":
        y = -2.0 * x + 1.0
    return {"x": x, "y": y}


VEC_CASES = [_vec_case(VEC_KINDS[i % len(VEC_KINDS)]) for i in range(44)]


def _r_vec_block():
    cases = ", ".join(f"list(x = {rnum(c['x'])}, y = {rnum(c['y'])})" for c in VEC_CASES)
    return f"""
vec_cases <- list({cases})
vec <- lapply(vec_cases, function(p) c(
  lapply(setNames(methods, methods), function(m) run(compute_similarity(p$x, p$y, method = m))),
  list(default = run(similarity(p$x, p$y)))))
"""


@requires_r
def test_random_vector_similarity_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(VEC_CASES, r["vec"])):
        for m in METHODS:
            py = run_py(lambda: compute_similarity(case["x"], case["y"], method=m))
            msg = same(py, expected[m], rtol=1e-9 if m != "dcov" else 1e-8)
            if msg:
                failures.append(f"vec case {i} {m}: {msg}")
        msg = same(run_py(lambda: similarity(case["x"], case["y"])), expected["default"])
        if msg:
            failures.append(f"vec case {i} default: {msg}")
    report(failures)


# ---------------------------------------------------------------------------
# B. similarity on random density maps (incl. EMD, saliency, lattices)
# ---------------------------------------------------------------------------

MAP_KINDS = ["smooth", "rand", "ties", "zeros", "const", "zero", "neg", "nan", "tiny"]


def _map_case(i):
    nx, ny = int(RNG.integers(2, 8)), int(RNG.integers(2, 7))
    gx, gy = rand_grid(nx), rand_grid(ny)
    k1 = MAP_KINDS[i % len(MAP_KINDS)]
    k2 = MAP_KINDS[int(RNG.integers(len(MAP_KINDS)))] if RNG.random() < 0.5 else "smooth"
    d1 = gen_density(gx, gy, rand_z(nx, ny, k1))
    gx2 = gx + 1.0 if RNG.random() < 0.08 else gx
    d2 = gen_density(gx2, gy, rand_z(nx, ny, k2))
    nonneg = all(np.all(np.nan_to_num(d.z) >= 0) and not np.isnan(d.z).any() for d in (d1, d2))
    sal = gen_density(gx, gy, rand_z(nx, ny, "rand") * RNG.uniform(0, 0.2)) if nonneg else None
    return {"d1": d1, "d2": d2, "emd": nonneg, "sal": sal}


MAP_CASES = [_map_case(i) for i in range(36)]


def _r_map_block():
    cases = ", ".join(
        f"list(d1 = {robj(c['d1'])}, d2 = {robj(c['d2'])}, emd = {robj(c['emd'])}, sal = {robj(c['sal'])})"
        for c in MAP_CASES)
    return f"""
map_cases <- list({cases})
maps <- lapply(map_cases, function(p) {{
  res <- lapply(setNames(methods, methods), function(m) run(similarity(p$d1, p$d2, method = m)))
  res$default <- run(similarity(p$d1, p$d2))
  if (p$emd) {{
    res$emd <- run(similarity(p$d1, p$d2, method = "emd"))
    res$saliency <- run(similarity(p$d1, p$d2, method = "emd", saliency_map = p$sal))
  }}
  res
}})
"""


@requires_r
def test_random_density_similarity_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(MAP_CASES, r["maps"])):
        checks = [(m, {"method": m}, 1e-9, 1e-12) for m in METHODS] + [("default", {}, 1e-9, 1e-12)]
        if case["emd"]:
            checks += [("emd", {"method": "emd"}, 0, 1e-6),
                       ("saliency", {"method": "emd", "saliency_map": case["sal"]}, 0, 1e-6)]
        for key, kw, rtol, atol in checks:
            py = run_py(lambda: similarity(case["d1"], case["d2"], **kw))
            msg = same(py, expected[key], rtol=rtol or 1e-9, atol=atol)
            if msg:
                failures.append(f"map case {i} {key}: {msg}")
    report(failures)


# ---------------------------------------------------------------------------
# C. multiscale similarity
# ---------------------------------------------------------------------------

def _ms(gx, gy, sigmas, kind="rand"):
    scales = []
    for s in sigmas:
        d = gen_density(gx, gy, rand_z(len(gx), len(gy), kind))
        d.sigma = float(s)
        scales.append(d)
    return EyeDensityMultiscale(scales=scales)


def _ms_case(i):
    nx, ny = int(RNG.integers(2, 6)), int(RNG.integers(2, 6))
    gx, gy = rand_grid(nx), rand_grid(ny)
    pool = [5.0, 10.0, 20.0, 40.0]
    sx = list(RNG.permutation(pool)[: int(RNG.integers(1, 4))])
    sy = list(RNG.permutation(pool)[: int(RNG.integers(1, 4))])
    if RNG.random() < 0.6:
        sy = list(RNG.permutation(sx))
    kind = ["rand", "ties", "smooth", "neg"][i % 4]
    return {"a": _ms(gx, gy, sx, kind), "b": _ms(gx, gy, sy, kind),
            "method": METHODS[i % len(METHODS)], "agg": ["mean", "none", "NONE", "mea", "max"][i % 5]}


MS_CASES = [_ms_case(i) for i in range(25)]


def _r_ms_block():
    cases = ", ".join(f"list(a = {robj(c['a'])}, b = {robj(c['b'])}, method = \"{c['method']}\", "
                      f"agg = \"{c['agg']}\")" for c in MS_CASES)
    return f"""
ms_cases <- list({cases})
ms <- lapply(ms_cases, function(p) run(unname(similarity(p$a, p$b, method = p$method,
                                                       multiscale_aggregation = p$agg))))
"""


@requires_r
def test_random_multiscale_similarity_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(MS_CASES, r["ms"])):
        py = run_py(lambda: similarity(case["a"], case["b"], method=case["method"],
                                       multiscale_aggregation=case["agg"]))
        msg = same(py, expected)
        if msg:
            failures.append(f"multiscale case {i} ({case['method']}, {case['agg']}): {msg}")
    report(failures)


# ---------------------------------------------------------------------------
# D. template_similarity on random tables
# ---------------------------------------------------------------------------

def _ts_case(i):
    nx, ny = int(RNG.integers(2, 6)), int(RNG.integers(2, 6))
    gx, gy = rand_grid(nx), rand_grid(ny)
    multiscale = i % 6 == 5
    method = (["pearson", "spearman", "fisherz", "cosine", "l1", "jaccard", "dcov", "emd"][i % 8]
              if not multiscale else METHODS[i % 7])
    n_ref = int(RNG.integers(3, 7))
    ref_keys = [f"k{j}" for j in range(n_ref)]
    if RNG.random() < 0.3:
        ref_keys[-1] = ref_keys[0]          # duplicated reference key: first wins
    if RNG.random() < 0.2:
        ref_keys[1] = None                  # missing key matches a missing source key
    pool = ref_keys + ["zz"]
    n_src = int(RNG.integers(3, 9))
    src_keys = [pool[int(RNG.integers(len(pool)))] for _ in range(n_src)]
    strata = ["a", "b", "c", None]
    ref_subj = [strata[int(RNG.integers(3))] for _ in range(n_ref)]
    src_subj = [strata[int(RNG.integers(4))] for _ in range(n_src)]
    kinds = ["smooth", "rand", "ties", "zeros"] if method in ("emd", "cosine") else \
        ["smooth", "rand", "ties", "zeros", "neg", "nan", "const"]
    sigmas = [10.0, 20.0]

    def make():
        if multiscale:
            return _ms(gx, gy, sigmas, kinds[int(RNG.integers(len(kinds)))])
        return gen_density(gx, gy, rand_z(nx, ny, kinds[int(RNG.integers(len(kinds)))]))

    ref_d = [make() for _ in range(n_ref)]
    src_d = [make() for _ in range(n_src)]
    if RNG.random() < 0.4:
        src_d[int(RNG.integers(n_src))] = None
    if RNG.random() < 0.3:
        ref_d[int(RNG.integers(n_ref))] = None
    return {
        "ref": {"key": ("chr", ref_keys), "subj": ("chr", ref_subj), "density": ("list", ref_d)},
        "src": {"key": ("chr", src_keys), "subj": ("chr", src_subj), "density": ("list", src_d)},
        "method": method,
        "perms": [0, 1000][i % 2],
        "permute_on": "subj" if i % 3 == 0 else None,
        "agg": "none" if (multiscale and i % 4 == 1) else "mean",
    }


TS_CASES = [_ts_case(i) for i in range(30)]

_R_TS_OUT = """function(res) list(key = res$key, eye_sim = res$eye_sim, perm_sim = res[["perm_sim"]],
                        eye_sim_diff = res[["eye_sim_diff"]], n_perm = res[["n_perm"]])"""


def _r_ts_block():
    cases = ", ".join(
        f"list(ref = {rtable(c['ref'])}, src = {rtable(c['src'])}, method = \"{c['method']}\", "
        f"perms = {c['perms']}, permute_on = {robj(c['permute_on'])}, agg = \"{c['agg']}\")"
        for c in TS_CASES)
    return f"""
ts_out <- {_R_TS_OUT}
ts_cases <- list({cases})
ts <- lapply(ts_cases, function(p) run(ts_out(template_similarity(
  p$ref, p$src, match_on = "key", permute_on = p$permute_on, method = p$method,
  permutations = p$perms, multiscale_aggregation = p$agg))))
"""


def _compare_ts(py_res, expected, label, failures, rtol=1e-9, atol=1e-12):
    if known_r_crash(expected, py_res):
        return
    if is_err(expected) or is_err(py_res):
        if not (is_err(expected) and is_err(py_res)):
            failures.append(f"{label}: R={expected!r} PY={py_res!r}")
        return
    py_keys = [None if (k is None or (isinstance(k, float) and np.isnan(k))) else k
               for k in py_res["key"]]
    if py_keys != as_list(expected["key"]):
        failures.append(f"{label} keys: R={expected['key']} PY={py_keys}")
        return
    for col in ("eye_sim", "perm_sim", "eye_sim_diff", "n_perm"):
        r_val = expected[col]
        if r_val is None:
            if col in py_res.columns:
                failures.append(f"{label}: PY has extra column {col}")
            continue
        py_col = py_res[col].tolist()
        py_flat = [np.asarray(v, dtype=float).ravel() for v in py_col]
        r_flat = [np.atleast_1d(as_float(v)).ravel() for v in r_val]
        for j, (a, b) in enumerate(zip(py_flat, r_flat)):
            msg = same(a, b, rtol=rtol, atol=atol)
            if msg:
                failures.append(f"{label} {col}[{j}]: {msg}")


@requires_r
def test_random_template_similarity_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(TS_CASES, r["ts"])):
        res = run_py(lambda: template_similarity(
            pytable(case["ref"]), pytable(case["src"]), match_on="key", permute_on=case["permute_on"],
            method=case["method"], permutations=case["perms"], multiscale_aggregation=case["agg"]))
        tol = 1e-6 if case["method"] == "emd" else 1e-9
        _compare_ts(res, expected, f"ts case {i} ({case['method']})", failures,
                    rtol=tol, atol=1e-6 if case["method"] == "emd" else 1e-12)
    report(failures)


# ---------------------------------------------------------------------------
# E/F. latent and geometric transforms through template_similarity(_cv)
# ---------------------------------------------------------------------------

def _transform_tables(n, nx, ny, kind="smooth", dup=False):
    gx, gy = np.linspace(0, 100, nx), np.linspace(0, 80, ny)
    keys = [f"k{j}" for j in range(n)]
    grp = [["g1", "g2"][j % 2] for j in range(n)]
    ref_d = [gen_density(gx, gy, rand_z(nx, ny, kind)) for _ in range(n)]
    # Source maps: the reference map plus structured distortion and noise.
    src_d = []
    for d in ref_d:
        z = np.roll(d.z, int(RNG.integers(0, 2)), axis=0) * RNG.uniform(0.5, 1.5) \
            + RNG.uniform(0, 0.3) * rand_z(nx, ny, kind)
        src_d.append(gen_density(gx, gy, z))
    src_keys = list(keys)
    src_grp = list(grp)
    if dup:
        src_keys.append(keys[0])
        src_grp.append(grp[0])
        src_d.append(gen_density(gx, gy, rand_z(nx, ny, kind)))
    ref = {"key": ("chr", keys), "grp": ("chr", grp), "density": ("list", ref_d)}
    src = {"key": ("chr", src_keys), "grp": ("chr", src_grp), "density": ("list", src_d)}
    return ref, src


def _tf_case(i):
    name = list(TRANSFORMS)[i % 5]
    n = int(RNG.integers(6, 11))
    ref, src = _transform_tables(n, int(RNG.integers(4, 7)), int(RNG.integers(3, 6)),
                                 kind="smooth", dup=RNG.random() < 0.3)
    args = {}
    if name in ("latent_pca_transform", "coral_transform", "cca_transform"):
        args["comps"] = int(RNG.integers(1, 4))
        args["center"] = bool(RNG.random() < 0.7)
        if RNG.random() < 0.3:
            args["scale."] = True
    if name != "latent_pca_transform":
        args["shrink"] = float(10.0 ** RNG.uniform(-6, -2))
        if RNG.random() < 0.4:
            args["fit_by"] = "grp"
    if name in ("cca_transform", "contract_transform", "affine_transform") and RNG.random() < 0.3:
        args["unique_match_only"] = True
    if "fit_by" in args and name != "cca_transform" and RNG.random() < 0.3:
        src["grp"][1][0] = None              # missing stratum: never fitted
    method = ["cosine", "jaccard"][i % 2] if name in ("latent_pca_transform", "coral_transform",
                                                       "cca_transform") else METHODS[i % 7]
    if i % 7 == 3:
        z = src["density"][1][0].z.copy()
        z[0, 0] = 0.0
        z[:, 0] = 0.0                        # zero cells: prcomp(scale. = TRUE) refuses
        src["density"][1][0] = gen_density(src["density"][1][0].x, src["density"][1][0].y, z)
        ref["density"] = ("list", [gen_density(d.x, d.y, np.where(np.arange(d.z.shape[1]) == 0, 0.0, d.z))
                                   for d in ref["density"][1]])
    return {"name": name, "args": args, "ref": ref, "src": src, "method": method,
            "perms": [0, 1000][i % 2], "as_string": i % 4 == 0}


TF_CASES = [_tf_case(i) for i in range(30)]


def _r_tf_block():
    cases = ", ".join(
        f"list(ref = {rtable(c['ref'])}, src = {rtable(c['src'])}, tf = "
        + (f"\"{c['name']}\"" if c["as_string"] else c["name"])
        + f", args = {rargs(c['args'])}, method = \"{c['method']}\", perms = {c['perms']})"
        for c in TF_CASES)
    return f"""
tf_cases <- list({cases})
tf <- lapply(tf_cases, function(p) run(ts_out(template_similarity(
  p$ref, p$src, match_on = "key", method = p$method, permutations = p$perms,
  similarity_transform = p$tf, similarity_transform_args = p$args))))
"""


@requires_r
def test_random_transformed_template_similarity_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(TF_CASES, r["tf"])):
        tf = case["name"] if case["as_string"] else TRANSFORMS[case["name"]]
        res = run_py(lambda: template_similarity(
            pytable(case["ref"]), pytable(case["src"]), match_on="key", method=case["method"],
            permutations=case["perms"], similarity_transform=tf,
            similarity_transform_args=dict(case["args"])))
        # Similarities of SVD/eigen outputs: 1e-7 covers conditioning of the
        # small eigenproblems (shrink down to 1e-6).
        _compare_ts(res, expected, f"tf case {i} ({case['name']} {case['args']} {case['method']})",
                    failures, rtol=1e-7, atol=1e-9)
    report(failures)


def _cv_case(i):
    tf_names = [None, *TRANSFORMS]
    name = tf_names[i % 6]
    n = int(RNG.integers(8, 13))
    ref, src = _transform_tables(n, 6, 4, kind="smooth", dup=i % 3 == 0)
    n_src = len(src["key"][1])
    src["subj"] = ("chr", [["a", "b"][j % 2] for j in range(n_src)])
    ref["subj"] = ("chr", [["a", "b"][j % 2] for j in range(len(ref["key"][1]))])
    args = {}
    if name in ("latent_pca_transform", "coral_transform", "cca_transform"):
        args["comps"] = 2
    if name not in (None, "latent_pca_transform"):
        args["shrink"] = 1e-3
    split_on = "key" if i % 2 == 0 else "grp2"
    src["grp2"] = ("chr", [f"s{j % 4}" if j != 2 or i % 5 else None for j in range(n_src)])
    fit_filter = [bool(RNG.random() < 0.85) for _ in range(n_src)] if i % 4 == 1 else None
    eval_filter = [bool(RNG.random() < 0.8) or j == 0 for j in range(n_src)] if i % 4 == 2 else None
    method = "cosine" if name in ("latent_pca_transform", "coral_transform", "cca_transform") \
        else METHODS[i % 7]
    return {"name": name, "args": args, "ref": ref, "src": src, "method": method,
            "split_on": split_on, "n_folds": [None, 2, 3, 4][i % 4], "seed": int(RNG.integers(1, 10_000)),
            "perms": [1000, 0][i % 2], "permute_on": "subj" if i % 3 == 1 else None,
            "fit": fit_filter, "eval": eval_filter}


CV_CASES = [_cv_case(i) for i in range(24)]


def _r_cv_block():
    def one(c):
        tf = "NULL" if c["name"] is None else c["name"]
        fit = "NULL" if c["fit"] is None else rlgl(c["fit"])
        ev = "NULL" if c["eval"] is None else rlgl(c["eval"])
        return (f"list(ref = {rtable(c['ref'])}, src = {rtable(c['src'])}, tf = {tf}, "
                f"args = {rargs(c['args'])}, method = \"{c['method']}\", split_on = \"{c['split_on']}\", "
                f"n_folds = {robj(c['n_folds']) if c['n_folds'] is not None else 'NULL'}, "
                f"seed = {c['seed']}, perms = {c['perms']}, permute_on = {robj(c['permute_on'])}, "
                f"fit = {fit}, eval = {ev})")
    return f"""
cv_cases <- list({", ".join(one(c) for c in CV_CASES)})
cv <- lapply(cv_cases, function(p) run({{
  res <- template_similarity_cv(p$ref, p$src, match_on = "key", permute_on = p$permute_on,
    method = p$method, permutations = p$perms, similarity_transform = p$tf,
    similarity_transform_args = p$args, split_on = p$split_on, n_folds = p$n_folds,
    seed = p$seed, fit_source_filter = p$fit, eval_source_filter = p$eval)
  c(ts_out(res), list(fold = res$.cv_fold))
}}))
"""


@requires_r
def test_random_template_similarity_cv_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(CV_CASES, r["cv"])):
        tf = None if case["name"] is None else TRANSFORMS[case["name"]]
        res = run_py(lambda: template_similarity_cv(
            pytable(case["ref"]), pytable(case["src"]), match_on="key", permute_on=case["permute_on"],
            method=case["method"], permutations=case["perms"], similarity_transform=tf,
            similarity_transform_args=dict(case["args"]), split_on=case["split_on"],
            n_folds=case["n_folds"], seed=case["seed"],
            fit_source_filter=None if case["fit"] is None else np.array(case["fit"]),
            eval_source_filter=None if case["eval"] is None else np.array(case["eval"])))
        label = f"cv case {i} ({case['name']}, {case['method']}, split_on={case['split_on']})"
        _compare_ts(res, expected, label, failures, rtol=1e-7, atol=1e-9)
        if not is_err(res) and not is_err(expected):
            msg = same(res[".cv_fold"].to_numpy(dtype=float), expected["fold"])
            if msg:
                failures.append(f"{label} .cv_fold: {msg}")
    report(failures)


# ---------------------------------------------------------------------------
# G. sample_density and sample_density_time
# ---------------------------------------------------------------------------

def _sd_case(i):
    nx, ny = int(RNG.integers(2, 8)), int(RNG.integers(2, 8))
    gx, gy = rand_grid(nx), rand_grid(ny)
    d = gen_density(gx, gy, rand_z(nx, ny, ["smooth", "rand", "ties", "zero", "neg", "nan"][i % 6]))
    fg = rand_fg(int(RNG.integers(1, 8)), lo=-10, hi=110)
    times = None if i % 3 == 0 else np.sort(RNG.integers(-100, 2500, size=int(RNG.integers(1, 12)))).astype(float)
    return {"d": d, "fg": fg, "times": times, "normalize": ["none", "max", "sum", "zscore"][i % 4]}


SD_CASES = [_sd_case(i) for i in range(30)]


def _sdt_case(i):
    nx, ny = int(RNG.integers(2, 7)), int(RNG.integers(2, 7))
    gx, gy = rand_grid(nx), rand_grid(ny)
    n_t = int(RNG.integers(2, 6))
    keys = [f"k{j}" for j in range(n_t)]
    subj = [["a", "b", "c"][int(RNG.integers(3))] for _ in range(n_t)]
    dens = [gen_density(gx, gy, rand_z(nx, ny, ["smooth", "rand", "ties", "zeros"][int(RNG.integers(4))]))
            for _ in range(n_t)]
    if i % 5 == 2:
        dens[1] = None
    if i % 7 == 3:                             # a template on another lattice
        dens[0] = gen_density(gx * 1.5, gy, rand_z(nx, ny, "rand"))
    n_s = int(RNG.integers(2, 8))
    pool = keys + ["zz"]
    s_keys = [pool[int(RNG.integers(len(pool)))] for _ in range(n_s)]
    s_subj = [["a", "b", "c"][int(RNG.integers(3))] for _ in range(n_s)]
    fgs = [rand_fg(int(RNG.integers(1, 7)), lo=-10, hi=110) for _ in range(n_s)]
    if i % 6 == 4:
        fgs[0] = None
    times = np.sort(np.unique(RNG.integers(-50, 2000, size=int(RNG.integers(3, 15))))).astype(float)
    bins = None if i % 4 == 0 else np.sort(np.unique(RNG.integers(-50, 2100, size=int(RNG.integers(2, 5))))).astype(float)
    if bins is not None and len(bins) < 2:
        bins = np.array([0.0, 1000.0])
    return {
        "tmpl": {"key": ("chr", keys), "subj": ("chr", subj), "density": ("list", dens)},
        "src": {"key": ("chr", s_keys), "subj": ("chr", s_subj), "fixgroup": ("list", fgs)},
        "times": times, "bins": bins, "perms": [0, 1000][i % 2],
        "permute_on": "subj" if i % 3 == 1 else None,
        "agg": ["mean", "median", "max"][i % 3], "normalize": ["none", "max", "sum", "zscore"][i % 4],
    }


SDT_CASES = [_sdt_case(i) for i in range(24)]
AGG = {"mean": np.mean, "median": np.median, "max": np.max}


def _r_sd_block():
    sd_cases = ", ".join(
        f"list(d = {robj(c['d'])}, fg = {robj(c['fg'])}, times = "
        f"{'NULL' if c['times'] is None else rnum(c['times'])}, normalize = \"{c['normalize']}\")"
        for c in SD_CASES)
    sdt_cases = ", ".join(
        f"list(tmpl = {rtable(c['tmpl'])}, src = {rtable(c['src'])}, times = {rnum(c['times'])}, "
        f"bins = {'NULL' if c['bins'] is None else rnum(c['bins'])}, perms = {c['perms']}, "
        f"permute_on = {robj(c['permute_on'])}, agg = {c['agg']}, normalize = \"{c['normalize']}\")"
        for c in SDT_CASES)
    return f"""
sd_cases <- list({sd_cases})
sd <- lapply(sd_cases, function(p) run(as.list(sample_density(p$d, p$fg, times = p$times,
                                                               normalize = p$normalize))))
sdt_cases <- list({sdt_cases})
sdt <- lapply(sdt_cases, function(p) run({{
  res <- sample_density_time(p$tmpl, p$src, match_on = "key", times = p$times, time_bins = p$bins,
    permutations = p$perms, permute_on = p$permute_on, aggregate_fun = p$agg, normalize = p$normalize)
  bincols <- grep("^(bin|perm_bin|diff_bin)_", names(res), value = TRUE)
  list(key = res$key, sampled = lapply(res$sampled, function(s) s$z),
       perm = if (is.null(res$perm_sampled)) NULL else lapply(res$perm_sampled, function(s) s$z),
       bins = lapply(setNames(bincols, bincols), function(b) res[[b]]))
}}))
"""


@requires_r
def test_random_sample_density_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(SD_CASES, r["sd"])):
        res = run_py(lambda: sample_density(case["d"], case["fg"], times=case["times"],
                                            normalize=case["normalize"]))
        if is_err(res) or is_err(expected):
            if not (is_err(res) and is_err(expected)):
                failures.append(f"sd case {i}: R={expected} PY={res}")
            continue
        for col in ("z", "time"):
            msg = same(res[col].to_numpy(dtype=float), expected[col])
            if msg:
                failures.append(f"sd case {i} {col}: {msg}")
    report(failures)


@requires_r
def test_random_sample_density_time_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(SDT_CASES, r["sdt"])):
        res = run_py(lambda: sample_density_time(
            pytable(case["tmpl"]), pytable(case["src"]), match_on="key", times=case["times"],
            time_bins=case["bins"], permutations=case["perms"], permute_on=case["permute_on"],
            aggregate_fun=AGG[case["agg"]], normalize=case["normalize"]))
        label = f"sdt case {i}"
        if is_err(res) or is_err(expected):
            if not (is_err(res) and is_err(expected)):
                failures.append(f"{label}: R={expected} PY={res}")
            continue
        if list(res["key"]) != as_list(expected["key"]):
            failures.append(f"{label} keys: R={expected['key']} PY={list(res['key'])}")
            continue
        for j, s in enumerate(res["sampled"]):
            msg = same(s["z"].to_numpy(dtype=float), expected["sampled"][j])
            if msg:
                failures.append(f"{label} sampled[{j}]: {msg}")
        if expected["perm"] is not None:
            for j, s in enumerate(res["perm_sampled"]):
                msg = same(s["z"].to_numpy(dtype=float), expected["perm"][j])
                if msg:
                    failures.append(f"{label} perm_sampled[{j}]: {msg}")
        r_bins = expected["bins"] or {}
        py_bins = [c for c in res.columns if c.split("_")[0] in ("bin", "perm", "diff") and c != "perm_sampled"]
        if sorted(py_bins) != sorted(r_bins):
            failures.append(f"{label} bin columns: R={sorted(r_bins)} PY={sorted(py_bins)}")
        for col, vals in r_bins.items():
            if col in res.columns:
                msg = same(res[col].to_numpy(dtype=float), vals)
                if msg:
                    failures.append(f"{label} {col}: {msg}")
    report(failures)


# ---------------------------------------------------------------------------
# H/I. fixation_overlap and fixation_similarity
# ---------------------------------------------------------------------------

def _ov_case(i):
    a = rand_fg(int(RNG.integers(1, 9)), hi=300, t0=float(RNG.integers(0, 200)))
    b = rand_fg(int(RNG.integers(1, 9)), hi=300, t0=float(RNG.integers(0, 200)))
    ts = None if i % 2 == 0 else np.arange(float(RNG.integers(-100, 50)), 2000.0, float(RNG.integers(7, 60)))
    return {"a": a, "b": b, "dthresh": float(RNG.uniform(10, 150)),
            "dist": ["euclidean", "manhattan"][i % 2 if i % 4 < 2 else 0], "ts": ts}


OV_CASES = [_ov_case(i) for i in range(30)]


def _fs_case(i):
    n = int(RNG.integers(3, 6))
    keys = [f"t{j}" for j in range(n)]
    ref_fg = [rand_fg(int(RNG.integers(2, 7)), hi=400) for _ in range(n)]
    src_keys = [keys[int(RNG.integers(n))] for _ in range(int(RNG.integers(2, 7)))]
    src_fg = [rand_fg(int(RNG.integers(2, 7)), hi=400) for _ in src_keys]
    method = ["sinkhorn", "overlap"][i % 2]
    extra = {}
    if method == "overlap":
        extra["time_samples"] = np.arange(0.0, 1500.0, float(RNG.integers(20, 80)))
        extra["dthresh"] = float(RNG.uniform(30, 120))
    elif i % 4 == 2:
        extra["lambda"] = float(RNG.uniform(0.05, 0.3))
        extra["tweight"] = float(RNG.uniform(0.2, 1.0))
    window = None if i % 3 else (0.0, float(RNG.integers(400, 1500)))
    return {"ref": {"trial": ("chr", keys), "fixgroup": ("list", ref_fg)},
            "src": {"trial": ("chr", src_keys), "fixgroup": ("list", src_fg)},
            "method": method, "extra": extra, "window": window, "perms": [0, 1000][(i // 2) % 2]}


FS_CASES = [_fs_case(i) for i in range(16)]


def _r_fix_block():
    ov = ", ".join(
        f"list(a = {robj(c['a'])}, b = {robj(c['b'])}, dthresh = {_num(c['dthresh'])}, "
        f"dist = \"{c['dist']}\", ts = {'NULL' if c['ts'] is None else rnum(c['ts'])})" for c in OV_CASES)
    fs = ", ".join(
        f"list(ref = {rtable(c['ref'])}, src = {rtable(c['src'])}, method = \"{c['method']}\", "
        f"extra = {rargs(c['extra'])}, window = {'NULL' if c['window'] is None else rnum(c['window'])}, "
        f"perms = {c['perms']})" for c in FS_CASES)
    return f"""
ov_cases <- list({ov})
ov <- lapply(ov_cases, function(p) run(if (is.null(p$ts))
  fixation_overlap(p$a, p$b, dthresh = p$dthresh, dist_method = p$dist) else
  fixation_overlap(p$a, p$b, dthresh = p$dthresh, time_samples = p$ts, dist_method = p$dist)))
fs_cases <- list({fs})
fs <- lapply(fs_cases, function(p) run({{
  res <- do.call(fixation_similarity, c(list(p$ref, p$src, match_on = "trial", permutations = p$perms,
                                             method = p$method, window = p$window), p$extra))
  list(key = res$trial, eye_sim = res$eye_sim, perm_sim = res[["perm_sim"]],
       eye_sim_diff = res[["eye_sim_diff"]], n_perm = res[["n_perm"]])
}}))
"""


@requires_r
def test_random_fixation_overlap_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(OV_CASES, r["ov"])):
        res = run_py(lambda: fixation_overlap(case["a"], case["b"], dthresh=case["dthresh"],
                                              time_samples=case["ts"], dist_method=case["dist"]))
        for key in ("overlap", "perc"):
            msg = same(res if is_err(res) else res[key], expected if is_err(expected) else expected[key])
            if msg:
                failures.append(f"overlap case {i} {key}: {msg}")
    report(failures)


@requires_r
def test_random_fixation_similarity_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(FS_CASES, r["fs"])):
        ref, src = pytable(case["ref"]), pytable(case["src"])
        ref = ref.rename(columns={"trial": "key"})
        src = src.rename(columns={"trial": "key"})
        res = run_py(lambda: fixation_similarity(ref, src, match_on="key", permutations=case["perms"],
                                                 method=case["method"], window=case["window"],
                                                 **case["extra"]))
        _compare_ts(res, expected, f"fixation_similarity case {i} ({case['method']})", failures,
                    rtol=1e-8, atol=1e-10)
    report(failures)


# ---------------------------------------------------------------------------
# J. multi_match and scanpath_similarity
# ---------------------------------------------------------------------------

MM_KEYS = ["mm_vector", "mm_direction", "mm_length", "mm_position", "mm_duration", "mm_position_emd"]


def _mm_case(i):
    screen = (float(RNG.integers(300, 1920)), float(RNG.integers(200, 1080)))
    n1 = 2 if i % 13 == 7 else int(RNG.integers(3, 16))
    n2 = int(RNG.integers(3, 16))
    a = rand_fg(n1, hi=min(screen), scan=True)
    b = a if i % 11 == 5 else rand_fg(n2, hi=min(screen), scan=True)
    return {"a": a, "b": b, "screen": screen}


MM_CASES = [_mm_case(i) for i in range(48)]


def _sps_case(i):
    n = int(RNG.integers(3, 6))
    keys = [f"t{j}" for j in range(n)]
    ref = [rand_fg(int(RNG.integers(3, 12)), hi=500, scan=True) for _ in range(n)]
    s_keys = [keys[int(RNG.integers(n))] for _ in range(int(RNG.integers(2, 6)))]
    src = [rand_fg(int(RNG.integers(3, 12)), hi=500, scan=True) for _ in s_keys]
    return {"ref": {"trial": ("chr", keys), "scanpath": ("list", ref)},
            "src": {"trial": ("chr", s_keys), "scanpath": ("list", src)},
            "window": None if i % 2 else (0.0, float(RNG.integers(800, 3000))),
            "perms": [1000, 0][i % 3 == 2], "screen": (600.0, 500.0)}


SPS_CASES = [_sps_case(i) for i in range(8)]


def _r_mm_block():
    mm = ", ".join(f"list(a = {robj(c['a'])}, b = {robj(c['b'])}, screen = {rnum(c['screen'])})"
                   for c in MM_CASES)
    sps = ", ".join(
        f"list(ref = {rtable(c['ref'])}, src = {rtable(c['src'])}, "
        f"window = {'NULL' if c['window'] is None else rnum(c['window'])}, perms = {c['perms']}, "
        f"screen = {rnum(c['screen'])})" for c in SPS_CASES)
    return f"""
mm_cases <- list({mm})
mm <- lapply(mm_cases, function(p) run(as.list(multi_match(p$a, p$b, p$screen))))
sps_cases <- list({sps})
sps <- lapply(sps_cases, function(p) run({{
  res <- scanpath_similarity(p$ref, p$src, match_on = "trial", permutations = p$perms,
                             window = p$window, screensize = p$screen)
  as.list(res[, grep("^mm_|^n_perm$|^trial$", names(res))])
}}))
"""


@requires_r
def test_random_multi_match_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(MM_CASES, r["mm"])):
        res = run_py(lambda: multi_match(case["a"], case["b"], screensize=case["screen"]))
        for key in MM_KEYS:
            tol = 1e-6 if key == "mm_position_emd" else 1e-10
            msg = same(res if is_err(res) else res[key], expected if is_err(expected) else expected[key],
                       rtol=1e-9, atol=tol)
            if msg:
                failures.append(f"multi_match case {i} (n={len(case['a'])},{len(case['b'])}) {key}: {msg}")
    report(failures)


@requires_r
def test_random_scanpath_similarity_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(SPS_CASES, r["sps"])):
        res = run_py(lambda: scanpath_similarity(
            pytable(case["ref"]), pytable(case["src"]), match_on="trial", permutations=case["perms"],
            window=case["window"], screensize=case["screen"]))
        if is_err(res) or is_err(expected):
            if not (is_err(res) and is_err(expected)):
                failures.append(f"scanpath case {i}: R={expected} PY={res}")
            continue
        r_cols = sorted(k for k in expected if k != "trial")
        py_cols = sorted(c for c in res.columns if c.startswith("mm_") or c == "n_perm")
        if r_cols != py_cols:
            failures.append(f"scanpath case {i} columns: R={r_cols} PY={py_cols}")
        for col in r_cols:
            if col in res.columns:
                tol = 1e-6 if "emd" in col else 1e-10
                msg = same(res[col].to_numpy(dtype=float), expected[col], atol=tol)
                if msg:
                    failures.append(f"scanpath case {i} {col}: {msg}")
    report(failures)


# ---------------------------------------------------------------------------
# K. repetitive_similarity
# ---------------------------------------------------------------------------

def _rep_case(i):
    nx, ny = int(RNG.integers(2, 6)), int(RNG.integers(2, 6))
    gx, gy = rand_grid(nx), rand_grid(ny)
    n = int(RNG.integers(3, 9))
    multiscale = i % 4 == 3
    conds = [["a", "b", "c", None][int(RNG.integers(4 if i % 3 == 0 else 3))] for _ in range(n)]
    method = (["spearman", "pearson", "fisherz", "cosine", "l1", "jaccard", "dcov", "emd"][i % 8]
              if not multiscale else METHODS[i % 7])
    kinds = ["smooth", "rand", "ties"] if method == "emd" else ["smooth", "rand", "ties", "neg", "nan", "const"]
    dens = []
    for _ in range(n):
        kind = kinds[int(RNG.integers(len(kinds)))]
        dens.append(_ms(gx, gy, [10.0, 20.0], kind) if multiscale
                    else gen_density(gx, gy, rand_z(nx, ny, kind)))
    if i % 5 == 1:
        dens[0] = None
    return {"tab": {"cond": ("chr", conds), "density": ("list", dens)}, "method": method,
            "agg": "none" if multiscale and i % 8 == 3 else "mean"}


REP_CASES = [_rep_case(i) for i in range(24)]


def _r_rep_block():
    cases = ", ".join(f"list(tab = {rtable(c['tab'])}, method = \"{c['method']}\", agg = \"{c['agg']}\")"
                      for c in REP_CASES)
    return f"""
rep_cases <- list({cases})
rep <- lapply(rep_cases, function(p) run({{
  res <- repetitive_similarity(p$tab, condition_var = "cond", method = p$method, pairwise = TRUE,
                               multiscale_aggregation = p$agg)
  list(repsim = res$repsim, othersim = res$othersim,
       pairwise = lapply(res$pairwise_repsim, function(v) unlist(lapply(v, as.numeric))))
}}))
"""


@requires_r
def test_random_repetitive_similarity_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(REP_CASES, r["rep"])):
        res = run_py(lambda: repetitive_similarity(pytable(case["tab"]), condition_var="cond",
                                                   method=case["method"], pairwise=True,
                                                   multiscale_aggregation=case["agg"]))
        if is_err(res) or is_err(expected):
            if not (is_err(res) and is_err(expected)):
                failures.append(f"rep case {i}: R={expected} PY={res}")
            continue
        tol = 1e-6 if case["method"] == "emd" else 1e-12
        for col in ("repsim", "othersim"):
            msg = same(res[col].to_numpy(dtype=float), expected[col], atol=tol)
            if msg:
                failures.append(f"rep case {i} ({case['method']}) {col}: {msg}")
        for j, (pv, rv) in enumerate(zip(res["pairwise_repsim"], expected["pairwise"])):
            # R na.omit()s each pair's value(s), so missing pairs drop out
            flat = np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)).ravel() for v in pv]) \
                if len(pv) else np.array([])
            msg = same(flat[~np.isnan(flat)], np.atleast_1d(as_float(rv if rv is not None else [])).ravel(),
                       atol=tol)
            if msg:
                failures.append(f"rep case {i} pairwise[{j}]: {msg}")
    report(failures)


# ---------------------------------------------------------------------------
# L/M. template_regression, template_multireg, template_sample
# ---------------------------------------------------------------------------

def _reg_case(i):
    nx, ny = int(RNG.integers(3, 7)), int(RNG.integers(3, 7))
    gx, gy = np.arange(nx, dtype=float), np.arange(ny, dtype=float)
    n = int(RNG.integers(2, 5))
    keys = [f"i{j}" for j in range(n)]
    method = ["lm", "rlm", "rank"][i % 3]
    base_maps = [gen_density(gx, gy, rand_z(nx, ny, "rand")) for _ in range(2)]
    if i % 5 == 1:
        base_maps[0] = gen_density(gx, gy, np.full((nx, ny), 0.5))   # aliased baseline
    ref = [gen_density(gx, gy, rand_z(nx, ny, "rand")) for _ in range(n)]
    src_keys = list(keys) + (["zz"] if i % 4 == 2 else [])
    src = []
    for j, _ in enumerate(src_keys):
        b = base_maps[j % 2].z
        r_z = ref[j % n].z
        z = 0.4 * b + 0.7 * r_z + 0.3 * RNG.normal(size=(nx, ny)) ** 3
        if i % 3 != 2 and i % 4 == 1 and j == 0:
            z[0, 0] = np.nan                                         # dropped by na.omit
        src.append(gen_density(gx, gy, z))
    base_keys = ["b0", "b1"]
    if i % 9 == 4:
        base_keys = ["b0", "b0"]                                     # duplicated key: error
    if i % 9 == 7:
        base_keys = ["b0", "bx"]                                     # missing key: error
    return {"ref": {"id": ("chr", keys), "density": ("list", ref)},
            "src": {"id": ("chr", src_keys), "base": ("chr", [["b0", "b1"][j % 2] for j in range(len(src_keys))]),
                    "density": ("list", src)},
            "base": {"base": ("chr", base_keys), "density": ("list", base_maps)}, "method": method}


REG_CASES = [_reg_case(i) for i in range(24)]


def _mr_case(i):
    nx, ny = int(RNG.integers(3, 7)), int(RNG.integers(3, 6))
    gx, gy = np.arange(nx, dtype=float), np.arange(ny, dtype=float)
    k = int(RNG.integers(1, 4))
    covs = [rand_z(nx, ny, "rand") for _ in range(k)]
    method = ["lm", "rlm", "nnls", "logistic"][i % 4]
    if i % 6 == 1 and k >= 1:
        covs[-1] = np.full((nx, ny), 1.0)                           # aliased with the intercept
    resp = sum(RNG.uniform(0, 1) * c for c in covs) + 0.2 * RNG.random((nx, ny)) ** 2
    cols = {"id": ("chr", ["r1"]), "resp": ("list", [gen_density(gx, gy, resp)])}
    names = []
    for j, c in enumerate(covs):
        cols[f"c{j}"] = ("list", [gen_density(gx, gy, c)])
        names.append(f"c{j}")
    return {"tab": cols, "covars": names, "method": method, "intercept": i % 5 != 2}


MR_CASES = [_mr_case(i) for i in range(24)]


def _tsamp_case(i):
    nx, ny = int(RNG.integers(2, 6)), int(RNG.integers(2, 6))
    gx, gy = rand_grid(nx), rand_grid(ny)
    n = int(RNG.integers(2, 5))
    tm = [gen_density(gx, gy, rand_z(nx, ny, "rand")) for _ in range(n)]
    fg = [rand_fg(int(RNG.integers(1, 6)), lo=-5, hi=105) for _ in range(n)]
    if i % 3 == 1:
        tm[0] = None
    if i % 4 == 2:
        fg[-1] = None
    time = None if i % 2 == 0 else np.sort(RNG.integers(0, 1500, size=4)).astype(float)
    return {"tab": {"id": ("chr", [f"r{j}" for j in range(n)]), "tmpl": ("list", tm), "fixgroup": ("list", fg)},
            "time": time}


TSAMP_CASES = [_tsamp_case(i) for i in range(10)]


def _r_reg_block():
    reg = ", ".join(f"list(ref = {rtable(c['ref'])}, src = {rtable(c['src'])}, base = {rtable(c['base'])}, "
                    f"method = \"{c['method']}\")" for c in REG_CASES)
    mr = ", ".join(f"list(tab = {rtable(c['tab'])}, covars = {rchr(c['covars'])}, method = \"{c['method']}\", "
                   f"intercept = {robj(c['intercept'])})" for c in MR_CASES)
    tsamp = ", ".join(f"list(tab = {rtable(c['tab'])}, time = {'NULL' if c['time'] is None else rnum(c['time'])})"
                      for c in TSAMP_CASES)
    return f"""
reg_cases <- list({reg})
reg <- lapply(reg_cases, function(p) run({{
  res <- template_regression(p$ref, p$src, "id", p$base, "base", method = p$method)
  list(id = res$id, b0 = res$beta_baseline, b1 = res$beta_source)
}}))
mr_cases <- list({mr})
mr <- lapply(mr_cases, function(p) run({{
  fit <- as.data.frame(template_multireg(p$tab, "resp", p$covars, method = p$method,
                                         intercept = p$intercept)$multireg[[1]])
  as.list(fit)
}}))
tsamp_cases <- list({tsamp})
tsamp <- lapply(tsamp_cases, function(p) run({{
  res <- template_sample(p$tab, "tmpl", time = p$time)
  list(id = res$id, z = lapply(res$sample_out, function(s) s$z), time = lapply(res$sample_out, function(s) s$time))
}}))
"""


@requires_r
def test_random_template_regression_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(REG_CASES, r["reg"])):
        res = run_py(lambda: template_regression(pytable(case["ref"]), pytable(case["src"]), "id",
                                                 pytable(case["base"]), "base", method=case["method"]))
        if is_err(res) or is_err(expected):
            if not (is_err(res) and is_err(expected)):
                failures.append(f"reg case {i} ({case['method']}): R={expected} PY={res}")
            continue
        if list(res["id"]) != as_list(expected["id"]):
            failures.append(f"reg case {i} ids: R={expected['id']} PY={list(res['id'])}")
            continue
        for col, key in (("beta_baseline", "b0"), ("beta_source", "b1")):
            msg = same(res[col].to_numpy(dtype=float), expected[key], rtol=1e-8, atol=1e-10)
            if msg:
                failures.append(f"reg case {i} ({case['method']}) {col}: {msg}")
    report(failures)


@requires_r
def test_random_template_multireg_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(MR_CASES, r["mr"])):
        res = run_py(lambda: template_multireg(pytable(case["tab"]), "resp", case["covars"],
                                               method=case["method"], intercept=case["intercept"]))
        label = f"multireg case {i} ({case['method']}, intercept={case['intercept']})"
        if is_err(res) or is_err(expected):
            if not (is_err(res) and is_err(expected)):
                failures.append(f"{label}: R={expected} PY={res}")
            continue
        fit = res["multireg"].iloc[0]
        if sorted(fit.columns) != sorted(expected):
            failures.append(f"{label} columns: R={sorted(expected)} PY={sorted(fit.columns)}")
        if list(fit["term"]) != as_list(expected["term"]):
            failures.append(f"{label} terms: R={expected['term']} PY={list(fit['term'])}")
            continue
        rtol = 1e-5 if case["method"] == "logistic" else 1e-8
        for col in expected:
            if col == "term" or col not in fit.columns:
                continue
            msg = same(fit[col].to_numpy(dtype=float), expected[col], rtol=rtol, atol=rtol * 1e-2)
            if msg:
                failures.append(f"{label} {col}: {msg}")
    report(failures)


@requires_r
def test_random_template_sample_matches_r(r):
    failures = []
    for i, (case, expected) in enumerate(zip(TSAMP_CASES, r["tsamp"])):
        res = run_py(lambda: template_sample(pytable(case["tab"]), "tmpl", time=case["time"]))
        if list(res["id"]) != as_list(expected["id"]):
            failures.append(f"template_sample case {i} ids: R={expected['id']} PY={list(res['id'])}")
            continue
        for j, s in enumerate(res["sample_out"]):
            for col in ("z", "time"):
                msg = same(s[col].to_numpy(dtype=float), expected[col][j])
                if msg:
                    failures.append(f"template_sample case {i} row {j} {col}: {msg}")
    report(failures)


# ---------------------------------------------------------------------------
# The single R script
# ---------------------------------------------------------------------------

R_CODE = (
    """
run <- function(expr) tryCatch(suppressMessages(suppressWarnings(expr)),
                               error = function(e) list(error = conditionMessage(e)))
methods <- c("pearson", "spearman", "fisherz", "cosine", "l1", "jaccard", "dcov")
"""
    + _r_vec_block() + _r_map_block() + _r_ms_block() + _r_ts_block() + _r_tf_block()
    + _r_cv_block() + _r_sd_block() + _r_fix_block() + _r_mm_block() + _r_rep_block()
    + _r_reg_block()
    + """
out <- list(vec = vec, maps = maps, ms = ms, ts = ts, tf = tf, cv = cv, sd = sd, sdt = sdt,
            ov = ov, fs = fs, mm = mm, sps = sps, rep = rep, reg = reg, mr = mr, tsamp = tsamp)
"""
)


@pytest.fixture(scope="module")
def r():
    return run_r(R_CODE)
