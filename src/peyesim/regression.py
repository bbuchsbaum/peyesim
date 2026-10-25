"""Template regression and multiple regression."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from peyesim._utils import match_keys, filter_unmatched
from peyesim.similarity import sample_density


def _partial_correlation(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    """Correlation between y and x after linear residualization on controls."""
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones(len(y)), controls])
    y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    if np.var(y_resid, ddof=1) <= np.finfo(float).eps:
        return np.nan
    if np.var(x_resid, ddof=1) <= np.finfo(float).eps:
        return np.nan
    return float(np.corrcoef(y_resid, x_resid)[0, 1])


def _partial_spearman_two_predictors(y: np.ndarray, baseline: np.ndarray, x2: np.ndarray) -> tuple[float, float]:
    """Match ppcor::pcor(..., method='spearman') for y, baseline, x2.

    As ppcor: covariance of average ranks, inverted with ``solve()`` unless its
    determinant is below machine epsilon (then ``MASS::ginv()``), and
    ``-cov2cor()`` of the inverse (``NaN`` where a variable has no variance).
    """
    ranked = np.column_stack([
        stats.rankdata(y, method="average"),
        stats.rankdata(baseline, method="average"),
        stats.rankdata(x2, method="average"),
    ])
    if not np.all(np.isfinite(ranked)) or np.isnan(np.column_stack([y, baseline, x2])).any():
        raise ValueError("missing value where TRUE/FALSE needed")
    cvx = np.cov(ranked, rowvar=False)
    if np.linalg.det(cvx) < np.finfo(float).eps:
        u, sv, vt = np.linalg.svd(cvx)
        keep = sv > max(np.sqrt(np.finfo(float).eps) * sv[0], 0)
        icvx = (vt[keep].T / sv[keep]) @ u[:, keep].T
    else:
        icvx = np.linalg.inv(cvx)
    d = np.sqrt(np.diag(icvx))
    with np.errstate(divide="ignore", invalid="ignore"):
        partial = -icvx / np.outer(d, d)
    return float(partial[1, 0]), float(partial[2, 0])


def _fit_logistic_tidy(y: np.ndarray, x: np.ndarray, terms: list[str], intercept: bool) -> pd.DataFrame:
    """``broom::tidy(glm(..., family = "binomial"))`` via R's ``glm.fit`` IRLS.

    Follows ``glm.fit`` step for step (``mustart``, the clamped logit link,
    the relative deviance test with ``epsilon = 1e-8``, ``maxit = 25``, and
    ``NA`` for coefficients aliased at the ``Cdqrls`` tolerance 1e-11).
    """
    design, out_terms = _design(x, terms, intercept)
    y = np.asarray(y, dtype=float)
    n, p = design.shape
    eps = np.finfo(float).eps
    thresh = 30.0

    def linkinv(eta):
        return np.where(eta < -thresh, eps, np.where(eta > thresh, 1 - eps, 1.0 / (1.0 + np.exp(-eta))))

    def mu_eta(eta):
        e = np.exp(np.clip(eta, -thresh, thresh))
        return np.where(np.abs(eta) > thresh, eps, e / (1.0 + e) ** 2)

    def deviance(mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
            t2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
        return float(np.sum(2 * (t1 + t2)))

    mu = (y + 0.5) / 2.0
    eta = np.log(mu / (1 - mu))
    devold = deviance(mu)
    coef = np.full(p, np.nan)
    kept = np.arange(p)
    wx = design
    for _ in range(25):
        me = mu_eta(eta)
        good = me != 0
        z = eta[good] + (y - mu)[good] / me[good]
        w = np.sqrt(me[good] ** 2 / (mu * (1 - mu))[good])
        wx = design[good] * w[:, None]
        kept = _r_qr_columns(wx, tol=1e-11)
        start = np.zeros(p)
        start[kept] = np.linalg.lstsq(wx[:, kept], z * w, rcond=None)[0]
        eta = design @ start
        mu = linkinv(eta)
        dev = deviance(mu)
        coef = start
        if abs(dev - devold) / (abs(dev) + 0.1) < 1e-8:
            break
        devold = dev
    estimate = np.full(p, np.nan)
    std_error = np.full(p, np.nan)
    estimate[kept] = coef[kept]
    if kept.size:
        _, rmat = np.linalg.qr(wx[:, kept])
        rinv = np.linalg.inv(np.triu(rmat))
        std_error[kept] = np.sqrt(np.sum(rinv ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = estimate / std_error
    p_value = 2 * stats.norm.sf(np.abs(statistic))
    return pd.DataFrame({
        "term": out_terms,
        "estimate": estimate,
        "std.error": std_error,
        "statistic": statistic,
        "p.value": p_value,
    })


def _r_qr_columns(design: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Columns R's ``qr()`` (LINPACK ``dqrdc2``, ``tol = 1e-7``) keeps, in order.

    A column is aliased when what remains of it after projecting out the
    columns kept before it is below ``tol`` times its original norm; ``lm()``
    then reports ``NA`` for its coefficient.
    """
    kept: list[int] = []
    for j in range(design.shape[1]):
        col = design[:, j]
        norm = np.linalg.norm(col)
        if norm == 0:
            continue
        if kept:
            basis = design[:, kept]
            resid = col - basis @ np.linalg.lstsq(basis, col, rcond=None)[0]
        else:
            resid = col
        if np.linalg.norm(resid) >= tol * norm:
            kept.append(j)
    return np.asarray(kept, dtype=int)


def _design(x: np.ndarray, terms: list[str], intercept: bool):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if intercept:
        return np.column_stack([np.ones(x.shape[0]), x]), ["(Intercept)", *terms]
    return x, list(terms)


def _fit_lm_tidy(y: np.ndarray, x: np.ndarray, terms: list[str], intercept: bool) -> pd.DataFrame:
    """``broom::tidy(lm(...))``: aliased terms get ``NA`` like R."""
    design, out_terms = _design(x, terms, intercept)
    p = design.shape[1]
    coef = np.full(p, np.nan)
    std_error = np.full(p, np.nan)
    statistic = np.full(p, np.nan)
    p_value = np.full(p, np.nan)
    kept = _r_qr_columns(design)
    if kept.size:
        xk = design[:, kept]
        bk = np.linalg.lstsq(xk, y, rcond=None)[0]
        coef[kept] = bk
        resid = y - xk @ bk
        df_resid = len(y) - kept.size
        if df_resid > 0:
            sigma2 = float(resid @ resid) / df_resid
            se = np.sqrt(np.maximum(np.diag(sigma2 * np.linalg.inv(xk.T @ xk)), 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                tval = bk / se
            std_error[kept] = se
            statistic[kept] = tval
            p_value[kept] = 2 * stats.t.sf(np.abs(tval), df=df_resid)
    return pd.DataFrame({
        "term": out_terms,
        "estimate": coef,
        "std.error": std_error,
        "statistic": statistic,
        "p.value": p_value,
    })


def _rlm_coefficients(y: np.ndarray, design: np.ndarray, maxit: int = 20,
                      k: float = 1.345, acc: float = 1e-4) -> dict:
    """``MASS::rlm.default`` (Huber psi, MAD scale, ``init = "ls"``) step for step."""
    if _r_qr_columns(design).size < design.shape[1]:
        raise ValueError("'x' is singular: singular fits are not implemented in 'rlm'")

    def wfit(w):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)[0]
        return coef, y - design @ coef

    coef, resid = wfit(np.ones(len(y)))
    scale = np.median(np.abs(resid - np.median(resid))) * 1.4826
    converged = False
    for _ in range(maxit):
        old = resid
        scale = np.median(np.abs(resid)) / 0.6745
        if scale == 0:
            converged = True
            break
        u = resid / scale
        with np.errstate(divide="ignore"):
            w = np.minimum(1.0, k / np.abs(u))
        coef, resid = wfit(w)
        delta = np.sqrt(np.sum((old - resid) ** 2) / max(1e-20, float(np.sum(old ** 2))))
        if delta <= acc:
            converged = True
            break
    if not converged:
        warnings.warn(f"'rlm' failed to converge in {maxit} steps")
    return {"coef": coef, "resid": resid, "scale": scale, "k": k}


def _fit_rlm_tidy(y: np.ndarray, x: np.ndarray, terms: list[str], intercept: bool,
                  maxit: int = 20) -> pd.DataFrame:
    """``broom::tidy(MASS::rlm(...))``: estimates and ``summary.rlm`` standard errors."""
    design, out_terms = _design(x, terms, intercept)
    fit = _rlm_coefficients(y, design, maxit=maxit)
    coef, resid, s, k = fit["coef"], fit["resid"], fit["scale"], fit["k"]
    n, p = design.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        u = resid / s
        w = np.minimum(1.0, k / np.abs(u))
    rdf = n - p
    S = float(np.sum((resid * w) ** 2)) / rdf if rdf > 0 else np.nan
    psiprime = (np.abs(u) <= k).astype(float)
    mn = float(np.mean(psiprime))
    kappa = 1 + p * float(np.var(psiprime, ddof=1)) / (n * mn ** 2) if n > 1 and mn > 0 else np.nan
    stddev = np.sqrt(S) * (kappa / mn) if mn > 0 else np.nan
    _, r = np.linalg.qr(design)
    rinv = np.linalg.solve(np.triu(r[:p, :p]), np.eye(p))
    rowlen = np.sqrt(np.sum(rinv ** 2, axis=1))
    std_error = rowlen * stddev
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = coef / std_error
    return pd.DataFrame({
        "term": out_terms,
        "estimate": np.asarray(coef, dtype=float),
        "std.error": std_error,
        "statistic": statistic,
    })


_MULTIREG_METHODS = ("lm", "rlm", "nnls", "logistic")


def _match_arg(method, choices, label):
    """R ``match.arg()``: ``None`` or the full choice vector selects the first."""
    if method is None or (isinstance(method, (list, tuple)) and list(method) == list(choices)):
        return choices[0]
    if method not in choices:
        raise ValueError(f"Unknown {label} method '{method}'. Supported methods: {', '.join(choices)}.")
    return method


def template_multireg(
    source_tab: pd.DataFrame,
    response: str,
    covars: list[str],
    method: str = "lm",
    intercept: bool = True,
) -> pd.DataFrame:
    """Multiple regression of density maps (mirrors R ``template_multireg``).

    Supported methods: 'lm' (default), 'rlm', 'nnls', 'logistic'.
    """
    method = _match_arg(method, _MULTIREG_METHODS, "regression")
    results = []
    for i in range(len(source_tab)):
        row = source_tab.iloc[i]
        y_dens = row[response]
        # As R: every map is divided by its sum (a zero or missing sum makes
        # the whole map missing), then model rows with a missing value are
        # dropped (na.omit).
        with np.errstate(divide="ignore", invalid="ignore"):
            y_vec = np.asarray(y_dens.z, dtype=float).ravel()
            y_vec = y_vec / y_vec.sum()
            X = np.column_stack([
                np.asarray(row[cv].z, dtype=float).ravel() / np.asarray(row[cv].z, dtype=float).sum()
                for cv in covars
            ])
        ok = np.isfinite(y_vec) & np.isfinite(X).all(axis=1)
        if method != "nnls":
            if not ok.any():
                raise ValueError("0 (non-NA) cases")
            y_vec = y_vec[ok]
            X = X[ok]

        if method == "lm":
            results.append({"multireg": _fit_lm_tidy(y_vec, X, list(covars), intercept=intercept)})
        elif method == "nnls":
            from scipy.optimize import nnls
            coef, _ = nnls(X, y_vec)
            results.append({
                "multireg": pd.DataFrame({
                    "term": list(covars),
                    "estimate": coef,
                })
            })
        elif method == "logistic":
            results.append({
                "multireg": _fit_logistic_tidy(y_vec, X, list(covars), intercept=intercept)
            })
        elif method == "rlm":
            results.append({
                "multireg": _fit_rlm_tidy(y_vec, X, list(covars), intercept=intercept)
            })
        else:
            raise ValueError(
                f"Unknown regression method '{method}'. Supported methods: lm, rlm, nnls, logistic."
            )

    out = source_tab.copy()
    out["multireg"] = [r["multireg"] for r in results]
    return out


def template_regression(
    ref_tab: pd.DataFrame,
    source_tab: pd.DataFrame,
    match_on: str,
    baseline_tab: pd.DataFrame,
    baseline_key: str,
    method: str = "lm",
    density_var: str = "density",
) -> pd.DataFrame:
    """Template regression with baseline control (mirrors R ``template_regression``).

    Each ``baseline_key`` value used by ``source_tab`` must appear in exactly one
    row of ``baseline_tab``; duplicated or missing keys are an error.
    ``density_var`` names the density column in all three tables.
    """
    method = _match_arg(method, ("lm", "rlm", "rank"), "template_regression")
    source_tab = source_tab.copy().reset_index(drop=True)
    matchind = match_keys(source_tab[match_on].values, ref_tab[match_on].values)
    source_tab, matchind = filter_unmatched(
        source_tab, matchind,
        warn_msg="did not find matching template map for all source maps. Removing non-matching elements.",
    )

    # Each baseline key used by a source row must identify exactly one map.
    baseline_keys = pd.Series(baseline_tab[baseline_key].values)
    used_keys = list(dict.fromkeys(source_tab[baseline_key].tolist()))
    duplicated = set(baseline_keys[baseline_keys.duplicated()].tolist())
    dup_keys = [k for k in used_keys if k in duplicated]
    if dup_keys:
        raise ValueError(
            f"template_regression(): baseline_tab has more than one row for {baseline_key} = "
            + ", ".join(str(k) for k in dup_keys)
            + ". Each baseline key must identify exactly one baseline map."
        )
    known = set(baseline_keys.tolist())
    missing_keys = [k for k in used_keys if k not in known]
    if missing_keys:
        raise ValueError(
            f"template_regression(): baseline_tab has no row for {baseline_key} = "
            + ", ".join(str(k) for k in missing_keys) + "."
        )

    b0_list = []
    b1_list = []

    for i, mi in enumerate(matchind):
        row = source_tab.iloc[i]
        bkey_val = row[baseline_key]
        b_idx = np.where(baseline_tab[baseline_key].values == bkey_val)[0]
        bdens = baseline_tab[density_var].iloc[b_idx[0]]
        d1 = ref_tab[density_var].iloc[mi]
        d2 = row[density_var]

        y_vec = d2.z.ravel()
        baseline_vec = bdens.z.ravel()
        x2_vec = d1.z.ravel()

        if method in ("lm", "rlm"):
            # lm()/rlm() drop rows with a missing value (na.omit)
            X = np.column_stack([baseline_vec, x2_vec])
            ok = ~(np.isnan(y_vec) | np.isnan(X).any(axis=1))
            if method == "lm":
                fit = _fit_lm_tidy(y_vec[ok], X[ok], ["baseline", "x2"], intercept=True)
            else:
                fit = _fit_rlm_tidy(y_vec[ok], X[ok], ["baseline", "x2"], intercept=True,
                                    maxit=100)
            estimates = fit.set_index("term")["estimate"]
            b0_list.append(estimates["baseline"])
            b1_list.append(estimates["x2"])
        elif method == "rank":
            b0, b1 = _partial_spearman_two_predictors(y_vec, baseline_vec, x2_vec)
            b0_list.append(b0)
            b1_list.append(b1)
        else:
            raise ValueError(f"Unknown method '{method}'. Supported methods: lm, rlm, rank.")

    source_tab["beta_baseline"] = b0_list
    source_tab["beta_source"] = b1_list
    return source_tab


def template_sample(
    source_tab: pd.DataFrame,
    template: str,
    fixgroup: str = "fixgroup",
    time=None,
    outcol: str = "sample_out",
) -> pd.DataFrame:
    """Sample template densities at fixation coordinates for each row in ``source_tab``.

    Rows whose template or fixation group is ``None`` cannot be sampled and are
    removed, with a warning.
    """
    if template not in source_tab.columns:
        raise KeyError(f"Column '{template}' not found in source_tab")
    if fixgroup not in source_tab.columns:
        raise KeyError(f"Column '{fixgroup}' not found in source_tab")

    # Rows whose template or fixation group is missing cannot be sampled.
    drop = np.array([dens is None or fg is None
                     for dens, fg in zip(source_tab[template], source_tab[fixgroup])], dtype=bool)
    if drop.any():
        warnings.warn(
            f"template_sample(): removing {int(drop.sum())} row(s) with a NULL template or fixation group."
        )
    out = source_tab.loc[~drop].copy()
    sampled = [sample_density(dens, fg, times=time) for dens, fg in zip(out[template], out[fixgroup])]
    out[outcol] = sampled
    return out
