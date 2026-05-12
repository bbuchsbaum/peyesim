"""Template regression and multiple regression."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LinearRegression

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
    """Match ppcor::pcor(..., method='spearman') for y, baseline, x2."""
    ranked = np.column_stack([
        stats.rankdata(y, method="average"),
        stats.rankdata(baseline, method="average"),
        stats.rankdata(x2, method="average"),
    ])
    corr = np.corrcoef(ranked, rowvar=False)
    inv_corr = np.linalg.pinv(corr)
    denom = np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
    partial = np.divide(
        -inv_corr,
        denom,
        out=np.full_like(inv_corr, np.nan, dtype=float),
        where=denom > np.finfo(float).eps,
    )
    np.fill_diagonal(partial, 1.0)
    return float(partial[1, 0]), float(partial[2, 0])


def _fit_logistic_tidy(y: np.ndarray, x: np.ndarray, terms: list[str], intercept: bool) -> pd.DataFrame:
    """Fit a binomial GLM with logit link and return broom-style coefficients."""
    if intercept:
        design = np.column_stack([np.ones(len(y)), x])
        out_terms = ["(Intercept)", *terms]
    else:
        design = x
        out_terms = terms

    def objective(beta):
        eta = design @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta):
        eta = design @ beta
        return design.T @ (expit(eta) - y)

    fit = minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-12, "maxiter": 10000},
    )
    coef = fit.x
    p = expit(design @ coef)
    hess = design.T @ ((p * (1.0 - p))[:, None] * design)
    cov = np.linalg.pinv(hess)
    std_error = np.sqrt(np.maximum(np.diag(cov), 0.0))
    statistic = np.divide(
        coef,
        std_error,
        out=np.full_like(coef, np.nan, dtype=float),
        where=std_error > np.finfo(float).eps,
    )
    p_value = 2 * stats.norm.sf(np.abs(statistic))
    return pd.DataFrame({
        "term": out_terms,
        "estimate": coef,
        "std.error": std_error,
        "statistic": statistic,
        "p.value": p_value,
    })


def _fit_rlm_tidy(y: np.ndarray, x: np.ndarray, terms: list[str], intercept: bool) -> pd.DataFrame:
    """Fit a Huber robust linear model and return broom-style coefficients."""
    import statsmodels.api as sm

    if intercept:
        design = np.column_stack([np.ones(len(y)), x])
        out_terms = ["(Intercept)", *terms]
    else:
        design = x
        out_terms = terms
    fit = sm.RLM(y, design, M=sm.robust.norms.HuberT(t=1.345)).fit(maxiter=100)
    return pd.DataFrame({
        "term": out_terms,
        "estimate": np.asarray(fit.params, dtype=float),
        "std.error": np.asarray(fit.bse, dtype=float),
        "statistic": np.asarray(fit.tvalues, dtype=float),
        "p.value": np.asarray(fit.pvalues, dtype=float),
    })


def template_multireg(
    source_tab: pd.DataFrame,
    response: str,
    covars: list[str],
    method: str = "lm",
    intercept: bool = True,
) -> pd.DataFrame:
    """Multiple regression of density maps (mirrors R ``template_multireg``).

    Supported methods: 'lm', 'rlm', 'logistic', 'nnls'.
    """
    results = []
    for i in range(len(source_tab)):
        row = source_tab.iloc[i]
        y_dens = row[response]
        y_vec = y_dens.z.ravel()
        y_sum = y_vec.sum()
        if y_sum > 0:
            y_vec = y_vec / y_sum

        X_cols = {}
        for cv in covars:
            cv_dens = row[cv]
            cv_vec = cv_dens.z.ravel()
            cv_sum = cv_vec.sum()
            if cv_sum > 0:
                cv_vec = cv_vec / cv_sum
            X_cols[cv] = cv_vec

        X = np.column_stack([X_cols[cv] for cv in covars])

        if method == "lm":
            terms = list(covars)
            if intercept:
                design = np.column_stack([np.ones(len(y_vec)), X])
                terms = ["(Intercept)", *terms]
            else:
                design = X
            coef, *_ = np.linalg.lstsq(design, y_vec, rcond=None)
            fitted = design @ coef
            resid = y_vec - fitted
            df_resid = max(len(y_vec) - design.shape[1], 0)
            if df_resid > 0:
                sigma2 = float((resid @ resid) / df_resid)
                cov = sigma2 * np.linalg.pinv(design.T @ design)
                std_error = np.sqrt(np.maximum(np.diag(cov), 0.0))
                statistic = np.divide(
                    coef,
                    std_error,
                    out=np.full_like(coef, np.nan, dtype=float),
                    where=std_error > np.finfo(float).eps,
                )
                p_value = 2 * stats.t.sf(np.abs(statistic), df=df_resid)
            else:
                std_error = np.full_like(coef, np.nan, dtype=float)
                statistic = np.full_like(coef, np.nan, dtype=float)
                p_value = np.full_like(coef, np.nan, dtype=float)
            results.append({
                "multireg": pd.DataFrame({
                    "term": terms,
                    "estimate": coef,
                    "std.error": std_error,
                    "statistic": statistic,
                    "p.value": p_value,
                })
            })
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
) -> pd.DataFrame:
    """Template regression with baseline control (mirrors R ``template_regression``)."""
    source_tab = source_tab.copy().reset_index(drop=True)
    matchind = match_keys(source_tab[match_on].values, ref_tab[match_on].values)
    source_tab, matchind = filter_unmatched(
        source_tab, matchind,
        warn_msg="did not find matching template map for all source maps. Removing non-matching elements.",
    )

    b0_list = []
    b1_list = []

    for i, mi in enumerate(matchind):
        row = source_tab.iloc[i]
        bkey_val = row[baseline_key]
        b_idx = np.where(baseline_tab[baseline_key].values == bkey_val)[0]
        if len(b_idx) == 0:
            b0_list.append(np.nan)
            b1_list.append(np.nan)
            continue

        bdens = baseline_tab["density"].iloc[b_idx[0]]
        d1 = ref_tab["density"].iloc[mi]
        d2 = row["density"]

        y_vec = d2.z.ravel()
        baseline_vec = bdens.z.ravel()
        x2_vec = d1.z.ravel()

        if method == "lm":
            X = np.column_stack([baseline_vec, x2_vec])
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y_vec)
            b0_list.append(model.coef_[0])
            b1_list.append(model.coef_[1])
        elif method == "rlm":
            X = np.column_stack([baseline_vec, x2_vec])
            fit = _fit_rlm_tidy(y_vec, X, ["baseline", "x2"], intercept=True)
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
    """Sample template densities at fixation coordinates for each row in ``source_tab``."""
    if template not in source_tab.columns:
        raise KeyError(f"Column '{template}' not found in source_tab")
    if fixgroup not in source_tab.columns:
        raise KeyError(f"Column '{fixgroup}' not found in source_tab")

    out = source_tab.copy()
    sampled = []
    for dens, fg in zip(out[template], out[fixgroup]):
        if dens is None or fg is None:
            sampled.append(None)
            continue
        sampled.append(sample_density(dens, fg, times=time))
    out[outcol] = sampled
    return out
