"""Template regression and multiple regression."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from peyesim._utils import match_keys, filter_unmatched
from peyesim.similarity import sample_density


def template_multireg(
    source_tab: pd.DataFrame,
    response: str,
    covars: list[str],
    method: str = "lm",
    intercept: bool = True,
) -> pd.DataFrame:
    """Multiple regression of density maps (mirrors R ``template_multireg``).

    Supported methods: 'lm', 'nnls'.
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
            model = LinearRegression(fit_intercept=intercept)
            model.fit(X, y_vec)
            coefs = dict(zip(covars, model.coef_))
            if intercept:
                coefs["(Intercept)"] = model.intercept_
            results.append({"multireg": coefs})
        elif method == "nnls":
            from scipy.optimize import nnls
            coef, _ = nnls(X, y_vec)
            results.append({"multireg": dict(zip(covars, coef))})
        else:
            raise ValueError(f"Unknown regression method '{method}'. Supported: lm, nnls.")

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
            # Fallback to OLS for now
            X = np.column_stack([baseline_vec, x2_vec])
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y_vec)
            b0_list.append(model.coef_[0])
            b1_list.append(model.coef_[1])
        else:
            raise ValueError(f"Unknown method '{method}'")

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
