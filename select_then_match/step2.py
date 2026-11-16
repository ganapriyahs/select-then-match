"""Step 2: regression to look at the association, with machine-learning screening of variables.

    python -m select_then_match.step2            # writes results/step2_regression.md

2.0  The association before any adjustment (crude).
2.1  Screen out variables with machine learning (screening.py): permutation importance of forests, and LASSO.
2.2  Adjusted regression: logistic regression (odds ratio) and linear regression with robust standard errors
     (risk difference), adjusted for all covariates or only for the kept ones.

The AUC (how well a model ranks people with diabetes above people without) is cross-validated, and the screening is
redone inside every training fold, so the people being scored never helped to choose the variables. Screening and the AUC
use a random sample of 20,000 people to keep the forests quick. The regressions use everybody; their intervals ignore that
the kept variables were chosen from the same data. A regression coefficient is an association. It is a causal effect only if
the covariates block every other path from cholesterol to diabetes. Survey weights are not used.
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import to_md
from .data import load_cdc
from .screening import keep_lasso, keep_variables
from .step1 import raw_gap

warnings.filterwarnings("ignore")


def _logit(A, y, steps=30):
    """Logistic regression by Newton's method (A has a column of ones first): coefficients and their covariance."""
    beta = np.zeros(A.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
        H = A.T @ (A * (p * (1 - p))[:, None]) + 1e-9 * np.eye(A.shape[1])
        step = np.linalg.solve(H, A.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(A @ beta, -30, 30)))
    return beta, np.linalg.inv(A.T @ (A * (p * (1 - p))[:, None]) + 1e-9 * np.eye(A.shape[1]))


def _ols_robust(A, y):
    """Linear regression with heteroskedasticity-robust (HC1) standard errors."""
    n, k = A.shape
    bread = np.linalg.inv(A.T @ A)
    beta = bread @ A.T @ y
    e = y - A @ beta
    return beta, bread @ (A.T @ (A * (e ** 2)[:, None])) @ bread * n / (n - k)


def adjusted_effect(X, t, y, cols):
    """Odds ratio and risk difference of the treatment adjusted for the covariates `cols` (95% intervals)."""
    A, y = np.column_stack([np.ones(len(t)), t, np.asarray(X[cols], dtype=float)]), np.asarray(y, dtype=float)
    b, V = _logit(A, y)
    rd, Vr = _ols_robust(A, y)
    z, se, ser = norm.ppf(0.975), np.sqrt(V[1, 1]), np.sqrt(Vr[1, 1])
    return {"odds ratio": np.exp(b[1]), "odds ratio 95%": f"{np.exp(b[1] - z * se):.2f} to {np.exp(b[1] + z * se):.2f}",
            "risk difference": rd[1], "risk difference 95%": f"{rd[1] - z * ser:.3f} to {rd[1] + z * ser:.3f}"}


def cv_auc(X, t, y, screen=None, seed=0, folds=5):
    """Cross-validated AUC of a logistic regression on all covariates plus the treatment (screen=None), or on the
    variables kept by `screen` ("perm" or "lasso"), with the screening redone in every training fold.
    Returns (AUC, average number of variables used)."""
    X, t, y = X.reset_index(drop=True), np.asarray(t), np.asarray(y)
    pred, used = np.zeros(len(y)), []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        Xtr = X.iloc[tr]
        cols = list(X.columns) if screen is None else (keep_variables if screen == "perm" else keep_lasso)(Xtr, t[tr], y[tr], seed)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        model.fit(np.column_stack([np.asarray(Xtr[cols], dtype=float), t[tr]]), y[tr])
        pred[te] = model.predict_proba(np.column_stack([np.asarray(X.iloc[te][cols], dtype=float), t[te]]))[:, 1]
        used.append(len(cols))
    return float(roc_auc_score(y, pred)), float(np.mean(used))


def build(path, out="results/step2_regression.md", n=20000, seed=0):
    d = load_cdc(path)
    s = d.subsample(n, seed)
    everything = list(d.X.columns)
    kept = {"permutation importance": keep_variables(s.X, s.t, s.y, seed), "LASSO": keep_lasso(s.X, s.t, s.y, seed)}
    crude = raw_gap(d.t, d.y)
    auc = {"all covariates": (cv_auc(s.X, s.t, s.y, None, seed)[0], len(everything)),
           "kept by permutation importance": cv_auc(s.X, s.t, s.y, "perm", seed), "kept by LASSO": cv_auc(s.X, s.t, s.y, "lasso", seed)}
    col_auc = f"AUC (n={len(s.y):,}, screening inside folds)"
    sets = {"all covariates": everything, "kept by permutation importance": kept["permutation importance"], "kept by LASSO": kept["LASSO"]}
    rows = [{"model": "no adjustment (crude)", "variables": 0, "odds ratio": crude["odds ratio"],
             "odds ratio 95%": f"{crude['odds ratio low']:.2f} to {crude['odds ratio high']:.2f}", "risk difference": crude["risk difference"],
             "risk difference 95%": f"{crude['risk difference low']:.3f} to {crude['risk difference high']:.3f}", col_auc: np.nan}]
    for name, cols in sets.items():
        rows.append({"model": f"adjusted for: {name}", "variables": len(cols), **adjusted_effect(d.X, d.t, d.y, cols), col_auc: auc[name][0]})
    res = pd.DataFrame(rows)
    sel = pd.DataFrame({"variable": everything, "permutation importance": ["kept" if c in kept["permutation importance"] else "removed" for c in everything],
                        "LASSO": ["kept" if c in kept["LASSO"] else "removed" for c in everything]})
    a_all, a_perm = res.iloc[1], res.iloc[2]
    text = ["# Step 2: regression and variable screening (CDC BRFSS 2015)\n",
            f"Outcome: diabetes. Exposure: high cholesterol. Regressions use all {len(d.y):,} people; screening and the AUC use a random sample of {len(s.y):,}.\n",
            "## 2.0 The association before adjustment\n", f"Risk difference {crude['risk difference']:.3f}, odds ratio {crude['odds ratio']:.2f}.\n",
            "## 2.1 Screening out variables with machine learning\n", to_md(sel, index=False),
            f"\nPermutation importance keeps {len(kept['permutation importance'])} of {len(everything)} variables, LASSO keeps {len(kept['LASSO'])}. "
            "Permutation importance scores each variable on people the forests were not trained on, once for the outcome and once for the treatment, scales each "
            "score so the largest is 1 and takes the geometric mean; a variable is kept when its score is at least 5% of the top, so it has to matter for both. "
            "\"Removed\" means it did not help prediction enough; it does not show the variable is unnecessary for causal adjustment.\n",
            "## 2.2 Adjusted regression\n", to_md(res, index=False, floatfmt="{:.3f}"),
            f"\nAdjusting for all {a_all['variables']} covariates moves the risk difference from {crude['risk difference']:.3f} to {a_all['risk difference']:.3f} "
            f"(odds ratio {crude['odds ratio']:.2f} to {a_all['odds ratio']:.2f}). Adjusting only for the {a_perm['variables']} kept variables gives {a_perm['risk difference']:.3f}. "
            f"The AUC of the screened model is {a_perm[col_auc]:.3f} against {a_all[col_auc]:.3f} with all variables (on average {auc['kept by permutation importance'][1]:.1f} "
            "variables kept per fold). The adjusted intervals ignore that the variables were chosen from the same data. "
            "These are associations, not effects: Step 3 asks what they become under matching and causal machine learning.\n"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(text))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    path = a.path or (sorted(glob.glob("data/*health_indicators_BRFSS2015.csv")) or [None])[0]
    if path is None:
        raise SystemExit("CDC file not found: put the BRFSS 2015 csv in data/ (see data/README.md)")
    print("wrote", build(path, n=a.n))


if __name__ == "__main__":
    main()
