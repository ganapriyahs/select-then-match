"""Stage 0 of the 3-stage TIM, and Step 2.1: screen out variables with machine learning.

Permutation importance of two random forests:
  * one forest predicts the outcome, one predicts the treatment;
  * a forest learns on one half of the people, then each variable is shuffled on the OTHER half, which the forest has
    never seen, and the rise in prediction error is the variable's importance (both halves play both roles);
  * each importance is scaled so that the top variable has 1, and the two are combined with a geometric mean, so a
    variable scores high only if it matters for BOTH outcome and treatment (which is what a confounder does);
  * a variable is kept if its score is at least 5% of the top score.
"Screened out" only means the variable did not help prediction enough. It does not prove the variable is unnecessary
for causal adjustment.
"""
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .common import is_binary

CUT = 0.05


def _error_increase(predict, Z, y, n_cols, repeats, rng):
    """How much the squared error of `predict` grows when one of the first n_cols columns of Z is shuffled."""
    nb = len(Z)
    base = float(np.mean((predict(Z) - y) ** 2))
    out = np.zeros(n_cols)
    for j in range(n_cols):
        big = np.tile(Z, (repeats, 1))
        for r in range(repeats):
            big[r * nb:(r + 1) * nb, j] = Z[rng.permutation(nb), j]
        out[j] = float(np.mean((predict(big) - np.tile(y, repeats)) ** 2)) - base
    return out


def scores(X, t, y, seed=0, repeats=3):
    """Score of every column of X, between 0 and 1 (the top variable has 1)."""
    import pandas as pd
    cols = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    t = np.asarray(t).astype(int)
    y = np.asarray(y, dtype=float)
    p = len(cols)
    rng = np.random.default_rng(seed)
    Forest = RandomForestClassifier if is_binary(y) else RandomForestRegressor
    kw = dict(n_estimators=200, min_samples_leaf=10, max_features="sqrt", random_state=seed, n_jobs=1)
    po, pt = np.zeros(p), np.zeros(p)
    for a, b in StratifiedKFold(2, shuffle=True, random_state=seed).split(Xv, t):
        Xo_a, Xo_b = np.column_stack([Xv[a], t[a]]), np.column_stack([Xv[b], t[b]])      # outcome forest also sees the treatment
        f_out = Forest(**kw).fit(Xo_a, y[a])
        f_trt = RandomForestClassifier(**kw).fit(Xv[a], t[a])
        out_pred = (lambda Z: f_out.predict_proba(Z)[:, 1]) if Forest is RandomForestClassifier else f_out.predict
        po += _error_increase(out_pred, Xo_b, y[b], p, repeats, rng)
        pt += _error_increase(lambda Z: f_trt.predict_proba(Z)[:, 1], Xv[b], t[b].astype(float), p, repeats, rng)
    po_c, pt_c = np.clip(po, 0, None), np.clip(pt, 0, None)
    po_c = po_c / po_c.max() if po_c.max() > 0 else po_c
    pt_c = pt_c / pt_c.max() if pt_c.max() > 0 else pt_c
    raw = po / max(abs(po).max(), 1e-12) + pt / max(abs(pt).max(), 1e-12)               # unclipped: only breaks ties
    tie = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    s = np.sqrt(po_c * pt_c) + 1e-6 * tie
    return pd.Series(s / s.max() if s.max() > 0 else s, index=cols)


def keep_variables(X, t, y, seed=0, cut=CUT):
    """Names of the columns that are kept (at least one is always kept)."""
    s = scores(X, t, y, seed)
    return [c for c in X.columns if s[c] >= cut] or [X.columns[0]]


def keep_lasso(X, t, y, seed=0):
    """LASSO screen for Step 2: keep a variable if the outcome model or the treatment model gives it a non-zero coefficient."""
    Z = StandardScaler().fit_transform(X)
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = LassoCV(cv=3, random_state=seed).fit(np.column_stack([Z, t]), y).coef_[:-1]
        trt = LassoCV(cv=3, random_state=seed).fit(Z, t).coef_
    return [c for c, a, b in zip(X.columns, out, trt) if a != 0 or b != 0] or [X.columns[0]]
