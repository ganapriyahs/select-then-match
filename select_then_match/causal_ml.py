"""Two causal machine learning methods, as references. Both use the same learner (gradient boosting) with 5-fold
cross-fitting (each person's prediction comes from a model that did not see that person).

Note on the target: `dml` fits ONE constant effect. When the effect differs between people this is a weighted average
that need not equal the effect on the treated, so part of its error in the simulations is a different target.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .common import is_binary

EPS = 0.02


def _learner(y, seed):
    kw = dict(max_depth=3, max_iter=100, min_samples_leaf=20, random_state=seed)
    return HistGradientBoostingClassifier(**kw) if is_binary(y) else HistGradientBoostingRegressor(**kw)


def _predict(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


def _crossfit(d, seed, cache):
    """Out-of-fold chance of treatment e, and outcome predictions mu0 (untreated), m (everybody). Shared by dml and tmle."""
    if "crossfit" in cache:
        return cache["crossfit"]
    X, t, y = np.asarray(d.X), np.asarray(d.t), np.asarray(d.y, dtype=float)
    e, mu0, m = (np.zeros(len(y)) for _ in range(3))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        e[te] = HistGradientBoostingClassifier(max_depth=3, max_iter=100, min_samples_leaf=20,
                                               random_state=seed).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
        m[te] = _predict(_learner(y, seed).fit(X[tr], y[tr]), X[te])
        untreated = tr[t[tr] == 0]
        mu0[te] = _predict(_learner(y, seed).fit(X[untreated], y[untreated]), X[te])
    cache["crossfit"] = (np.clip(e, EPS, 1 - EPS), mu0, m)
    return cache["crossfit"]


def dml(d, seed=0, cache=None):
    """Double machine learning (partially linear): take what the covariates explain out of both treatment and outcome,
    then regress what is left of the outcome on what is left of the treatment."""
    e, mu0, m = _crossfit(d, seed, {} if cache is None else cache)
    rt, ry = np.asarray(d.t) - e, np.asarray(d.y, dtype=float) - m
    return dict(att=float((rt * ry).sum() / (rt ** 2).sum()), share=1.0)


def tmle(d, seed=0, cache=None):
    """Targeted maximum likelihood (ATT form). The untreated-outcome model is nudged, in one logistic step along a
    'clever covariate', until the weighted residuals of the untreated people average to zero. The effect is the treated
    average outcome minus the average of the nudged prediction. A number outcome is rescaled to [0, 1] and scaled back."""
    e, mu0, m = _crossfit(d, seed, {} if cache is None else cache)
    t, y = np.asarray(d.t), np.asarray(d.y, dtype=float)
    lo, hi = (0.0, 1.0) if is_binary(y) else (y.min(), y.max())
    ys, q0 = (y - lo) / (hi - lo), np.clip((mu0 - lo) / (hi - lo), 1e-3, 1 - 1e-3)
    h = -e / (1 - e) / t.mean()                                   # clever covariate
    off, c = np.log(q0 / (1 - q0)), t == 0
    eps = 0.0
    for _ in range(25):                                           # Newton steps for the one-number nudge
        p = 1 / (1 + np.exp(-(off[c] + eps * h[c])))
        step = (h[c] * (ys[c] - p)).sum() / (-(h[c] ** 2 * p * (1 - p)).sum())
        eps -= step
        if abs(step) < 1e-9:
            break
    q0_star = 1 / (1 + np.exp(-(off + eps * h)))
    return dict(att=float((ys[t == 1].mean() - q0_star[t == 1].mean()) * (hi - lo)), share=1.0)
