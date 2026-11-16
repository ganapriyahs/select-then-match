"""Three simple baselines. Every method has the same shape: fn(d, seed, cache) -> {"att": estimate, "share": share of
treated people used}, where d is a Data object (data.py)."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

EPS = 0.02          # propensity scores are kept inside [0.02, 0.98]
MAX_PAIRS = 4e7     # robust matching builds a treated x untreated table; refuse a table bigger than this


def _propensity(X, t):
    """Chance of being treated given the covariates (logistic regression)."""
    Z = StandardScaler().fit_transform(X)
    return np.clip(LogisticRegression(C=1.0, max_iter=1000).fit(Z, t).predict_proba(Z)[:, 1], EPS, 1 - EPS)


def naive(d, seed=0, cache=None):
    """No adjustment: the difference in average outcome between treated and untreated people."""
    return dict(att=float(d.y[d.t == 1].mean() - d.y[d.t == 0].mean()), share=1.0)


def psm(d, seed=0, cache=None, caliper=0.2):
    """Propensity score matching: each treated person gets the 1 nearest untreated person (with replacement) on the logit
    of the propensity score. A pair further apart than 0.2 standard deviations is not used (a caliper)."""
    e = _propensity(d.X, d.t)
    s = np.log(e / (1 - e))
    i1, i0 = np.flatnonzero(d.t == 1), np.flatnonzero(d.t == 0)
    dist, pos = NearestNeighbors(n_neighbors=1).fit(s[i0].reshape(-1, 1)).kneighbors(s[i1].reshape(-1, 1))
    ok = dist[:, 0] <= caliper * s.std()
    if not ok.any():
        return dict(att=np.nan, share=0.0)
    return dict(att=float(np.mean(d.y[i1[ok]] - d.y[i0[pos[ok, 0]]])), share=float(ok.mean()))


def robust(d, seed=0, cache=None, caliper=0.2):
    """Robust matching: optimal 1:1 matching (the pairing with the smallest total Mahalanobis distance, without
    replacement) that may leave a treated person unmatched. A pair is allowed only if the propensity scores are within
    the caliper, as in psm. Treated people with no allowed partner are dropped. (Without a caliper, 1:1 matching with
    about as many treated as untreated people must use almost everybody and ends up near the naive difference.)
    Written as an assignment problem, which has the same optimum as an integer program for pair matching."""
    Z = np.asarray(d.X, dtype=float)
    VI = np.linalg.pinv(np.atleast_2d(np.cov(Z.T)))
    W = Z @ np.linalg.cholesky(VI + 1e-9 * np.eye(Z.shape[1]))     # plain distance on W = Mahalanobis distance on X
    i1, i0 = np.flatnonzero(d.t == 1), np.flatnonzero(d.t == 0)
    if len(i1) * len(i0) > MAX_PAIRS:
        raise ValueError(f"robust matching needs a {len(i1)} x {len(i0)} distance table: use a smaller sample")
    e = _propensity(d.X, d.t)
    s = np.log(e / (1 - e))
    allowed = np.abs(s[i1][:, None] - s[i0][None, :]) <= caliper * s.std()
    D = cdist(W[i1], W[i0])
    top = float(D.max()) + 1.0
    cost = np.hstack([np.where(allowed, D, 4 * top),                  # a forbidden pair costs more than staying unmatched
                      np.full((len(i1), len(i1)), 2 * top)])          # one "stay unmatched" column per treated person
    r, c = linear_sum_assignment(cost)
    real = (c < len(i0)) & allowed[r, np.minimum(c, len(i0) - 1)]
    if not real.any():
        return dict(att=np.nan, share=0.0)
    return dict(att=float(np.mean(d.y[i1[r[real]]] - d.y[i0[c[real]]])), share=float(real.sum() / len(i1)))
