"""MALTS-style matching: Matching After Learning To Stretch (Parikh, Sarkar, Volfovsky, Rudin; JMLR 2022). A compact
re-implementation written for this project, NOT the authors' package (pymalts).

How it works
  * all variables are standardized. A distance between two people is  sum_k w_k * (x_ik - x_jk)^2,  with one weight w_k >= 0
    per variable (a "stretch": a big weight makes a difference in that variable count more);
  * the weights are LEARNED on the learning half so that people who are close in this distance have similar outcomes:
    inside each treatment group, the outcome of a person is predicted by a kernel average of the other people's outcomes
    (nearby people count more), and the weights minimize the prediction error plus a small penalty on the sum of weights
    (which pushes the weights of useless variables to zero);
  * on the matching half, each treated person is compared with their k = 10 nearest untreated people in the learned
    distance: effect = own outcome minus the average outcome of those 10; the ATT is the average over treated people.

Simplifications: one shared set of weights for both groups and for yes/no and number variables alike (the paper learns
them separately), a fixed penalty, and at most 500 people per group are used to learn the weights.
The halves are swapped, as for TIM (common.swap).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

from .common import swap

K = 10              # untreated people compared with each treated person
LAM = 0.01          # penalty on the sum of the weights
MAX_LEARN = 500     # people per group used to learn the weights


def _arms(Z, t, y, seed):
    """Per treatment group: pairwise squared differences S[k, i, j] and the outcomes (standardized)."""
    rng = np.random.default_rng(seed)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    out = []
    for g in (0, 1):
        idx = np.flatnonzero(t == g)
        if len(idx) > MAX_LEARN:
            idx = rng.choice(idx, MAX_LEARN, replace=False)
        Zg = Z[idx]
        S = (Zg.T[:, :, None] - Zg.T[:, None, :]) ** 2
        out.append((S, ys[idx]))
    return out


def loss_and_gradient(w, arms, lam=LAM):
    """Prediction error of the kernel average (leave one out) plus the penalty, and its gradient with respect to w."""
    total_n = sum(len(y) for _, y in arms)
    loss, grad = lam * w.sum(), lam * np.ones_like(w)
    for S, y in arms:
        D = np.tensordot(w, S, axes=(0, 0))
        Kern = np.exp(-D)
        np.fill_diagonal(Kern, 0.0)                                   # a person does not predict their own outcome
        den = Kern.sum(1) + 1e-12
        pred = Kern @ y / den
        r = y - pred
        loss += float((r ** 2).sum()) / total_n
        G = (2.0 / total_n) * (r / den)[:, None] * Kern * (y[None, :] - pred[:, None])
        grad += np.tensordot(S, G, axes=([1, 2], [0, 1]))
    return loss, grad


def learn_weights(Z, t, y, seed=0):
    p = Z.shape[1]
    res = minimize(loss_and_gradient, np.full(p, 1.0 / p), args=(_arms(Z, t, y, seed),), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, 50.0)] * p, options=dict(maxiter=60))
    return res.x


def _fit_half(L, M, seed):
    mu = L.X.mean()
    sd = L.X.std(ddof=0).replace(0, 1.0)
    ZL, ZM = ((L.X - mu) / sd).to_numpy(), ((M.X - mu) / sd).to_numpy()
    w = learn_weights(ZL, L.t, L.y, seed)
    i1, i0 = np.flatnonzero(M.t == 1), np.flatnonzero(M.t == 0)
    if len(i1) == 0 or len(i0) == 0:
        return dict(att=np.nan, n=0)
    root = np.sqrt(w)
    nn = NearestNeighbors(n_neighbors=min(K, len(i0))).fit(ZM[i0] * root).kneighbors(ZM[i1] * root)[1]
    return dict(att=float(np.mean(M.y[i1] - M.y[i0][nn].mean(axis=1))), n=int(len(i1)), weights=w)


def malts(d, seed=0, cache=None):
    att, share, _ = swap(d, seed, lambda L, M: _fit_half(L, M, seed))
    return dict(att=att, share=share)
