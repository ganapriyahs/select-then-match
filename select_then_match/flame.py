"""FLAME-style matching: Almost Matching Exactly with learned variable dropping (Wang, Morucci, Awan, Liu, Roy, Rudin,
Volfovsky; JMLR 2021). A compact re-implementation written for this project, NOT the authors' package (dame-flame).

How it works
  * continuous variables are cut into bins first (FLAME needs categories), with the same bin rule as TIM;
  * match exactly on all variables. Treated and untreated people in a cell that has both are matched and set aside;
  * for the people still unmatched, choose ONE variable to drop: the one with the best "matching quality"
        quality = C * (share of unmatched people who would find a match) - prediction error
    where the prediction error comes from ridge regressions of the outcome (one for untreated, one for treated people)
    that use the variables kept so far, scored by cross-validation on a separate learning half. A variable that the
    outcome depends on costs a lot of prediction error to drop, so FLAME drops unimportant variables first;
  * repeat until every treated person is matched, or until the prediction error has grown by more than 25% compared with
    using all variables (early stop). Treated people left unmatched are not used.

Difference from TIM: FLAME re-chooses the variable to drop after every round, using the outcome. TIM ranks the variables
once, and TIM keeps going until all treated are matched.
Learning and matching halves are swapped, as for TIM (common.swap). Ridge on standardized bin codes, C = 0.1 and the 25%
early stop follow the package defaults as far as I know them.
"""
import numpy as np
import pandas as pd

from .common import swap
from .tim import _group_ids, apply_cuts, bin_cuts

C = 0.1
EARLY_STOP = 0.25
FOLDS = 3


def _codes(D, cuts, continuous):
    """Integer bin code of every variable (continuous: bin number; others: value)."""
    return np.column_stack([apply_cuts(D.X[c], cuts[c]) if c in continuous else D.X[c].to_numpy(dtype=float) for c in D.X.columns])


def _ridge_cv_mse(Z, y, alpha=1.0):
    """Cross-validated mean squared error of a ridge regression of y on Z."""
    n = len(y)
    if n < 2 * FOLDS or Z.shape[1] == 0:
        return float(np.var(y)) if n else 0.0
    order = np.random.default_rng(0).permutation(n)
    err = 0.0
    for f in range(FOLDS):
        te = np.zeros(n, dtype=bool)
        te[order[f::FOLDS]] = True
        mu, sd = Z[~te].mean(0), Z[~te].std(0)
        A, B = (Z[~te] - mu) / np.where(sd > 0, sd, 1.0), (Z[te] - mu) / np.where(sd > 0, sd, 1.0)
        yc = y[~te] - y[~te].mean()
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ yc)
        err += float(((y[te] - (B @ w + y[~te].mean())) ** 2).sum())
    return err / n


def _prediction_error(codeL, tL, yL, cols):
    """Ridge prediction error on the learning half, for untreated and treated people, using the variables `cols`."""
    Z = codeL[:, cols]
    return sum(_ridge_cv_mse(Z[tL == g], yL[tL == g]) for g in (0, 1))


def _cells(codeM, tM, idx, cols):
    """For the unmatched people `idx`: group ids on `cols`, and a yes/no flag for 'cell has treated and untreated people'."""
    gid = _group_ids(codeM[idx][:, cols])
    g = int(gid.max()) + 1
    nt = np.bincount(gid, weights=(tM[idx] == 1), minlength=g)
    nc = np.bincount(gid, weights=(tM[idx] == 0), minlength=g)
    return gid, ((nt > 0) & (nc > 0))[gid]


def _fit_half(L, M, continuous):
    cuts = bin_cuts(L.X, L.t, L.y, continuous)
    codeL, codeM = _codes(L, cuts, continuous), _codes(M, cuts, continuous)
    tM, yM = M.t, M.y
    active = list(range(codeM.shape[1]))
    pe_all = _prediction_error(codeL, L.t, L.y, active)
    remaining = np.ones(len(yM), dtype=bool)
    effects, weights = [], []
    while active and (remaining & (tM == 1)).any():
        idx = np.flatnonzero(remaining)
        gid, ok = _cells(codeM, tM, idx, active)                      # match exactly on the variables still in use
        if ok.any():
            ids, gs = idx[ok], gid[ok]
            o = np.argsort(gs, kind="stable")
            ids, gs = ids[o], gs[o]
            for m in np.split(ids, np.flatnonzero(np.diff(gs)) + 1):
                tr, co = m[tM[m] == 1], m[tM[m] == 0]
                effects.append(yM[tr].mean() - yM[co].mean())
                weights.append(len(tr))
            remaining[ids] = False
        idx = np.flatnonzero(remaining)
        if len(active) == 1 or not (tM[idx] == 1).any():
            break
        best, best_q, best_pe = None, -np.inf, None
        for j in active:                                              # which variable is cheapest to drop?
            rest = [c for c in active if c != j]
            _, ok = _cells(codeM, tM, idx, rest)
            is_t = tM[idx] == 1
            bf = ok[is_t].mean() + (ok[~is_t].mean() if (~is_t).any() else 0.0)     # share of unmatched treated + untreated that would match
            pe = _prediction_error(codeL, L.t, L.y, rest)
            q = C * bf - pe
            if q > best_q:
                best, best_q, best_pe = j, q, pe
        if best_pe > pe_all * (1 + EARLY_STOP):                       # early stop: dropping more costs too much accuracy
            break
        active.remove(best)
    return dict(att=float(np.average(effects, weights=weights)) if effects else np.nan, n=int(sum(weights)))


def flame(d, seed=0, cache=None):
    att, share, _ = swap(d, seed, lambda L, M: _fit_half(L, M, d.continuous))
    return dict(att=att, share=share)
