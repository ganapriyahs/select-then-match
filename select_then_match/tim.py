"""TIM, Two-Stage Interpretable Matching (Shikalgar and Noor-E-Alam 2025), and the two additions of this project.

TIM (stage 1 and stage 2)
    rank     rank the variables with a standardized linear model (outcome and treatment, scores averaged);
    bin      cut continuous variables into equal-width bins (the `cem` rule of the public TIM code);
    stage 1  match exactly on ALL variables. A treated person with no match drops the LEAST important variable and the
             rest try again, until every treated person that can be matched is matched. A group needs at least one
             treated and one untreated person;
    stage 2  inside a group matched on fewer variables, untreated people closer to the treated people on the dropped
             variables get more weight (the farthest gets 0);
    effect   treated average outcome minus weighted untreated average outcome, averaged over matched treated people (ATT).

Addition 1, `screen=True` (the 3-stage TIM): before ranking, screen out variables (screening.py), so matching starts
    from the important variables only and drops from there.
Addition 2, `correct=True` (bias correction): a boosting model trained on untreated people predicts what each person's
    outcome would be without treatment (mu0). In every group the untreated average is shifted by
    (treated mu0 - matched untreated mu0), which removes the leftover gap on the variables that were dropped or binned.
    The idea is Abadie and Imbens (2011); their theory is not claimed for this combination. The model sees ALL covariates.

Honest fitting: the ranking, screening, bin rule and correction model are learned on one half of the data and applied to
the other half, then the halves swap (common.swap). The price is that each matching pool has half the people.

Three optional upgrades, all OFF by default (the defaults give exactly the numbers of the final run). They are tested in
improve.py, with the pass/fail rule written before the run:
    full=True      no swap: learn on everybody and match everybody (a matching pool twice as large). The correction model
                   is cross-fitted (5 folds over the untreated people), so no untreated person's own outcome is used to
                   predict his or her own mu0. The ranking, bins and screening still see the outcomes of the people they
                   are applied to (that is the price; the simulations show whether it matters);
    repeats=R      repeat the two-half swap R times with different random splits and average the R estimates;
    mu0="avg"      the correction model is the average of the boosting model and a linear model (ridge, or logistic for a
                   yes/no outcome) instead of boosting alone.
One more option, `rank=` (Step 1 of TIM: which variable to drop first; default "linear" = TIM's average of the two scaled
regression scores): "out_only", "trt_only", "max", "geo", "ovb", "ridge_raw" (see rank_scores) and "forest" (the screening score
of screening.py). Tested in improve_rank.py.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .common import is_binary, swap
from .screening import CUT, keep_variables, scores


# ----------------------------------------------------------------------------- step 1: ranking
RANKS = ("linear", "geo", "max", "ovb", "out_only", "trt_only", "ridge_raw", "forest")


def rank_scores(X, t, y, how="linear"):
    """Step 1 of TIM: one importance score per variable; the variable with the lowest score is dropped first.

    The default, how="linear", is TIM's own: logistic (yes/no outcome) or ridge (number outcome) coefficients for the
    outcome, logistic coefficients for the treatment, all on standardized columns; each set of coefficients is scaled to a
    maximum of 1, then the two are averaged. Other ways to use or combine the same two models (all tested in improve_rank.py):
        out_only   only the outcome score (does the variable help to predict the outcome?)
        trt_only   only the treatment score (does the variable help to predict who is treated?)
        max        the larger of the two scores: high if the variable matters for EITHER
        geo        geometric mean of the two scores: high only if the variable matters for BOTH
        ovb        |outcome coefficient| x |difference in the variable's average between treated and untreated people|
                   (standardized): roughly how much bias dropping this variable would cause
        ridge_raw  a close version of the public TIM code: ridge for both models on the RAW columns (penalty chosen by
                   cross-validation), the two |coefficients| added and divided by the largest. The score depends on the
                   units of each column; that is how that code behaves.
    """
    norm = lambda a: np.abs(a) / max(np.abs(a).max(), 1e-12)
    if how == "ridge_raw":
        Xr, alphas = np.asarray(X, dtype=float), np.logspace(-3, 3, 13)
        s = np.abs(RidgeCV(alphas=alphas).fit(Xr, np.asarray(y, dtype=float)).coef_) + np.abs(RidgeCV(alphas=alphas).fit(Xr, np.asarray(t, dtype=float)).coef_)
        return pd.Series(s / max(s.max(), 1e-12), index=list(X.columns))
    Z = StandardScaler().fit_transform(X)
    Zo = np.column_stack([Z, t])
    if is_binary(y):
        out = LogisticRegression(C=1.0, max_iter=500).fit(Zo, y).coef_[0][:-1]
    else:
        out = Ridge(alpha=1.0).fit(Zo, (y - np.mean(y)) / (np.std(y) + 1e-12)).coef_[:-1]
    trt = LogisticRegression(C=1.0, max_iter=500).fit(Z, t).coef_[0]
    if how == "linear":
        s = (norm(out) + norm(trt)) / 2
    elif how == "out_only":
        s = norm(out)
    elif how == "trt_only":
        s = norm(trt)
    elif how == "max":
        s = np.maximum(norm(out), norm(trt))
    elif how == "geo":
        s = np.sqrt(norm(out) * norm(trt))
    elif how == "ovb":
        t = np.asarray(t)
        s = norm(np.abs(out) * np.abs(Z[t == 1].mean(axis=0) - Z[t == 0].mean(axis=0)))
    else:
        raise ValueError(f"unknown ranking {how!r}; choose from {RANKS}")
    return pd.Series(s, index=list(X.columns))


# ----------------------------------------------------------------------------- step 2: bins
def apply_cuts(x, cuts):
    """Bin number of every value of x (bins are closed on the right, like pandas.cut)."""
    return np.searchsorted(np.asarray(cuts, dtype=float), np.asarray(x, dtype=float), side="left")


def _l1_imbalance(codes, t):
    """Multivariate L1 imbalance (Iacus, King, Porro): half the sum over cells of |share of treated - share of controls|."""
    gid = pd.DataFrame(np.asarray(codes)).groupby(list(range(np.asarray(codes).shape[1])), sort=False).ngroup().to_numpy()
    g = int(gid.max()) + 1
    a, b = np.bincount(gid[t == 1], minlength=g).astype(float), np.bincount(gid[t == 0], minlength=g).astype(float)
    return float(np.abs(a / a.sum() - b / b.sum()).sum() / 2.0)


def bin_cuts(X, t, y, continuous, lower=4, upper=10):
    """The bin rule of the public TIM code (the R package `cem`, cem.coarsen): equal-width bins, the same number H of
    bins for every continuous variable, H chosen from 4..10 as the one whose imbalance is the median of the seven.
    Returns {variable: cut points}. Written in pandas and NumPy; checked against the cem 1.1.0 source."""
    cont = [c for c in X.columns if c in set(continuous)]
    base = {c: pd.factorize(X[c])[0] for c in X.columns if c not in set(cont)}
    base["__y__"] = pd.factorize(np.asarray(y))[0]
    imb = {}
    for H in range(lower, upper + 1):
        cols = dict(base)
        for c in cont:
            cols[c] = pd.cut(X[c], bins=min(int(X[c].nunique()), H), labels=False).to_numpy()
        imb[H] = _l1_imbalance(np.column_stack([cols[k] for k in cols]), t)
    imb = pd.Series(imb).round(10)
    H = int((imb.sort_values(ascending=False, kind="stable") <= imb.quantile(0.5)).idxmax())
    return {c: np.asarray(pd.cut(X[c], bins=min(int(X[c].nunique()), H), retbins=True)[1][1:-1], dtype=float) for c in cont}


# ----------------------------------------------------------------------------- stage 2: distance on dropped variables
def _omega(codes, discrete):
    """Distance between two values of a discrete variable (Ahmad-Dey style): how differently the other discrete
    variables are distributed among people with each value."""
    table = {}
    for i in discrete:
        others = [j for j in discrete if j != i]
        for a, b in combinations(np.unique(codes[i]), 2):
            tot = 0.0
            for j in others:
                ct = pd.crosstab(codes[i], codes[j])
                pa, pb = ct.loc[a] / ct.loc[a].sum(), ct.loc[b] / ct.loc[b].sum()
                tot += float(np.maximum(pa, pb).sum() - 1.0)
            table[(i, a, b)] = table[(i, b, a)] = tot / len(others) if others else 1.0
    return table


def _nearest_gap(xc, xt):
    """For every value in xc, the distance to the closest value in xt."""
    s = np.sort(xt)
    pos = np.searchsorted(s, xc)
    return np.minimum(np.abs(xc - s[np.clip(pos - 1, 0, len(s) - 1)]), np.abs(xc - s[np.clip(pos, 0, len(s) - 1)]))


def _group_ids(sub):
    return pd.DataFrame(sub).groupby(list(range(sub.shape[1])), sort=False).ngroup().to_numpy()


def _control_gap(X, codes, omega, dropped, tr, co, continuous):
    """Distance of each untreated person in a group from the group's treated people, summed over the dropped variables."""
    d = np.zeros(len(co))
    for c in dropped:
        if c in continuous:
            d += _nearest_gap(X[c].to_numpy()[co], X[c].to_numpy()[tr])
        else:
            cc = codes[c].to_numpy()
            tv = np.unique(cc[tr])
            per_value = {v: min(0.0 if v == u else omega.get((c, v, u), 1.0) for u in tv) for v in np.unique(cc[co])}
            d += np.array([per_value[v] for v in cc[co]])
    return d


# ----------------------------------------------------------------------------- the method
class TIM:
    def __init__(self, continuous, discrete, screen=False, correct=False, seed=0, mu0="boost", repeats=1, full=False, rank="linear"):
        assert mu0 in ("boost", "avg") and repeats >= 1 and rank in RANKS
        self.continuous, self.discrete = list(continuous), list(discrete)
        self.screen, self.correct, self.seed = screen, correct, seed
        self.mu0, self.repeats, self.full, self.rank = mu0, int(repeats), bool(full), rank
        self._seed = seed                           # the seed of the split being fitted (changes between repeats)

    def fit(self, d):
        """d is a Data object. Sets att_ (the estimate), share_ (share of treated people matched) and vars_used_
        (average number of variables a treated person was matched on: fewer = coarser groups)."""
        if self.full:                               # learn on everybody, match everybody
            self._seed = self.seed
            p = self._fit_half(d, d)
            self.att_ = p["att"] if p["n"] > 0 else np.nan
            self.share_ = p["n"] / max(int((d.t == 1).sum()), 1)
            parts = [p]
        else:
            runs = []
            for r in range(self.repeats):
                self._seed = self.seed + 10007 * r  # repeat 0 uses the plain seed, so repeats=1 is the original method
                runs.append(swap(d, self._seed, self._fit_half))
            self._seed = self.seed
            atts = [a for a, _, _ in runs if np.isfinite(a)]
            self.att_ = float(np.mean(atts)) if atts else np.nan
            self.share_ = float(np.mean([sh for _, sh, _ in runs]))
            parts = [p for _, _, ps in runs for p in ps]
        k = np.concatenate([p["k_used"] for p in parts])
        self.vars_used_ = float(k.mean()) if len(k) else np.nan
        self.kept_ = parts[0]["kept"]
        return self

    # ------------------------------------------------------------------------- the correction model
    def _boost(self, Xtr, ytr, Xte):
        kw = dict(max_depth=3, max_iter=100, min_samples_leaf=20, random_state=self._seed)
        if is_binary(ytr):
            return HistGradientBoostingClassifier(**kw).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        return HistGradientBoostingRegressor(**kw).fit(Xtr, ytr).predict(Xte)

    def _linear(self, Xtr, ytr, Xte):
        sc = StandardScaler().fit(Xtr)
        if is_binary(ytr):
            return LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xte))[:, 1]
        return Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))

    def _predict_mu0(self, Xtr, ytr, Xte):
        """Train on untreated people (Xtr, ytr), predict the untreated outcome of the rows of Xte."""
        if len(np.unique(ytr)) < 2 and is_binary(ytr):
            return np.full(len(Xte), float(np.mean(ytr)))        # a yes/no outcome with only one answer: nothing to learn
        boost = self._boost(Xtr, ytr, Xte)
        return boost if self.mu0 == "boost" else 0.5 * (boost + self._linear(Xtr, ytr, Xte))

    def _mu0_all(self, L, M):
        """mu0 for every row of M. Normal case: the model learns on the untreated people of L and predicts M (a different
        half). full=True (L is M): untreated people get a cross-fitted prediction (5 folds), treated people the average of
        the five fold models; nobody is predicted by a model that saw his or her own outcome."""
        XL, tL, yL = L.X.to_numpy(), L.t, L.y
        ctrl = np.flatnonzero(tL == 0)
        if L is not M:
            return self._predict_mu0(XL[ctrl], yL[ctrl], M.X.to_numpy())
        yc = yL[ctrl]
        folds = StratifiedKFold(5, shuffle=True, random_state=self._seed).split(XL[ctrl], yc) if is_binary(yc) \
            else KFold(5, shuffle=True, random_state=self._seed).split(XL[ctrl])
        mu0, treated = np.zeros(len(yL)), np.flatnonzero(tL == 1)
        for a, b in folds:
            pred = self._predict_mu0(XL[ctrl[a]], yc[a], XL)
            mu0[ctrl[b]] = pred[ctrl[b]]
            mu0[treated] += pred[treated] / 5.0
        return mu0

    def _fit_half(self, L, M):
        XL, tL, yL = L.X, L.t, L.y                  # learn half: ranking, screening, bins, correction model
        X, t, y = M.X, M.t, M.y                     # match half: matching and the effect
        # ranking and screening (learned on L)
        if self.rank == "forest":                   # rank by the screening score itself (held-out permutation importance)
            sc = scores(XL, tL, yL, self._seed)
            cols = list(sc.sort_values(ascending=False, kind="stable").index)
            keep = [c for c in XL.columns if sc[c] >= CUT] or [XL.columns[0]]      # the same rule as keep_variables
        else:
            cols = list(rank_scores(XL, tL, yL, self.rank).sort_values(ascending=False, kind="stable").index)
            keep = keep_variables(XL, tL, yL, self._seed) if self.screen else None
        if self.screen:
            cols = [c for c in cols if c in keep] or cols[:1]
        # bins (rule learned on L, applied to M)
        cuts = bin_cuts(XL, tL, yL, self.continuous)
        codes = pd.DataFrame({c: apply_cuts(X[c], cuts[c]) if c in self.continuous else pd.factorize(X[c], sort=True)[0] for c in cols})
        omega = _omega(codes, [c for c in cols if c in self.discrete])
        # stage 1: exact matching, dropping the least important variable until the treated are matched
        arr, K = codes.to_numpy(), len(cols)
        remaining = np.ones(len(X), dtype=bool)
        groups = []                                 # (number of variables used, treated rows, untreated rows)
        for k in range(K, 0, -1):
            idx = np.flatnonzero(remaining)
            if not (t[idx] == 1).any():
                break
            gid = _group_ids(arr[idx, :k])
            g = int(gid.max()) + 1
            nt = np.bincount(gid, weights=(t[idx] == 1), minlength=g)
            nc = np.bincount(gid, weights=(t[idx] == 0), minlength=g)
            ok = ((nt > 0) & (nc > 0))[gid]
            if not ok.any():
                continue
            ids, gs = idx[ok], gid[ok]
            o = np.argsort(gs, kind="stable")
            ids, gs = ids[o], gs[o]
            for m in np.split(ids, np.flatnonzero(np.diff(gs)) + 1):
                groups.append((k, m[t[m] == 1], m[t[m] == 0]))
            remaining[ids] = False
        # bias correction model: what would each person's outcome be without treatment?
        mu0 = self._mu0_all(L, M) if self.correct else None
        # stage 2 and the effect
        effects, weights, k_used = [], [], []
        for k, tr, co in groups:
            w = np.ones(len(co))
            if cols[k:]:
                dist = _control_gap(X, codes, omega, cols[k:], tr, co, self.continuous)
                w = 1.0 - (dist - dist.min()) / (dist.max() - dist.min()) if dist.max() > dist.min() else np.ones(len(co))
            if w.sum() <= 0:
                w = np.ones(len(co))
            wn = w / w.sum()
            control = float((wn * y[co]).sum())
            if mu0 is not None:
                control += mu0[tr].mean() - float((wn * mu0[co]).sum())
            effects.append(y[tr].mean() - control)
            weights.append(len(tr))
            k_used += [k] * len(tr)
        return dict(att=float(np.average(effects, weights=weights)) if effects else np.nan, n=int(sum(weights)),
                    k_used=np.array(k_used, dtype=int), kept=cols)
