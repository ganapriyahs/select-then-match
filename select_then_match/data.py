"""The two kinds of data in this project.

1. Simulated data (the truth is known, so every method can be scored).
   Columns: C0..C3 real confounders (they push both treatment and outcome), P0..P2 predict only the outcome,
   I0..I2 predict only the treatment, N0..N9 pure noise. Some columns are yes/no. 20 covariates in total.
   Twelve settings: outcome yes/no or number, outcome model linear or nonlinear, and for each of the four
   a smaller sample (n = 500) and poor overlap (treated and untreated people look very different).

2. The CDC diabetes data (BRFSS 2015): treatment = high cholesterol, outcome = diabetes, 20 other columns as covariates
   (as in the TIM paper). No true effect is known there.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Data:
    X: pd.DataFrame
    t: np.ndarray            # 0/1 treatment
    y: np.ndarray            # outcome (0/1 or a number)
    continuous: list         # columns that matching cuts into bins
    discrete: list           # yes/no or category columns
    att: float = None        # true effect on the treated (simulations only)

    def subset(self, idx):
        return Data(self.X.iloc[idx].reset_index(drop=True), self.t[idx], self.y[idx], self.continuous, self.discrete, self.att)

    def subsample(self, n, seed):
        rng = np.random.default_rng(seed)
        return self.subset(np.sort(rng.choice(len(self.y), size=min(n, len(self.y)), replace=False)))


# ----------------------------------------------------------------------------- simulation
CONFOUNDERS = ["C0", "C1", "C2", "C3"]
SIGNAL = CONFOUNDERS + ["P0", "P1", "P2"] + ["I0", "I1", "I2"]
YES_NO = ["C2", "C3", "P1", "P2", "I1", "I2"]
NOISE = 10          # N0..N9 -> 20 covariates; N6..N9 are yes/no


def _yes_no(c):
    return c in YES_NO or (c.startswith("N") and int(c[1:]) % 10 >= 6)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_sim(setting, seed):
    """One simulated data set of a setting in SETTINGS. The same seed always gives the same data."""
    s = SETTINGS[setting]
    n, outcome, nonlinear, confounding = s["n"], s["outcome"], s["nonlinear"], s["confounding"]
    rng = np.random.default_rng(seed)
    cols = SIGNAL + [f"N{i}" for i in range(NOISE)]
    p = len(cols)
    cov = np.full((p, p), 0.3)                       # covariates are correlated (rho = 0.3)
    np.fill_diagonal(cov, 1.0)
    # Cholesky (not the default SVD): its factor is unique, so the same seed gives the same data on every machine.
    X = pd.DataFrame(rng.multivariate_normal(np.zeros(p), cov, size=n, method="cholesky"), columns=cols)
    for c in cols:
        if _yes_no(c):
            X[c] = (X[c] > 0).astype(float)
    b = lambda c: X[c].to_numpy() - 0.5              # centred yes/no columns

    treat = confounding * (0.6 * (X.C0 + X.C1 + b("C2") * 2 + b("C3") * 2) + 0.6 * (X.I0 + b("I1") * 2 + b("I2") * 2)).to_numpy()
    e = _sigmoid(-0.3 + treat)                        # true chance of treatment
    t = rng.binomial(1, e)

    base = (0.5 * (X.P0 + b("P1") * 2 + b("P2") * 2) + 0.8 * (b("C2") * 2 + b("C3") * 2)).to_numpy()
    if nonlinear:                                    # a linear model cannot see C0 and C1 here
        score0 = -0.5 + base + (1.0 * X.C0 * X.C1 + 0.8 * (X.C0 ** 2 - 1)).to_numpy()
    else:
        score0 = -0.5 + base + (0.7 * (X.C0 + X.C1)).to_numpy()
    delta = 0.8 + 0.8 * b("C3") * 2                  # the effect differs between people

    if outcome == "binary":
        p0, p1 = _sigmoid(score0), _sigmoid(score0 + delta)
        y = np.where(t == 1, rng.random(n) < p1, rng.random(n) < p0).astype(int)
        ite = p1 - p0                                # true effect for each person: a risk difference
    else:
        delta = np.broadcast_to(np.asarray(delta, dtype=float), (n,)).copy()
        y = score0 + t * delta + rng.normal(0.0, 1.0, n)
        ite = delta
    return Data(X, t.astype(int), y, [c for c in cols if not _yes_no(c)], [c for c in cols if _yes_no(c)],
                att=float(ite[t == 1].mean()))


SETTINGS = {}
for _outcome in ("binary", "continuous"):
    for _form in ("linear", "nonlinear"):
        _b = dict(outcome=_outcome, nonlinear=(_form == "nonlinear"), n=1000, confounding=1.0)
        SETTINGS[f"{_outcome}_{_form}"] = dict(_b)
        SETTINGS[f"{_outcome}_{_form}_n500"] = dict(_b, n=500)
        SETTINGS[f"{_outcome}_{_form}_poor_overlap"] = dict(_b, confounding=2.0)


# ----------------------------------------------------------------------------- CDC
CDC_TREATMENT = "HighChol"
CDC_NUMBERS = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"]   # cut into bins like continuous columns


def load_cdc(path):
    """Read the CDC file (253,680 people). Works with the 3-level file (diabetes = level 2 only) and the binary file."""
    raw = pd.read_csv(path)
    if "Diabetes_binary" in raw.columns:
        y, drop = raw["Diabetes_binary"].to_numpy().astype(int), "Diabetes_binary"
    elif "Diabetes_012" in raw.columns:
        y, drop = (raw["Diabetes_012"].to_numpy() == 2).astype(int), "Diabetes_012"
    else:
        raise ValueError(f"{path}: no Diabetes_binary or Diabetes_012 column")
    X = raw.drop(columns=[CDC_TREATMENT, drop]).astype(float).reset_index(drop=True)
    numbers = [c for c in CDC_NUMBERS if c in X.columns]
    return Data(X, raw[CDC_TREATMENT].to_numpy().astype(int), y, numbers, [c for c in X.columns if c not in numbers])
