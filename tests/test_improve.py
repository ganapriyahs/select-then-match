"""Tests for the three optional upgrades (tim.py) and the pre-declared rule (improve.py). Run:  python -m pytest -q tests/test_improve.py"""
import numpy as np
import pandas as pd
import pytest

from select_then_match import improve
from select_then_match.data import Data, make_sim
from select_then_match.tim import TIM


def _small():
    return make_sim("continuous_linear_n500", 21)


def test_defaults_are_the_original_method():
    d = _small()
    a = TIM(d.continuous, d.discrete, correct=True, seed=2).fit(d)
    b = TIM(d.continuous, d.discrete, correct=True, seed=2, mu0="boost", repeats=1, full=False).fit(d)
    assert a.att_ == b.att_ and a.share_ == b.share_


def test_repeats_average_the_single_runs_with_shifted_seeds():
    d = _small()
    singles = [TIM(d.continuous, d.discrete, correct=True, seed=5 + 10007 * r).fit(d).att_ for r in range(3)]
    rep = TIM(d.continuous, d.discrete, correct=True, seed=5, repeats=3).fit(d)
    assert rep.att_ == pytest.approx(np.mean(singles))
    assert TIM(d.continuous, d.discrete, correct=True, seed=5, repeats=1).fit(d).att_ == pytest.approx(singles[0])


def test_mu0_avg_is_the_mean_of_boosting_and_linear():
    d = _small()
    m = TIM(d.continuous, d.discrete, correct=True, mu0="avg", seed=1)
    c = np.flatnonzero(d.t == 0)
    X = d.X.to_numpy()
    boost, lin = m._boost(X[c], d.y[c], X), m._linear(X[c], d.y[c], X)
    assert m._predict_mu0(X[c], d.y[c], X) == pytest.approx(0.5 * (boost + lin))
    assert np.corrcoef(lin, d.y)[0, 1] > 0.3                              # the linear part is a real model, not noise


def test_mu0_avg_works_for_a_yes_no_outcome():
    d = make_sim("binary_linear_n500", 22)
    m = TIM(d.continuous, d.discrete, correct=True, mu0="avg", seed=1)
    c = np.flatnonzero(d.t == 0)
    p = m._predict_mu0(d.X.to_numpy()[c], d.y[c], d.X.to_numpy())
    assert ((p >= 0) & (p <= 1)).all()


def test_full_mode_cross_fits_so_an_untreated_person_never_predicts_himself():
    d = _small()
    m = TIM(d.continuous, d.discrete, correct=True, full=True, seed=3)
    base = m._mu0_all(d, d)
    i = int(np.flatnonzero(d.t == 0)[7])
    y2 = d.y.copy()
    y2[i] += 1000.0                                                        # change this person's own outcome a lot
    d2 = Data(d.X, d.t, y2, d.continuous, d.discrete, d.att)
    changed = TIM(d.continuous, d.discrete, correct=True, full=True, seed=3)._mu0_all(d2, d2)
    assert changed[i] == pytest.approx(base[i])                            # his own prediction does not move ...
    assert not np.allclose(np.delete(changed, i), np.delete(base, i))      # ... while other people's do (the model did see him)


def test_full_mode_uses_everybody_and_gives_a_sensible_estimate():
    d = make_sim("continuous_linear", 23)
    half = TIM(d.continuous, d.discrete, correct=True, seed=0).fit(d)
    full = TIM(d.continuous, d.discrete, correct=True, full=True, seed=0).fit(d)
    assert np.isfinite(full.att_) and 0 < full.share_ <= 1
    assert abs(full.att_ - d.att) < 1.0                                    # not wildly off (true effect is about 0.8)
    assert half.share_ <= 1


def test_full_mode_survives_a_rare_yes_no_outcome():
    d = make_sim("binary_linear_n500", 24)
    y = np.zeros_like(d.y)
    y[np.flatnonzero(d.t == 0)[:2]] = 1                                    # two untreated people with the outcome
    d2 = Data(d.X, d.t, y, d.continuous, d.discrete, d.att)
    m = TIM(d2.continuous, d2.discrete, correct=True, full=True, seed=0).fit(d2)
    assert 0 <= m.share_ <= 1                                              # no crash is the point


# ----------------------------------------------------------------------------- the rule
def _ratio(vals):
    return pd.Series(vals, index=list(improve.SETTINGS))


def test_rule_passes_only_when_all_three_conditions_hold():
    good = _ratio([0.8] * 9 + [1.0] * 3)
    assert improve.judge(good)["verdict"] == "PASS"
    assert improve.judge(_ratio([0.94] * 8 + [0.96] * 4))["verdict"] == "PASS"       # median 0.94, 8 better
    assert improve.judge(_ratio([0.80] * 7 + [0.97] * 5))["verdict"] == "no reliable gain"   # only 7 better
    assert improve.judge(_ratio([0.80] * 10 + [1.2] * 2))["verdict"] == "no reliable gain"   # 2 worse
    assert improve.judge(_ratio([0.96] * 12))["verdict"] == "no reliable gain"               # median above 0.95
    assert improve.judge(_ratio([1.0] * 12))["verdict"] == "no reliable gain"


def test_rule_counts():
    j = improve.judge(_ratio([0.5, 0.9, 0.96, 1.0, 1.04, 1.06, 1.3, 0.94, 0.7, 1.0, 1.0, 1.0]))
    assert (j["better"], j["tie"], j["worse"]) == (4, 6, 2)


def test_rule_needs_all_twelve_settings():
    assert improve.judge(pd.Series([0.5] * 11))["verdict"] == "no reliable gain"
