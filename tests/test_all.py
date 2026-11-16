"""Tests. Run:  python -m pytest -q   (about a minute)."""
import numpy as np
import pandas as pd
import pytest

from select_then_match import common, flame, malts, screening, step1, step2, step3
from select_then_match.data import SETTINGS, Data, make_sim
from select_then_match.methods import METHODS
from select_then_match.tim import TIM, apply_cuts, bin_cuts, rank_scores


# ----------------------------------------------------------------------------- data
def test_twelve_settings_and_reproducible_data():
    assert len(SETTINGS) == 12
    a, b = make_sim("binary_linear", 7), make_sim("binary_linear", 7)
    assert a.X.equals(b.X) and (a.y == b.y).all() and a.att == b.att
    assert not make_sim("binary_linear", 8).X.equals(a.X)
    assert a.X.shape == (1000, 20) and set(np.unique(a.y)) <= {0, 1}
    assert make_sim("continuous_nonlinear_n500", 1).X.shape[0] == 500
    assert 0 < a.att < 1                                           # a risk difference


def test_poor_overlap_pushes_propensity_apart():
    good, poor = make_sim("binary_linear", 3), make_sim("binary_linear_poor_overlap", 3)
    gap = lambda d: abs(d.X.loc[d.t == 1, "C0"].mean() - d.X.loc[d.t == 0, "C0"].mean())
    assert gap(poor) > gap(good)


def test_subset_keeps_columns_and_truth():
    d = make_sim("binary_linear", 1).subset(np.arange(10))
    assert len(d.y) == 10 and d.att is not None and d.X.index.tolist() == list(range(10))


# ----------------------------------------------------------------------------- common
def test_halves_are_stratified_and_swap_combines_by_treated_count():
    t = np.array([0] * 60 + [1] * 40)
    half = common.split_halves(t, 0)
    assert half[t == 1].sum() == 20 and half[t == 0].sum() == 30
    d = Data(pd.DataFrame({"a": np.arange(100.0)}), t, np.zeros(100), ["a"], [])
    parts = iter([dict(att=1.0, n=10), dict(att=4.0, n=30)])
    att, share, _ = common.swap(d, 0, lambda L, M: next(parts))
    assert att == pytest.approx((1 * 10 + 4 * 30) / 40) and share == pytest.approx(40 / 40)


def test_swap_with_one_empty_half_uses_the_other():
    d = make_sim("binary_linear_n500", 1)
    parts = iter([dict(att=np.nan, n=0), dict(att=0.3, n=50)])
    att, share, _ = common.swap(d, 0, lambda L, M: next(parts))
    assert att == pytest.approx(0.3)


# ----------------------------------------------------------------------------- screening
def test_screening_keeps_signal_and_drops_most_noise():
    d = make_sim("continuous_linear", 11)
    keep = screening.keep_variables(d.X, d.t, d.y, seed=0)
    assert {"C0", "C1"} <= set(keep)
    assert len([c for c in keep if c.startswith("N")]) <= 3          # at most 3 of the 10 noise columns survive
    assert len(keep) < 20


def test_screening_needs_both_outcome_and_treatment():
    # a variable that only drives the treatment (I0 here) must score far below the confounders
    d = make_sim("continuous_linear", 11)
    s = screening.scores(d.X, d.t, d.y, seed=0)
    assert s["I0"] < 0.5 * max(s["C0"], s["C1"]) and s.max() == pytest.approx(1.0)


# ----------------------------------------------------------------------------- TIM, by hand
def _learn_half():
    """A learning half where 'a' drives both treatment and outcome and 'b' is noise, so the ranking is a > b."""
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 2, 600).astype(float), rng.integers(0, 2, 600).astype(float)
    t = (rng.random(600) < 0.2 + 0.6 * a).astype(int)
    y = 2 * a + rng.normal(0, 0.3, 600)
    return Data(pd.DataFrame({"a": a, "b": b}), t, y, [], ["a", "b"])


def test_ranking_puts_the_confounder_first():
    L = _learn_half()
    s = rank_scores(L.X, L.t, L.y)
    assert s["a"] == pytest.approx(1.0) and s["a"] > s["b"]


def test_tim_toy_matches_on_everything_first_then_drops_the_least_important_variable():
    L = _learn_half()
    # treated #1 (a=0,b=0) has two untreated twins; treated #2 (a=1,b=0) has no twin until b is dropped
    M = Data(pd.DataFrame({"a": [0, 0, 0, 1, 1.0], "b": [0, 0, 0, 0, 1.0]}), np.array([1, 0, 0, 1, 0]),
             np.array([10.0, 4.0, 6.0, 3.0, 1.0]), [], ["a", "b"])
    r = TIM([], ["a", "b"])._fit_half(L, M)
    assert r["n"] == 2 and sorted(r["k_used"]) == [1, 2]
    assert r["att"] == pytest.approx(((10 - 5) + (3 - 1)) / 2)


def test_tim_with_no_possible_match_returns_no_estimate():
    L = _learn_half()
    M = Data(pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 0.0]}), np.array([1, 1]), np.array([1.0, 2.0]), [], ["a", "b"])
    r = TIM([], ["a", "b"])._fit_half(L, M)
    assert r["n"] == 0 and np.isnan(r["att"])


def test_bins_are_right_closed_and_rule_returns_cut_points():
    assert list(apply_cuts([1, 2, 3, 4], [2.0, 3.0])) == [0, 0, 1, 2]
    d = make_sim("continuous_linear", 2)
    cuts = bin_cuts(d.X, d.t, d.y, d.continuous)
    assert set(cuts) == set(d.continuous) and all(len(c) >= 3 for c in cuts.values())


def test_correction_removes_most_of_tims_overshoot():
    plain, fixed = [], []
    for seed in (101, 102, 103):
        d = make_sim("continuous_linear", seed)
        plain.append(abs(TIM(d.continuous, d.discrete, seed=seed).fit(d).att_ - d.att))
        fixed.append(abs(TIM(d.continuous, d.discrete, seed=seed, correct=True).fit(d).att_ - d.att))
    assert np.mean(fixed) < 0.5 * np.mean(plain)


def test_three_stage_matches_on_fewer_variables():
    d = make_sim("binary_linear", 21)
    m = TIM(d.continuous, d.discrete, screen=True, correct=True, seed=1).fit(d)
    assert len(m.kept_) < 20 and 0 < m.share_ <= 1 and np.isfinite(m.att_)


# ----------------------------------------------------------------------------- FLAME and MALTS
def test_flame_toy_matches_exactly_when_it_can():
    L = _learn_half()
    M = Data(pd.DataFrame({"a": [0, 0, 0.0], "b": [0, 0, 0.0]}), np.array([1, 0, 0]), np.array([10.0, 4.0, 6.0]), [], ["a", "b"])
    r = flame._fit_half(L, M, [])
    assert r["n"] == 1 and r["att"] == pytest.approx(5.0)


def test_flame_drops_the_noise_variable_before_the_important_one():
    L = _learn_half()
    # the only way to match treated #2 is to drop a variable; FLAME must drop b (noise), keeping a
    M = Data(pd.DataFrame({"a": [1, 1, 1.0], "b": [0, 1, 1.0]}), np.array([1, 0, 0]), np.array([5.0, 2.0, 4.0]), [], ["a", "b"])
    r = flame._fit_half(L, M, [])
    assert r["n"] == 1 and r["att"] == pytest.approx(5.0 - 3.0)


def test_malts_gradient_is_right_and_weights_favour_signal():
    from scipy.optimize import check_grad
    d = make_sim("continuous_linear", 5)
    Z = ((d.X - d.X.mean()) / d.X.std(ddof=0)).to_numpy()
    arms = malts._arms(Z[:300], d.t[:300], d.y[:300], 0)
    w0 = np.random.default_rng(1).uniform(0.02, 0.2, Z.shape[1])
    f, g = (lambda w: malts.loss_and_gradient(w, arms)[0]), (lambda w: malts.loss_and_gradient(w, arms)[1])
    assert check_grad(f, g, w0) < 1e-5
    w = pd.Series(malts.learn_weights(Z[:500], d.t[:500], d.y[:500], 0), index=d.X.columns)
    assert w[["C0", "C1", "C2", "C3"]].mean() > 5 * w[[c for c in w.index if c.startswith("N")]].mean()


# ----------------------------------------------------------------------------- the ten methods
def test_every_method_gives_a_finite_estimate_and_a_share():
    d = make_sim("binary_linear_n500", 31)
    cache = {}
    for name, fn in METHODS.items():
        r = fn(d, 31, cache)
        assert np.isfinite(r["att"]) and 0 < r["share"] <= 1, name


def test_naive_is_the_difference_in_means_and_adjusting_moves_it_toward_the_truth():
    d = make_sim("binary_linear", 41)
    naive = METHODS["naive"](d, 41)["att"]
    assert naive == pytest.approx(d.y[d.t == 1].mean() - d.y[d.t == 0].mean())
    assert abs(METHODS["dml"](d, 41, {})["att"] - d.att) < abs(naive - d.att)


def test_psm_and_robust_use_only_treated_people_with_a_close_partner():
    d = make_sim("binary_linear_poor_overlap", 51)
    assert METHODS["robust"](d, 51)["share"] < 0.8 and METHODS["psm"](d, 51)["share"] <= 1.0


# ----------------------------------------------------------------------------- Steps 1 to 3
def test_step1_raw_gap_on_a_known_table():
    t = np.array([1] * 100 + [0] * 100)
    y = np.array([1] * 30 + [0] * 70 + [1] * 10 + [0] * 90)
    g = step1.raw_gap(t, y)
    assert g["risk difference"] == pytest.approx(0.20) and g["risk ratio"] == pytest.approx(3.0)
    assert g["odds ratio"] == pytest.approx((30 / 70) / (10 / 90))


def test_step1_smd_is_the_standardized_gap():
    X = pd.DataFrame({"x": [0.0, 2.0, 0.0, 2.0, 4.0, 6.0]})
    t = np.array([0, 0, 0, 1, 1, 1])
    tab = step1.covariate_table(X, t)
    sd = np.sqrt((np.var([0, 2, 0]) + np.var([2, 4, 6])) / 2)
    assert tab.SMD[0] == pytest.approx((4.0 - 2 / 3) / sd)


def test_step2_logistic_and_robust_ols_recover_a_known_effect():
    rng = np.random.default_rng(0)
    n = 40000
    x = rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.5 * x + 1.0 * t)))).astype(int)
    e = step2.adjusted_effect(pd.DataFrame({"x": x}), t, y, ["x"])
    assert e["odds ratio"] == pytest.approx(np.exp(1.0), rel=0.1) and 0.1 < e["risk difference"] < 0.35


def test_step3_compare_counts_settings_with_a_tie_band():
    rmse = pd.DataFrame({"a": [0.5, 1.0, 2.0, 1.02], "b": [1.0, 1.0, 1.0, 1.0]})
    c = step3._compare(rmse, "a", "b")
    assert (c["better"], c["tie"], c["worse"]) == (1, 2, 1) and c["median_ratio"] == pytest.approx(1.01)


def test_step3_report_builds_from_result_files(tmp_path):
    rows = []
    for seed in (1, 2):
        for m in METHODS:
            rows.append(dict(setting="binary_linear", seed=seed, att_true=0.1, method=m, att=0.1 + 0.01 * seed, share=1.0, vars_used=np.nan, seconds=0.1,
                             err=0.01 * seed))
    (tmp_path / "sim").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "sim" / "binary_linear.csv", index=False)
    out = step3.build(str(tmp_path / "sim"), str(tmp_path / "none.csv"), str(tmp_path / "RESULTS.md"))
    text = open(out).read()
    assert "RMSE" in text and "tim_3stage_bc" in text and "Does the correction help TIM?" in text
