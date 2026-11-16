"""Tests for the Step 1 options (tim.py, rank_scores) and the test script improve_rank.py. Run:  python -m pytest -q tests/test_rank.py"""
import numpy as np
import pandas as pd
import pytest

from select_then_match import improve, improve_rank
from select_then_match import tim as tim_module
from select_then_match.data import make_sim
from select_then_match.screening import CUT, keep_variables, scores
from select_then_match.tim import RANKS, TIM, rank_scores

# Numbers of the ORIGINAL package (before any upgrade option existed), seed=2: the default must still give exactly these.
# The "fixed_screen" rows use a fixed list of 13 kept variables instead of the forest screening: the forest scores of the noise
# variables sit right at the 0.05 cut, so a tiny difference between computers (library versions) can move a variable across it.
ORIGINAL = [
    ("continuous_linear_n500", 21, "plain", 1.8679078483414895, 0.8362068965517241),
    ("continuous_linear_n500", 21, "correct", 0.8801230244549803, 0.8362068965517241),
    ("continuous_linear_n500", 21, "fixed_screen_correct", 0.9424991892930101, 0.8362068965517241),
    ("binary_linear_n500", 22, "plain", 0.24657051398923852, 0.9186991869918699),
    ("binary_linear_n500", 22, "correct", 0.02726787854564587, 0.9186991869918699),
    ("binary_linear_n500", 22, "fixed_screen_correct", -0.04524050648606489, 0.8455284552845529),
]
KW = {"plain": {}, "correct": dict(correct=True), "fixed_screen_correct": dict(screen=True, correct=True)}
KEEP = ["C0", "C1", "C2", "C3", "P0", "P1", "P2", "I0", "I1", "I2", "N0", "N1", "N2"]


@pytest.mark.parametrize("setting,seed,kind,att,share", ORIGINAL)
def test_default_ranking_gives_exactly_the_original_numbers(setting, seed, kind, att, share, monkeypatch):
    if kind == "fixed_screen_correct":
        monkeypatch.setattr(tim_module, "keep_variables", lambda X, t, y, seed=0, cut=0.05: [c for c in X.columns if c in KEEP])
    d = make_sim(setting, seed)
    m = TIM(d.continuous, d.discrete, seed=2, **KW[kind]).fit(d)                     # rank="linear" is the default
    assert m.att_ == pytest.approx(att, rel=1e-9, abs=1e-12)
    assert m.share_ == pytest.approx(share, rel=1e-9, abs=1e-12)


def _data(setting="continuous_linear_n500", seed=31):
    d = make_sim(setting, seed)
    return d.X, d.t, d.y


@pytest.mark.parametrize("how", [h for h in RANKS if h != "forest"])
def test_every_score_is_a_finite_non_negative_series_over_the_columns(how):
    for setting in ("continuous_linear_n500", "binary_nonlinear_n500"):
        X, t, y = _data(setting)
        s = rank_scores(X, t, y, how)
        assert list(s.index) == list(X.columns)
        assert np.isfinite(s.to_numpy()).all() and (s >= 0).all() and s.max() <= 1 + 1e-12


def test_unknown_score_is_refused():
    X, t, y = _data()
    with pytest.raises(ValueError):
        rank_scores(X, t, y, "nonsense")
    with pytest.raises(AssertionError):
        TIM(["C0"], [], rank="nonsense")


def test_the_scores_are_built_from_the_same_two_parts():
    for setting in ("continuous_linear_n500", "binary_linear_n500"):
        X, t, y = _data(setting)
        o, a = rank_scores(X, t, y, "out_only"), rank_scores(X, t, y, "trt_only")
        assert o.max() == pytest.approx(1) and a.max() == pytest.approx(1)
        assert rank_scores(X, t, y, "linear").to_numpy() == pytest.approx(((o + a) / 2).to_numpy())     # TIM's own score
        assert rank_scores(X, t, y, "max").to_numpy() == pytest.approx(np.maximum(o, a).to_numpy())
        assert rank_scores(X, t, y, "geo").to_numpy() == pytest.approx(np.sqrt(o * a).to_numpy())
        g, lin, mx = (rank_scores(X, t, y, h).to_numpy() for h in ("geo", "linear", "max"))
        assert (g <= lin + 1e-12).all() and (lin <= mx + 1e-12).all()                                   # geometric <= average <= max


def test_ovb_is_zero_for_a_variable_with_no_gap_between_treated_and_untreated():
    rng = np.random.default_rng(0)
    n = 400
    t = np.repeat([0, 1], n // 2)
    base = rng.normal(size=n // 2)
    X = pd.DataFrame({"same": np.concatenate([base, base]),                 # identical values in both groups: gap exactly 0
                      "gap": rng.normal(size=n) + 1.5 * t})
    y = 2.0 * X["same"].to_numpy() + 1.0 * X["gap"].to_numpy() + rng.normal(size=n)
    s = rank_scores(X, t, y, "ovb")
    assert s["same"] < 1e-9 and s["gap"] == pytest.approx(1.0)
    assert rank_scores(X, t, y, "out_only")["same"] > 0.5                   # while the outcome score alone ranks it high


def test_ridge_raw_depends_on_units_and_the_standardized_scores_do_not():
    X, t, y = _data()
    X2 = X.copy()
    X2["C0"] = X2["C0"] * 1000.0
    for how in ("linear", "out_only", "trt_only", "max", "geo", "ovb"):
        assert rank_scores(X2, t, y, how).to_numpy() == pytest.approx(rank_scores(X, t, y, how).to_numpy(), rel=1e-6, abs=1e-6)
    a, b = rank_scores(X, t, y, "ridge_raw"), rank_scores(X2, t, y, "ridge_raw")
    assert not np.allclose(a.to_numpy(), b.to_numpy(), atol=0.05)


def test_forest_rank_keeps_the_same_variables_as_the_screening_rule():
    X, t, y = _data()
    sc = scores(X, t, y, 4)
    assert ([c for c in X.columns if sc[c] >= CUT] or [X.columns[0]]) == keep_variables(X, t, y, 4)


@pytest.mark.parametrize("how", RANKS)
def test_every_ranking_runs_the_whole_pipeline_and_gives_a_sensible_effect(how):
    d = make_sim("continuous_linear_n500", 32)
    m = TIM(d.continuous, d.discrete, screen=True, correct=True, seed=1, rank=how).fit(d)
    assert np.isfinite(m.att_) and 0 < m.share_ <= 1
    assert abs(m.att_ - d.att) < 1.5


def test_ranking_option_works_together_with_the_other_options():
    d = make_sim("binary_linear_n500", 33)
    for kw in (dict(full=True), dict(repeats=2), dict(mu0="avg")):
        m = TIM(d.continuous, d.discrete, screen=True, correct=True, seed=1, rank="max", **kw).fit(d)
        assert np.isfinite(m.att_)


# ----------------------------------------------------------------------------- the test script
def test_the_rule_is_the_one_from_the_first_test():
    assert improve_rank.judge is improve.judge
    assert (improve_rank.MEDIAN_MAX, improve_rank.BETTER_MIN, improve_rank.WORSE_MAX, improve_rank.BAND) == (0.95, 8, 1, 0.05)


def test_each_candidate_changes_only_the_ranking():
    for base in improve_rank.BASES:
        v = improve_rank.variants(base)
        assert v["current"] == improve.VARIANTS[base] and "rank" not in v["current"]
        for name in improve_rank.CANDIDATES + improve_rank.CONTEXT:
            assert v[name] == {**improve.VARIANTS[base], "rank": name}


def test_a_small_run_writes_results_and_a_report(tmp_path):
    out = str(tmp_path / "sim")
    improve_rank.run_sim(2, 13000, 1, out, "max,ovb", ["continuous_linear_n500"])
    df = pd.read_csv(f"{out}/continuous_linear_n500.csv")
    assert set(df.method) == {"current", "max", "ovb"} and sorted(df.seed.unique()) == [13000, 13001]
    assert df.att.notna().all() and np.allclose(df.err, df.att - df.att_true, atol=1e-12)
    with pytest.raises(SystemExit):                                          # same folder, different base: refused
        improve_rank.run_sim(2, 13000, 1, out, "max", ["continuous_linear_n500"], base="full")
    with pytest.raises(SystemExit):                                          # same folder, different seeds: refused
        improve_rank.run_sim(2, 14000, 1, out, "max", ["continuous_linear_n500"])
    improve_rank.run_ranks(1, 13000, 1, out=str(tmp_path / "ranks.csv"))
    r = pd.read_csv(tmp_path / "ranks.csv")
    assert set(r.score) == set(improve_rank.SCORES) and set(r.group) == set("CPIN")
    path = improve_rank.build(out, cdc_csv=str(tmp_path / "none.csv"), ranks_csv=str(tmp_path / "ranks.csv"), out=str(tmp_path / "R.md"))
    text = open(path).read()
    assert "## Verdict" in text and "Where each score puts each kind of variable" in text and "**current**" in text
