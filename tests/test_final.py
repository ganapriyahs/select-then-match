"""Tests for the final comparison (final.py). Run:  python -m pytest -q tests/test_final.py"""
import os

import numpy as np
import pandas as pd
import pytest

from select_then_match import final
from select_then_match.data import SETTINGS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORED = os.path.join(HERE, "results", "sim")
PKG = os.path.join(HERE, "results", "package_sim.csv")


@pytest.mark.skipif(not (os.path.isdir(STORED) and os.path.exists(PKG)), reason="the stored final run is not in results/")
def test_the_stored_final_run_gives_the_published_ranking():
    """Whatever final.py does with the new rows, the stored rows must give the ranks of the final run (seeds 6000 to 6019)."""
    stored = pd.concat([pd.read_csv(os.path.join(STORED, f)) for f in sorted(os.listdir(STORED)) if f.endswith(".csv")], ignore_index=True)
    stored = stored[stored.method.isin(final.TEN)]
    pk = pd.read_csv(PKG)
    pk["err"] = pk.att - pk.att_true
    rmse = final._rmse(pd.concat([stored, pk], ignore_index=True))
    r, n = final._ranks(rmse, final.TEN)
    assert n == 12
    assert r["dml"] == pytest.approx(2.00, abs=0.005) and r["tim_3stage_bc"] == pytest.approx(5.17, abs=0.005)
    assert r["naive"] == pytest.approx(10.00, abs=0.005) and r["malts_pkg"] == pytest.approx(3.75, abs=0.005)


def _fake(tmp_path, bonus=0.5):
    """A tiny stored run (2 settings x 3 seeds, all the methods) and new rows; the new 3-stage rows are `bonus` times the old error."""
    rng = np.random.default_rng(0)
    rows = []
    for s in list(SETTINGS)[:2]:
        for seed in (1, 2, 3):
            for k, m in enumerate(final.TEN):
                att_true = 0.2
                att = att_true + rng.normal() * (0.05 + 0.02 * k)
                rows.append(dict(setting=s, seed=seed, att_true=att_true, method=m, att=att, share=0.9, vars_used=4.0, seconds=1.0, err=att - att_true))
    allrows = pd.DataFrame(rows)
    (tmp_path / "sim").mkdir()
    own = allrows[~allrows.method.isin(["flame_pkg", "malts_pkg"])]
    for s, g in own.groupby("setting"):
        g.to_csv(tmp_path / "sim" / f"{s}.csv", index=False)
    allrows[allrows.method.isin(["flame_pkg", "malts_pkg"])].drop(columns=["err", "vars_used"]).to_csv(tmp_path / "package_sim.csv", index=False)
    new = []
    for name in final.NEW:
        h = allrows[allrows.method == "tim_3stage_bc"].copy()
        h["method"], h["err"] = name, h.err * bonus
        h["att"] = h.att_true + h.err
        new.append(h)
    (tmp_path / "v2").mkdir()
    pd.concat(new).to_csv(tmp_path / "v2" / "all.csv", index=False)


def test_report_builds_and_puts_a_better_pipeline_higher(tmp_path):
    _fake(tmp_path, bonus=0.2)
    out = final.build(str(tmp_path / "sim"), str(tmp_path / "package_sim.csv"), str(tmp_path / "v2"), str(tmp_path / "x.csv"), str(tmp_path / "y.csv"),
                      str(tmp_path / "z.csv"), str(tmp_path / "FINAL.md"))
    text = open(out).read()
    assert "Average rank" in text and "## The questions" in text and "3-stage (5 splits)" in text
    df, stored, new = final.load_all(str(tmp_path / "sim"), str(tmp_path / "package_sim.csv"), str(tmp_path / "v2"))
    r, n = final._ranks(final._rmse(df), final.TEN_R5)
    assert n == 2 and r["tim_3stage_bc_r5"] < 2.0                          # error is 5 times smaller than the single split: near the top
    r_old, _ = final._ranks(final._rmse(df), final.TEN)
    assert "tim_3stage_bc_r5" not in r_old.index and len(r_old) == 10


def test_only_the_data_sets_of_the_new_rows_are_compared(tmp_path):
    _fake(tmp_path)
    new = pd.read_csv(tmp_path / "v2" / "all.csv")
    new = new[new.seed != 3]
    new.to_csv(tmp_path / "v2" / "all.csv", index=False)
    df, stored, _ = final.load_all(str(tmp_path / "sim"), str(tmp_path / "package_sim.csv"), str(tmp_path / "v2"))
    assert set(stored.seed) == {1, 2} and set(df.seed) == {1, 2}


def test_a_small_run_writes_results_and_refuses_other_seeds(tmp_path):
    out = str(tmp_path / "v2")
    final.run_sim(2, 15000, 1, out, ["continuous_linear_n500"])
    df = pd.read_csv(f"{out}/continuous_linear_n500.csv")
    assert set(df.method) == set(final.NEW) and sorted(df.seed.unique()) == [15000, 15001]
    assert df.att.notna().all() and np.allclose(df.err, df.att - df.att_true, atol=1e-12)
    with pytest.raises(SystemExit):
        final.run_sim(2, 16000, 1, out, ["continuous_linear_n500"])
