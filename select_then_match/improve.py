"""Can the 3-stage pipeline be made stronger? Three upgrades, tested on NEW data with the rule written BEFORE the run.

The pipeline being tested (`current`) is tim_3stage_bc of the final run: screen, TIM, bias correction. Nothing about it
changes. Each candidate changes ONE thing (see tim.py for the details):

    full      no two-half swap: learn on everybody and match everybody (matching pool twice as large), with a
              cross-fitted correction model;
    repeat5   the two-half swap repeated 5 times with different random splits, the 5 estimates averaged;
    mu0avg    the correction model is the average of the boosting model and a linear model.

Two more rows are shown for context only and are not part of the rule: `plain_tim` (TIM as designed) and `tim_bc`
(TIM + correction, without screening).

THE RULE (fixed before the run; it is not changed after seeing results)
    Data: the 12 simulated settings, 20 repeats each, seeds 9000 to 9019 (never used before). Error = estimate minus the
    true effect on ALL treated people. For every setting, ratio = RMSE of the candidate / RMSE of `current`.
    A candidate PASSES only if all three hold:
        1. the median ratio over the 12 settings is at most 0.95;
        2. the ratio is below 0.95 in at least 8 of the 12 settings;
        3. the ratio is above 1.05 in at most 1 of the 12 settings.
    Three candidates are looked at, so a pass is not yet an adoption. A candidate that passes is run again, unchanged, on
    seeds 10000 to 10019 (`--seed0 10000`); it is adopted only if it passes again. A candidate that does not pass is reported as
    "no reliable gain". All three are reported either way.
    The counts are counts of settings, not tests; the settings are variants of four base cases and are not independent.
    The paired z column is information only.
    Seeds 7000 to 7002 were used once, on 3 settings, only to check that the code runs; they are not used for any result.

Run (from the project folder; about 45 to 90 minutes on 2 cores):
    python -m select_then_match.improve sim --seed0 9000 --reps 20
    python -m select_then_match.improve report                              # results/IMPROVE.md
Confirmation run, only for a candidate that passed (here `full`):
    python -m select_then_match.improve sim --seed0 10000 --reps 20 --variants full --outdir results/improve_confirm
    python -m select_then_match.improve report --simdir results/improve_confirm --out results/IMPROVE_CONFIRM.md
Real data (optional, needs the CDC csv in data/):
    python -m select_then_match.improve cdc --n 4000 --reps 10
"""
import argparse
import glob
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import to_md
from .data import SETTINGS, load_cdc, make_sim
from .tim import TIM

BAND = 0.05
MEDIAN_MAX, BETTER_MIN, WORSE_MAX = 0.95, 8, 1            # the rule

VARIANTS = {
    "current": dict(screen=True, correct=True),
    "full": dict(screen=True, correct=True, full=True),
    "repeat5": dict(screen=True, correct=True, repeats=5),
    "mu0avg": dict(screen=True, correct=True, mu0="avg"),
    "plain_tim": dict(),
    "tim_bc": dict(correct=True),
}
CANDIDATES = ["full", "repeat5", "mu0avg"]
CONTEXT = ["plain_tim", "tim_bc"]
WHAT = {
    "current": "the final pipeline: screen, TIM, correction (two-half swap, boosting correction)",
    "full": "no swap: learn on everybody, match everybody, cross-fitted correction",
    "repeat5": "two-half swap repeated 5 times, estimates averaged",
    "mu0avg": "correction model = average of boosting and a linear model",
    "plain_tim": "context: TIM as designed",
    "tim_bc": "context: TIM + correction, no screening",
}


def run_variants(d, seed, names):
    rows = []
    for name in names:
        t0 = time.time()
        try:
            m = TIM(d.continuous, d.discrete, seed=seed, **VARIANTS[name]).fit(d)
            rows.append(dict(method=name, att=m.att_, share=m.share_, vars_used=m.vars_used_, seconds=time.time() - t0))
        except Exception as ex:                                     # one failing fit must not stop a long run
            rows.append(dict(method=name, att=np.nan, share=np.nan, vars_used=np.nan, seconds=time.time() - t0, error=str(ex)[:100]))
    return rows


def _sim_job(args):
    setting, seed, names = args
    with threadpool_limits(1):
        d = make_sim(setting, seed)
        return [dict(setting=setting, seed=seed, att_true=d.att, **r) for r in run_variants(d, seed, names)]


def _cdc_job(args):
    path, n, rep, names = args
    with threadpool_limits(1):
        d = load_cdc(path).subsample(n, rep)
        return [dict(rep=rep, n=len(d.y), **r) for r in run_variants(d, rep, names)]


def _map(fn, jobs, workers):
    if workers <= 1:
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(workers) as ex:
        yield from ex.map(fn, jobs, chunksize=1)


def _names(variants):
    names = ["current"] + [v for v in (variants.split(",") if variants else CANDIDATES + CONTEXT) if v != "current"]
    bad = [v for v in names if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; choose from {list(VARIANTS)}")
    return names


def run_sim(reps, seed0, workers, outdir, variants=None, settings=None, force=False):
    os.makedirs(outdir, exist_ok=True)
    names, seeds, todo = _names(variants), list(range(seed0, seed0 + reps)), []
    for s in (settings or list(SETTINGS)):
        path = f"{outdir}/{s}.csv"
        if os.path.exists(path) and not force:
            have = sorted(pd.read_csv(path).seed.unique())
            if have != seeds:
                raise SystemExit(f"{path} holds seeds {have[0]}..{have[-1]} ({len(have)}), not {seeds[0]}..{seeds[-1]}: use --force to redo it")
            print("skip (done)", s, flush=True)
            continue
        todo.append(s)
    buf, t0 = {}, time.time()
    for rows in _map(_sim_job, [(s, seed, names) for s in todo for seed in seeds], workers):
        s = rows[0]["setting"]
        buf.setdefault(s, []).extend(rows)
        if len({r["seed"] for r in buf[s]}) == reps:
            df = pd.DataFrame(buf.pop(s))
            df["err"] = df.att - df.att_true
            df.to_csv(f"{outdir}/{s}.csv", index=False)
            print(f"done {s}  ({time.time() - t0:.0f}s)", flush=True)


def run_cdc(path, n, reps, workers, variants=None, out="results/improve_cdc.csv"):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    names = _names(variants)
    rows = [r for rs in _map(_cdc_job, [(path, n, rep, names) for rep in range(reps)], workers) for r in rs]
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


# ----------------------------------------------------------------------------- the rule and the report
def judge(ratio):
    """ratio: RMSE(candidate) / RMSE(current) per setting. Returns the counts and the verdict of the pre-declared rule."""
    r = pd.Series(ratio).dropna()
    med, better, worse = float(r.median()), int((r < 1 - BAND).sum()), int((r > 1 + BAND).sum())
    tie = int(len(r) - better - worse)
    passed = len(r) == len(SETTINGS) and med <= MEDIAN_MAX and better >= BETTER_MIN and worse <= WORSE_MAX
    return dict(median_ratio=med, better=better, tie=tie, worse=worse, verdict="PASS" if passed else "no reliable gain")


def _paired_z(df, cand):
    """Settings where the candidate is clearly better / clearly worse than `current` on the same data sets (paired z of
    the absolute errors, |z| > 2). Information only."""
    up = down = 0
    for _, g in df.groupby("setting"):
        a = g[g.method == "current"].set_index("seed").err.abs()
        b = g[g.method == cand].set_index("seed").err.abs()
        d = (a - b).dropna()
        if len(d) > 2 and d.std(ddof=1) > 0:
            z = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            up += z > 2
            down += z < -2
    return int(up), int(down)


def build(simdir="results/improve", cdc_csv="results/improve_cdc.csv", out="results/IMPROVE.md"):
    files = sorted(glob.glob(f"{simdir}/*.csv"))
    if not files:
        raise SystemExit(f"no result files in {simdir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ok = df[df.att.notna()]
    rmse = ok.groupby(["setting", "method"]).err.apply(lambda e: float(np.sqrt(np.mean(e ** 2)))).unstack("method").reindex(list(SETTINGS)).dropna(how="all")
    bias = ok.groupby(["setting", "method"]).err.mean().unstack("method").reindex(rmse.index)
    cols = [c for c in ["current"] + CANDIDATES + CONTEXT if c in rmse.columns]
    cands = [c for c in CANDIDATES if c in rmse.columns]
    ratio = pd.DataFrame({c: rmse[c] / rmse["current"] for c in cands})
    reps, seeds = df.groupby("setting").seed.nunique(), (int(df.seed.min()), int(df.seed.max()))
    parts = [f"# Can the 3-stage pipeline be made stronger?\n",
             f"{df.setting.nunique()} simulated settings, {int(reps.min()) if reps.min() == reps.max() else f'{int(reps.min())} to {int(reps.max())}'} repeats each, "
             f"seeds {seeds[0]} to {seeds[1]}. Error = estimate minus the true effect on ALL treated people. "
             f"`current` is the final pipeline (screen, TIM, correction). Each candidate changes one thing.\n",
             "## The rule (written before the run)\n\n"
             f"A candidate PASSES only if, against `current`, (1) the median RMSE ratio over the 12 settings is at most {MEDIAN_MAX}, "
             f"(2) the ratio is below {1 - BAND:.2f} in at least {BETTER_MIN} settings, and (3) it is above {1 + BAND:.2f} in at most {WORSE_MAX}. "
             "A pass on seeds 9000 to 9019 is not an adoption: the candidate is run again, unchanged, on seeds 10000 to 10019 and adopted only if it "
             "passes again. The counts are counts of settings, not tests.\n",
             "## The versions\n\n" + to_md(pd.DataFrame({"version": cols, "what it is": [WHAT[c] for c in cols]}), index=False) + "\n"]
    rows = []
    for c in cands:
        j = judge(ratio[c])
        up, down = _paired_z(ok, c)
        rows.append({"candidate": c, "median ratio": j["median_ratio"], "better": j["better"], "tie": j["tie"], "worse": j["worse"],
                     "clearly better / worse (paired z, info)": f"{up} / {down}", "verdict": j["verdict"]})
    parts.append("## Verdict\n\n`ratio` = RMSE of the candidate / RMSE of `current`; below 1 means the candidate is better.\n\n" + to_md(pd.DataFrame(rows), index=False) + "\n")
    parts.append("## RMSE by setting (lower is better)\n\n" + to_md(rmse[cols]) + "\n")
    parts.append("## Ratio to `current` by setting\n\n" + to_md(ratio) + "\n")
    parts.append("## Average error (bias; positive = too high)\n\n" + to_md(bias[cols]) + "\n")
    summ = pd.DataFrame({"treated used": ok.groupby("method").share.mean(), "variables used per treated person": ok.groupby("method").vars_used.mean(),
                         "seconds per data set": df.groupby("method").seconds.mean()}).reindex(cols)
    summ.index.name = "version"
    parts.append("## Cost\n\n" + to_md(summ) + "\n")
    bad = df[df.att.isna()]
    if len(bad):
        parts.append("No estimate or a failed fit, left out above: " + ", ".join(f"`{m}` ({k} of {int((df.method == m).sum())})" for m, k in bad.groupby("method").size().items()) + ".\n")
    if os.path.exists(cdc_csv):
        c = pd.read_csv(cdc_csv)
        g = c[c.att.notna()].groupby("method")
        t = pd.DataFrame({"effect (mean over samples)": g.att.mean(), "sd across samples": g.att.std(), "treated used": g.share.mean(), "samples": g.size()})
        t = t.reindex([m for m in ["current"] + CANDIDATES + CONTEXT if m in t.index])
        t.index.name = "version"
        parts.append(f"## CDC: high cholesterol and diabetes ({int(c.n.iloc[0]):,} people per sample, {c.rep.nunique()} samples)\n\nNo true effect is known; "
                     "this only shows whether a version moves the answer.\n\n" + to_md(t, floatfmt="{:.4f}") + "\n")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["sim", "cdc", "report"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--variants", default=None, help="comma separated, default all (current is always included)")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--outdir", default="results/improve")
    ap.add_argument("--simdir", default="results/improve")
    ap.add_argument("--out", default="results/IMPROVE.md")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.which == "sim":
        run_sim(a.reps or 20, a.seed0, a.workers, a.outdir, a.variants, a.settings.split(",") if a.settings else None, a.force)
    elif a.which == "cdc":
        path = a.path or (sorted(glob.glob("data/*health_indicators_BRFSS2015.csv")) or [None])[0]
        if path is None:
            raise SystemExit("CDC file not found: put the BRFSS 2015 csv in data/ (see data/README.md)")
        run_cdc(path, a.n, a.reps or 10, a.workers, a.variants)
    else:
        print("wrote", build(a.simdir, out=a.out))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
