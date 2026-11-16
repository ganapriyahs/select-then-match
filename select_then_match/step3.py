"""Step 3: the ten methods on simulated data (truth known) and on the CDC data (no truth), and the report.

    python -m select_then_match.step3 sim    --reps 20 --seed0 5000      # results/sim/<setting>.csv
    python -m select_then_match.step3 cdc    --n 4000 --reps 10          # results/cdc.csv
    python -m select_then_match.step3 report                             # results/RESULTS.md

Error = estimate minus the true effect on ALL treated people (so a method that throws treated people away is not
rewarded for it). RMSE = the typical size of that error over the repeats. A failed fit is recorded in the `error` column
and does not stop the run. A finished setting is skipped on a re-run; use --force to run it again.
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
from .methods import GROUP, METHODS, WHAT

BAND = 0.05     # a ratio of RMSEs within 5% of 1 is called a tie


def run_methods(d, seed):
    """One data set, every method. Returns a list of rows."""
    cache, rows = {}, []
    for name, fn in METHODS.items():
        t0 = time.time()
        try:
            r = fn(d, seed, cache)
            rows.append(dict(method=name, att=r["att"], share=r["share"], vars_used=r.get("vars_used", np.nan), seconds=time.time() - t0))
        except Exception as ex:                                   # one failing fit must not stop a long run
            rows.append(dict(method=name, att=np.nan, share=np.nan, seconds=time.time() - t0, error=str(ex)[:100]))
    return rows


def _sim_job(args):
    setting, seed = args
    with threadpool_limits(1):
        d = make_sim(setting, seed)
        return [dict(setting=setting, seed=seed, att_true=d.att, **r) for r in run_methods(d, seed)]


def _cdc_job(args):
    path, n, rep = args
    with threadpool_limits(1):
        d = load_cdc(path).subsample(n, rep)
        return [dict(rep=rep, n=len(d.y), **r) for r in run_methods(d, rep)]


def _map(fn, jobs, workers):
    if workers <= 1:
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(workers) as ex:
        yield from ex.map(fn, jobs, chunksize=1)


def run_sim(reps, seed0, workers, outdir="results/sim", settings=None, force=False):
    os.makedirs(outdir, exist_ok=True)
    seeds = list(range(seed0, seed0 + reps))
    todo = []
    for s in (settings or list(SETTINGS)):
        path = f"{outdir}/{s}.csv"
        if os.path.exists(path) and not force:
            have = sorted(pd.read_csv(path).seed.unique())
            if have != seeds:
                raise SystemExit(f"{path} holds seeds {have[0]}..{have[-1]} ({len(have)}), not {seeds[0]}..{seeds[-1]}: use --force to redo it")
            print("skip (done)", s, flush=True)
            continue
        todo.append(s)
    jobs = [(s, seed) for s in todo for seed in seeds]
    buf, t0 = {}, time.time()
    for rows in _map(_sim_job, jobs, workers):
        s = rows[0]["setting"]
        buf.setdefault(s, []).extend(rows)
        if len({r["seed"] for r in buf[s]}) == reps:
            df = pd.DataFrame(buf.pop(s))
            df["err"] = df.att - df.att_true
            df.to_csv(f"{outdir}/{s}.csv", index=False)
            print(f"done {s}  ({time.time() - t0:.0f}s)", flush=True)


def run_cdc(path, n, reps, workers, out="results/cdc.csv"):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows = [r for rs in _map(_cdc_job, [(path, n, rep) for rep in range(reps)], workers) for r in rs]
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


# ----------------------------------------------------------------------------- report
def _compare(rmse, a, b):
    """Setting by setting RMSE ratio of method a to method b (below 1: a is better)."""
    r = rmse[a] / rmse[b]
    return dict(median_ratio=float(r.median()), better=int((r < 1 - BAND).sum()), tie=int(((r >= 1 - BAND) & (r <= 1 + BAND)).sum()),
                worse=int((r > 1 + BAND).sum()))


def _sim_tables(df):
    ok = df[df.att.notna()]
    rmse = ok.groupby(["setting", "method"]).err.apply(lambda e: float(np.sqrt(np.mean(e ** 2)))).unstack("method")
    bias = ok.groupby(["setting", "method"]).err.mean().unstack("method")
    order = [m for m in METHODS if m in rmse.columns]
    rmse, bias = rmse[order].reindex(list(SETTINGS)).dropna(how="all"), bias[order].reindex(list(SETTINGS)).dropna(how="all")
    summary = pd.DataFrame({"group": [GROUP[m] for m in order], "average RMSE rank": rmse.rank(axis=1).mean().loc[order].values,
                            "worst RMSE / best": (rmse.div(rmse.min(axis=1), axis=0)).max().loc[order].values,
                            "treated used": ok.groupby("method").share.mean().loc[order].values,
                            "seconds per data set": df.groupby("method").seconds.mean().loc[order].values}, index=order)
    summary.index.name = "method"
    return rmse, bias, summary


def _questions(rmse):
    rows = []
    def add(question, a, b):
        if a in rmse and b in rmse:
            c = _compare(rmse, a, b)
            rows.append({"question": question, "compared": f"{a} / {b}", "median RMSE ratio": c["median_ratio"],
                         "a better": c["better"], "tie": c["tie"], "a worse": c["worse"]})
    add("Does the correction help TIM?", "tim_2stage_bc", "tim_2stage")
    add("Is the 3-stage pipeline (screen, match, correct) better than TIM?", "tim_3stage_bc", "tim_2stage")
    add("Does screening add anything once the correction is on?", "tim_3stage_bc", "tim_2stage_bc")
    for r in ("psm", "robust", "flame", "malts", "dml", "tmle"):
        add(f"Corrected TIM against {r}", "tim_2stage_bc", r)
    for r in ("psm", "robust", "flame", "malts", "dml", "tmle"):
        add(f"3-stage pipeline against {r}", "tim_3stage_bc", r)
    return pd.DataFrame(rows)


def build(simdir="results/sim", cdc_csv="results/cdc.csv", out="results/RESULTS.md"):
    parts = ["# Step 3 results\n",
             "Ten methods, on simulated data (truth known) and on CDC (truth unknown). Error = estimate minus the true effect on ALL "
             "treated people. `dml` fits one constant effect, which differs from the effect on the treated when the effect differs "
             "between people (in the continuous settings by about 0.3), so part of its error is a different target. `psm`, `robust`, "
             "`flame`, `malts` and TIM answer for the treated people they use (see `treated used`). `flame` and `malts` are compact "
             "re-implementations written for this project, not the authors' packages.\n",
             "## The methods\n\n" + to_md(pd.DataFrame({"method": list(WHAT), "group": [GROUP[m] for m in WHAT], "what it is": list(WHAT.values())}), index=False) + "\n"]
    files = sorted(glob.glob(f"{simdir}/*.csv"))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        rmse, bias, summary = _sim_tables(df)
        reps = df.groupby("setting").seed.nunique()
        parts.append(f"## 1. Simulated data: {df.setting.nunique()} settings, {int(reps.min()) if reps.min() == reps.max() else f"{int(reps.min())} to {int(reps.max())}"} repeats each, seeds {int(df.seed.min())} to {int(df.seed.max())}\n")
        parts.append("### RMSE (lower is better)\n\n" + to_md(rmse) + "\n")
        parts.append("### Average error (bias; positive = too high)\n\n" + to_md(bias) + "\n")
        parts.append("### Summary\n\nRank 1 = lowest RMSE in a setting; the rank is averaged over settings. `treated used` is the share of treated "
                     "people the estimate is based on.\n\n" + to_md(summary) + "\n")
        parts.append(f"### The questions\n\n`median RMSE ratio` is the ratio of the first method's RMSE to the second's, setting by setting, then the median "
                     f"(below 1: the first is better). `a better` / `tie` / `a worse` count settings where the ratio is below {1 - BAND:.2f}, within "
                     f"{BAND:.0%} of 1, or above {1 + BAND:.2f}. These are counts of settings, not tests: the settings are variants of four base cases "
                     "and are not independent.\n\n" + to_md(_questions(rmse), index=False) + "\n")
        none = df[df.att.isna()]
        if len(none):
            parts.append("No estimate (no treated person could be used) or a failed fit, left out above: " +
                         ", ".join(f"`{m}` ({k} of {int((df.method == m).sum())})" for m, k in none.groupby("method").size().items()) + ".\n")
    if os.path.exists(cdc_csv):
        c = pd.read_csv(cdc_csv)
        g = c[c.att.notna()].groupby("method")
        order = [m for m in METHODS if m in g.groups]
        t = pd.DataFrame({"group": [GROUP[m] for m in order], "effect (mean over samples)": g.att.mean().loc[order].values,
                          "sd across samples": g.att.std().loc[order].values, "treated used": g.share.mean().loc[order].values,
                          "samples": g.size().loc[order].values}, index=order)
        t.index.name = "method"
        parts.append(f"## 2. CDC: high cholesterol and diabetes ({int(c.n.iloc[0]):,} people per sample, {c.rep.nunique()} samples)\n\n"
                     "No true effect is known. The TIM paper reports 0.079. Look at how close the methods are and how much each moves between "
                     "samples. These are adjusted associations in the released sample (no survey weights), and every method assumes the 20 "
                     "covariates capture all confounding.\n\n" + to_md(t, floatfmt="{:.4f}") + "\n")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["sim", "cdc", "report"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--n", type=int, default=4000, help="CDC sample size")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--settings", default=None, help="comma separated names, default all twelve")
    ap.add_argument("--path", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.which == "sim":
        run_sim(a.reps or 20, a.seed0, a.workers, settings=a.settings.split(",") if a.settings else None, force=a.force)
    elif a.which == "cdc":
        path = a.path or (sorted(glob.glob("data/*health_indicators_BRFSS2015.csv")) or [None])[0]
        if path is None:
            raise SystemExit("CDC file not found: put the BRFSS 2015 csv in data/ (see data/README.md)")
        run_cdc(path, a.n, a.reps or 10, a.workers)
    else:
        print("wrote", build())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
