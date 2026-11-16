"""The final comparison: the adopted pipeline (the 5-split version, `repeats=5`) against the stored results of the final run.

Nothing of the other methods is run again. The final run (seeds 6000 to 6019, 12 settings x 20 data sets) is stored in
results/sim/*.csv (the methods written for this project) and results/package_sim.csv (the authors' FLAME and MALTS packages).
Here only the two corrected TIM pipelines are run again, on exactly the same 240 data sets, with the 5-split average:

    tim_3stage_bc_r5   screen, TIM, correction, two-half swap repeated 5 times, the 5 estimates averaged  (the adopted pipeline)
    tim_2stage_bc_r5   the same without screening

Seeds 6000 to 6019 were not used to pick the 5-split version (it was picked on seeds 9000 to 9019 and confirmed on 10000 to
10019), so this is a clean comparison, and because the data sets are the same ones the other methods saw, it is a paired one.
The first of the 5 splits is the split of the stored single-split run (same seed), so the two differ only by the 4 extra splits.

    python -m select_then_match.final sim        # results/final_v2/<setting>.csv       (about 40 to 60 minutes on 2 cores)
    python -m select_then_match.final cdc        # results/final_v2_cdc.csv             (the 10 CDC samples of 4,000 people)
    python -m select_then_match.final report     # results/FINAL.md

A finished setting is skipped on a re-run. Error = estimate minus the true effect on ALL treated people.
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

BAND = 0.05                                       # a ratio of RMSEs within 5% of 1 is called a tie
SEED0, REPS = 6000, 20                            # the seeds of the final run
NEW = {"tim_3stage_bc_r5": dict(screen=True, correct=True, repeats=5), "tim_2stage_bc_r5": dict(correct=True, repeats=5)}
TEN = ["tim_2stage", "tim_2stage_bc", "tim_3stage_bc", "psm", "robust", "naive", "dml", "tmle", "flame_pkg", "malts_pkg"]    # the final run
TEN_R5 = ["tim_3stage_bc_r5" if m == "tim_3stage_bc" else m for m in TEN]                                                   # with the new pipeline
RIVALS = ["psm", "robust", "dml", "tmle", "malts_pkg", "flame_pkg"]
NAMES = {"tim_2stage": "plain TIM", "tim_2stage_bc": "2-stage + correction", "tim_3stage_bc": "3-stage (single split)",
         "tim_3stage_bc_r5": "3-stage (5 splits)", "tim_2stage_bc_r5": "2-stage + correction (5 splits)"}


# ----------------------------------------------------------------------------- running
def _fit(d, seed):
    rows = []
    for name, kw in NEW.items():
        t0 = time.time()
        try:
            m = TIM(d.continuous, d.discrete, seed=seed, **kw).fit(d)
            rows.append(dict(method=name, att=m.att_, share=m.share_, vars_used=m.vars_used_, seconds=time.time() - t0))
        except Exception as ex:                                     # one failing fit must not stop a long run
            rows.append(dict(method=name, att=np.nan, share=np.nan, vars_used=np.nan, seconds=time.time() - t0, error=str(ex)[:100]))
    return rows


def _sim_job(args):
    setting, seed = args
    with threadpool_limits(1):
        d = make_sim(setting, seed)
        return [dict(setting=setting, seed=seed, att_true=d.att, **r) for r in _fit(d, seed)]


def _cdc_job(args):
    path, n, rep = args
    with threadpool_limits(1):
        d = load_cdc(path).subsample(n, rep)
        return [dict(rep=rep, n=len(d.y), **r) for r in _fit(d, rep)]


def _map(fn, jobs, workers):
    if workers <= 1:
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(workers) as ex:
        yield from ex.map(fn, jobs, chunksize=1)


def run_sim(reps=REPS, seed0=SEED0, workers=2, outdir="results/final_v2", settings=None, force=False):
    os.makedirs(outdir, exist_ok=True)
    seeds, todo = list(range(seed0, seed0 + reps)), []
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
    for rows in _map(_sim_job, [(s, seed) for s in todo for seed in seeds], workers):
        s = rows[0]["setting"]
        buf.setdefault(s, []).extend(rows)
        if len({r["seed"] for r in buf[s]}) == reps:
            df = pd.DataFrame(buf.pop(s))
            df["err"] = df.att - df.att_true
            df.to_csv(f"{outdir}/{s}.csv", index=False)
            print(f"done {s}  ({time.time() - t0:.0f}s)", flush=True)


def run_cdc(path, n=4000, reps=10, workers=2, out="results/final_v2_cdc.csv"):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows = [r for rs in _map(_cdc_job, [(path, n, rep) for rep in range(reps)], workers) for r in rs]
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


# ----------------------------------------------------------------------------- the report
def _rmse(df):
    ok = df[df.att.notna()]
    return ok.groupby(["setting", "method"]).err.apply(lambda e: float(np.sqrt(np.mean(e ** 2)))).unstack("method").reindex(list(SETTINGS)).dropna(how="all")


def load_all(simdir="results/sim", pkgcsv="results/package_sim.csv", newdir="results/final_v2"):
    """The stored final run (the ten methods) plus the new rows, on the seeds the new rows were run on."""
    new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{newdir}/*.csv"))], ignore_index=True)
    stored = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{simdir}/*.csv"))], ignore_index=True)
    stored = stored[stored.method.isin(TEN)]
    if os.path.exists(pkgcsv):
        pk = pd.read_csv(pkgcsv)
        pk["err"] = pk.att - pk.att_true
        stored = pd.concat([stored, pk], ignore_index=True)
    key = new[["setting", "seed"]].drop_duplicates()
    stored = stored.merge(key, on=["setting", "seed"])               # the same data sets only
    return pd.concat([stored, new], ignore_index=True), stored, new


def _ranks(rmse, methods):
    cols = [m for m in methods if m in rmse.columns]
    r = rmse[cols].dropna()
    return r.rank(axis=1).mean(), len(r)


def _compare(rmse, a, b):
    r = (rmse[a] / rmse[b]).dropna()
    return dict(median_ratio=float(r.median()), better=int((r < 1 - BAND).sum()), tie=int(((r >= 1 - BAND) & (r <= 1 + BAND)).sum()), worse=int((r > 1 + BAND).sum()))


def build(simdir="results/sim", pkgcsv="results/package_sim.csv", newdir="results/final_v2", cdc_old="results/cdc.csv",
          cdc_pkg="results/package_cdc.csv", cdc_new="results/final_v2_cdc.csv", out="results/FINAL.md"):
    df, stored, new = load_all(simdir, pkgcsv, newdir)
    rmse = _rmse(df)
    bias = df[df.att.notna()].groupby(["setting", "method"]).err.mean().unstack("method").reindex(rmse.index)
    reps = new.groupby("setting").seed.nunique()
    r_old, n_old = _ranks(rmse, TEN)
    r_new, n_new = _ranks(rmse, TEN_R5)
    parts = ["# Final comparison: the adopted pipeline against the other methods\n",
             f"{new.setting.nunique()} simulated settings, {int(reps.min()) if reps.min() == reps.max() else f'{int(reps.min())} to {int(reps.max())}'} data sets each, "
             f"seeds {int(new.seed.min())} to {int(new.seed.max())}: the SAME data sets as the final run. The results of the other methods are the stored ones "
             "(they were not run again); only the two 5-split TIM rows are new. Error = estimate minus the true effect on ALL treated people. "
             "`dml` fits one constant effect, which differs from the effect on the treated in the continuous settings (by about 0.3), so part of its error "
             "is a different target. FLAME and MALTS are the authors' packages at their default settings.\n"]
    rows = []
    for m in sorted(set(TEN) | set(TEN_R5), key=lambda m: (r_new.get(m, 99), r_old.get(m, 99))):
        rows.append({"method": NAMES.get(m, m), "average RMSE rank, the ten of the final run": r_old.get(m, np.nan),
                     "average RMSE rank, the ten with the 5-split pipeline": r_new.get(m, np.nan)})
    parts.append(f"## Average rank (1 = lowest RMSE in a setting, averaged over {n_new} settings)\n\nFirst column: the ten methods of the final run, as published. "
                 "Second column: the same ten, with the 3-stage pipeline replaced by its 5-split version. Ranks are only comparable within a column.\n\n"
                 + to_md(pd.DataFrame(rows), index=False) + "\n")
    cols = [m for m in ["tim_2stage", "tim_2stage_bc", "tim_2stage_bc_r5", "tim_3stage_bc", "tim_3stage_bc_r5", "psm", "robust", "dml", "tmle", "malts_pkg", "flame_pkg", "naive"] if m in rmse.columns]
    parts.append("## RMSE by setting (lower is better)\n\n" + to_md(rmse[cols]) + "\n")
    parts.append("## Average error (bias; positive = too high)\n\n" + to_md(bias[cols]) + "\n")
    qs = []
    def add(question, a, b):
        if a in rmse and b in rmse:
            c = _compare(rmse, a, b)
            qs.append({"question": question, "compared": f"{a} / {b}", "median RMSE ratio": c["median_ratio"], "a better": c["better"], "tie": c["tie"], "a worse": c["worse"]})
    add("Does repeating the split 5 times help the 3-stage pipeline (same data sets)?", "tim_3stage_bc_r5", "tim_3stage_bc")
    add("Does it help the 2-stage pipeline with correction?", "tim_2stage_bc_r5", "tim_2stage_bc")
    add("Does screening add anything once the correction and the 5 splits are on?", "tim_3stage_bc_r5", "tim_2stage_bc_r5")
    add("5-split 3-stage pipeline against plain TIM", "tim_3stage_bc_r5", "tim_2stage")
    for r in RIVALS:
        add(f"5-split 3-stage pipeline against {r}", "tim_3stage_bc_r5", r)
    for r in RIVALS:
        add(f"single-split 3-stage pipeline against {r} (the final run)", "tim_3stage_bc", r)
    parts.append(f"## The questions\n\n`median RMSE ratio` is the ratio of the first method's RMSE to the second's, setting by setting, then the median (below 1: the first is "
                 f"better). `a better` / `tie` / `a worse` count settings where the ratio is below {1 - BAND:.2f}, within {BAND:.0%} of 1, or above {1 + BAND:.2f}. These are "
                 "counts of settings, not tests: the settings are variants of four base cases and are not independent.\n\n" + to_md(pd.DataFrame(qs), index=False) + "\n")
    ok = df[df.att.notna()]
    summ = pd.DataFrame({"treated used": ok.groupby("method").share.mean(), "seconds per data set": df.groupby("method").seconds.mean()}).reindex(cols)
    summ.index.name = "method"
    parts.append("## Treated people used and time\n\n" + to_md(summ) + "\n")
    bad = new[new.att.isna()]
    if len(bad):
        parts.append("No estimate or a failed fit, left out above: " + ", ".join(f"`{m}` ({k} of {int((new.method == m).sum())})" for m, k in bad.groupby("method").size().items()) + ".\n")
    if os.path.exists(cdc_new):
        c = pd.concat([pd.read_csv(p) for p in (cdc_old, cdc_pkg, cdc_new) if os.path.exists(p)], ignore_index=True)
        c = c[c.method.isin(TEN + list(NEW))]
        g = c[c.att.notna()].groupby("method")
        order = [m for m in ["tim_2stage", "tim_2stage_bc", "tim_2stage_bc_r5", "tim_3stage_bc", "tim_3stage_bc_r5", "psm", "robust", "dml", "tmle", "flame_pkg", "naive"] if m in g.groups]
        t = pd.DataFrame({"effect (mean over samples)": g.att.mean().loc[order].values, "sd across samples": g.att.std().loc[order].values,
                          "treated used": g.share.mean().loc[order].values, "samples": g.size().loc[order].values}, index=order)
        t.index.name = "method"
        parts.append(f"## CDC: high cholesterol and diabetes ({int(c.n.iloc[0]):,} people per sample, {c.rep.nunique()} samples)\n\nNo true effect is known. The TIM paper reports 0.079. "
                     "These are adjusted associations in the released sample (no survey weights), and every method assumes the 20 covariates capture all confounding. "
                     "The MALTS package was not run on CDC (too slow).\n\n" + to_md(t, floatfmt="{:.4f}") + "\n")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(parts))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["sim", "cdc", "report"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--n", type=int, default=4000, help="CDC sample size")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--settings", default=None, help="comma separated names, default all twelve")
    ap.add_argument("--path", default=None)
    ap.add_argument("--outdir", default="results/final_v2")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.which == "sim":
        run_sim(a.reps or REPS, a.seed0, a.workers, a.outdir, a.settings.split(",") if a.settings else None, a.force)
    elif a.which == "cdc":
        path = a.path or (sorted(glob.glob("data/*health_indicators_BRFSS2015.csv")) or [None])[0]
        if path is None:
            raise SystemExit("CDC file not found: put the BRFSS 2015 csv in data/ (see data/README.md)")
        run_cdc(path, a.n, a.reps or 10, a.workers)
    else:
        print("wrote", build(newdir=a.outdir))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
