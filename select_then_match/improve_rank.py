"""Does a different Step 1 make the pipeline stronger? Step 1 of TIM is the importance score that decides which variable is
dropped first when exact matches run out. TIM's own score is the average of a linear (outcome) and a logistic (treatment)
regression score. Four other scores are tested on NEW data with the rule written BEFORE the run.

Only the score changes. Screening, bins, matching, the correction and everything else stay as in `current`. Each candidate
changes ONE thing (details in tim.py, function rank_scores):

    ridge_raw   a close version of the public TIM code: ridge on the raw columns for both models, scores added. Depends on the
                units of the columns. (A first look at this idea on an older version of our code, seeds 6000 to 6019, gave
                0.83 with the correction; the cause is not known and it was never repeated on new seeds. This is that repeat.
                It is a close version of the public code, not the same code.)
    out_only    only the outcome score (does the variable help to predict the outcome?)
    max         the larger of the outcome score and the treatment score
    ovb         |outcome coefficient| x |gap between treated and untreated people in the variable's average|

One more row is shown for context only and is not part of the rule: `trt_only` (only the treatment score).
Geometric mean, forest and permutation importance were tested before (error moved by less than about 6%), so they are not
repeated here.

THE BASE. `current` is the final pipeline (screen, TIM, correction, two-half swap), and this test is run on it. The first upgrade
test (improve.py) found that repeating the swap 5 times (`repeat5`) passed its first round; whether it is adopted depends on its
confirmation run. The Step 1 test is NOT run on top of it: that would cost 5 times the time, and the two ideas are separate. If both a
Step 1 change and `repeat5` are adopted, the pair is run together once at the end (`--base repeat5` runs this script on that base).
The base is written in the report.

THE RULE (fixed before the run; it is not changed after seeing results)
    Data: the 12 simulated settings, 20 repeats each, seeds 11000 to 11019 (never used before). Error = estimate minus the true
    effect on ALL treated people. For every setting, ratio = RMSE of the candidate / RMSE of `current`. The same rule as in
    improve.py (it is imported from there, so it cannot drift). A candidate PASSES only if all three hold:
        1. the median ratio over the 12 settings is at most 0.95;
        2. the ratio is below 0.95 in at least 8 of the 12 settings;
        3. the ratio is above 1.05 in at most 1 of the 12 settings.
    Four candidates are looked at, so a pass is not yet an adoption. A candidate that passes is run again, unchanged, on seeds
    12000 to 12019; it is adopted only if it passes again. A candidate that does not pass is reported as "no reliable gain".
    All four are reported either way. The counts are counts of settings, not tests; the settings are variants of four base cases
    and are not independent. The paired z column is information only.
    Seeds 13000 and above were used once, on tiny runs, only to check that the code runs; they are not used for any result.

Also written (information only, not part of the rule): `ranks` shows where every score puts each kind of variable in the
simulated data (C = real confounders, P = predict only the outcome, I = predict only the treatment, N = noise).

Run (from the project folder; about 1 hour on 2 cores):
    python -m select_then_match.improve_rank sim --seed0 11000 --reps 20
    python -m select_then_match.improve_rank ranks --seed0 11000 --reps 20
    python -m select_then_match.improve_rank report                         # results/IMPROVE_RANK.md
Confirmation run, only for a candidate that passed (here `ridge_raw`):
    python -m select_then_match.improve_rank sim --seed0 12000 --reps 20 --variants ridge_raw --outdir results/improve_rank_confirm
    python -m select_then_match.improve_rank report --simdir results/improve_rank_confirm --out results/IMPROVE_RANK_CONFIRM.md
Real data (optional, needs the CDC csv in data/):
    python -m select_then_match.improve_rank cdc --n 4000 --reps 10
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
from .improve import BAND, BETTER_MIN, MEDIAN_MAX, VARIANTS as UPGRADES, WORSE_MAX, _paired_z, judge   # the same rule as test 1
from .tim import TIM, rank_scores

BASES = ["current", "full", "repeat5", "mu0avg"]
CANDIDATES = ["ridge_raw", "out_only", "max", "ovb"]
CONTEXT = ["trt_only"]
WHAT = {
    "current": "TIM's Step 1: average of the scaled outcome score and treatment score",
    "ridge_raw": "ridge on raw columns for both models, scores added (close version of the public code)",
    "out_only": "only the outcome score",
    "max": "the larger of the outcome score and the treatment score",
    "ovb": "outcome effect x gap between treated and untreated in the variable's average",
    "trt_only": "context: only the treatment score",
}
SCORES = ["linear", "out_only", "trt_only", "max", "geo", "ovb", "ridge_raw"]      # for the `ranks` table


def variants(base="current"):
    """name -> TIM arguments. `current` is the base itself; every other row is the base with only the Step 1 score changed."""
    if base not in BASES:
        raise SystemExit(f"unknown base {base!r}; choose from {BASES}")
    b = dict(UPGRADES[base])
    return {"current": b, **{n: {**b, "rank": n} for n in CANDIDATES + CONTEXT}}


def run_variants(d, seed, names, base="current"):
    v, rows = variants(base), []
    for name in names:
        t0 = time.time()
        try:
            m = TIM(d.continuous, d.discrete, seed=seed, **v[name]).fit(d)
            rows.append(dict(method=name, att=m.att_, share=m.share_, vars_used=m.vars_used_, seconds=time.time() - t0))
        except Exception as ex:                                     # one failing fit must not stop a long run
            rows.append(dict(method=name, att=np.nan, share=np.nan, vars_used=np.nan, seconds=time.time() - t0, error=str(ex)[:100]))
    return rows


def _sim_job(args):
    setting, seed, names, base = args
    with threadpool_limits(1):
        d = make_sim(setting, seed)
        return [dict(setting=setting, seed=seed, att_true=d.att, **r) for r in run_variants(d, seed, names, base)]


def _cdc_job(args):
    path, n, rep, names, base = args
    with threadpool_limits(1):
        d = load_cdc(path).subsample(n, rep)
        return [dict(rep=rep, n=len(d.y), **r) for r in run_variants(d, rep, names, base)]


def _rank_job(args):
    setting, seed = args
    with threadpool_limits(1):
        d = make_sim(setting, seed)
        rows = []
        for how in SCORES:
            pos = rank_scores(d.X, d.t, d.y, how).rank(ascending=False, method="average")         # 1 = kept longest
            for g in "CPIN":
                rows.append(dict(setting=setting, seed=seed, score=how, group=g, position=float(pos[[c for c in pos.index if c[0] == g]].mean())))
        return rows


def _map(fn, jobs, workers):
    if workers <= 1:
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(workers) as ex:
        yield from ex.map(fn, jobs, chunksize=1)


def _names(vs):
    names = ["current"] + [v for v in (vs.split(",") if vs else CANDIDATES + CONTEXT) if v != "current"]
    bad = [v for v in names if v not in CANDIDATES + CONTEXT + ["current"]]
    if bad:
        raise SystemExit(f"unknown variant(s) {bad}; choose from {CANDIDATES + CONTEXT}")
    return names


def run_sim(reps, seed0, workers, outdir, variants_arg=None, settings=None, force=False, base="current"):
    os.makedirs(outdir, exist_ok=True)
    flag = f"{outdir}/BASE.txt"
    if os.path.exists(flag) and open(flag).read().strip() != base and not force:
        raise SystemExit(f"{outdir} holds results for base {open(flag).read().strip()!r}, not {base!r}: use another --outdir or --force")
    open(flag, "w").write(base + "\n")
    names, seeds, todo = _names(variants_arg), list(range(seed0, seed0 + reps)), []
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
    for rows in _map(_sim_job, [(s, seed, names, base) for s in todo for seed in seeds], workers):
        s = rows[0]["setting"]
        buf.setdefault(s, []).extend(rows)
        if len({r["seed"] for r in buf[s]}) == reps:
            df = pd.DataFrame(buf.pop(s))
            df["err"] = df.att - df.att_true
            df.to_csv(f"{outdir}/{s}.csv", index=False)
            print(f"done {s}  ({time.time() - t0:.0f}s)", flush=True)


def run_cdc(path, n, reps, workers, variants_arg=None, out="results/improve_rank_cdc.csv", base="current"):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    names = _names(variants_arg)
    rows = [r for rs in _map(_cdc_job, [(path, n, rep, names, base) for rep in range(reps)], workers) for r in rs]
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


def run_ranks(reps, seed0, workers, out="results/improve_rank_ranks.csv"):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows = [r for rs in _map(_rank_job, [(s, seed) for s in SETTINGS for seed in range(seed0, seed0 + reps)], workers) for r in rs]
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


# ----------------------------------------------------------------------------- the report
def build(simdir="results/improve_rank", cdc_csv="results/improve_rank_cdc.csv", ranks_csv="results/improve_rank_ranks.csv", out="results/IMPROVE_RANK.md"):
    files = sorted(glob.glob(f"{simdir}/*.csv"))
    if not files:
        raise SystemExit(f"no result files in {simdir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ok = df[df.att.notna()]
    base = open(f"{simdir}/BASE.txt").read().strip() if os.path.exists(f"{simdir}/BASE.txt") else "current"
    rmse = ok.groupby(["setting", "method"]).err.apply(lambda e: float(np.sqrt(np.mean(e ** 2)))).unstack("method").reindex(list(SETTINGS)).dropna(how="all")
    bias = ok.groupby(["setting", "method"]).err.mean().unstack("method").reindex(rmse.index)
    cols = [c for c in ["current"] + CANDIDATES + CONTEXT if c in rmse.columns]
    cands = [c for c in CANDIDATES if c in rmse.columns]
    ratio = pd.DataFrame({c: rmse[c] / rmse["current"] for c in cands})
    reps, seeds = df.groupby("setting").seed.nunique(), (int(df.seed.min()), int(df.seed.max()))
    parts = ["# Does a different Step 1 make the pipeline stronger?\n",
             f"{df.setting.nunique()} simulated settings, {int(reps.min()) if reps.min() == reps.max() else f'{int(reps.min())} to {int(reps.max())}'} repeats each, "
             f"seeds {seeds[0]} to {seeds[1]}. Error = estimate minus the true effect on ALL treated people. "
             f"`current` is the base pipeline (**{base}**). Each candidate changes only the Step 1 score (the variable ranking).\n",
             "## The rule (written before the run)\n\n"
             f"A candidate PASSES only if, against `current`, (1) the median RMSE ratio over the 12 settings is at most {MEDIAN_MAX}, "
             f"(2) the ratio is below {1 - BAND:.2f} in at least {BETTER_MIN} settings, and (3) it is above {1 + BAND:.2f} in at most {WORSE_MAX}. "
             "A pass on seeds 11000 to 11019 is not an adoption: the candidate is run again, unchanged, on seeds 12000 to 12019 and adopted only if it "
             "passes again. The counts are counts of settings, not tests.\n",
             "## The versions\n\n" + to_md(pd.DataFrame({"version": cols, "Step 1 score": [WHAT[c] for c in cols]}), index=False) + "\n"]
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
    if os.path.exists(ranks_csv):
        r = pd.read_csv(ranks_csv)
        t = r.groupby(["score", "group"]).position.mean().unstack("group").reindex(SCORES)[list("CPIN")]
        t.columns = ["C real confounders (4)", "P outcome only (3)", "I treatment only (3)", "N noise (10)"]
        t.index.name = "score"
        parts.append("## Where each score puts each kind of variable (information only)\n\nAverage position over all settings and data sets "
                     "(1 = kept longest, 20 = dropped first; 10.5 would be no information). The simulated data has 4 real confounders, 3 variables that "
                     "predict only the outcome, 3 that predict only the treatment, and 10 noise variables. Scores are computed on the whole data set here, "
                     "not on a learning half.\n\n" + to_md(t) + "\n")
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
    ap.add_argument("which", choices=["sim", "cdc", "ranks", "report"])
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=11000)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base", default="current", help="current, or the upgrade from improve.py that passed its confirmation: full, repeat5, mu0avg")
    ap.add_argument("--variants", default=None, help="comma separated, default all (current is always included)")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--outdir", default="results/improve_rank")
    ap.add_argument("--simdir", default="results/improve_rank")
    ap.add_argument("--out", default="results/IMPROVE_RANK.md")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.which == "sim":
        run_sim(a.reps or 20, a.seed0, a.workers, a.outdir, a.variants, a.settings.split(",") if a.settings else None, a.force, a.base)
    elif a.which == "cdc":
        path = a.path or (sorted(glob.glob("data/*health_indicators_BRFSS2015.csv")) or [None])[0]
        if path is None:
            raise SystemExit("CDC file not found: put the BRFSS 2015 csv in data/ (see data/README.md)")
        run_cdc(path, a.n, a.reps or 10, a.workers, a.variants, base=a.base)
    elif a.which == "ranks":
        run_ranks(a.reps or 20, a.seed0, a.workers)
    else:
        print("wrote", build(a.simdir, out=a.out))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
