"""Run the authors' FLAME and MALTS packages on the same data as the main study (package_check.py must already work).

    python -m select_then_match.package_run sim --workers 4      # 12 settings x 20 repeats, seeds 6000-6019: FLAME and MALTS
    python -m select_then_match.package_run cdc --workers 4      # the 10 CDC samples of 4,000 people: FLAME only

Each finished run is added to results/package_sim.csv (or results/package_cdc.csv) at once. You can stop with Ctrl+C and
start again later: finished runs are skipped. Seeds run in order, so a part-finished run still has every setting.
"""
import argparse
import glob
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")

from . import package_check as pc
from .data import SETTINGS, load_cdc, make_sim

FUNCS = {"flame": pc.flame_package, "malts": pc.malts_package}


def _job(a):
    kind, setting, seed, which, path, n = a
    t0 = time.time()
    row = dict(method=which + "_pkg")
    row.update(dict(setting=setting, seed=seed) if kind == "sim" else dict(rep=seed, n=n))
    try:
        with threadpool_limits(1):
            d = make_sim(setting, seed) if kind == "sim" else load_cdc(path).subsample(n, seed)
            r = FUNCS[which](d, seed)
        row.update(att=r["att"], share=r["share"])
        if kind == "sim":
            row["att_true"] = d.att
    except Exception as ex:
        row.update(att=float("nan"), share=float("nan"), error=f"{type(ex).__name__}: {str(ex)[:150]}")
    row["seconds"] = time.time() - t0
    return row


def run(kind, jobs, out, workers):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    keys = ["setting", "seed", "method"] if kind == "sim" else ["rep", "method"]
    have = set()
    if os.path.exists(out):
        old = pd.read_csv(out)
        have = {tuple(str(v) for v in r) for r in old[old.att.notna()][keys].to_numpy()}
    todo = [j for j in jobs if tuple(str(v) for v in ((j[1], j[2], j[3] + "_pkg") if kind == "sim" else (j[2], j[3] + "_pkg"))) not in have]
    print(f"{len(todo)} runs to do ({len(jobs) - len(todo)} already done), {workers} at a time", flush=True)
    t0, n_done, errors = time.time(), 0, {}
    with ProcessPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(_job, j) for j in todo]):
            row = f.result()
            pd.DataFrame([row]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            n_done += 1
            if "error" in row and row["method"] not in errors:
                errors[row["method"]] = row["error"]
                print(f"  first error in {row['method']}: {row['error']}", flush=True)
            if n_done % 5 == 0 or n_done == len(todo):
                el = time.time() - t0
                print(f"  {n_done}/{len(todo)} done, {el / 60:.1f} min, about {el / n_done * (len(todo) - n_done) / 60:.0f} min left", flush=True)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["sim", "cdc"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--seed0", type=int, default=6000)
    ap.add_argument("--n", type=int, default=4000, help="CDC sample size")
    ap.add_argument("--methods", default=None, help="flame, malts or flame,malts (default: both for sim, flame for cdc)")
    ap.add_argument("--settings", default=None, help="comma separated names, default all twelve")
    ap.add_argument("--path", default=None)
    a = ap.parse_args()
    if a.which == "sim":
        methods = (a.methods or "flame,malts").split(",")
        reps = a.reps or 20
        settings = a.settings.split(",") if a.settings else list(SETTINGS)
        jobs = [("sim", s, seed, m, None, None) for seed in range(a.seed0, a.seed0 + reps) for s in settings for m in methods]
        run("sim", jobs, "results/package_sim.csv", a.workers)
    else:
        path = a.path or (sorted(glob.glob("data/*health_indicators_BRFSS2015.csv")) or [None])[0]
        if path is None:
            raise SystemExit("CDC file not found: put the BRFSS 2015 csv in data/ (see data/README.md)")
        methods = (a.methods or "flame").split(",")
        jobs = [("cdc", None, rep, m, path, a.n) for rep in range(a.reps or 10) for m in methods]
        run("cdc", jobs, "results/package_cdc.csv", a.workers)


if __name__ == "__main__":
    main()
