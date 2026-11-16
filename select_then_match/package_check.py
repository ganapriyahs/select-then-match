"""Check my compact FLAME-style and MALTS-style methods against the authors' packages, at the packages' default settings.

    pip install dame-flame pymalts
    python -m select_then_match.package_check                                   # 3 settings x 5 repeats, a few minutes
    python -m select_then_match.package_check --reps 20 --settings all          # the full set, much slower

It runs on the same simulated data sets (seeds 6000 upward) as the main study and writes results/package_check.md and
results/package_check.csv: RMSE, average error and share of treated people used, for
    flame (mine)   flame (package)   malts (mine)   malts (package)   tim_2stage_bc   tim_3stage_bc
so you can see whether the package numbers tell a different story from mine.

How the packages are called
  * dame-flame (FLAME): needs categories, so continuous variables are cut with the same bins as TIM; one half of the people
    is the holdout set the package needs, the other half is matched, then the halves swap (as for every method here).
    FLAME(verbose=0) and nothing else is set: all other settings are the package defaults.
  * pymalts (MALTS): the package cuts the data into folds itself, so it gets the whole data set and its default settings
    (only the list of yes/no variables is passed). The ATT is the mean estimated effect of the treated people.
This file was written without being able to install the packages, so it was never run against the real ones. If a call fails
the error is printed and the run goes on; send the error message and I can fix the call.
"""
import argparse
import os
import time
import traceback
import warnings
from importlib import metadata
import pymalts2 as pymalts
import numpy as np
import pandas as pd

from .common import swap, to_md
from .data import SETTINGS, make_sim
from .flame import _codes
from .methods import METHODS
from .tim import bin_cuts

warnings.filterwarnings("ignore")

DEFAULT_SETTINGS = ["binary_linear", "continuous_linear", "binary_nonlinear"]
ERRORS = []     # (method, message) of the first failure of each method, printed at the end


# ----------------------------------------------------------------------------- FLAME (dame-flame package)
def _frame(D, cuts, continuous):
    """The data the package wants: integer category codes, a 0/1 `treated` column and an `outcome` column."""
    df = pd.DataFrame(_codes(D, cuts, continuous).astype(int), columns=list(D.X.columns))
    df["treated"], df["outcome"] = D.t.astype(int), D.y.astype(float)
    return df


def _matched_ids(model, df):
    """Row labels of the people that ended up in a matched group."""
    md = getattr(model, "matched_data", None)
    if md is not None and len(md):
        return pd.Index(md.index).unique()
    groups = getattr(model, "units_per_group", None)
    if groups:
        return pd.Index(sorted({u for g in groups for u in g}))
    return pd.Index([])


def _att_from_groups(model, df):
    """ATT worked out from the matched groups (used only if the package's own ATT() call fails)."""
    effects, weights = [], []
    for g in getattr(model, "units_per_group", None) or []:
        sub = df.loc[list(g)]
        tr, co = sub[sub.treated == 1], sub[sub.treated == 0]
        if len(tr) and len(co):
            effects.append(tr.outcome.mean() - co.outcome.mean())
            weights.append(len(tr))
    return float(np.average(effects, weights=weights)) if effects else np.nan


def _flame_package_half(L, M, continuous):
    import dame_flame
    from dame_flame.utils import post_processing

    cuts = bin_cuts(L.X, L.t, L.y, continuous)
    hold, mat = _frame(L, cuts, continuous), _frame(M, cuts, continuous)
    model = dame_flame.matching.FLAME(verbose=0)                      # every other setting stays at the package default
    model.fit(holdout_data=hold, treatment_column_name="treated", outcome_column_name="outcome")
    model.predict(mat)
    ids = _matched_ids(model, mat)
    n = int((mat.loc[ids, "treated"] == 1).sum()) if len(ids) else 0
    if n == 0:
        return dict(att=np.nan, n=0)
    try:
        att = float(post_processing.ATT(model))
    except Exception:
        att = _att_from_groups(model, mat)
    return dict(att=att, n=n)


def flame_package(d, seed=0, cache=None):
    att, share, _ = swap(d, seed, lambda L, M: _flame_package_half(L, M, d.continuous))
    return dict(att=att, share=share)


# ----------------------------------------------------------------------------- MALTS (pymalts package)
def _old_pandas_numpy_shims():
    """pymalts2 was written for pandas 1.x. Give it back DataFrame.append, Series.append, iteritems and the old numpy names."""
    def df_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if isinstance(other, dict):
            other = pd.DataFrame([other])
        elif isinstance(other, pd.Series):
            other = other.to_frame().T
        elif isinstance(other, (list, tuple)):
            other = [o.to_frame().T if isinstance(o, pd.Series) else o for o in other]
            if other and not isinstance(other[0], pd.DataFrame):
                other = pd.DataFrame(other)
        parts = other if isinstance(other, list) else [other]
        return pd.concat([self] + parts, ignore_index=ignore_index, verify_integrity=verify_integrity, sort=sort)

    def series_append(self, to_append, ignore_index=False, verify_integrity=False):
        parts = list(to_append) if isinstance(to_append, (list, tuple)) else [to_append]
        return pd.concat([self] + parts, ignore_index=ignore_index, verify_integrity=verify_integrity)

    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = df_append
    if not hasattr(pd.Series, "append"):
        pd.Series.append = series_append
    for cls in (pd.DataFrame, pd.Series):
        if not hasattr(cls, "iteritems"):
            cls.iteritems = cls.items
    for name, typ in (("float", float), ("int", int), ("bool", bool), ("object", object)):
        if not hasattr(np, name):
            setattr(np, name, typ)


def malts_package(d, seed=0, cache=None):
    _old_pandas_numpy_shims()
    import pymalts2 as pymalts

    np.random.seed(seed)
    df = d.X.reset_index(drop=True).copy()
    df["outcome"], df["treated"] = d.y.astype(float), d.t.astype(int)
    m = pymalts.malts_mf("outcome", "treated", data=df, discrete=list(d.discrete))    # every other setting stays at the package default
    cate = m.CATE_df
    col = "avg.CATE" if "avg.CATE" in cate.columns else next(c for c in cate.columns if "CATE" in str(c))
    treated = cate[col].reindex(df.index[df.treated == 1]).astype(float)
    ok = treated.notna()
    return dict(att=float(treated[ok].mean()) if ok.any() else np.nan, share=float(ok.mean()))


# ----------------------------------------------------------------------------- the comparison
def methods():
    return {"flame (mine)": METHODS["flame"], "flame (package)": flame_package, "malts (mine)": METHODS["malts"],
            "malts (package)": malts_package, "tim_2stage_bc": METHODS["tim_2stage_bc"], "tim_3stage_bc": METHODS["tim_3stage_bc"]}


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def run(settings, reps, seed0, out="results/package_check"):
    for pkg in ("dame-flame", "pymalts2"):
        if version(pkg) == "NOT INSTALLED":
            raise SystemExit(f"{pkg} is not installed: run  pip install dame-flame pymalts2")
    rows = []
    for s in settings:
        for seed in range(seed0, seed0 + reps):
            d = make_sim(s, seed)
            for name, fn in methods().items():
                t0 = time.time()
                try:
                    r = fn(d, seed, {})
                    rows.append(dict(setting=s, seed=seed, method=name, att=r["att"], share=r["share"], att_true=d.att, seconds=time.time() - t0))
                except Exception as ex:
                    rows.append(dict(setting=s, seed=seed, method=name, att=np.nan, share=np.nan, att_true=d.att, seconds=time.time() - t0))
                    if not any(m == name for m, _ in ERRORS):
                        tb = traceback.extract_tb(ex.__traceback__)[-1]
                        ERRORS.append((name, f"{type(ex).__name__}: {str(ex)[:300]}   (in {os.path.basename(tb.filename)}, line {tb.lineno}: {tb.line})"))
            print(f"done {s} seed {seed}", flush=True)
    df = pd.DataFrame(rows)
    df["err"] = df.att - df.att_true
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out + ".csv", index=False)
    ok = df[df.att.notna()]
    order = list(methods())
    rmse = ok.groupby(["setting", "method"]).err.apply(lambda e: float(np.sqrt(np.mean(e ** 2)))).unstack().reindex(columns=order)
    bias = ok.groupby(["setting", "method"]).err.mean().unstack().reindex(columns=order)
    used = ok.groupby(["setting", "method"]).share.mean().unstack().reindex(columns=order)
    failed = df[df.att.isna()].groupby("method").size()
    text = ["# Package check: my FLAME / MALTS against the authors' packages\n",
            f"dame-flame {version('dame-flame')}, pymalts {version('pymalts2')}. Package default settings. {reps} repeats per setting, seeds {seed0} to {seed0 + reps - 1}, "
            "same data as the main study. Error = estimate minus the true effect on all treated people.\n",
            "## RMSE (lower is better)\n", to_md(rmse), "\n## Average error (positive = too high)\n", to_md(bias),
            "\n## Share of treated people used\n", to_md(used)]
    if len(failed):
        text.append("\nNo estimate (failed or no match): " + ", ".join(f"`{m}` ({k} of {int((df.method == m).sum())})" for m, k in failed.items()) + ".\n")
    with open(out + ".md", "w") as f:
        f.write("\n".join(text))
    print("\n".join(text))
    for name, msg in ERRORS:
        print(f"\nERROR in {name}: {msg}")
    print(f"\nwrote {out}.md and {out}.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=6000)
    ap.add_argument("--settings", default=",".join(DEFAULT_SETTINGS), help="comma separated names, or 'all'")
    a = ap.parse_args()
    run(list(SETTINGS) if a.settings == "all" else a.settings.split(","), a.reps, a.seed0)


if __name__ == "__main__":
    main()
