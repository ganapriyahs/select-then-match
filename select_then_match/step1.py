"""Step 1: descriptive statistics of the CDC data (high cholesterol -> diabetes).

    python -m select_then_match.step1            # writes results/step1_descriptive.md

How many people, the diabetes rate in each group, the raw gap (risk difference, risk ratio, odds ratio), and one row per
covariate with its average in the treated and in the untreated group and the standardized mean difference (SMD = the gap
divided by the pooled standard deviation; above 0.1 means the groups clearly differ). Matching tries to shrink the SMDs.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from .common import to_md
from .data import load_cdc


def raw_gap(t, y):
    """Risk difference, risk ratio and odds ratio of treated against untreated, with 95% intervals (Wald)."""
    n1, n0 = int((t == 1).sum()), int((t == 0).sum())
    a, b = int(y[t == 1].sum()), int(y[t == 0].sum())
    p1, p0 = a / n1, b / n0
    rd_se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    rr_se = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n0)
    or_se = np.sqrt(1 / a + 1 / (n1 - a) + 1 / b + 1 / (n0 - b))
    rr, odds = p1 / p0, (a / (n1 - a)) / (b / (n0 - b))
    return {"rate in treated": p1, "rate in untreated": p0,
            "risk difference": p1 - p0, "risk difference low": p1 - p0 - 1.96 * rd_se, "risk difference high": p1 - p0 + 1.96 * rd_se,
            "risk ratio": rr, "risk ratio low": rr * np.exp(-1.96 * rr_se), "risk ratio high": rr * np.exp(1.96 * rr_se),
            "odds ratio": odds, "odds ratio low": odds * np.exp(-1.96 * or_se), "odds ratio high": odds * np.exp(1.96 * or_se)}


def covariate_table(X, t):
    """One row per covariate: average (the share, for a yes/no column) in each group, and the SMD."""
    rows = []
    for c in X.columns:
        x = X[c].to_numpy(dtype=float)
        sd = np.sqrt((x[t == 1].var() + x[t == 0].var()) / 2)
        rows.append({"variable": c, "type": "yes/no" if set(np.unique(x)) <= {0.0, 1.0} else "number",
                     "treated": x[t == 1].mean(), "untreated": x[t == 0].mean(), "SMD": (x[t == 1].mean() - x[t == 0].mean()) / sd if sd > 0 else 0.0})
    return pd.DataFrame(rows)


def build(path, out="results/step1_descriptive.md"):
    d = load_cdc(path)
    t, y = d.t, d.y
    g = raw_gap(t, y)
    tab = covariate_table(d.X, t)
    tab = tab.reindex(tab.SMD.abs().sort_values(ascending=False).index)
    groups = pd.DataFrame({"": ["people", "share of everyone", "had diabetes", "diabetes rate"],
                           "treated (high cholesterol)": [f"{int((t == 1).sum()):,}", f"{(t == 1).mean():.1%}", f"{int(y[t == 1].sum()):,}", f"{g['rate in treated']:.1%}"],
                           "untreated": [f"{int((t == 0).sum()):,}", f"{(t == 0).mean():.1%}", f"{int(y[t == 0].sum()):,}", f"{g['rate in untreated']:.1%}"]})
    text = ["# Step 1: descriptive statistics (CDC BRFSS 2015)\n",
            f"{len(y):,} people. Treatment: high cholesterol. Outcome: diabetes (prediabetes counts as no). Missing values: {int(d.X.isna().sum().sum())}.\n",
            "## The two groups\n", to_md(groups, index=False),
            "\n## The raw gap (no adjustment)\n",
            f"People with high cholesterol have a diabetes rate {g['risk difference']:.3f} higher ({g['risk difference low']:.3f} to {g['risk difference high']:.3f}); "
            f"risk ratio {g['risk ratio']:.2f} ({g['risk ratio low']:.2f} to {g['risk ratio high']:.2f}); odds ratio {g['odds ratio']:.2f} "
            f"({g['odds ratio low']:.2f} to {g['odds ratio high']:.2f}). A comparison, not yet an effect: the groups differ in many other ways.\n",
            f"## Covariates, largest difference first ({int((tab.SMD.abs() > 0.1).sum())} of {len(tab)} have |SMD| above 0.1)\n",
            to_md(tab, index=False),
            "\n`treated` and `untreated` are the average (for yes/no columns, the share with a yes).\n"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(text))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None)
    a = ap.parse_args()
    path = a.path or (sorted(glob.glob("data/*health_indicators_BRFSS2015.csv")) or [None])[0]
    if path is None:
        raise SystemExit("CDC file not found: put the BRFSS 2015 csv in data/ (see data/README.md)")
    print("wrote", build(path))


if __name__ == "__main__":
    main()
