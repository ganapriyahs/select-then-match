"""Small helpers used by several files."""
import numpy as np


def is_binary(y):
    """True when the outcome only holds 0 and 1 (a yes/no outcome)."""
    return bool(np.isin(y, (0, 1)).all())


def split_halves(t, seed):
    """Random split of the people in two halves, stratified by treatment. Returns a yes/no mask for half A."""
    rng = np.random.default_rng(seed)
    half = np.zeros(len(t), dtype=bool)
    for g in (0, 1):
        idx = np.flatnonzero(t == g)
        rng.shuffle(idx)
        half[idx[: len(idx) // 2]] = True
    return half


def swap(d, seed, fit_half):
    """Two-fold swap, the honest way to fit a method that learns something from the outcome.

    Learn on half A and estimate on half B, then learn on half B and estimate on half A, and combine the two estimates
    (each weighted by the number of treated people it is based on). Nothing is ever applied to the people it learned from.

    fit_half(learn, match) gets two Data objects and returns a dict with `att` (estimate) and `n` (number of treated
    people the estimate is based on; 0 when there is no estimate). Returns (att, share of treated people used, parts).
    """
    half = split_halves(d.t, seed)
    parts = [fit_half(d.subset(np.flatnonzero(learn)), d.subset(np.flatnonzero(~learn))) for learn in (half, ~half)]
    used = [p for p in parts if p["n"] > 0]
    att = float(np.average([p["att"] for p in used], weights=[p["n"] for p in used])) if used else np.nan
    return att, sum(p["n"] for p in parts) / max(int((d.t == 1).sum()), 1), parts


def to_md(df, floatfmt="{:.3f}", index=True):
    """A pandas table as a markdown table (no extra package needed)."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    line = "|" + "|".join("---" for _ in d.columns) + "|"
    cell = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if v != v else str(v))
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, line] + rows)
