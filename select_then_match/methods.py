"""The ten methods compared in Step 3, in one place. Each is  fn(d, seed, cache) -> {"att", "share"}.

TIM family (the project)           tim_2stage      TIM as designed (rank, bin, match with dropping, refine)
                                   tim_2stage_bc   TIM + outcome-model bias correction
                                   tim_3stage_bc   screen variables first, then TIM, then bias correction (the project's pipeline)
Matching rivals                    psm             propensity score matching, caliper
                                   robust          optimal 1:1 matching inside a propensity caliper
                                   flame           FLAME-style (own compact version, see flame.py)
                                   malts           MALTS-style (own compact version, see malts.py)
References                         naive           no adjustment (shows how much adjusting matters)
                                   dml             double machine learning
                                   tmle            targeted maximum likelihood
"""
from .causal_ml import dml, tmle
from .flame import flame
from .malts import malts
from .matching import naive, psm, robust
from .tim import TIM


def _tim(**kw):
    def fn(d, seed=0, cache=None):
        m = TIM(d.continuous, d.discrete, seed=seed, **kw).fit(d)
        return dict(att=m.att_, share=m.share_, vars_used=m.vars_used_)
    return fn


METHODS = {
    "tim_2stage": _tim(), "tim_2stage_bc": _tim(correct=True), "tim_3stage_bc": _tim(screen=True, correct=True),
    "psm": psm, "robust": robust, "flame": flame, "malts": malts,
    "naive": naive, "dml": dml, "tmle": tmle,
}
GROUP = {"tim_2stage": "TIM", "tim_2stage_bc": "TIM", "tim_3stage_bc": "TIM", "psm": "matching rival", "robust": "matching rival",
         "flame": "matching rival", "malts": "matching rival", "naive": "reference", "dml": "reference", "tmle": "reference"}
WHAT = {
    "tim_2stage": "TIM: rank, bin, exact match with dropping, refine",
    "tim_2stage_bc": "TIM + outcome-model bias correction",
    "tim_3stage_bc": "screen variables, then TIM, then bias correction",
    "psm": "propensity score, 1 nearest neighbour, caliper",
    "robust": "optimal 1:1 matching inside a propensity caliper",
    "flame": "FLAME-style: exact matching, learned dropping of variables (own compact version)",
    "malts": "MALTS-style: learned distance, 10 nearest neighbours (own compact version)",
    "naive": "difference in means, no adjustment",
    "dml": "double machine learning (one constant effect)",
    "tmle": "targeted maximum likelihood (ATT)",
}
