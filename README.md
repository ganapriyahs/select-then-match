# select-then-match

**Question.** Can a 3-stage version of TIM (screen the variables, match, correct what is left over) estimate a treatment effect
more accurately than TIM itself, with groups that stay easy to explain?

TIM is *A Two-Stage Interpretable Matching Framework for Causal Inference* (Shikalgar and Noor-E-Alam, 2025,
[arXiv 2504.09635](https://arxiv.org/abs/2504.09635)). This project builds on it for the course IE/OR 7374, Causal AI and
Analytics (Northeastern). It follows the course in three steps, on the CDC BRFSS 2015 data (253,680 adults; treatment = high
cholesterol, outcome = diabetes) and on simulated data where the true effect is known.

## Start here

```bash
caffeinate -i bash START_HERE.sh      # macOS; on Linux: bash START_HERE.sh
```

`START_HERE.sh` makes its own Python environment in this folder, runs the 68 tests, runs the adopted 5-split pipeline on the
same 240 simulated data sets as the final run, runs it on the 10 CDC samples, and writes **`results/FINAL.md`**: the final
comparison. It takes about 1 hour on 2 cores and can be stopped and started again (finished settings are skipped). The CDC file is
not part of the repository: put it in `data/` first (see `data/README.md`).

Which file to read:

| file | what it holds |
|---|---|
| `results/FINAL.md` | **the final comparison**: ranks, RMSE and bias by setting, the paired questions, CDC |
| `results/IMPROVE.md`, `IMPROVE_CONFIRM.md` | the three upgrades tested on new data, and the confirmation of the one that passed |
| `results/IMPROVE_RANK.md` | four other ways to rank the variables inside TIM (none passed) |
| `results/package_sim.csv`, `package_cdc.csv` | stored results of the FLAME and MALTS packages |
| `results/RESULTS.md`, `results/sim/` | the earlier ten-method run (Step 3, `step3.py`). It uses my own compact FLAME and MALTS and the single-split pipeline. It is kept because the final comparison re-uses its stored results; the final comparison is `FINAL.md` |

| step | what | code | output |
|---|---|---|---|
| 1 | Descriptive statistics: the two groups, the raw gap, how different they are on each covariate | `step1.py` | `results/step1_descriptive.md` |
| 2 | Regression and machine-learning screening of variables | `step2.py` | `results/step2_regression.md` |
| 3 | Ten causal methods side by side: 12 simulated settings, then CDC (first run) | `step3.py` | `results/RESULTS.md` |
| final | The adopted pipeline against the stored final run, and on CDC | `final.py` | `results/FINAL.md` |

## The idea: three stages

1. **Screen.** Two random forests, one for the outcome and one for the treatment, are scored on held-out people by shuffling each
   variable (permutation importance). A variable is kept only if it matters for both (score of at least 0.05).
2. **Match (TIM).** Cut continuous variables into bins, match treated people exactly to untreated people on all kept variables, and
   when a treated person has no match, drop the *least important* variable and try again. Then give more weight to untreated people
   that are closer on the dropped variables.
3. **Correct.** A gradient boosting model trained on untreated people predicts the outcome without treatment. In every matched
   group the control outcome is moved by the gap between the model's prediction for the treated people and for their matches (the
   idea of Abadie and Imbens, 2011, used inside TIM; their guarantees are not claimed for this version).

Learning (ranking, bins, screening, the correction model) is done on one random half of the people and matching on the other half;
then the halves swap and the two answers are combined, weighted by the number of treated people each used. **The adopted
pipeline repeats this over 5 random splits and averages the five answers** (`tim_3stage_bc_r5`). The one-split version
(`tim_3stage_bc`) is reported next to it. Repeating the split is a known technique (for example in double machine learning); it is
tested here, not claimed as new.

## The ten methods in the final comparison

The target is the effect on the treated people (ATT). For a yes/no outcome it is a risk difference.

| group | name in the code | what it is |
|---|---|---|
| TIM | `tim_2stage` | plain TIM: rank, bin, exact match with dropping, weights (my own Python version) |
| TIM | `tim_2stage_bc` | TIM plus the correction, no screening |
| TIM | `tim_3stage_bc` | screen, TIM, correction, one two-half swap |
| TIM | `tim_3stage_bc_r5` | **the adopted pipeline**: the same, averaged over 5 random splits |
| matching | `psm` | propensity score, 1 nearest neighbour, caliper (own code) |
| matching | `robust` | optimal 1:1 matching inside a propensity caliper (own code) |
| matching | `flame_pkg` | the public package `dame-flame` 0.81, default settings; continuous variables cut with the TIM bins |
| other | `malts_pkg` | the public package `pymalts2` 0.0.2 (`malts_mf`), default settings |
| model-based | `dml` | double machine learning, gradient boosting, 5-fold cross-fitting; fits one constant effect (own code) |
| model-based | `tmle` | targeted maximum likelihood, same boosting learner (own code) |
| reference | `naive` | difference in means, no adjustment |

In the ranking the pipeline appears once, as the 5-split version, so there are ten methods. `flame.py` and `malts.py` are my earlier
compact versions; they are used only in the first run (`step3.py`), not in the final comparison.

## What was fixed before the run

- 12 simulated settings: {yes/no, number outcome} × {linear, nonlinear} × {n = 1000, n = 500, poor overlap}; 20 covariates
  (4 confounders, 3 outcome-only, 3 treatment-only, 10 noise).
- 20 repeats per setting, seeds 6000 to 6019, written before the first run of the final code. CDC: 10 random samples of 4,000 people.
- Error = estimate minus the true effect on **all** treated people, so a method that quietly drops treated people is not rewarded.
- Report rule: for each pair of methods the ratio of RMSEs, setting by setting; below 0.95 is better, within 5% a tie, above 1.05
  worse. Counts of settings, not significance tests (the settings are variants of four base cases, not independent).
- Rule for improving the pipeline, written before any of those runs: an idea passes only if, against the current pipeline, the
  median ratio is at most 0.95, the ratio is below 0.95 in at least 8 of 12 settings, and above 1.05 in at most 1; a pass must be
  repeated on new data. Seeds: 9000 to 9019 (first round), 10000 to 10019 (confirmation), 11000 to 11019 (ranking tests).
  The final comparison re-uses seeds 6000 to 6019, which were not used to test or pick any idea.

## Results

All numbers are from `results/FINAL.md`. Simulated data: 12 settings × 20 data sets (seeds 6000 to 6019). Rank 1 = lowest RMSE in a
setting, averaged over the 12 settings. A rank is only comparable inside one list of ten.

| method | average rank (of 10) | share of treated people used |
|---|---|---|
| `dml` | 2.08 | 100% |
| `robust` | 2.42 | 36% |
| `malts_pkg` | 3.83 | 100% |
| **`tim_3stage_bc_r5`** | **3.92** | 86% |
| `tmle` | 4.33 | 100% |
| `tim_2stage_bc` | 5.58 | 86% |
| `psm` | 6.50 | 93% |
| `tim_2stage` (plain TIM) | 8.00 | 86% |
| `flame_pkg` | 8.33 | 26% |
| `naive` | 10.00 | 100% |

In the first final run, with the one-split pipeline `tim_3stage_bc` in this list instead, it ranked fifth (5.17).

What each part adds (ratio = RMSE of the first method over the second, per setting; counts are settings with ratio below 0.95 /
within 5% / above 1.05 out of 12):

| question | median ratio | better / tie / worse |
|---|---|---|
| Does the correction help TIM? (`tim_2stage_bc` / `tim_2stage`) | 0.50 | 11 / 0 / 1 |
| Is the whole pipeline (1 split) better than plain TIM? (`tim_3stage_bc` / `tim_2stage`) | 0.45 | 11 / 0 / 1 |
| Does screening add anything, 1 split? (`tim_3stage_bc` / `tim_2stage_bc`) | 0.98 | 3 / 6 / 3 |
| Does screening add anything, 5 splits? (`tim_3stage_bc_r5` / `tim_2stage_bc_r5`) | 0.998 | 2 / 9 / 1 |
| 5 splits against 1 split, final data sets | 0.839 | 9 / 1 / 2 |
| 5 splits against 1 split, new seeds 9000 to 9019 | 0.792 | 12 / 0 / 0 |
| 5 splits against 1 split, new seeds 10000 to 10019 | 0.832 | 12 / 0 / 0 |

Five splits lower the spread of the estimate (to about two thirds) and leave the bias about the same. They cost about 5 times the
computing time (9.6 s against 1.9 s per data set on one machine), and the five splits give different matched groups. On the final
data sets both losses are the 500-person settings, so the gain is not in every setting.

The adopted pipeline against each other method:

| against | 5 splits: better / tie / worse | 5 splits: median ratio | 1 split: better / tie / worse | 1 split: median ratio |
|---|---|---|---|---|
| `tim_2stage` (plain TIM) | 11 / 0 / 1 | 0.375 | 11 / 0 / 1 | 0.450 |
| `psm` | 12 / 0 / 0 | 0.748 | 8 / 1 / 3 | 0.816 |
| `flame_pkg` | 12 / 0 / 0 | 0.368 | 12 / 0 / 0 | 0.485 |
| `tmle` | 6 / 2 / 4 | 0.969 | 3 / 0 / 9 | 1.205 |
| `malts_pkg` | 5 / 0 / 7 | 1.080 | 4 / 1 / 7 | 1.208 |
| `dml` | 1 / 1 / 10 | 1.492 | 0 / 2 / 10 | 1.761 |
| `robust` | 3 / 0 / 9 | 1.739 | 0 / 0 / 12 | 2.241 |

Ideas tested to improve the pipeline (new data, rule above):

| idea | what it changes | median ratio | better / tie / worse | result |
|---|---|---|---|---|
| `repeat5` | the swap repeated over 5 random splits | 0.792 | 12 / 0 / 0 | passed, and again on new data (0.832) |
| `full` | no halves: learn and match on everybody | 0.868 | 8 / 0 / 4 | did not pass |
| `mu0avg` | correction = average of boosting and a linear model | 1.082 | 5 / 1 / 6 | did not pass |
| `ridge_raw` | rank with ridge on raw columns | 0.940 | 7 / 4 / 1 | did not pass |
| `out_only` | rank by the outcome score only | 0.887 | 7 / 2 / 3 | did not pass |
| `max` | rank by the larger of the two scores | 0.974 | 5 / 2 / 5 | did not pass |
| `ovb` | outcome effect × treated-vs-untreated gap | 0.952 | 5 / 6 / 1 | did not pass |

What this says, in plain words:

1. **The correction (stage 3) is what improves TIM.** Plain TIM is biased upwards in all 12 settings; the correction roughly
   halves its error.
2. **Screening (stage 1) adds no reliable gain**: the ratio stayed between 0.98 and 1.02 in four looks. One possible reason is that
   plain TIM already matches on only about 4 to 5 of the 20 variables. This is a reading of the numbers, not a tested cause.
3. **The pipeline is a clear improvement over TIM, but not the best method.** It is clearly ahead of plain TIM, `psm` and
   `flame_pkg`, about level with `tmle`, slightly behind `malts_pkg`, and clearly behind `dml` and `robust`. `dml` leads the yes/no
   settings and `robust` the number settings. `robust` uses only 36% of the treated people, so it answers a narrower question.

CDC (10 samples of 4,000 people, mean estimate of the risk difference, spread across samples in brackets; no true value is
known, the TIM paper reports 0.079 with its own setup, the Step 2 regression on everyone gives 0.056):

| method | mean (spread) | method | mean (spread) |
|---|---|---|---|
| `dml` | 0.050 (0.015) | `psm` | 0.068 (0.022) |
| `tim_2stage_bc` | 0.050 (0.030) | `tim_2stage` | 0.068 (0.025) |
| `tmle` | 0.053 (0.021) | `flame_pkg` | 0.119 (0.019) |
| `tim_3stage_bc_r5` | 0.057 (0.020) | `naive` | 0.138 (0.011) |
| `tim_3stage_bc` | 0.059 (0.031) | `robust` | 0.067 (0.014) |

Eight of the ten methods shown here land between 0.050 and 0.068; the naive difference stays near the raw gap (0.140) and the
FLAME package is high and uses only 68% of the treated people. With 5 splits the spread of the pipeline across samples falls from
0.031 to 0.020, as in the simulations. The MALTS package was not run on CDC (too slow at this size). This is a sanity check, not a
claim about what cholesterol does.

## Layout

```
START_HERE.sh    one command: environment, tests, final comparison, report
run_all.sh       the earlier Steps 1 to 3 run (step1, step2, step3)
select_then_match/
  data.py        simulated data generator (12 settings) and the CDC loader
  common.py      half split, the two-half swap, small helpers
  screening.py   stage 1: forest permutation importance (and LASSO for Step 2)
  tim.py         stages 2 and 3: TIM, with the correction, screening and repeated splits as switches
  matching.py    naive, psm, robust
  causal_ml.py   dml, tmle
  methods.py     the methods of the first run in one place
  final.py       the final comparison: runs the 5-split pipelines, builds results/FINAL.md
  improve.py     test of three upgrades (rule written first)       improve_rank.py   test of four other variable scores
  package_check.py, package_run.py   run the FLAME and MALTS packages on the same data
  flame.py  malts.py             my earlier compact versions (first run only)
  step1.py  step2.py  step3.py   the three steps of the course
tests/           68 tests
results/         the output of the runs (tables as markdown / csv)
```

The FLAME and MALTS packages are needed only to run them again (`pip install dame-flame pymalts2`); their stored results are in
`results/package_sim.csv` and `results/package_cdc.csv`, so the final comparison does not need them.

## Limits

- One data generator, made by the author. Conclusions are about these 12 settings, not about all data.
- 20 repeats per setting and no confidence intervals on the ranks; the 12 settings are variants of four base cases. Small
  differences should not be read as real.
- FLAME and MALTS ran at default settings and were not tuned. `psm`, `robust`, `dml`, `tmle` and TIM are my own code.
- Every method assumes that no important confounder is missing.
- CDC has no known true effect. The numbers show how far methods agree, not which is right. Survey weights are not used, and some
  covariates (general health, heart disease, blood pressure, BMI) may be results of cholesterol or diabetes.
- `dml` fits one constant effect, so part of its error can come from a different target than the effect on the treated.
- The screening rule keeps a variable only if it matters for both treatment and outcome, so it can drop a real confounder. Why
  screening did not help was not tested.
- The 5-split gain is not in every setting (9 / 1 / 2 on the final data sets), costs about 5 times the time, and the groups differ
  from split to split.
- The correction is a boosting model, so the adjustment is harder to explain than the matched groups. Explainability was not
  measured.

## References

- Abadie, A. and Imbens, G. W. (2011). Bias-corrected matching estimators for average treatment effects. *Journal of Business & Economic Statistics* 29(1), 1–11.
- Breiman, L. (2001). Random forests. *Machine Learning* 45(1), 5–32.
- CDC Diabetes Health Indicators (2017). UCI Machine Learning Repository. doi:10.24432/C53919.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal* 21(1), C1–C68.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics* 29(5), 1189–1232.
- Iacus, S. M., King, G. and Porro, G. (2012). Causal inference without balance checking: coarsened exact matching. *Political Analysis* 20(1), 1–24.
- Parikh, H., Rudin, C. and Volfovsky, A. (2022). MALTS: matching after learning to stretch. *JMLR* 23(240), 1–42.
- Rosenbaum, P. R. and Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika* 70(1), 41–55.
- Shikalgar, S. and Noor-E-Alam, M. (2025). A two-stage interpretable matching framework for causal inference. arXiv:2504.09635.
- Stuart, E. A. (2010). Matching methods for causal inference: a review and a look forward. *Statistical Science* 25(1), 1–21.
- van der Laan, M. J. and Rubin, D. (2006). Targeted maximum likelihood learning. *The International Journal of Biostatistics* 2(1), Article 11.
- Wang, T. et al. (2021). FLAME: a fast large-scale almost matching exactly approach to causal inference. *JMLR* 22(31), 1–41.
