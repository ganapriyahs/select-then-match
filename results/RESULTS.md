# Step 3 results

Ten methods, on simulated data (truth known) and on CDC (truth unknown). Error = estimate minus the true effect on ALL treated people. `dml` fits one constant effect, which differs from the effect on the treated when the effect differs between people (in the continuous settings by about 0.3), so part of its error is a different target. `psm`, `robust`, `flame`, `malts` and TIM answer for the treated people they use (see `treated used`). `flame` and `malts` are compact re-implementations written for this project, not the authors' packages.

## The methods

| method | group | what it is |
|---|---|---|
| tim_2stage | TIM | TIM: rank, bin, exact match with dropping, refine |
| tim_2stage_bc | TIM | TIM + outcome-model bias correction |
| tim_3stage_bc | TIM | screen variables, then TIM, then bias correction |
| psm | matching rival | propensity score, 1 nearest neighbour, caliper |
| robust | matching rival | optimal 1:1 matching inside a propensity caliper |
| flame | matching rival | FLAME-style: exact matching, learned dropping of variables (own compact version) |
| malts | matching rival | MALTS-style: learned distance, 10 nearest neighbours (own compact version) |
| naive | reference | difference in means, no adjustment |
| dml | reference | double machine learning (one constant effect) |
| tmle | reference | targeted maximum likelihood (ATT) |

## 1. Simulated data: 12 settings, 20 repeats each, seeds 6000 to 6019

### RMSE (lower is better)

| setting | tim_2stage | tim_2stage_bc | tim_3stage_bc | psm | robust | flame | malts | naive | dml | tmle |
|---|---|---|---|---|---|---|---|---|---|---|
| binary_linear | 0.250 | 0.115 | 0.090 | 0.082 | 0.083 | 0.271 | 0.127 | 0.442 | 0.034 | 0.074 |
| binary_linear_n500 | 0.246 | 0.116 | 0.118 | 0.083 | 0.082 | 0.308 | 0.197 | 0.439 | 0.051 | 0.066 |
| binary_linear_poor_overlap | 0.346 | 0.129 | 0.137 | 0.171 | 0.095 | 0.388 | 0.231 | 0.515 | 0.062 | 0.049 |
| binary_nonlinear | 0.138 | 0.095 | 0.108 | 0.087 | 0.075 | 0.181 | 0.134 | 0.302 | 0.036 | 0.088 |
| binary_nonlinear_n500 | 0.147 | 0.089 | 0.086 | 0.122 | 0.068 | 0.173 | 0.198 | 0.297 | 0.051 | 0.092 |
| binary_nonlinear_poor_overlap | 0.127 | 0.174 | 0.181 | 0.270 | 0.079 | 0.229 | 0.219 | 0.344 | 0.054 | 0.147 |
| continuous_linear | 1.485 | 0.337 | 0.327 | 0.326 | 0.117 | 0.190 | 0.751 | 3.171 | 0.339 | 0.210 |
| continuous_linear_n500 | 1.478 | 0.371 | 0.354 | 0.476 | 0.125 | 0.294 | 1.073 | 3.230 | 0.304 | 0.328 |
| continuous_linear_poor_overlap | 1.862 | 0.480 | 0.478 | 0.774 | 0.207 | 0.249 | 1.221 | 3.637 | 0.392 | 0.446 |
| continuous_nonlinear | 0.860 | 0.460 | 0.359 | 0.434 | 0.138 | 0.359 | 1.037 | 2.135 | 0.346 | 0.395 |
| continuous_nonlinear_n500 | 0.838 | 0.493 | 0.621 | 0.719 | 0.191 | 0.675 | 1.373 | 2.208 | 0.337 | 0.516 |
| continuous_nonlinear_poor_overlap | 0.771 | 0.564 | 0.528 | 0.950 | 0.242 | 0.374 | 1.479 | 2.407 | 0.439 | 0.806 |

### Average error (bias; positive = too high)

| setting | tim_2stage | tim_2stage_bc | tim_3stage_bc | psm | robust | flame | malts | naive | dml | tmle |
|---|---|---|---|---|---|---|---|---|---|---|
| binary_linear | 0.230 | 0.005 | -0.007 | 0.010 | 0.076 | 0.263 | 0.123 | 0.441 | 0.018 | 0.024 |
| binary_linear_n500 | 0.226 | 0.064 | 0.061 | 0.015 | 0.059 | 0.302 | 0.190 | 0.438 | 0.015 | 0.027 |
| binary_linear_poor_overlap | 0.330 | 0.108 | 0.108 | 0.038 | 0.077 | 0.383 | 0.226 | 0.515 | 0.043 | -0.013 |
| binary_nonlinear | 0.110 | 0.039 | 0.058 | 0.005 | 0.066 | 0.171 | 0.127 | 0.300 | 0.009 | 0.043 |
| binary_nonlinear_n500 | 0.113 | 0.044 | 0.042 | 0.014 | 0.041 | 0.164 | 0.189 | 0.295 | -0.003 | 0.041 |
| binary_nonlinear_poor_overlap | 0.098 | 0.137 | 0.152 | 0.128 | 0.047 | 0.224 | 0.215 | 0.343 | 0.015 | 0.064 |
| continuous_linear | 1.420 | 0.229 | 0.185 | -0.079 | -0.023 | 0.069 | 0.734 | 3.167 | -0.323 | 0.120 |
| continuous_linear_n500 | 1.411 | 0.307 | 0.293 | 0.142 | -0.012 | 0.209 | 1.060 | 3.224 | -0.266 | 0.230 |
| continuous_linear_poor_overlap | 1.771 | 0.387 | 0.385 | 0.183 | -0.132 | 0.144 | 1.212 | 3.636 | -0.375 | 0.083 |
| continuous_nonlinear | 0.730 | 0.330 | 0.267 | -0.018 | -0.041 | 0.233 | 1.018 | 2.127 | -0.335 | 0.294 |
| continuous_nonlinear_n500 | 0.742 | 0.430 | 0.554 | 0.271 | -0.008 | 0.544 | 1.353 | 2.196 | -0.304 | 0.347 |
| continuous_nonlinear_poor_overlap | 0.483 | 0.362 | 0.362 | 0.438 | -0.145 | 0.235 | 1.472 | 2.402 | -0.419 | 0.491 |

### Summary

Rank 1 = lowest RMSE in a setting; the rank is averaged over settings. `treated used` is the share of treated people the estimate is based on.

| method | group | average RMSE rank | worst RMSE / best | treated used | seconds per data set |
|---|---|---|---|---|---|
| tim_2stage | TIM | 7.583 | 12.719 | 0.856 | 1.274 |
| tim_2stage_bc | TIM | 5.167 | 3.413 | 0.856 | 1.412 |
| tim_3stage_bc | TIM | 4.917 | 3.365 | 0.864 | 5.938 |
| psm | matching rival | 5.833 | 5.039 | 0.926 | 0.009 |
| robust | matching rival | 1.833 | 2.479 | 0.357 | 0.023 |
| flame | matching rival | 5.750 | 8.073 | 0.738 | 1.695 |
| malts | matching rival | 7.917 | 8.600 | 1.000 | 0.143 |
| naive | reference | 10.000 | 27.156 | 1.000 | 0.000 |
| dml | reference | 2.250 | 2.906 | 1.000 | 1.048 |
| tmle | reference | 3.750 | 3.336 | 1.000 | 0.000 |

### The questions

`median RMSE ratio` is the ratio of the first method's RMSE to the second's, setting by setting, then the median (below 1: the first is better). `a better` / `tie` / `a worse` count settings where the ratio is below 0.95, within 5% of 1, or above 1.05. These are counts of settings, not tests: the settings are variants of four base cases and are not independent.

| question | compared | median RMSE ratio | a better | tie | a worse |
|---|---|---|---|---|---|
| Does the correction help TIM? | tim_2stage_bc / tim_2stage | 0.504 | 11 | 0 | 1 |
| Is the 3-stage pipeline (screen, match, correct) better than TIM? | tim_3stage_bc / tim_2stage | 0.448 | 11 | 0 | 1 |
| Does screening add anything once the correction is on? | tim_3stage_bc / tim_2stage_bc | 0.984 | 3 | 6 | 3 |
| Corrected TIM against psm | tim_2stage_bc / psm | 0.767 | 7 | 1 | 4 |
| Corrected TIM against robust | tim_2stage_bc / robust | 2.265 | 0 | 0 | 12 |
| Corrected TIM against flame | tim_2stage_bc / flame | 0.745 | 7 | 0 | 5 |
| Corrected TIM against malts | tim_2stage_bc / malts | 0.449 | 12 | 0 | 0 |
| Corrected TIM against dml | tim_2stage_bc / dml | 1.606 | 0 | 1 | 11 |
| Corrected TIM against tmle | tim_2stage_bc / tmle | 1.146 | 1 | 2 | 9 |
| 3-stage pipeline against psm | tim_3stage_bc / psm | 0.816 | 8 | 1 | 3 |
| 3-stage pipeline against robust | tim_3stage_bc / robust | 2.241 | 0 | 0 | 12 |
| 3-stage pipeline against flame | tim_3stage_bc / flame | 0.854 | 7 | 1 | 4 |
| 3-stage pipeline against malts | tim_3stage_bc / malts | 0.444 | 12 | 0 | 0 |
| 3-stage pipeline against dml | tim_3stage_bc / dml | 1.761 | 0 | 2 | 10 |
| 3-stage pipeline against tmle | tim_3stage_bc / tmle | 1.205 | 3 | 0 | 9 |

## 2. CDC: high cholesterol and diabetes (4,000 people per sample, 10 samples)

No true effect is known. The TIM paper reports 0.079. Look at how close the methods are and how much each moves between samples. These are adjusted associations in the released sample (no survey weights), and every method assumes the 20 covariates capture all confounding.

| method | group | effect (mean over samples) | sd across samples | treated used | samples |
|---|---|---|---|---|---|
| tim_2stage | TIM | 0.0681 | 0.0250 | 0.9819 | 10 |
| tim_2stage_bc | TIM | 0.0503 | 0.0304 | 0.9819 | 10 |
| tim_3stage_bc | TIM | 0.0585 | 0.0306 | 0.9744 | 10 |
| psm | matching rival | 0.0682 | 0.0219 | 0.9989 | 10 |
| robust | matching rival | 0.0673 | 0.0143 | 0.8425 | 10 |
| flame | matching rival | 0.1023 | 0.0133 | 0.9987 | 10 |
| malts | matching rival | 0.0738 | 0.0185 | 1.0000 | 10 |
| naive | reference | 0.1377 | 0.0109 | 1.0000 | 10 |
| dml | reference | 0.0504 | 0.0154 | 1.0000 | 10 |
| tmle | reference | 0.0533 | 0.0214 | 1.0000 | 10 |
