# Final comparison: the adopted pipeline against the other methods

12 simulated settings, 20 data sets each, seeds 6000 to 6019: the SAME data sets as the final run. The results of the other methods are the stored ones (they were not run again); only the two 5-split TIM rows are new. Error = estimate minus the true effect on ALL treated people. `dml` fits one constant effect, which differs from the effect on the treated in the continuous settings (by about 0.3), so part of its error is a different target. FLAME and MALTS are the authors' packages at their default settings.

## Average rank (1 = lowest RMSE in a setting, averaged over 12 settings)

First column: the ten methods of the final run, as published. Second column: the same ten, with the 3-stage pipeline replaced by its 5-split version. Ranks are only comparable within a column.

| method | average RMSE rank, the ten of the final run | average RMSE rank, the ten with the 5-split pipeline |
|---|---|---|
| dml | 2.000 | 2.083 |
| robust | 2.167 | 2.417 |
| malts_pkg | 3.750 | 3.833 |
| 3-stage (5 splits) |  | 3.917 |
| tmle | 4.000 | 4.333 |
| 2-stage + correction | 5.417 | 5.583 |
| psm | 6.167 | 6.500 |
| plain TIM | 8.000 | 8.000 |
| flame_pkg | 8.333 | 8.333 |
| naive | 10.000 | 10.000 |
| 3-stage (single split) | 5.167 |  |

## RMSE by setting (lower is better)

| setting | tim_2stage | tim_2stage_bc | tim_2stage_bc_r5 | tim_3stage_bc | tim_3stage_bc_r5 | psm | robust | dml | tmle | malts_pkg | flame_pkg | naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_linear | 0.250 | 0.115 | 0.059 | 0.090 | 0.053 | 0.082 | 0.083 | 0.034 | 0.074 | 0.058 | 0.231 | 0.442 |
| binary_linear_n500 | 0.246 | 0.116 | 0.072 | 0.118 | 0.071 | 0.083 | 0.082 | 0.051 | 0.066 | 0.065 | 0.205 | 0.439 |
| binary_linear_poor_overlap | 0.346 | 0.129 | 0.111 | 0.137 | 0.111 | 0.171 | 0.095 | 0.062 | 0.049 | 0.086 | 0.286 | 0.515 |
| binary_nonlinear | 0.138 | 0.095 | 0.074 | 0.108 | 0.071 | 0.087 | 0.075 | 0.036 | 0.088 | 0.058 | 0.220 | 0.302 |
| binary_nonlinear_n500 | 0.147 | 0.089 | 0.092 | 0.086 | 0.091 | 0.122 | 0.068 | 0.051 | 0.092 | 0.084 | 0.196 | 0.297 |
| binary_nonlinear_poor_overlap | 0.127 | 0.174 | 0.135 | 0.181 | 0.136 | 0.270 | 0.079 | 0.054 | 0.147 | 0.092 | 0.196 | 0.344 |
| continuous_linear | 1.485 | 0.337 | 0.290 | 0.327 | 0.294 | 0.326 | 0.117 | 0.339 | 0.210 | 0.375 | 1.326 | 3.171 |
| continuous_linear_n500 | 1.478 | 0.371 | 0.417 | 0.354 | 0.402 | 0.476 | 0.125 | 0.304 | 0.328 | 0.303 | 0.615 | 3.230 |
| continuous_linear_poor_overlap | 1.862 | 0.480 | 0.424 | 0.478 | 0.423 | 0.774 | 0.207 | 0.392 | 0.446 | 0.532 | 1.271 | 3.637 |
| continuous_nonlinear | 0.860 | 0.460 | 0.352 | 0.359 | 0.369 | 0.434 | 0.138 | 0.346 | 0.395 | 0.450 | 1.147 | 2.135 |
| continuous_nonlinear_n500 | 0.838 | 0.493 | 0.481 | 0.621 | 0.541 | 0.719 | 0.191 | 0.337 | 0.516 | 0.498 | 1.122 | 2.208 |
| continuous_nonlinear_poor_overlap | 0.771 | 0.564 | 0.445 | 0.528 | 0.422 | 0.950 | 0.242 | 0.439 | 0.806 | 0.696 | 0.860 | 2.407 |

## Average error (bias; positive = too high)

| setting | tim_2stage | tim_2stage_bc | tim_2stage_bc_r5 | tim_3stage_bc | tim_3stage_bc_r5 | psm | robust | dml | tmle | malts_pkg | flame_pkg | naive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| binary_linear | 0.230 | 0.005 | 0.019 | -0.007 | 0.020 | 0.010 | 0.076 | 0.018 | 0.024 | -0.003 | 0.220 | 0.441 |
| binary_linear_n500 | 0.226 | 0.064 | 0.036 | 0.061 | 0.033 | 0.015 | 0.059 | 0.015 | 0.027 | 0.008 | 0.142 | 0.438 |
| binary_linear_poor_overlap | 0.330 | 0.108 | 0.093 | 0.108 | 0.094 | 0.038 | 0.077 | 0.043 | -0.013 | 0.022 | 0.270 | 0.515 |
| binary_nonlinear | 0.110 | 0.039 | 0.053 | 0.058 | 0.050 | 0.005 | 0.066 | 0.009 | 0.043 | -0.008 | 0.211 | 0.300 |
| binary_nonlinear_n500 | 0.113 | 0.044 | 0.059 | 0.042 | 0.061 | 0.014 | 0.041 | -0.003 | 0.041 | 0.004 | 0.118 | 0.295 |
| binary_nonlinear_poor_overlap | 0.098 | 0.137 | 0.123 | 0.152 | 0.127 | 0.128 | 0.047 | 0.015 | 0.064 | 0.055 | 0.185 | 0.343 |
| continuous_linear | 1.420 | 0.229 | 0.214 | 0.185 | 0.213 | -0.079 | -0.023 | -0.323 | 0.120 | 0.323 | 1.237 | 3.167 |
| continuous_linear_n500 | 1.411 | 0.307 | 0.378 | 0.293 | 0.371 | 0.142 | -0.012 | -0.266 | 0.230 | 0.229 | 0.273 | 3.224 |
| continuous_linear_poor_overlap | 1.771 | 0.387 | 0.374 | 0.385 | 0.377 | 0.183 | -0.132 | -0.375 | 0.083 | 0.490 | 1.153 | 3.636 |
| continuous_nonlinear | 0.730 | 0.330 | 0.255 | 0.267 | 0.280 | -0.018 | -0.041 | -0.335 | 0.294 | 0.376 | 1.127 | 2.127 |
| continuous_nonlinear_n500 | 0.742 | 0.430 | 0.451 | 0.554 | 0.503 | 0.271 | -0.008 | -0.304 | 0.347 | 0.382 | 0.374 | 2.196 |
| continuous_nonlinear_poor_overlap | 0.483 | 0.362 | 0.378 | 0.362 | 0.374 | 0.438 | -0.145 | -0.419 | 0.491 | 0.677 | 0.752 | 2.402 |

## The questions

`median RMSE ratio` is the ratio of the first method's RMSE to the second's, setting by setting, then the median (below 1: the first is better). `a better` / `tie` / `a worse` count settings where the ratio is below 0.95, within 5% of 1, or above 1.05. These are counts of settings, not tests: the settings are variants of four base cases and are not independent.

| question | compared | median RMSE ratio | a better | tie | a worse |
|---|---|---|---|---|---|
| Does repeating the split 5 times help the 3-stage pipeline (same data sets)? | tim_3stage_bc_r5 / tim_3stage_bc | 0.839 | 9 | 1 | 2 |
| Does it help the 2-stage pipeline with correction? | tim_2stage_bc_r5 / tim_2stage_bc | 0.825 | 9 | 2 | 1 |
| Does screening add anything once the correction and the 5 splits are on? | tim_3stage_bc_r5 / tim_2stage_bc_r5 | 0.998 | 2 | 9 | 1 |
| 5-split 3-stage pipeline against plain TIM | tim_3stage_bc_r5 / tim_2stage | 0.375 | 11 | 0 | 1 |
| 5-split 3-stage pipeline against psm | tim_3stage_bc_r5 / psm | 0.748 | 12 | 0 | 0 |
| 5-split 3-stage pipeline against robust | tim_3stage_bc_r5 / robust | 1.739 | 3 | 0 | 9 |
| 5-split 3-stage pipeline against dml | tim_3stage_bc_r5 / dml | 1.492 | 1 | 1 | 10 |
| 5-split 3-stage pipeline against tmle | tim_3stage_bc_r5 / tmle | 0.969 | 6 | 2 | 4 |
| 5-split 3-stage pipeline against malts_pkg | tim_3stage_bc_r5 / malts_pkg | 1.080 | 5 | 0 | 7 |
| 5-split 3-stage pipeline against flame_pkg | tim_3stage_bc_r5 / flame_pkg | 0.368 | 12 | 0 | 0 |
| single-split 3-stage pipeline against psm (the final run) | tim_3stage_bc / psm | 0.816 | 8 | 1 | 3 |
| single-split 3-stage pipeline against robust (the final run) | tim_3stage_bc / robust | 2.241 | 0 | 0 | 12 |
| single-split 3-stage pipeline against dml (the final run) | tim_3stage_bc / dml | 1.761 | 0 | 2 | 10 |
| single-split 3-stage pipeline against tmle (the final run) | tim_3stage_bc / tmle | 1.205 | 3 | 0 | 9 |
| single-split 3-stage pipeline against malts_pkg (the final run) | tim_3stage_bc / malts_pkg | 1.208 | 4 | 1 | 7 |
| single-split 3-stage pipeline against flame_pkg (the final run) | tim_3stage_bc / flame_pkg | 0.485 | 12 | 0 | 0 |

## Treated people used and time

| method | treated used | seconds per data set |
|---|---|---|
| tim_2stage | 0.856 | 1.274 |
| tim_2stage_bc | 0.856 | 1.412 |
| tim_2stage_bc_r5 | 0.860 | 1.904 |
| tim_3stage_bc | 0.864 | 5.938 |
| tim_3stage_bc_r5 | 0.864 | 10.048 |
| psm | 0.926 | 0.009 |
| robust | 0.357 | 0.023 |
| dml | 1.000 | 1.048 |
| tmle | 1.000 | 0.000 |
| malts_pkg | 1.000 | 45.052 |
| flame_pkg | 0.256 | 2.161 |
| naive | 1.000 | 0.000 |

## CDC: high cholesterol and diabetes (4,000 people per sample, 10 samples)

No true effect is known. The TIM paper reports 0.079. These are adjusted associations in the released sample (no survey weights), and every method assumes the 20 covariates capture all confounding. The MALTS package was not run on CDC (too slow).

| method | effect (mean over samples) | sd across samples | treated used | samples |
|---|---|---|---|---|
| tim_2stage | 0.0681 | 0.0250 | 0.9819 | 10 |
| tim_2stage_bc | 0.0503 | 0.0304 | 0.9819 | 10 |
| tim_2stage_bc_r5 | 0.0573 | 0.0207 | 0.9848 | 10 |
| tim_3stage_bc | 0.0585 | 0.0306 | 0.9744 | 10 |
| tim_3stage_bc_r5 | 0.0572 | 0.0202 | 0.9811 | 10 |
| psm | 0.0682 | 0.0219 | 0.9989 | 10 |
| robust | 0.0673 | 0.0143 | 0.8425 | 10 |
| dml | 0.0504 | 0.0154 | 1.0000 | 10 |
| tmle | 0.0533 | 0.0214 | 1.0000 | 10 |
| flame_pkg | 0.1187 | 0.0185 | 0.6793 | 10 |
| naive | 0.1377 | 0.0109 | 1.0000 | 10 |
