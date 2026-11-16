# Does a different Step 1 make the pipeline stronger?

12 simulated settings, 20 repeats each, seeds 11000 to 11019. Error = estimate minus the true effect on ALL treated people. `current` is the base pipeline (**current**). Each candidate changes only the Step 1 score (the variable ranking).

## The rule (written before the run)

A candidate PASSES only if, against `current`, (1) the median RMSE ratio over the 12 settings is at most 0.95, (2) the ratio is below 0.95 in at least 8 settings, and (3) it is above 1.05 in at most 1. A pass on seeds 11000 to 11019 is not an adoption: the candidate is run again, unchanged, on seeds 12000 to 12019 and adopted only if it passes again. The counts are counts of settings, not tests.

## The versions

| version | Step 1 score |
|---|---|
| current | TIM's Step 1: average of the scaled outcome score and treatment score |
| ridge_raw | ridge on raw columns for both models, scores added (close version of the public code) |
| out_only | only the outcome score |
| max | the larger of the outcome score and the treatment score |
| ovb | outcome effect x gap between treated and untreated in the variable's average |
| trt_only | context: only the treatment score |

## Verdict

`ratio` = RMSE of the candidate / RMSE of `current`; below 1 means the candidate is better.

| candidate | median ratio | better | tie | worse | clearly better / worse (paired z, info) | verdict |
|---|---|---|---|---|---|---|
| ridge_raw | 0.940 | 7 | 4 | 1 | 0 / 0 | no reliable gain |
| out_only | 0.887 | 7 | 2 | 3 | 0 / 0 | no reliable gain |
| max | 0.974 | 5 | 2 | 5 | 1 / 1 | no reliable gain |
| ovb | 0.952 | 5 | 6 | 1 | 0 / 0 | no reliable gain |

## RMSE by setting (lower is better)

| setting | current | ridge_raw | out_only | max | ovb | trt_only |
|---|---|---|---|---|---|---|
| binary_linear | 0.090 | 0.061 | 0.068 | 0.072 | 0.089 | 0.086 |
| binary_linear_n500 | 0.101 | 0.102 | 0.084 | 0.130 | 0.097 | 0.089 |
| binary_linear_poor_overlap | 0.128 | 0.106 | 0.109 | 0.107 | 0.116 | 0.129 |
| binary_nonlinear | 0.087 | 0.075 | 0.066 | 0.092 | 0.075 | 0.091 |
| binary_nonlinear_n500 | 0.086 | 0.090 | 0.077 | 0.112 | 0.062 | 0.102 |
| binary_nonlinear_poor_overlap | 0.161 | 0.159 | 0.166 | 0.156 | 0.142 | 0.197 |
| continuous_linear | 0.342 | 0.323 | 0.380 | 0.226 | 0.336 | 0.279 |
| continuous_linear_n500 | 0.430 | 0.403 | 0.436 | 0.525 | 0.540 | 0.475 |
| continuous_linear_poor_overlap | 0.514 | 0.429 | 0.407 | 0.452 | 0.511 | 0.484 |
| continuous_nonlinear | 0.435 | 0.467 | 0.476 | 0.365 | 0.414 | 0.435 |
| continuous_nonlinear_n500 | 0.627 | 0.616 | 0.677 | 0.616 | 0.580 | 0.753 |
| continuous_nonlinear_poor_overlap | 0.618 | 0.513 | 0.543 | 0.822 | 0.594 | 0.814 |

## Ratio to `current` by setting

| setting | ridge_raw | out_only | max | ovb |
|---|---|---|---|---|
| binary_linear | 0.683 | 0.758 | 0.804 | 0.987 |
| binary_linear_n500 | 1.004 | 0.828 | 1.284 | 0.954 |
| binary_linear_poor_overlap | 0.828 | 0.853 | 0.830 | 0.904 |
| binary_nonlinear | 0.869 | 0.759 | 1.058 | 0.870 |
| binary_nonlinear_n500 | 1.047 | 0.897 | 1.302 | 0.727 |
| binary_nonlinear_poor_overlap | 0.984 | 1.027 | 0.966 | 0.882 |
| continuous_linear | 0.943 | 1.109 | 0.659 | 0.981 |
| continuous_linear_n500 | 0.937 | 1.016 | 1.220 | 1.256 |
| continuous_linear_poor_overlap | 0.835 | 0.793 | 0.879 | 0.994 |
| continuous_nonlinear | 1.072 | 1.094 | 0.839 | 0.950 |
| continuous_nonlinear_n500 | 0.983 | 1.080 | 0.982 | 0.925 |
| continuous_nonlinear_poor_overlap | 0.830 | 0.878 | 1.331 | 0.961 |

## Average error (bias; positive = too high)

| setting | current | ridge_raw | out_only | max | ovb | trt_only |
|---|---|---|---|---|---|---|
| binary_linear | 0.027 | 0.005 | 0.019 | 0.019 | 0.014 | 0.054 |
| binary_linear_n500 | 0.019 | 0.022 | 0.023 | 0.026 | 0.017 | 0.017 |
| binary_linear_poor_overlap | 0.030 | 0.076 | 0.052 | 0.073 | 0.060 | 0.090 |
| binary_nonlinear | 0.040 | 0.044 | 0.029 | 0.058 | 0.027 | 0.050 |
| binary_nonlinear_n500 | 0.054 | 0.041 | 0.026 | 0.075 | 0.019 | 0.064 |
| binary_nonlinear_poor_overlap | 0.136 | 0.138 | 0.125 | 0.131 | 0.109 | 0.169 |
| continuous_linear | 0.196 | 0.184 | 0.194 | 0.103 | 0.240 | 0.201 |
| continuous_linear_n500 | 0.282 | 0.277 | 0.310 | 0.437 | 0.394 | 0.414 |
| continuous_linear_poor_overlap | 0.430 | 0.358 | 0.337 | 0.338 | 0.436 | 0.358 |
| continuous_nonlinear | 0.318 | 0.393 | 0.376 | 0.277 | 0.345 | 0.365 |
| continuous_nonlinear_n500 | 0.528 | 0.562 | 0.563 | 0.504 | 0.442 | 0.662 |
| continuous_nonlinear_poor_overlap | 0.451 | 0.417 | 0.440 | 0.699 | 0.509 | 0.750 |

## Cost

| version | treated used | variables used per treated person | seconds per data set |
|---|---|---|---|
| current | 0.867 | 4.635 | 1.975 |
| ridge_raw | 0.841 | 5.086 | 1.963 |
| out_only | 0.869 | 4.958 | 1.988 |
| max | 0.840 | 4.705 | 1.984 |
| ovb | 0.868 | 4.886 | 1.976 |
| trt_only | 0.851 | 4.576 | 1.976 |

## Where each score puts each kind of variable (information only)

Average position over all settings and data sets (1 = kept longest, 20 = dropped first; 10.5 would be no information). The simulated data has 4 real confounders, 3 variables that predict only the outcome, 3 that predict only the treatment, and 10 noise variables. Scores are computed on the whole data set here, not on a learning half.

| score | C real confounders (4) | P outcome only (3) | I treatment only (3) | N noise (10) |
|---|---|---|---|---|
| linear | 2.780 | 9.107 | 5.818 | 15.410 |
| out_only | 4.107 | 5.106 | 13.643 | 13.732 |
| trt_only | 3.979 | 14.328 | 4.050 | 13.895 |
| max | 3.622 | 9.146 | 4.604 | 15.426 |
| geo | 2.975 | 8.746 | 7.883 | 14.821 |
| ovb | 3.708 | 5.474 | 12.446 | 14.141 |
| ridge_raw | 4.552 | 6.254 | 7.258 | 15.125 |
