# Can the 3-stage pipeline be made stronger?

12 simulated settings, 20 repeats each, seeds 9000 to 9019. Error = estimate minus the true effect on ALL treated people. `current` is the final pipeline (screen, TIM, correction). Each candidate changes one thing.

## The rule (written before the run)

A candidate PASSES only if, against `current`, (1) the median RMSE ratio over the 12 settings is at most 0.95, (2) the ratio is below 0.95 in at least 8 settings, and (3) it is above 1.05 in at most 1. A pass on seeds 9000 to 9019 is not an adoption: the candidate is run again, unchanged, on seeds 10000 to 10019 and adopted only if it passes again. The counts are counts of settings, not tests.

## The versions

| version | what it is |
|---|---|
| current | the final pipeline: screen, TIM, correction (two-half swap, boosting correction) |
| full | no swap: learn on everybody, match everybody, cross-fitted correction |
| repeat5 | two-half swap repeated 5 times, estimates averaged |
| mu0avg | correction model = average of boosting and a linear model |
| plain_tim | context: TIM as designed |
| tim_bc | context: TIM + correction, no screening |

## Verdict

`ratio` = RMSE of the candidate / RMSE of `current`; below 1 means the candidate is better.

| candidate | median ratio | better | tie | worse | clearly better / worse (paired z, info) | verdict |
|---|---|---|---|---|---|---|
| full | 0.868 | 8 | 0 | 4 | 0 / 0 | no reliable gain |
| repeat5 | 0.792 | 12 | 0 | 0 | 3 / 0 | PASS |
| mu0avg | 1.082 | 5 | 1 | 6 | 3 / 6 | no reliable gain |

## RMSE by setting (lower is better)

| setting | current | full | repeat5 | mu0avg | plain_tim | tim_bc |
|---|---|---|---|---|---|---|
| binary_linear | 0.080 | 0.060 | 0.046 | 0.075 | 0.235 | 0.067 |
| binary_linear_n500 | 0.114 | 0.163 | 0.076 | 0.116 | 0.298 | 0.130 |
| binary_linear_poor_overlap | 0.113 | 0.098 | 0.078 | 0.104 | 0.350 | 0.121 |
| binary_nonlinear | 0.097 | 0.104 | 0.064 | 0.111 | 0.097 | 0.101 |
| binary_nonlinear_n500 | 0.105 | 0.089 | 0.075 | 0.130 | 0.158 | 0.111 |
| binary_nonlinear_poor_overlap | 0.125 | 0.098 | 0.107 | 0.169 | 0.095 | 0.114 |
| continuous_linear | 0.324 | 0.275 | 0.247 | 0.235 | 1.355 | 0.294 |
| continuous_linear_n500 | 0.445 | 0.493 | 0.397 | 0.293 | 1.574 | 0.430 |
| continuous_linear_poor_overlap | 0.504 | 0.354 | 0.470 | 0.287 | 2.043 | 0.517 |
| continuous_nonlinear | 0.364 | 0.388 | 0.299 | 0.703 | 0.739 | 0.354 |
| continuous_nonlinear_n500 | 0.688 | 0.609 | 0.597 | 0.951 | 0.977 | 0.633 |
| continuous_nonlinear_poor_overlap | 0.625 | 0.542 | 0.543 | 1.065 | 0.697 | 0.616 |

## Ratio to `current` by setting

| setting | full | repeat5 | mu0avg |
|---|---|---|---|
| binary_linear | 0.749 | 0.581 | 0.938 |
| binary_linear_n500 | 1.427 | 0.671 | 1.016 |
| binary_linear_poor_overlap | 0.870 | 0.690 | 0.919 |
| binary_nonlinear | 1.073 | 0.656 | 1.149 |
| binary_nonlinear_n500 | 0.850 | 0.715 | 1.236 |
| binary_nonlinear_poor_overlap | 0.785 | 0.858 | 1.347 |
| continuous_linear | 0.850 | 0.762 | 0.727 |
| continuous_linear_n500 | 1.108 | 0.893 | 0.658 |
| continuous_linear_poor_overlap | 0.704 | 0.933 | 0.570 |
| continuous_nonlinear | 1.066 | 0.823 | 1.932 |
| continuous_nonlinear_n500 | 0.884 | 0.867 | 1.381 |
| continuous_nonlinear_poor_overlap | 0.866 | 0.869 | 1.703 |

## Average error (bias; positive = too high)

| setting | current | full | repeat5 | mu0avg | plain_tim | tim_bc |
|---|---|---|---|---|---|---|
| binary_linear | -0.002 | 0.005 | 0.012 | 0.008 | 0.221 | 0.014 |
| binary_linear_n500 | 0.057 | 0.051 | 0.051 | 0.052 | 0.281 | 0.076 |
| binary_linear_poor_overlap | 0.079 | 0.053 | 0.069 | 0.064 | 0.340 | 0.089 |
| binary_nonlinear | 0.016 | 0.009 | 0.012 | 0.063 | 0.055 | 0.012 |
| binary_nonlinear_n500 | 0.046 | 0.029 | 0.057 | 0.099 | 0.120 | 0.054 |
| binary_nonlinear_poor_overlap | 0.079 | 0.053 | 0.077 | 0.145 | 0.068 | 0.059 |
| continuous_linear | 0.203 | 0.052 | 0.212 | 0.100 | 1.301 | 0.167 |
| continuous_linear_n500 | 0.285 | 0.200 | 0.334 | 0.127 | 1.472 | 0.291 |
| continuous_linear_poor_overlap | 0.386 | 0.203 | 0.411 | 0.143 | 1.919 | 0.390 |
| continuous_nonlinear | 0.266 | 0.078 | 0.230 | 0.627 | 0.580 | 0.266 |
| continuous_nonlinear_n500 | 0.489 | 0.163 | 0.463 | 0.823 | 0.821 | 0.483 |
| continuous_nonlinear_poor_overlap | 0.499 | 0.327 | 0.482 | 0.987 | 0.461 | 0.478 |

## Cost

| version | treated used | variables used per treated person | seconds per data set |
|---|---|---|---|
| current | 0.873 | 4.699 | 2.102 |
| full | 0.884 | 5.195 | 1.506 |
| repeat5 | 0.873 | 4.640 | 10.480 |
| mu0avg | 0.873 | 4.699 | 2.085 |
| plain_tim | 0.863 | 4.741 | 0.330 |
| tim_bc | 0.863 | 4.741 | 0.368 |
