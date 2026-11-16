# Can the 3-stage pipeline be made stronger?

12 simulated settings, 20 repeats each, seeds 10000 to 10019. Error = estimate minus the true effect on ALL treated people. `current` is the final pipeline (screen, TIM, correction). Each candidate changes one thing.

## The rule (written before the run)

A candidate PASSES only if, against `current`, (1) the median RMSE ratio over the 12 settings is at most 0.95, (2) the ratio is below 0.95 in at least 8 settings, and (3) it is above 1.05 in at most 1. A pass on seeds 9000 to 9019 is not an adoption: the candidate is run again, unchanged, on seeds 10000 to 10019 and adopted only if it passes again. The counts are counts of settings, not tests.

## The versions

| version | what it is |
|---|---|
| current | the final pipeline: screen, TIM, correction (two-half swap, boosting correction) |
| repeat5 | two-half swap repeated 5 times, estimates averaged |

## Verdict

`ratio` = RMSE of the candidate / RMSE of `current`; below 1 means the candidate is better.

| candidate | median ratio | better | tie | worse | clearly better / worse (paired z, info) | verdict |
|---|---|---|---|---|---|---|
| repeat5 | 0.832 | 12 | 0 | 0 | 1 / 0 | PASS |

## RMSE by setting (lower is better)

| setting | current | repeat5 |
|---|---|---|
| binary_linear | 0.067 | 0.038 |
| binary_linear_n500 | 0.111 | 0.087 |
| binary_linear_poor_overlap | 0.112 | 0.098 |
| binary_nonlinear | 0.078 | 0.067 |
| binary_nonlinear_n500 | 0.125 | 0.091 |
| binary_nonlinear_poor_overlap | 0.117 | 0.110 |
| continuous_linear | 0.257 | 0.229 |
| continuous_linear_n500 | 0.438 | 0.379 |
| continuous_linear_poor_overlap | 0.517 | 0.476 |
| continuous_nonlinear | 0.457 | 0.337 |
| continuous_nonlinear_n500 | 0.504 | 0.407 |
| continuous_nonlinear_poor_overlap | 0.876 | 0.616 |

## Ratio to `current` by setting

| setting | repeat5 |
|---|---|
| binary_linear | 0.558 |
| binary_linear_n500 | 0.783 |
| binary_linear_poor_overlap | 0.872 |
| binary_nonlinear | 0.857 |
| binary_nonlinear_n500 | 0.726 |
| binary_nonlinear_poor_overlap | 0.939 |
| continuous_linear | 0.888 |
| continuous_linear_n500 | 0.864 |
| continuous_linear_poor_overlap | 0.920 |
| continuous_nonlinear | 0.738 |
| continuous_nonlinear_n500 | 0.807 |
| continuous_nonlinear_poor_overlap | 0.703 |

## Average error (bias; positive = too high)

| setting | current | repeat5 |
|---|---|---|
| binary_linear | -0.006 | 0.013 |
| binary_linear_n500 | 0.031 | 0.043 |
| binary_linear_poor_overlap | 0.064 | 0.065 |
| binary_nonlinear | 0.028 | 0.038 |
| binary_nonlinear_n500 | 0.085 | 0.063 |
| binary_nonlinear_poor_overlap | 0.082 | 0.091 |
| continuous_linear | 0.171 | 0.192 |
| continuous_linear_n500 | 0.316 | 0.340 |
| continuous_linear_poor_overlap | 0.412 | 0.428 |
| continuous_nonlinear | 0.254 | 0.317 |
| continuous_nonlinear_n500 | 0.349 | 0.343 |
| continuous_nonlinear_poor_overlap | 0.672 | 0.553 |

## Cost

| version | treated used | variables used per treated person | seconds per data set |
|---|---|---|---|
| current | 0.871 | 4.580 | 1.913 |
| repeat5 | 0.865 | 4.591 | 9.596 |
