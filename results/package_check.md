# Package check: my FLAME / MALTS against the authors' packages

dame-flame 0.81, pymalts 0.0.2. Package default settings. 5 repeats per setting, seeds 6000 to 6004, same data as the main study. Error = estimate minus the true effect on all treated people.

## RMSE (lower is better)

| setting | flame (mine) | flame (package) | malts (mine) | malts (package) | tim_2stage_bc | tim_3stage_bc |
|---|---|---|---|---|---|---|
| binary_linear | 0.214 | 0.207 | 0.111 | 0.076 | 0.124 | 0.108 |
| binary_nonlinear | 0.134 | 0.213 | 0.124 | 0.054 | 0.098 | 0.128 |
| continuous_linear | 0.189 | 1.275 | 0.740 | 0.376 | 0.264 | 0.305 |

## Average error (positive = too high)

| setting | flame (mine) | flame (package) | malts (mine) | malts (package) | tim_2stage_bc | tim_3stage_bc |
|---|---|---|---|---|---|---|
| binary_linear | 0.208 | 0.195 | 0.109 | -0.048 | 0.023 | 0.005 |
| binary_nonlinear | 0.125 | 0.201 | 0.116 | -0.037 | -0.001 | 0.066 |
| continuous_linear | 0.099 | 1.089 | 0.723 | 0.336 | 0.194 | 0.167 |

## Share of treated people used

| setting | flame (mine) | flame (package) | malts (mine) | malts (package) | tim_2stage_bc | tim_3stage_bc |
|---|---|---|---|---|---|---|
| binary_linear | 0.856 | 0.234 | 1.000 | 1.000 | 0.873 | 0.889 |
| binary_nonlinear | 0.935 | 0.475 | 1.000 | 1.000 | 0.909 | 0.920 |
| continuous_linear | 0.504 | 0.368 | 1.000 | 1.000 | 0.856 | 0.869 |