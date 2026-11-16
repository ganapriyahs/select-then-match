# Step 2: regression and variable screening (CDC BRFSS 2015)

Outcome: diabetes. Exposure: high cholesterol. Regressions use all 253,680 people; screening and the AUC use a random sample of 20,000.

## 2.0 The association before adjustment

Risk difference 0.140, odds ratio 3.25.

## 2.1 Screening out variables with machine learning

| variable | permutation importance | LASSO |
|---|---|---|
| HighBP | kept | kept |
| CholCheck | removed | kept |
| BMI | kept | kept |
| Smoker | removed | kept |
| Stroke | removed | kept |
| HeartDiseaseorAttack | kept | kept |
| PhysActivity | removed | kept |
| Fruits | removed | kept |
| Veggies | removed | kept |
| HvyAlcoholConsump | removed | kept |
| AnyHealthcare | removed | kept |
| NoDocbcCost | removed | kept |
| GenHlth | kept | kept |
| MentHlth | removed | kept |
| PhysHlth | removed | kept |
| DiffWalk | removed | kept |
| Sex | removed | kept |
| Age | kept | kept |
| Education | removed | kept |
| Income | removed | kept |

Permutation importance keeps 5 of 20 variables, LASSO keeps 20. Permutation importance scores each variable on people the forests were not trained on, once for the outcome and once for the treatment, scales each score so the largest is 1 and takes the geometric mean; a variable is kept when its score is at least 5% of the top, so it has to matter for both. "Removed" means it did not help prediction enough; it does not show the variable is unnecessary for causal adjustment.

## 2.2 Adjusted regression

| model | variables | odds ratio | odds ratio 95% | risk difference | risk difference 95% | AUC (n=20,000, screening inside folds) |
|---|---|---|---|---|---|---|
| no adjustment (crude) | 0 | 3.255 | 3.18 to 3.33 | 0.140 | 0.137 to 0.143 |  |
| adjusted for: all covariates | 20 | 1.783 | 1.74 to 1.83 | 0.056 | 0.053 to 0.059 | 0.829 |
| adjusted for: kept by permutation importance | 5 | 1.789 | 1.74 to 1.84 | 0.056 | 0.053 to 0.059 | 0.822 |
| adjusted for: kept by LASSO | 20 | 1.783 | 1.74 to 1.83 | 0.056 | 0.053 to 0.059 | 0.829 |

Adjusting for all 20 covariates moves the risk difference from 0.140 to 0.056 (odds ratio 3.25 to 1.78). Adjusting only for the 5 kept variables gives 0.056. The AUC of the screened model is 0.822 against 0.829 with all variables (on average 5.2 variables kept per fold). The adjusted intervals ignore that the variables were chosen from the same data. These are associations, not effects: Step 3 asks what they become under matching and causal machine learning.
