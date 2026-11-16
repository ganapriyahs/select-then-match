# Step 1: descriptive statistics (CDC BRFSS 2015)

253,680 people. Treatment: high cholesterol. Outcome: diabetes (prediabetes counts as no). Missing values: 0.

## The two groups

|  | treated (high cholesterol) | untreated |
|---|---|---|
| people | 107,591 | 146,089 |
| share of everyone | 42.4% | 57.6% |
| had diabetes | 23,686 | 11,660 |
| diabetes rate | 22.0% | 8.0% |

## The raw gap (no adjustment)

People with high cholesterol have a diabetes rate 0.140 higher (0.137 to 0.143); risk ratio 2.76 (2.70 to 2.82); odds ratio 3.25 (3.18 to 3.33). A comparison, not yet an effect: the groups differ in many other ways.

## Covariates, largest difference first (14 of 20 have |SMD| above 0.1)

| variable | type | treated | untreated | SMD |
|---|---|---|---|---|
| HighBP | yes/no | 0.601 | 0.302 | 0.629 |
| Age | number | 9.001 | 7.318 | 0.582 |
| GenHlth | number | 2.771 | 2.320 | 0.430 |
| HeartDiseaseorAttack | yes/no | 0.156 | 0.049 | 0.358 |
| DiffWalk | yes/no | 0.231 | 0.122 | 0.290 |
| PhysHlth | number | 5.479 | 3.331 | 0.244 |
| BMI | number | 29.204 | 27.777 | 0.217 |
| Smoker | yes/no | 0.496 | 0.404 | 0.185 |
| Stroke | yes/no | 0.062 | 0.025 | 0.182 |
| CholCheck | yes/no | 0.982 | 0.949 | 0.180 |
| Income | number | 5.848 | 6.206 | -0.173 |
| PhysActivity | yes/no | 0.718 | 0.785 | -0.157 |
| Education | number | 4.969 | 5.110 | -0.143 |
| MentHlth | number | 3.721 | 2.790 | 0.124 |
| AnyHealthcare | yes/no | 0.962 | 0.943 | 0.087 |
| Fruits | yes/no | 0.611 | 0.651 | -0.083 |
| Veggies | yes/no | 0.793 | 0.825 | -0.080 |
| Sex | yes/no | 0.458 | 0.427 | 0.063 |
| NoDocbcCost | yes/no | 0.088 | 0.081 | 0.027 |
| HvyAlcoholConsump | yes/no | 0.053 | 0.058 | -0.023 |

`treated` and `untreated` are the average (for yes/no columns, the share with a yes).
