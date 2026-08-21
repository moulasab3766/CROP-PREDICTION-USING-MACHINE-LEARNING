# Research Findings

## 1. Top-K Evaluation

On the evaluated 440-row benchmark test split, Top-1, Top-2, and Top-3 accuracy were
0.995455, 1.000000, and
1.000000. Top-K offers model-ranked alternatives, but it
does not establish that the alternatives are agronomically equivalent.

## 2. Probability Calibration

The raw RF log loss/ECE were 0.050389/0.037568; the
sigmoid-calibrated values were 0.083758/
0.070277. The predeclared reliability rule evaluated
calibration improvement as **False**.
Calibration does not convert these outputs into agricultural-success probabilities.

## 3. Local Explainability

Global mean-absolute SHAP ranked humidity, rainfall, K
as its three leading features. RF impurity importance and SHAP had Spearman rank
correlation 0.928571; they measure
different aspects of fitted-model behavior. In the documented local example,
orange was predicted at 0.970000; its
strongest contribution was K with SHAP
+0.337898. SHAP is not causal evidence.

## 4. Hyperparameter Tuning

The bounded training-only search selected `{"class_weight": "balanced", "max_depth": null, "max_features": "sqrt", "min_samples_leaf": 1, "min_samples_split": 10, "n_estimators": 200}`.
Against the production baseline, held-out macro-F1 was classified as
**tied**, with exact delta
+0.000000000000. The production model was not replaced.

## 5. Feature Ablation

The largest observed macro-F1 degradation occurred for **Without Rainfall**
(-0.027615). This indicates benchmark model dependence, not
agricultural causality. Ablation, impurity importance, and SHAP evidence are distinct.

## 6. Robustness Analysis

Within the tested controlled numerical ranges, **humidity** had the
largest sensitivity ordering, with flip rate 0.013636
and mean absolute top-probability change
0.067273. These perturbations are
sensitivity tests, not predictions of future field conditions.

## 7. Model Disagreement

The modal-vote distribution for agreement counts 1–6 was
`{"1": 0, "2": 0, "3": 2, "4": 8, "5": 13, "6": 417}`. There were
10 samples with at most four of six
models voting for the modal crop. This is inter-model disagreement, not formal
uncertainty.

## 8. Error Analysis

The baseline RF made 2 held-out errors. Every error is present
in `error_analysis.csv`; no cases were manufactured. Because this is a very small
group, probability, margin, agreement, and SHAP contrasts are descriptive only.

## 9. Major Limitations

- Results are specific to one public benchmark dataset and one fixed split.
- High benchmark accuracy does not establish farm, seasonal, regional, or sensor robustness.
- Raw and calibrated probabilities are model outputs, not success guarantees.
- SHAP, impurity importance, and ablation do not establish causal agronomic effects.
- Controlled perturbations and classifier disagreement are limited reliability indicators.
- External labeled and field validation remain unresolved requirements.
