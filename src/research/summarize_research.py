"""Assemble measured experiment outputs into tables and cautious findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from src.research.common import (
    RESEARCH_MODELS_DIR,
    RESEARCH_RESULTS_DIR,
    class_positions,
    ensure_research_directories,
    load_baseline_artifacts,
    load_research_split,
    probability_metrics,
    validate_probability_matrix,
    write_json,
)
from src.research.evaluate_topk import top_k_correctness


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required research artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _topk_for_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    classes = np.asarray(model.classes_)
    probabilities = validate_probability_matrix(
        model.predict_proba(X_test),
        expected_rows=len(X_test),
        expected_classes=len(classes),
    )
    return {
        f"top_{k}_accuracy": float(
            np.mean(top_k_correctness(probabilities, y_test, classes, k=k))
        )
        for k in (1, 2, 3)
    }


def generate_research_summary(
    *,
    output_dir: str | Path = RESEARCH_RESULTS_DIR,
    models_dir: str | Path = RESEARCH_MODELS_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Create actual cross-experiment summary files and findings markdown."""

    ensure_research_directories()
    destination = Path(output_dir)
    model_destination = Path(models_dir)
    split = load_research_split()
    baseline_model, encoder = load_baseline_artifacts()
    encoded_classes, _ = class_positions(baseline_model, encoder)
    baseline_probabilities = baseline_model.predict_proba(split.X_test)
    baseline_metrics = probability_metrics(
        split.y_test, baseline_probabilities, encoded_classes
    )
    baseline_topk = _topk_for_model(baseline_model, split.X_test, split.y_test)

    calibrated_model = joblib.load(model_destination / "random_forest_calibrated.joblib")
    tuned_model = joblib.load(model_destination / "random_forest_tuned.joblib")
    models = [
        ("Baseline Random Forest", baseline_model, baseline_metrics, baseline_topk),
        (
            "Tuned Random Forest",
            tuned_model,
            probability_metrics(
                split.y_test,
                tuned_model.predict_proba(split.X_test),
                tuned_model.classes_,
            ),
            _topk_for_model(tuned_model, split.X_test, split.y_test),
        ),
        (
            "Sigmoid-Calibrated Random Forest",
            calibrated_model,
            probability_metrics(
                split.y_test,
                calibrated_model.predict_proba(split.X_test),
                calibrated_model.classes_,
            ),
            _topk_for_model(calibrated_model, split.X_test, split.y_test),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, _, metrics, topk in models:
        rows.append(
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Macro Precision": metrics["macro_precision"],
                "Macro Recall": metrics["macro_recall"],
                "Macro F1": metrics["macro_f1"],
                "Weighted F1": metrics["weighted_f1"],
                "Log Loss": metrics["log_loss"],
                "Top-1 Accuracy": topk["top_1_accuracy"],
                "Top-2 Accuracy": topk["top_2_accuracy"],
                "Top-3 Accuracy": topk["top_3_accuracy"],
            }
        )
    csv_path = destination / "research_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.12f")

    topk = _read_json(destination / "top_k_metrics.json")
    calibration = _read_json(destination / "calibration_summary.json")
    shap_summary = _read_json(destination / "shap_summary.json")
    tuning = _read_json(destination / "tuning_summary.json")
    ablation = _read_json(destination / "feature_ablation_summary.json")
    robustness = _read_json(destination / "robustness_summary.json")
    disagreement = _read_json(destination / "model_disagreement_summary.json")
    errors = _read_json(destination / "error_analysis_summary.json")
    payload = {
        "experiment": "Step-1500 research summary",
        "models": rows,
        "top_k": topk["metrics"],
        "calibration": {
            "improved": calibration["probability_quality_improved_by_decision_rule"],
            "best_method": calibration["best_calibration_method"],
        },
        "shap": {
            "top_global_features": shap_summary["top_global_features"],
            "rank_correlation_with_impurity": shap_summary[
                "impurity_vs_shap_rank_spearman"
            ],
        },
        "tuning": {
            "best_parameters": tuning["best_parameters"],
            "outcome": tuning["outcome_by_held_out_macro_f1"],
        },
        "ablation": {
            "largest_macro_f1_degradation": ablation[
                "largest_macro_f1_degradation"
            ]
        },
        "robustness": {
            "most_sensitivity_causing_feature": robustness[
                "most_sensitivity_causing_feature"
            ]
        },
        "inter_model_disagreement": {
            "agreement_distribution": disagreement["agreement_distribution"],
            "low_agreement_count": disagreement["low_agreement_at_most_four_count"],
        },
        "error_analysis": {
            "error_count": errors["error_count"],
            "descriptives": errors["correct_vs_incorrect_descriptives"],
        },
    }
    json_path = destination / "research_summary.json"
    write_json(json_path, payload)

    top_metrics = topk["metrics"]
    calibration_methods = {row["model"]: row for row in calibration["methods"]}
    raw = calibration_methods["Baseline Random Forest"]
    calibrated = calibration_methods["Sigmoid-Calibrated Random Forest"]
    local = shap_summary["local_examples"][0]
    largest = ablation["largest_macro_f1_degradation"]
    sensitive = robustness["most_sensitivity_causing_feature"]
    findings = f"""# Research Findings

## 1. Top-K Evaluation

On the evaluated 440-row benchmark test split, Top-1, Top-2, and Top-3 accuracy were
{top_metrics['top_1_accuracy']:.6f}, {top_metrics['top_2_accuracy']:.6f}, and
{top_metrics['top_3_accuracy']:.6f}. Top-K offers model-ranked alternatives, but it
does not establish that the alternatives are agronomically equivalent.

## 2. Probability Calibration

The raw RF log loss/ECE were {raw['log_loss']:.6f}/{raw['top_label_ece']:.6f}; the
sigmoid-calibrated values were {calibrated['log_loss']:.6f}/
{calibrated['top_label_ece']:.6f}. The predeclared reliability rule evaluated
calibration improvement as **{calibration['probability_quality_improved_by_decision_rule']}**.
Calibration does not convert these outputs into agricultural-success probabilities.

## 3. Local Explainability

Global mean-absolute SHAP ranked {', '.join(row['feature'] for row in shap_summary['top_global_features'])}
as its three leading features. RF impurity importance and SHAP had Spearman rank
correlation {shap_summary['impurity_vs_shap_rank_spearman']:.6f}; they measure
different aspects of fitted-model behavior. In the documented local example,
{local['predicted_crop']} was predicted at {local['model_probability']:.6f}; its
strongest contribution was {local['contributions'][0]['feature']} with SHAP
{local['contributions'][0]['shap_contribution']:+.6f}. SHAP is not causal evidence.

## 4. Hyperparameter Tuning

The bounded training-only search selected `{json.dumps(tuning['best_parameters'], sort_keys=True)}`.
Against the production baseline, held-out macro-F1 was classified as
**{tuning['outcome_by_held_out_macro_f1']}**, with exact delta
{tuning['held_out_macro_f1_delta']:+.12f}. The production model was not replaced.

## 5. Feature Ablation

The largest observed macro-F1 degradation occurred for **{largest['configuration']}**
({largest['macro_f1_delta']:+.6f}). This indicates benchmark model dependence, not
agricultural causality. Ablation, impurity importance, and SHAP evidence are distinct.

## 6. Robustness Analysis

Within the tested controlled numerical ranges, **{sensitive['feature']}** had the
largest sensitivity ordering, with flip rate {sensitive['prediction_flip_rate']:.6f}
and mean absolute top-probability change
{sensitive['average_absolute_top_probability_change']:.6f}. These perturbations are
sensitivity tests, not predictions of future field conditions.

## 7. Model Disagreement

The modal-vote distribution for agreement counts 1–6 was
`{json.dumps(disagreement['agreement_distribution'], sort_keys=True)}`. There were
{disagreement['low_agreement_at_most_four_count']} samples with at most four of six
models voting for the modal crop. This is inter-model disagreement, not formal
uncertainty.

## 8. Error Analysis

The baseline RF made {errors['error_count']} held-out errors. Every error is present
in `error_analysis.csv`; no cases were manufactured. Because this is a very small
group, probability, margin, agreement, and SHAP contrasts are descriptive only.

## 9. Major Limitations

- Results are specific to one public benchmark dataset and one fixed split.
- High benchmark accuracy does not establish farm, seasonal, regional, or sensor robustness.
- Raw and calibrated probabilities are model outputs, not success guarantees.
- SHAP, impurity importance, and ablation do not establish causal agronomic effects.
- Controlled perturbations and classifier disagreement are limited reliability indicators.
- External labeled and field validation remain unresolved requirements.
"""
    findings_path = destination / "findings.md"
    findings_path.write_text(findings, encoding="utf-8")
    # Re-serialize every research JSON with the common path policy so artifacts
    # stay portable and never record a developer's absolute workspace path.
    for json_artifact in destination.glob("*.json"):
        write_json(json_artifact, _read_json(json_artifact))
    if verbose:
        print(json.dumps(payload, indent=2))
        print(f"Saved research findings to {findings_path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RESULTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=RESEARCH_MODELS_DIR)
    args = parser.parse_args(argv)
    generate_research_summary(
        output_dir=args.output_dir, models_dir=args.models_dir, verbose=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
