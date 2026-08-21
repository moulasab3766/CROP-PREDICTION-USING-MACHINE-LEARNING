"""Evaluate sigmoid calibration without changing production inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.research.common import (
    BASELINE_MODEL_PATH,
    RESEARCH_MODELS_DIR,
    RESEARCH_RESULTS_DIR,
    class_positions,
    ensure_research_directories,
    expected_calibration_error,
    load_baseline_artifacts,
    load_research_split,
    probability_metrics,
    sha256_file,
    top_label_calibration_bins,
    validate_probability_matrix,
    write_json,
)
from src.train import RANDOM_STATE, build_random_forest


CALIBRATION_FOLDS = 5
ECE_BINS = 10
MAX_ACCEPTABLE_MACRO_F1_DROP = 0.01


def evaluate_probability_model(
    name: str,
    y_true: np.ndarray,
    probabilities: Any,
    classes: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    """Calculate common metrics and equal-width top-label ECE."""

    matrix = validate_probability_matrix(
        probabilities,
        expected_rows=len(y_true),
        expected_classes=len(classes),
    )
    metrics: dict[str, Any] = {"model": name}
    metrics.update(probability_metrics(y_true, matrix, classes))
    bins = top_label_calibration_bins(
        y_true, matrix, classes, n_bins=ECE_BINS
    )
    metrics["top_label_ece"] = expected_calibration_error(bins)
    metrics["ece_bins"] = ECE_BINS
    return metrics, bins


def calibration_improved(
    baseline: dict[str, Any],
    calibrated: dict[str, Any],
) -> bool:
    """Apply the predeclared conservative future-UI consideration rule."""

    return bool(
        float(calibrated["log_loss"]) < float(baseline["log_loss"])
        and float(calibrated["top_label_ece"]) < float(baseline["top_label_ece"])
        and float(calibrated["macro_f1"])
        >= float(baseline["macro_f1"]) - MAX_ACCEPTABLE_MACRO_F1_DROP
    )


def _save_reliability_plot(
    baseline_bins: list[dict[str, float | int]],
    calibrated_bins: list[dict[str, float | int]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Ideal")
    for name, bins, color, marker in (
        ("Baseline RF", baseline_bins, "#245b3a", "o"),
        ("Sigmoid calibrated RF", calibrated_bins, "#b05c33", "s"),
    ):
        populated = [row for row in bins if int(row["count"]) > 0]
        axis.plot(
            [float(row["mean_confidence"]) for row in populated],
            [float(row["observed_accuracy"]) for row in populated],
            marker=marker,
            linewidth=2,
            label=name,
            color=color,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean predicted top-label probability")
    axis.set_ylabel("Observed top-label accuracy")
    axis.set_title("Multiclass Top-Label Reliability (10 Equal-Width Bins)")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def calibrate_probabilities(
    *,
    output_dir: str | Path = RESEARCH_RESULTS_DIR,
    model_output: str | Path = RESEARCH_MODELS_DIR / "random_forest_calibrated.joblib",
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit sigmoid calibration on training-only CV and evaluate once on test data."""

    ensure_research_directories()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    calibrated_model_path = Path(model_output)
    calibrated_model_path.parent.mkdir(parents=True, exist_ok=True)

    split = load_research_split()
    baseline_model, encoder = load_baseline_artifacts()
    encoded_classes, _ = class_positions(baseline_model, encoder)
    baseline_hash_before = sha256_file(BASELINE_MODEL_PATH)

    baseline_probabilities = baseline_model.predict_proba(split.X_test)
    baseline_metrics, baseline_bins = evaluate_probability_model(
        "Baseline Random Forest", split.y_test, baseline_probabilities, encoded_classes
    )

    cv = StratifiedKFold(
        n_splits=CALIBRATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=build_random_forest(),
        method="sigmoid",
        cv=cv,
        n_jobs=1,
        ensemble=True,
    )
    calibrated_model.fit(split.X_train, split.y_train)
    calibrated_probabilities = validate_probability_matrix(
        calibrated_model.predict_proba(split.X_test),
        expected_rows=440,
        expected_classes=len(encoded_classes),
    )
    if not np.array_equal(calibrated_model.classes_, encoded_classes):
        raise RuntimeError("Calibrated model class order differs from the baseline.")
    calibrated_metrics, calibrated_bins = evaluate_probability_model(
        "Sigmoid-Calibrated Random Forest",
        split.y_test,
        calibrated_probabilities,
        encoded_classes,
    )
    joblib.dump(calibrated_model, calibrated_model_path)

    metrics_path = destination / "calibration_metrics.csv"
    metric_columns = [
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "log_loss",
        "mean_true_class_probability",
        "multiclass_brier_score",
        "mean_top_probability",
        "top_label_ece",
        "ece_bins",
    ]
    pd.DataFrame([baseline_metrics, calibrated_metrics], columns=metric_columns).to_csv(
        metrics_path, index=False, float_format="%.12f"
    )

    plot_path = destination / "calibration_reliability.png"
    _save_reliability_plot(baseline_bins, calibrated_bins, plot_path)
    improved = calibration_improved(baseline_metrics, calibrated_metrics)
    baseline_hash_after = sha256_file(BASELINE_MODEL_PATH)
    if baseline_hash_before != baseline_hash_after:
        raise RuntimeError("Baseline model changed during calibration research.")

    payload = {
        "experiment": "Training-only cross-validated probability calibration",
        "best_calibration_method": "sigmoid" if improved else "baseline raw probabilities",
        "probability_quality_improved_by_decision_rule": improved,
        "future_ui_decision": (
            "Sigmoid calibration merits later UI evaluation."
            if improved
            else "Keep raw production probabilities pending stronger calibration evidence."
        ),
        "decision_rule": {
            "requires_lower_log_loss": True,
            "requires_lower_top_label_ece": True,
            "maximum_macro_f1_drop": MAX_ACCEPTABLE_MACRO_F1_DROP,
        },
        "methods": [baseline_metrics, calibrated_metrics],
        "reliability_methodology": {
            "transformation": "multiclass top-label calibration",
            "confidence": "maximum predicted class probability per sample",
            "outcome": "whether the top predicted label is correct",
            "binning": "10 fixed equal-width bins over [0, 1]",
            "ece": "count-weighted absolute difference between bin accuracy and confidence",
            "held_out_labels_used_for_fitting": False,
            "calibration_training": "5-fold stratified CV inside the 1,760-row training partition",
        },
        "isotonic_status": {
            "tested": False,
            "reason": (
                "Skipped prospectively because each training CV calibration fold has "
                "limited per-class support; non-parametric isotonic calibration is prone "
                "to overfitting with small calibration samples."
            ),
        },
        "limitations": [
            "Calibration does not turn model output into agricultural-success probability.",
            "The comparison applies only to this dataset and fixed held-out split.",
            "Top-label ECE combines classes and can hide crop-specific calibration patterns.",
        ],
        "artifacts": {
            "metrics_csv": str(metrics_path),
            "reliability_plot": str(plot_path),
            "calibrated_model": str(calibrated_model_path),
        },
        "baseline_model_sha256_before": baseline_hash_before,
        "baseline_model_sha256_after": baseline_hash_after,
    }
    summary_path = destination / "calibration_summary.json"
    write_json(summary_path, payload)
    if verbose:
        print(json.dumps(payload, indent=2))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RESULTS_DIR)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=RESEARCH_MODELS_DIR / "random_forest_calibrated.joblib",
    )
    args = parser.parse_args(argv)
    calibrate_probabilities(
        output_dir=args.output_dir,
        model_output=args.model_output,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
