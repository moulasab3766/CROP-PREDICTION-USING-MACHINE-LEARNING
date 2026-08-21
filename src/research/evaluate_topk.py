"""Evaluate Top-K behavior of the protected baseline Random Forest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.research.common import (
    BASELINE_MODEL_PATH,
    RESEARCH_RESULTS_DIR,
    class_positions,
    ensure_research_directories,
    load_baseline_artifacts,
    load_research_split,
    mean_or_none,
    sha256_file,
    validate_probability_matrix,
    write_json,
)


NEAR_TIE_MARGIN = 0.05


def ranked_class_positions(probabilities: Any) -> np.ndarray:
    """Return descending class-column positions for every probability row."""

    matrix = validate_probability_matrix(probabilities)
    return np.argsort(-matrix, axis=1, kind="stable")


def top_k_correctness(
    probabilities: Any,
    y_true: Sequence[int] | np.ndarray,
    classes: Sequence[int] | np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Return whether each true encoded class appears among the top ``k``."""

    class_array = np.asarray(classes)
    truth = np.asarray(y_true)
    matrix = validate_probability_matrix(
        probabilities,
        expected_rows=len(truth),
        expected_classes=len(class_array),
    )
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(class_array):
        raise ValueError("k must be an integer within the available class count.")
    order = ranked_class_positions(matrix)
    ranked_classes = class_array[order[:, :k]]
    return np.any(ranked_classes == truth[:, None], axis=1)


def calculate_top_k_summary(
    probabilities: Any,
    y_true: Sequence[int] | np.ndarray,
    classes: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Calculate monotonic Top-1/2/3/5 and descriptive margin statistics."""

    truth = np.asarray(y_true)
    class_array = np.asarray(classes)
    matrix = validate_probability_matrix(
        probabilities,
        expected_rows=len(truth),
        expected_classes=len(class_array),
    )
    top1 = top_k_correctness(matrix, truth, class_array, k=1)
    top2 = top_k_correctness(matrix, truth, class_array, k=2)
    top3 = top_k_correctness(matrix, truth, class_array, k=3)
    top5 = top_k_correctness(matrix, truth, class_array, k=min(5, len(class_array)))
    accuracies = [float(np.mean(values)) for values in (top1, top2, top3, top5)]
    if not all(left <= right for left, right in zip(accuracies, accuracies[1:])):
        raise AssertionError("Top-K accuracy must be monotonic as K increases.")

    order = ranked_class_positions(matrix)
    ordered_probabilities = np.take_along_axis(matrix, order, axis=1)
    top_probability = ordered_probabilities[:, 0]
    margin = ordered_probabilities[:, 0] - ordered_probabilities[:, 1]
    correct_probabilities = top_probability[top1]
    incorrect_probabilities = top_probability[~top1]
    correct_margins = margin[top1]
    incorrect_margins = margin[~top1]

    return {
        "sample_count": int(len(truth)),
        "class_count": int(len(class_array)),
        "probability_matrix_shape": [int(value) for value in matrix.shape],
        "top_1_accuracy": accuracies[0],
        "top_2_accuracy": accuracies[1],
        "top_3_accuracy": accuracies[2],
        "top_5_accuracy": accuracies[3],
        "top_1_correct_count": int(np.sum(top1)),
        "top_2_correct_count": int(np.sum(top2)),
        "top_3_correct_count": int(np.sum(top3)),
        "top_5_correct_count": int(np.sum(top5)),
        "additional_correct_top_1_to_2": int(np.sum(top2) - np.sum(top1)),
        "additional_correct_top_2_to_3": int(np.sum(top3) - np.sum(top2)),
        "outside_top_3_count": int(np.sum(~top3)),
        "top_1_probability": {
            "mean": float(np.mean(top_probability)),
            "median": float(np.median(top_probability)),
            "minimum": float(np.min(top_probability)),
            "maximum": float(np.max(top_probability)),
            "mean_when_correct": mean_or_none(correct_probabilities.tolist()),
            "mean_when_incorrect": mean_or_none(incorrect_probabilities.tolist()),
        },
        "top_1_minus_top_2_margin": {
            "definition": "top_1_probability - top_2_probability",
            "mean": float(np.mean(margin)),
            "median": float(np.median(margin)),
            "mean_when_correct": mean_or_none(correct_margins.tolist()),
            "mean_when_incorrect": mean_or_none(incorrect_margins.tolist()),
            "near_tie_threshold": NEAR_TIE_MARGIN,
            "near_tie_count": int(np.sum(margin <= NEAR_TIE_MARGIN)),
        },
    }


def _save_plot(summary: dict[str, Any], path: Path) -> None:
    labels = ["Top-1", "Top-2", "Top-3"]
    values = [summary[f"top_{index}_accuracy"] for index in (1, 2, 3)]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(labels, values, color=["#245b3a", "#3d7f55", "#6aa374"])
    axis.set_title("Baseline Random Forest Top-K Accuracy")
    axis.set_ylabel("Held-out accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.015, 1.025),
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def evaluate_top_k(
    *,
    output_dir: str | Path = RESEARCH_RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full held-out Top-K experiment without fitting the baseline model."""

    ensure_research_directories()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split = load_research_split()
    model, encoder = load_baseline_artifacts()
    encoded_classes, decoded_classes = class_positions(model, encoder)
    if not np.array_equal(encoder.classes_, split.label_encoder.classes_):
        raise RuntimeError("Saved encoder class order differs from reconstructed encoding.")

    model_hash_before = sha256_file(BASELINE_MODEL_PATH)
    probabilities = validate_probability_matrix(
        model.predict_proba(split.X_test),
        expected_rows=440,
        expected_classes=len(decoded_classes),
    )
    summary = calculate_top_k_summary(probabilities, split.y_test, encoded_classes)
    order = ranked_class_positions(probabilities)
    ordered_probabilities = np.take_along_axis(probabilities, order, axis=1)
    ranked_encoded = encoded_classes[order]
    ranked_decoded = decoded_classes[order]
    top1_correct = ranked_encoded[:, 0] == split.y_test
    top2_correct = np.any(ranked_encoded[:, :2] == split.y_test[:, None], axis=1)
    top3_correct = np.any(ranked_encoded[:, :3] == split.y_test[:, None], axis=1)

    records: list[dict[str, Any]] = []
    for position, source_index in enumerate(split.X_test.index.tolist()):
        true_crop = str(encoder.inverse_transform([split.y_test[position]])[0])
        records.append(
            {
                "test_position": position,
                "source_row_index": int(source_index),
                "true_crop": true_crop,
                "top_1_crop": str(ranked_decoded[position, 0]),
                "top_1_probability": float(ordered_probabilities[position, 0]),
                "top_2_crop": str(ranked_decoded[position, 1]),
                "top_2_probability": float(ordered_probabilities[position, 1]),
                "top_3_crop": str(ranked_decoded[position, 2]),
                "top_3_probability": float(ordered_probabilities[position, 2]),
                "top_1_correct": bool(top1_correct[position]),
                "top_2_correct": bool(top2_correct[position]),
                "top_3_correct": bool(top3_correct[position]),
                "top_1_minus_top_2_margin": float(
                    ordered_probabilities[position, 0] - ordered_probabilities[position, 1]
                ),
            }
        )
    predictions_path = destination / "top_k_predictions.csv"
    pd.DataFrame(records).to_csv(predictions_path, index=False, float_format="%.12f")
    plot_path = destination / "top_k_accuracy.png"
    _save_plot(summary, plot_path)

    outside_top3 = [record for record in records if not record["top_3_correct"]]
    payload = {
        "experiment": "Scientific Top-K evaluation",
        "methodology": {
            "baseline_model_retrained": False,
            "held_out_samples": 440,
            "probability_source": "RandomForestClassifier.predict_proba",
            "class_order_verified_against_saved_encoder": True,
            "top_k_interpretation": (
                "Decision-support alternatives from model probability ranking; "
                "not evidence that crops are agronomically equivalent."
            ),
        },
        "metrics": summary,
        "outside_top_3_samples": outside_top3,
        "limitations": [
            "The probabilities are raw model outputs and are not guaranteed confidence.",
            "Results apply only to this dataset and fixed split.",
            "Top-K ranking is a decision-support enhancement, not a novel classifier.",
        ],
        "artifacts": {
            "predictions_csv": str(predictions_path),
            "accuracy_plot": str(plot_path),
        },
        "baseline_model_sha256_before": model_hash_before,
        "baseline_model_sha256_after": sha256_file(BASELINE_MODEL_PATH),
    }
    if payload["baseline_model_sha256_before"] != payload["baseline_model_sha256_after"]:
        raise RuntimeError("Baseline model hash changed during Top-K evaluation.")
    metrics_path = destination / "top_k_metrics.json"
    write_json(metrics_path, payload)
    if verbose:
        print(json.dumps(summary, indent=2))
        print(f"Saved Top-K artifacts under {destination}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RESULTS_DIR)
    args = parser.parse_args(argv)
    evaluate_top_k(output_dir=args.output_dir, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
