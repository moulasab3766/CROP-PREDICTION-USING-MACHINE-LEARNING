"""Generate local and global SHAP explanations for the baseline Random Forest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.explain import (
    contribution_records,
    local_explanation_text,
    normalize_multiclass_shap,
)
from src.preprocessing import FEATURE_NAMES
from src.research.common import (
    BASELINE_MODEL_PATH,
    RESEARCH_RESULTS_DIR,
    class_positions,
    ensure_research_directories,
    load_baseline_artifacts,
    load_research_split,
    python_value,
    sha256_file,
    validate_probability_matrix,
    write_json,
)

def explain_one_sample(
    model: Any,
    encoder: Any,
    shap_values: np.ndarray,
    sample_position: int,
    sample: pd.Series,
) -> dict[str, Any]:
    """Build one predicted-class local explanation from precomputed SHAP values."""

    encoded_classes, decoded_classes = class_positions(model, encoder)
    frame = sample.to_frame().T.loc[:, FEATURE_NAMES]
    probabilities = validate_probability_matrix(
        model.predict_proba(frame), expected_rows=1, expected_classes=len(encoded_classes)
    )[0]
    predicted_column = int(np.argmax(probabilities))
    predicted_encoded = int(encoded_classes[predicted_column])
    predicted_crop = str(decoded_classes[predicted_column])
    contributions = shap_values[sample_position, :, predicted_column]
    records = contribution_records(sample.loc[list(FEATURE_NAMES)].to_numpy(), contributions)
    return {
        "sample_position": int(sample_position),
        "predicted_encoded_class": predicted_encoded,
        "predicted_crop": predicted_crop,
        "model_probability": float(probabilities[predicted_column]),
        "contributions": records,
        "text": local_explanation_text(predicted_crop, records),
    }


def _plot_global(rows: pd.DataFrame, path: Path) -> None:
    ordered = rows.sort_values("mean_absolute_shap", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(ordered["feature"], ordered["mean_absolute_shap"], color="#276749")
    axis.set_title("Global SHAP Importance Across Held-Out Samples and Classes")
    axis.set_xlabel("Mean absolute SHAP contribution")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_local_examples(examples: Sequence[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(len(examples), 1, figsize=(10, 4.1 * len(examples)))
    if len(examples) == 1:
        axes = [axes]
    for axis, example in zip(axes, examples):
        records = list(reversed(example["contributions"]))
        values = [float(row["shap_contribution"]) for row in records]
        colors = ["#2f855a" if value >= 0 else "#c05640" for value in values]
        axis.barh([str(row["feature"]) for row in records], values, color=colors)
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_xlabel("SHAP contribution to predicted-class model output")
        correctness = "correct" if example["correct"] else "misclassified"
        axis.set_title(
            f"{example['selection']}: predicted {example['predicted_crop']} "
            f"({example['model_probability']:.3f}), actual {example['true_crop']} "
            f"— {correctness}"
        )
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Local SHAP Explanations (Model Behavior, Not Causality)", y=1.0)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_shap_analysis(
    *,
    output_dir: str | Path = RESEARCH_RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate actual global and selected local held-out SHAP artifacts."""

    ensure_research_directories()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split = load_research_split()
    model, encoder = load_baseline_artifacts()
    encoded_classes, decoded_classes = class_positions(model, encoder)
    model_hash_before = sha256_file(BASELINE_MODEL_PATH)

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(split.X_test)
    shap_values = normalize_multiclass_shap(
        raw_values,
        n_samples=len(split.X_test),
        n_features=len(FEATURE_NAMES),
        n_classes=len(encoded_classes),
    )
    probabilities = validate_probability_matrix(
        model.predict_proba(split.X_test),
        expected_rows=440,
        expected_classes=len(encoded_classes),
    )
    predicted_columns = np.argmax(probabilities, axis=1)
    predictions = encoded_classes[predicted_columns]
    ordered_probabilities = np.sort(probabilities, axis=1)[:, ::-1]
    margins = ordered_probabilities[:, 0] - ordered_probabilities[:, 1]
    correct_mask = predictions == split.y_test

    mean_absolute = np.mean(np.abs(shap_values), axis=(0, 2))
    impurity = np.asarray(model.feature_importances_, dtype=float)
    shap_rank = pd.Series(-mean_absolute).rank(method="min").astype(int).to_numpy()
    impurity_rank = pd.Series(-impurity).rank(method="min").astype(int).to_numpy()
    global_rows = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "mean_absolute_shap": mean_absolute,
            "shap_rank": shap_rank,
            "rf_impurity_importance": impurity,
            "rf_impurity_rank": impurity_rank,
            "rank_difference": shap_rank - impurity_rank,
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    global_csv = destination / "shap_global_importance.csv"
    global_rows.to_csv(global_csv, index=False, float_format="%.12f")
    global_plot = destination / "shap_global_importance.png"
    _plot_global(global_rows, global_plot)

    correct_position = int(np.flatnonzero(correct_mask)[0])
    low_margin_order = np.argsort(margins, kind="stable")
    low_margin_position = int(
        next(position for position in low_margin_order if int(position) != correct_position)
    )
    incorrect_positions = np.flatnonzero(~correct_mask)
    selections: list[tuple[str, int]] = [
        ("Correctly predicted example", correct_position),
        ("Lowest-margin distinct example", low_margin_position),
    ]
    if len(incorrect_positions):
        incorrect_position = int(incorrect_positions[0])
        if incorrect_position not in {correct_position, low_margin_position}:
            selections.append(("Misclassified example", incorrect_position))
        else:
            selections[-1] = ("Low-margin misclassified example", incorrect_position)

    examples: list[dict[str, Any]] = []
    for selection, position in selections:
        explanation = explain_one_sample(
            model,
            encoder,
            shap_values,
            position,
            split.X_test.iloc[position],
        )
        true_crop = str(encoder.inverse_transform([split.y_test[position]])[0])
        explanation.update(
            {
                "selection": selection,
                "source_row_index": int(split.X_test.index[position]),
                "true_crop": true_crop,
                "correct": bool(correct_mask[position]),
                "top_1_minus_top_2_margin": float(margins[position]),
            }
        )
        examples.append(explanation)

    local_plot = destination / "shap_local_example.png"
    _plot_local_examples(examples, local_plot)
    local_json = destination / "shap_local_examples.json"
    write_json(local_json, python_value(examples))
    correlation = spearmanr(mean_absolute, impurity).statistic
    model_hash_after = sha256_file(BASELINE_MODEL_PATH)
    if model_hash_before != model_hash_after:
        raise RuntimeError("Baseline model changed during SHAP analysis.")

    payload = {
        "experiment": "Baseline Random Forest SHAP analysis",
        "shap_version": shap.__version__,
        "normalized_shap_shape": [int(value) for value in shap_values.shape],
        "class_count": len(decoded_classes),
        "global_method": "mean absolute SHAP across held-out samples and all class outputs",
        "top_global_features": global_rows.head(3).to_dict(orient="records"),
        "impurity_vs_shap_rank_spearman": float(correlation),
        "comparison_note": (
            "Impurity importance summarizes split usage inside fitted trees; mean absolute "
            "SHAP summarizes contribution magnitudes over evaluated samples and class outputs. "
            "They measure different behavior and need not have identical rankings."
        ),
        "local_examples": python_value(examples),
        "limitations": [
            "SHAP explains model behavior and is not an agronomic causal analysis.",
            "SHAP does not prove crop suitability or justify fertilizer advice.",
            "Global values are conditional on the fitted model and evaluated benchmark split.",
            "A dense multiclass beeswarm was omitted to avoid an unreadable 22-output plot.",
        ],
        "artifacts": {
            "global_csv": str(global_csv),
            "global_plot": str(global_plot),
            "local_examples_json": str(local_json),
            "local_plot": str(local_plot),
        },
        "baseline_model_sha256_before": model_hash_before,
        "baseline_model_sha256_after": model_hash_after,
    }
    write_json(destination / "shap_summary.json", python_value(payload))
    if verbose:
        print(json.dumps(python_value(payload), indent=2))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RESULTS_DIR)
    args = parser.parse_args(argv)
    run_shap_analysis(output_dir=args.output_dir, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
