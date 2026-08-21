"""Analyze every held-out baseline Random Forest error without inventing cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import shap

from src.preprocessing import FEATURE_NAMES
from src.research.common import (
    RESEARCH_RESULTS_DIR,
    class_positions,
    ensure_research_directories,
    load_baseline_artifacts,
    load_research_split,
    validate_probability_matrix,
    write_json,
)
from src.research.model_disagreement import run_model_disagreement
from src.research.shap_explain import contribution_records, normalize_multiclass_shap


def grouped_error_descriptives(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Compare probability, margin, and agreement for correct versus error groups."""

    required = {"correct", "top_1_probability", "top_1_vs_top_2_margin", "agreement_count"}
    if not required.issubset(frame.columns):
        raise ValueError("Error comparison frame is missing required columns.")
    rows = []
    for correct, group in frame.groupby("correct", sort=False):
        rows.append(
            {
                "group": "correct" if bool(correct) else "incorrect",
                "sample_count": int(len(group)),
                "mean_top_1_probability": float(group["top_1_probability"].mean()),
                "mean_top_1_vs_top_2_margin": float(group["top_1_vs_top_2_margin"].mean()),
                "mean_agreement_count": float(group["agreement_count"].mean()),
            }
        )
    return rows


def run_error_analysis(
    *,
    output_dir: str | Path = RESEARCH_RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Record Top-3, margin, agreement, and SHAP evidence for all RF mistakes."""

    ensure_research_directories()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    disagreement_path = destination / "model_disagreement.csv"
    if not disagreement_path.is_file():
        run_model_disagreement(output_dir=destination, verbose=False)
    disagreement = pd.read_csv(disagreement_path).set_index("test_position")

    split = load_research_split()
    model, encoder = load_baseline_artifacts()
    encoded_classes, decoded_classes = class_positions(model, encoder)
    probabilities = validate_probability_matrix(
        model.predict_proba(split.X_test),
        expected_rows=440,
        expected_classes=len(encoded_classes),
    )
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranked_probabilities = np.take_along_axis(probabilities, order, axis=1)
    ranked_encoded = encoded_classes[order]
    ranked_names = decoded_classes[order]
    predictions = ranked_encoded[:, 0]
    correct = predictions == split.y_test
    error_positions = np.flatnonzero(~correct)

    explainer = shap.TreeExplainer(model)
    error_shap: np.ndarray | None = None
    if len(error_positions):
        raw = explainer.shap_values(split.X_test.iloc[error_positions])
        error_shap = normalize_multiclass_shap(
            raw,
            n_samples=len(error_positions),
            n_features=len(FEATURE_NAMES),
            n_classes=len(encoded_classes),
        )

    comparison_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for position in range(len(split.X_test)):
        base = {
            "correct": bool(correct[position]),
            "top_1_probability": float(ranked_probabilities[position, 0]),
            "top_1_vs_top_2_margin": float(
                ranked_probabilities[position, 0] - ranked_probabilities[position, 1]
            ),
            "agreement_count": int(disagreement.loc[position, "agreement_count"]),
        }
        comparison_rows.append(base)

    for error_offset, position in enumerate(error_positions):
        predicted_column = int(order[position, 0])
        assert error_shap is not None
        contributions = contribution_records(
            split.X_test.iloc[position].loc[list(FEATURE_NAMES)].to_numpy(),
            error_shap[error_offset, :, predicted_column],
        )
        strongest = contributions[0]
        true_crop = str(encoder.inverse_transform([split.y_test[position]])[0])
        true_in_top3 = bool(np.any(ranked_encoded[position, :3] == split.y_test[position]))
        error_rows.append(
            {
                "test_position": int(position),
                "source_row_index": int(split.X_test.index[position]),
                "actual_crop": true_crop,
                "predicted_crop": str(ranked_names[position, 0]),
                "top_1_probability": float(ranked_probabilities[position, 0]),
                "second_crop": str(ranked_names[position, 1]),
                "second_probability": float(ranked_probabilities[position, 1]),
                "third_crop": str(ranked_names[position, 2]),
                "third_probability": float(ranked_probabilities[position, 2]),
                "top_1_vs_top_2_margin": float(
                    ranked_probabilities[position, 0] - ranked_probabilities[position, 1]
                ),
                "true_crop_in_top_3": true_in_top3,
                "agreement_count": int(disagreement.loc[position, "agreement_count"]),
                "models_agreeing_with_random_forest": int(
                    disagreement.loc[position, "models_agreeing_with_random_forest"]
                ),
                "strongest_predicted_class_shap_feature": strongest["feature"],
                "strongest_predicted_class_shap_contribution": strongest[
                    "shap_contribution"
                ],
                "strongest_predicted_class_shap_direction": strongest["direction"],
            }
        )

    errors_csv = destination / "error_analysis.csv"
    pd.DataFrame(error_rows).to_csv(errors_csv, index=False, float_format="%.12f")
    comparison = grouped_error_descriptives(pd.DataFrame(comparison_rows))
    payload = {
        "experiment": "Held-out baseline Random Forest error analysis",
        "held_out_samples": len(split.X_test),
        "error_count": int(len(error_rows)),
        "errors": error_rows,
        "correct_vs_incorrect_descriptives": comparison,
        "interpretation": (
            "Every observed error is reported. Because the number is very small, "
            "differences from correct cases are descriptive and must not be generalized."
        ),
        "shap_note": (
            "The strongest SHAP contribution is for the predicted class and explains "
            "model behavior, not an agronomic cause."
        ),
        "limitations": [
            "Very few errors make subgroup averages unstable.",
            "The benchmark may not represent field, seasonal, sensor, or regional shifts.",
            "Inter-model agreement and probability margins are reliability indicators, not formal uncertainty.",
        ],
        "artifacts": {"errors_csv": str(errors_csv)},
    }
    write_json(destination / "error_analysis_summary.json", payload)
    if verbose:
        print(json.dumps(payload, indent=2))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RESULTS_DIR)
    args = parser.parse_args(argv)
    run_error_analysis(output_dir=args.output_dir, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
