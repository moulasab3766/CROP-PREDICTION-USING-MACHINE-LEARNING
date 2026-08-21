"""Controlled held-out input-perturbation sensitivity analysis."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


SAMPLES_PER_CLASS = 5


def perturbation_specification() -> OrderedDict[str, dict[str, Any]]:
    """Return documented, modest numerical changes for every baseline feature."""

    return OrderedDict(
        [
            ("N", {"kind": "absolute", "changes": [-10.0, -5.0, 0.0, 5.0, 10.0]}),
            ("P", {"kind": "absolute", "changes": [-10.0, -5.0, 0.0, 5.0, 10.0]}),
            ("K", {"kind": "absolute", "changes": [-10.0, -5.0, 0.0, 5.0, 10.0]}),
            (
                "temperature",
                {"kind": "absolute", "changes": [-2.0, -1.0, 0.0, 1.0, 2.0]},
            ),
            (
                "humidity",
                {"kind": "absolute", "changes": [-5.0, -2.5, 0.0, 2.5, 5.0]},
            ),
            ("ph", {"kind": "absolute", "changes": [-0.2, -0.1, 0.0, 0.1, 0.2]}),
            (
                "rainfall",
                {"kind": "relative", "changes": [-0.10, -0.05, 0.0, 0.05, 0.10]},
            ),
        ]
    )


def stratified_sample_positions(y: Sequence[int] | np.ndarray, per_class: int) -> np.ndarray:
    """Select the first deterministic held-out positions from each encoded class."""

    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 1:
        raise ValueError("per_class must be a positive integer.")
    labels = np.asarray(y)
    positions: list[int] = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f"Class {label} has fewer than {per_class} samples.")
        positions.extend(int(position) for position in candidates[:per_class])
    return np.asarray(sorted(positions), dtype=int)


def apply_controlled_change(
    values: np.ndarray,
    change: float,
    *,
    kind: str,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one change and clip to training-observed bounds, reporting clips."""

    numeric = np.asarray(values, dtype=float)
    if kind == "absolute":
        proposed = numeric + change
    elif kind == "relative":
        proposed = numeric * (1.0 + change)
    else:
        raise ValueError("Perturbation kind must be 'absolute' or 'relative'.")
    clipped = np.clip(proposed, minimum, maximum)
    return clipped, ~np.isclose(clipped, proposed, rtol=0.0, atol=1e-12)


def _ranking_names(
    probabilities: np.ndarray,
    decoded_classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, axis=1, kind="stable")
    ranked_names = decoded_classes[order]
    ranked_probabilities = np.take_along_axis(probabilities, order, axis=1)
    return order, ranked_names, ranked_probabilities


def _save_plot(summary: pd.DataFrame, path: Path) -> None:
    figure, first = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(summary))
    bars = first.bar(
        positions,
        summary["prediction_flip_rate"],
        color="#276749",
        label="Top-1 flip rate",
    )
    first.set_xticks(positions, summary["feature"], rotation=25, ha="right")
    first.set_ylim(0.0, max(0.05, float(summary["prediction_flip_rate"].max()) * 1.2))
    first.set_ylabel("Prediction flip rate")
    first.set_title("Controlled Numerical Perturbation Sensitivity")
    first.grid(axis="y", alpha=0.25)
    second = first.twinx()
    second.plot(
        positions,
        summary["average_absolute_top_probability_change"],
        color="#b05c33",
        marker="o",
        linewidth=2,
        label="Mean |top-probability change|",
    )
    second.set_ylabel("Mean absolute top-probability change")
    handles = [bars, second.lines[0]]
    first.legend(handles, [handle.get_label() for handle in handles], loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_robustness_analysis(
    *,
    output_dir: str | Path = RESEARCH_RESULTS_DIR,
    samples_per_class: int = SAMPLES_PER_CLASS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Measure prediction changes under bounded, non-forecast perturbations."""

    ensure_research_directories()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split = load_research_split()
    model, encoder = load_baseline_artifacts()
    encoded_classes, decoded_classes = class_positions(model, encoder)
    selected_positions = stratified_sample_positions(split.y_test, samples_per_class)
    selected = split.X_test.iloc[selected_positions].copy().astype(float)
    original_probabilities = validate_probability_matrix(
        model.predict_proba(selected),
        expected_rows=len(selected),
        expected_classes=len(encoded_classes),
    )
    original_order, original_names, original_ranked_probabilities = _ranking_names(
        original_probabilities, decoded_classes
    )
    original_encoded = encoded_classes[original_order[:, 0]]

    bounds = {
        feature: (
            float(split.X_train[feature].min()),
            float(split.X_train[feature].max()),
        )
        for feature in FEATURE_NAMES
    }
    detailed: list[dict[str, Any]] = []
    for feature, specification in perturbation_specification().items():
        for change in specification["changes"]:
            perturbed = selected.copy()
            changed_values, clipped = apply_controlled_change(
                selected[feature].to_numpy(),
                float(change),
                kind=str(specification["kind"]),
                minimum=bounds[feature][0],
                maximum=bounds[feature][1],
            )
            perturbed.loc[:, feature] = changed_values
            perturbed_probabilities = validate_probability_matrix(
                model.predict_proba(perturbed),
                expected_rows=len(perturbed),
                expected_classes=len(encoded_classes),
            )
            perturbed_order, perturbed_names, perturbed_ranked_probabilities = _ranking_names(
                perturbed_probabilities, decoded_classes
            )
            perturbed_encoded = encoded_classes[perturbed_order[:, 0]]
            for local_position, test_position in enumerate(selected_positions):
                original_class_column = int(original_order[local_position, 0])
                original_class_rank = int(
                    np.flatnonzero(
                        perturbed_order[local_position] == original_class_column
                    )[0]
                )
                detailed.append(
                    {
                        "test_position": int(test_position),
                        "source_row_index": int(split.X_test.index[test_position]),
                        "true_crop": str(
                            encoder.inverse_transform([split.y_test[test_position]])[0]
                        ),
                        "feature": feature,
                        "change_kind": str(specification["kind"]),
                        "requested_change": float(change),
                        "original_feature_value": float(selected.iloc[local_position][feature]),
                        "perturbed_feature_value": float(changed_values[local_position]),
                        "clipped_to_training_range": bool(clipped[local_position]),
                        "original_predicted_crop": str(original_names[local_position, 0]),
                        "perturbed_predicted_crop": str(perturbed_names[local_position, 0]),
                        "original_top_probability": float(
                            original_ranked_probabilities[local_position, 0]
                        ),
                        "perturbed_top_probability": float(
                            perturbed_ranked_probabilities[local_position, 0]
                        ),
                        "original_top_3": "|".join(original_names[local_position, :3]),
                        "perturbed_top_3": "|".join(perturbed_names[local_position, :3]),
                        "top_1_changed": bool(
                            perturbed_encoded[local_position] != original_encoded[local_position]
                        ),
                        "original_top_1_minus_top_2_margin": float(
                            original_ranked_probabilities[local_position, 0]
                            - original_ranked_probabilities[local_position, 1]
                        ),
                        "perturbed_top_1_minus_top_2_margin": float(
                            perturbed_ranked_probabilities[local_position, 0]
                            - perturbed_ranked_probabilities[local_position, 1]
                        ),
                        "absolute_top_probability_change": float(
                            abs(
                                perturbed_ranked_probabilities[local_position, 0]
                                - original_ranked_probabilities[local_position, 0]
                            )
                        ),
                        "original_top_1_rank_displacement": original_class_rank,
                    }
                )

    detailed_frame = pd.DataFrame(detailed)
    nonzero = detailed_frame[~np.isclose(detailed_frame["requested_change"], 0.0)]
    summary = (
        nonzero.groupby("feature", sort=False)
        .agg(
            evaluated_perturbations=("top_1_changed", "size"),
            prediction_flips=("top_1_changed", "sum"),
            prediction_flip_rate=("top_1_changed", "mean"),
            average_absolute_top_probability_change=(
                "absolute_top_probability_change",
                "mean",
            ),
            average_original_top_1_rank_displacement=(
                "original_top_1_rank_displacement",
                "mean",
            ),
            clipped_values=("clipped_to_training_range", "sum"),
        )
        .reset_index()
    )
    detailed_csv = destination / "robustness_detailed.csv"
    summary_csv = destination / "robustness_summary.csv"
    detailed_frame.to_csv(detailed_csv, index=False, float_format="%.12f")
    summary.to_csv(summary_csv, index=False, float_format="%.12f")
    plot_path = destination / "robustness_analysis.png"
    _save_plot(summary, plot_path)
    most_sensitive = summary.sort_values(
        ["prediction_flip_rate", "average_absolute_top_probability_change"],
        ascending=False,
    ).iloc[0].to_dict()
    payload = {
        "experiment": "Controlled benchmark input perturbation sensitivity",
        "selected_samples": int(len(selected_positions)),
        "samples_per_crop_class": samples_per_class,
        "selection": "first deterministic held-out positions within each encoded class",
        "perturbations": perturbation_specification(),
        "clipping_policy": (
            "Perturbed values are clipped to the minimum and maximum observed in the "
            "training partition; every clipped value is flagged in the detailed CSV."
        ),
        "summary": summary.to_dict(orient="records"),
        "most_sensitivity_causing_feature": most_sensitive,
        "metric_definitions": {
            "flip_rate": "fraction of non-zero perturbations changing Top-1 crop",
            "probability_change": "absolute change in the maximum probability",
            "ranking_change": "zero-based displacement of the original Top-1 crop",
        },
        "limitations": [
            "Perturbations are numerical sensitivity tests, not forecasts or sensor-error models.",
            "Results do not establish robustness to all real agricultural conditions.",
            "Training-range clipping can reduce the effective perturbation near boundaries.",
        ],
        "artifacts": {
            "summary_csv": str(summary_csv),
            "detailed_csv": str(detailed_csv),
            "plot": str(plot_path),
        },
    }
    write_json(destination / "robustness_summary.json", payload)
    if verbose:
        print(json.dumps(payload, indent=2))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RESULTS_DIR)
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    args = parser.parse_args(argv)
    run_robustness_analysis(
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
