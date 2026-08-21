"""Analyze inter-model disagreement across the six baseline classifiers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.compare_models import build_model_candidates
from src.research.common import (
    RESEARCH_RESULTS_DIR,
    ensure_research_directories,
    load_research_split,
    validate_probability_matrix,
    write_json,
)


MODEL_COLUMN_NAMES = {
    "Random Forest": "random_forest_prediction",
    "Gaussian Naive Bayes": "gaussian_nb_prediction",
    "Support Vector Machine": "svm_prediction",
    "Decision Tree": "decision_tree_prediction",
    "K-Nearest Neighbors": "knn_prediction",
    "Logistic Regression": "logistic_regression_prediction",
}


def consensus_agreement(predictions: Sequence[Any]) -> tuple[int, Any]:
    """Return maximum vote count and deterministic modal prediction."""

    if not predictions:
        raise ValueError("At least one model prediction is required.")
    counts = Counter(predictions)
    maximum = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == maximum)
    return int(maximum), winners[0]


def _save_plot(frame: pd.DataFrame, path: Path) -> None:
    counts = frame["agreement_count"].value_counts().sort_index()
    groups = sorted(frame["agreement_count"].unique())
    mean_margin = frame.groupby("agreement_count")["rf_top_1_minus_top_2_margin"].mean()
    figure, first = plt.subplots(figsize=(9, 5.8))
    positions = np.asarray(groups)
    bars = first.bar(positions, [int(counts[group]) for group in groups], color="#276749")
    first.set_xlabel("Models agreeing on the modal crop (out of 6)")
    first.set_ylabel("Held-out sample count")
    first.set_xticks(range(1, 7))
    first.set_title("Inter-Model Agreement and Random Forest Margin")
    first.grid(axis="y", alpha=0.25)
    second = first.twinx()
    line = second.plot(
        positions,
        [float(mean_margin[group]) for group in groups],
        color="#b05c33",
        marker="o",
        linewidth=2,
        label="Mean RF top-two margin",
    )[0]
    second.set_ylabel("Mean RF Top-1 minus Top-2 probability")
    first.legend([bars, line], ["Sample count", line.get_label()], loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_model_disagreement(
    *,
    output_dir: str | Path = RESEARCH_RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit the original six definitions on one split and compare predictions."""

    ensure_research_directories()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split = load_research_split()
    candidates = build_model_candidates()
    encoded_predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}
    for name, model in candidates.items():
        model.fit(split.X_train, split.y_train)
        encoded_predictions[name] = np.asarray(model.predict(split.X_test), dtype=int)
        fitted[name] = model

    rf = fitted["Random Forest"]
    rf_probabilities = validate_probability_matrix(
        rf.predict_proba(split.X_test),
        expected_rows=440,
        expected_classes=len(split.label_encoder.classes_),
    )
    rf_order = np.argsort(-rf_probabilities, axis=1, kind="stable")
    rf_ranked_probabilities = np.take_along_axis(rf_probabilities, rf_order, axis=1)
    decoded_predictions = {
        name: split.label_encoder.inverse_transform(values)
        for name, values in encoded_predictions.items()
    }

    rows: list[dict[str, Any]] = []
    for position, source_index in enumerate(split.X_test.index.tolist()):
        votes = [decoded_predictions[name][position] for name in candidates]
        agreement_count, consensus_crop = consensus_agreement(votes)
        rf_crop = str(decoded_predictions["Random Forest"][position])
        row: dict[str, Any] = {
            "test_position": position,
            "source_row_index": int(source_index),
            "true_crop": str(
                split.label_encoder.inverse_transform([split.y_test[position]])[0]
            ),
            "agreement_count": agreement_count,
            "consensus_crop": str(consensus_crop),
            "models_agreeing_with_random_forest": int(sum(vote == rf_crop for vote in votes)),
            "rf_top_probability": float(rf_ranked_probabilities[position, 0]),
            "rf_top_1_minus_top_2_margin": float(
                rf_ranked_probabilities[position, 0] - rf_ranked_probabilities[position, 1]
            ),
        }
        for name in candidates:
            row[MODEL_COLUMN_NAMES[name]] = str(decoded_predictions[name][position])
        rows.append(row)
    frame = pd.DataFrame(rows)
    csv_path = destination / "model_disagreement.csv"
    frame.to_csv(csv_path, index=False, float_format="%.12f")
    plot_path = destination / "model_disagreement.png"
    _save_plot(frame, plot_path)

    distribution = {
        str(count): int((frame["agreement_count"] == count).sum())
        for count in range(1, 7)
    }
    low_agreement = frame[frame["agreement_count"] <= 4]
    correlation = spearmanr(
        frame["rf_top_1_minus_top_2_margin"], frame["agreement_count"]
    ).statistic
    payload = {
        "experiment": "Inter-model disagreement using original six classifier definitions",
        "sample_count": len(frame),
        "agreement_definition": (
            "agreement_count is the number of models voting for the modal crop; ties in "
            "the modal crop name are broken alphabetically only for deterministic reporting."
        ),
        "agreement_distribution": distribution,
        "six_of_six_count": distribution["6"],
        "five_of_six_count": distribution["5"],
        "four_of_six_count": distribution["4"],
        "low_agreement_at_most_four_count": int(len(low_agreement)),
        "mean_rf_probability_by_agreement": {
            str(key): float(value)
            for key, value in frame.groupby("agreement_count")["rf_top_probability"].mean().items()
        },
        "spearman_rf_margin_vs_agreement_count": float(correlation),
        "interpretation": (
            "This is inter-model disagreement among six established classifiers, not a "
            "formal posterior uncertainty estimate."
        ),
        "limitations": [
            "The six models are not independent samples from a probabilistic model family.",
            "Agreement can be high even when all models share dataset-specific biases.",
            "Results apply only to the fixed benchmark split and model definitions.",
        ],
        "artifacts": {"csv": str(csv_path), "plot": str(plot_path)},
    }
    write_json(destination / "model_disagreement_summary.json", payload)
    if verbose:
        print(json.dumps(payload, indent=2))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RESULTS_DIR)
    args = parser.parse_args(argv)
    run_model_disagreement(output_dir=args.output_dir, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
