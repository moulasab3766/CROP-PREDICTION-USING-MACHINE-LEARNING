"""Measure held-out RF performance after removing one feature at a time."""

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
    ensure_research_directories,
    load_research_split,
    write_json,
)
from src.train import build_random_forest, calculate_classification_metrics


def ablation_configurations() -> OrderedDict[str, tuple[str, ...]]:
    """Return the baseline and seven deterministic leave-one-feature-out sets."""

    configurations: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    configurations["All 7 Features"] = FEATURE_NAMES
    display = {
        "N": "N",
        "P": "P",
        "K": "K",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "ph": "pH",
        "rainfall": "Rainfall",
    }
    for removed in FEATURE_NAMES:
        configurations[f"Without {display[removed]}"] = tuple(
            feature for feature in FEATURE_NAMES if feature != removed
        )
    return configurations


def add_ablation_deltas(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add signed differences from the all-seven-feature result."""

    records = [dict(row) for row in rows]
    baseline = next(
        (row for row in records if row.get("configuration") == "All 7 Features"),
        None,
    )
    if baseline is None:
        raise ValueError("Ablation rows require an All 7 Features baseline.")
    for row in records:
        row["accuracy_delta"] = float(row["accuracy"]) - float(baseline["accuracy"])
        row["macro_f1_delta"] = float(row["macro_f1"]) - float(baseline["macro_f1"])
    return records


def _save_plot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    names = [str(row["configuration"]) for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    macro_f1 = [float(row["macro_f1"]) for row in rows]
    positions = np.arange(len(names))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 6.5))
    axis.bar(positions - width / 2, accuracy, width, label="Accuracy", color="#276749")
    axis.bar(positions + width / 2, macro_f1, width, label="Macro F1", color="#6b8e5f")
    axis.set_xticks(positions, names, rotation=28, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Held-out metric")
    axis.set_title("Random Forest Leave-One-Feature-Out Ablation")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_feature_ablation(
    *,
    output_dir: str | Path = RESEARCH_RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit eight training-only RFs on one unchanged row split and compare them."""

    ensure_research_directories()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split = load_research_split()
    rows: list[dict[str, Any]] = []
    for configuration, features in ablation_configurations().items():
        model = build_random_forest()
        model.fit(split.X_train.loc[:, features], split.y_train)
        predictions = model.predict(split.X_test.loc[:, features])
        metrics = calculate_classification_metrics(split.y_test, predictions)
        removed = next((name for name in FEATURE_NAMES if name not in features), None)
        rows.append(
            {
                "configuration": configuration,
                "removed_feature": removed or "none",
                "feature_count": len(features),
                "feature_order": "|".join(features),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )
    rows = add_ablation_deltas(rows)
    csv_path = destination / "feature_ablation.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.12f")
    plot_path = destination / "feature_ablation.png"
    _save_plot(rows, plot_path)

    ablated = rows[1:]
    largest_degradation = min(ablated, key=lambda row: float(row["macro_f1_delta"]))
    improvements = [row for row in ablated if float(row["macro_f1_delta"]) > 1e-12]
    payload = {
        "experiment": "Leave-one-feature-out Random Forest ablation",
        "same_train_test_rows_for_every_configuration": True,
        "largest_macro_f1_degradation": largest_degradation,
        "removals_improving_macro_f1": improvements,
        "interpretation": (
            "Ablation measures dependence of this fitted model family on a feature under "
            "the benchmark split; it does not establish agricultural causality."
        ),
        "evidence_distinction": (
            "Ablation performance, RF impurity importance, and global SHAP importance "
            "measure different aspects of model behavior and are compared separately."
        ),
        "rows": rows,
        "artifacts": {"csv": str(csv_path), "plot": str(plot_path)},
    }
    write_json(destination / "feature_ablation_summary.json", payload)
    if verbose:
        print(json.dumps(payload, indent=2))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RESULTS_DIR)
    args = parser.parse_args(argv)
    run_feature_ablation(output_dir=args.output_dir, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
