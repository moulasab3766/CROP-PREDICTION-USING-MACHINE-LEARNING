"""Fair held-out comparison of six crop-recommendation classifiers."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # Supports module and direct-file execution.
    from .preprocessing import (
        DEFAULT_DATASET_PATH,
        DatasetValidationError,
        FEATURE_NAMES,
        preprocess_data,
    )
    from .train import (
        RANDOM_STATE,
        FeatureLeakageError,
        build_random_forest,
        calculate_classification_metrics,
        create_train_test_split,
        require_no_feature_overlap,
    )
except ImportError:  # pragma: no cover - direct `python src/compare_models.py`.
    from preprocessing import (  # type: ignore
        DEFAULT_DATASET_PATH,
        DatasetValidationError,
        FEATURE_NAMES,
        preprocess_data,
    )
    from train import (  # type: ignore
        RANDOM_STATE,
        FeatureLeakageError,
        build_random_forest,
        calculate_classification_metrics,
        create_train_test_split,
        require_no_feature_overlap,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
TIE_TOLERANCE = 1e-12

COMPARISON_METRICS: tuple[str, ...] = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
)
CSV_COLUMN_NAMES = {
    "model": "Model",
    "accuracy": "Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
    "weighted_precision": "Weighted Precision",
    "weighted_recall": "Weighted Recall",
    "weighted_f1": "Weighted F1",
}


def build_model_candidates() -> OrderedDict[str, BaseEstimator]:
    """Build the six specified classifiers with fair preprocessing pipelines."""

    return OrderedDict(
        [
            (
                "Logistic Regression",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=5000,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "Decision Tree",
                DecisionTreeClassifier(random_state=RANDOM_STATE),
            ),
            (
                "K-Nearest Neighbors",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("classifier", KNeighborsClassifier(n_neighbors=5)),
                    ]
                ),
            ),
            (
                "Support Vector Machine",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            SVC(probability=True, random_state=RANDOM_STATE),
                        ),
                    ]
                ),
            ),
            ("Gaussian Naive Bayes", GaussianNB()),
            ("Random Forest", build_random_forest()),
        ]
    )


def _names_at_maximum(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    candidates: Sequence[str] | None = None,
) -> tuple[float, list[str]]:
    allowed = set(candidates) if candidates is not None else None
    eligible = [row for row in rows if allowed is None or row["model"] in allowed]
    maximum = max(float(row[metric]) for row in eligible)
    names = [
        str(row["model"])
        for row in eligible
        if np.isclose(
            float(row[metric]),
            maximum,
            rtol=0.0,
            atol=TIE_TOLERANCE,
        )
    ]
    return maximum, names


def select_model_from_results(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the measured accuracy-first, macro-F1 tie-break selection rule."""

    if not rows:
        raise ValueError("At least one model result is required for selection")

    highest_accuracy, accuracy_leaders = _names_at_maximum(rows, "accuracy")
    highest_macro_f1, macro_f1_leaders = _names_at_maximum(rows, "macro_f1")
    remaining = list(accuracy_leaders)
    tie_break_trace: list[dict[str, Any]] = []

    # Macro F1 is the mandatory first tie-break. Additional required metrics are
    # supporting tie-breaks only if the preceding metric remains exactly tied.
    supporting_order = (
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "weighted_precision",
        "weighted_recall",
    )
    if len(remaining) > 1:
        for metric in supporting_order:
            best_value, leaders = _names_at_maximum(
                rows, metric, candidates=remaining
            )
            tie_break_trace.append(
                {
                    "metric": metric,
                    "best_value_among_remaining": best_value,
                    "remaining_models": leaders,
                }
            )
            remaining = leaders
            if len(remaining) == 1:
                break

    if len(accuracy_leaders) == 1:
        selected_model = accuracy_leaders[0]
        status = "selected"
        reason = (
            f"{selected_model} was selected because it achieved the sole highest "
            f"held-out test accuracy ({highest_accuracy:.6f})."
        )
    elif len(remaining) == 1:
        selected_model = remaining[0]
        status = "selected_after_tie_break"
        first_resolving_metric = tie_break_trace[-1]["metric"]
        reason = (
            "Held-out accuracy was tied among "
            f"{', '.join(accuracy_leaders)} at {highest_accuracy:.6f}. "
            f"{selected_model} was selected using {first_resolving_metric} "
            "after applying macro F1 first and the remaining reported metrics "
            "only as needed."
        )
    else:
        selected_model = None
        status = "unresolved_metric_tie"
        reason = (
            "No unique best model is claimed: held-out accuracy was tied among "
            f"{', '.join(accuracy_leaders)}, and all specified supporting metrics "
            f"remained tied for {', '.join(remaining)}."
        )

    return {
        "status": status,
        "selected_model": selected_model,
        "reason": reason,
        "highest_test_accuracy": highest_accuracy,
        "highest_accuracy_models": accuracy_leaders,
        "highest_macro_f1": highest_macro_f1,
        "highest_macro_f1_models": macro_f1_leaders,
        "tie_break_trace": tie_break_trace,
        "unresolved_tied_models": remaining if selected_model is None else [],
    }


def _save_accuracy_chart(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    plot_rows = sorted(
        rows,
        key=lambda row: (float(row["accuracy"]), float(row["macro_f1"])),
        reverse=True,
    )
    names = [str(row["model"]) for row in plot_rows]
    accuracies = [float(row["accuracy"]) for row in plot_rows]
    figure, axis = plt.subplots(figsize=(11, 6.5))
    bars = axis.bar(names, accuracies, color="#2F6B9A")
    axis.set_title("Held-Out Test Accuracy by Classifier")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.tick_params(axis="x", labelrotation=25)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    for bar, accuracy in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(accuracy + 0.015, 1.025),
            f"{accuracy:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def compare_models(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit all six models on one split and save metrics and selection evidence."""

    X, y, label_encoder = preprocess_data(dataset_path)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    leakage = require_no_feature_overlap(X_train, X_test)

    candidates = build_model_candidates()
    fitted_models: dict[str, BaseEstimator] = {}
    rows: list[dict[str, Any]] = []
    for name, estimator in candidates.items():
        # Pipelines ensure each scaler sees X_train only. Unscaled estimators are
        # fitted directly on the identical X_train partition.
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        all_metrics = calculate_classification_metrics(y_test, predictions)
        row: dict[str, Any] = {"model": name}
        row.update({metric: float(all_metrics[metric]) for metric in COMPARISON_METRICS})
        rows.append(row)
        fitted_models[name] = estimator
        if verbose:
            print(
                f"{name}: accuracy={row['accuracy']:.6f}, "
                f"macro_f1={row['macro_f1']:.6f}"
            )

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            -float(row["accuracy"]),
            -float(row["macro_f1"]),
            str(row["model"]),
        ),
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank
    selection = select_model_from_results(rows)

    output_dir = Path(results_dir)
    model_output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "model_comparison.csv"
    json_path = output_dir / "model_comparison.json"
    chart_path = output_dir / "model_accuracy_comparison.png"

    csv_records = []
    for row in ranked_rows:
        record = {"Rank": row["rank"]}
        for key in ("model", *COMPARISON_METRICS):
            record[CSV_COLUMN_NAMES[key]] = row[key]
        csv_records.append(record)
    pd.DataFrame(csv_records).to_csv(csv_path, index=False, float_format="%.12f")
    _save_accuracy_chart(rows, chart_path)

    # Preserve the reproducible Random Forest required by the explainability and
    # prediction modules, regardless of whether another classifier wins.
    random_forest_path = model_output_dir / "random_forest_crop.joblib"
    encoder_path = model_output_dir / "label_encoder.joblib"
    joblib.dump(fitted_models["Random Forest"], random_forest_path)
    joblib.dump(label_encoder, encoder_path)

    selected_path: Path | None = None
    if selection["selected_model"] is not None:
        selected_path = model_output_dir / "selected_crop_model.joblib"
        joblib.dump(fitted_models[str(selection["selected_model"])], selected_path)

    result: dict[str, Any] = {
        "experiment": "Six-model held-out comparison",
        "dataset_path": str(dataset_path or DEFAULT_DATASET_PATH),
        "feature_order": list(FEATURE_NAMES),
        "split": {
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "shared_by_all_models": True,
        },
        "scaling": {
            "training_partition_only": True,
            "scaled_inside_pipeline": [
                "Logistic Regression",
                "K-Nearest Neighbors",
                "Support Vector Machine",
            ],
            "not_scaled": [
                "Decision Tree",
                "Gaussian Naive Bayes",
                "Random Forest",
            ],
        },
        "metrics": list(COMPARISON_METRICS),
        "results": ranked_rows,
        "selection": selection,
        "leakage_check": leakage,
        "limitations": [
            "High accuracy on this public dataset does not establish real-world agricultural performance.",
            "External validation is required before field deployment.",
            "A tied accuracy is reported as a tie and is not presented as a unique accuracy win.",
        ],
        "artifacts": {
            "comparison_csv": str(csv_path),
            "comparison_json": str(json_path),
            "accuracy_chart": str(chart_path),
            "random_forest_model": str(random_forest_path),
            "label_encoder": str(encoder_path),
            "selected_model": str(selected_path) if selected_path else None,
        },
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    if verbose:
        print(selection["reason"])
        print(
            "Highest held-out accuracy models: "
            + ", ".join(selection["highest_accuracy_models"])
        )
        print(
            "Highest macro-F1 models: "
            + ", ".join(selection["highest_macro_f1_models"])
        )
        print(
            "Exact train/test feature-vector overlaps: "
            f"{leakage['overlapping_unique_feature_vectors']}"
        )
        print(f"Saved comparison artifacts under: {output_dir}")
        if selected_path:
            print(f"Saved measured selected model: {selected_path}")
        else:
            print("No single selected-model artifact was written because the metric tie remained unresolved.")

    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare six classifiers on one 80/20 stratified split."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        compare_models(
            args.dataset,
            results_dir=args.results_dir,
            models_dir=args.models_dir,
            verbose=True,
        )
    except (FileNotFoundError, DatasetValidationError, FeatureLeakageError, ValueError) as exc:
        print(f"Model comparison stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
