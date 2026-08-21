"""Training-only Random Forest tuning with one final held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.research.common import (
    BASELINE_MODEL_PATH,
    RESEARCH_MODELS_DIR,
    RESEARCH_RESULTS_DIR,
    class_positions,
    ensure_research_directories,
    load_baseline_artifacts,
    load_research_split,
    probability_metrics,
    python_value,
    sha256_file,
    validate_probability_matrix,
    write_json,
)
from src.train import RANDOM_STATE, build_random_forest


FULL_SEARCH_ITERATIONS = 16
QUICK_SEARCH_ITERATIONS = 2
CV_FOLDS = 5


def parameter_search_space() -> dict[str, list[Any]]:
    """Return the deliberately bounded, reproducible RF search space."""

    return {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [None, 8, 12, 16, 24],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": ["sqrt", "log2", None, 0.75],
        "class_weight": [None, "balanced"],
    }


def format_search_results(cv_results: dict[str, Any]) -> pd.DataFrame:
    """Create a compact, traceable table from RandomizedSearchCV results."""

    required = (
        "params",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "rank_test_macro_f1",
        "mean_test_accuracy",
        "std_test_accuracy",
        "rank_test_accuracy",
        "mean_fit_time",
    )
    missing = [name for name in required if name not in cv_results]
    if missing:
        raise ValueError(f"Search results are missing: {', '.join(missing)}")
    records: list[dict[str, Any]] = []
    for index, parameters in enumerate(cv_results["params"]):
        record: dict[str, Any] = {
            "candidate": index + 1,
            "mean_cv_macro_f1": float(cv_results["mean_test_macro_f1"][index]),
            "std_cv_macro_f1": float(cv_results["std_test_macro_f1"][index]),
            "rank_cv_macro_f1": int(cv_results["rank_test_macro_f1"][index]),
            "mean_cv_accuracy": float(cv_results["mean_test_accuracy"][index]),
            "std_cv_accuracy": float(cv_results["std_test_accuracy"][index]),
            "rank_cv_accuracy": int(cv_results["rank_test_accuracy"][index]),
            "mean_fit_time_seconds": float(cv_results["mean_fit_time"][index]),
        }
        record.update({f"param_{key}": value for key, value in parameters.items()})
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["rank_cv_macro_f1", "rank_cv_accuracy", "candidate"]
    )


def _save_confusion_plot(matrix: np.ndarray, labels: Sequence[str], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Greens", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    axis.set_xticks(ticks, labels, rotation=90)
    axis.set_yticks(ticks, labels)
    axis.set_xlabel("Predicted crop")
    axis.set_ylabel("Actual crop")
    axis.set_title("Tuned Random Forest Held-Out Confusion Matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def tune_random_forest(
    *,
    output_dir: str | Path = RESEARCH_RESULTS_DIR,
    model_output: str | Path = RESEARCH_MODELS_DIR / "random_forest_tuned.joblib",
    quick: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Tune only on training CV, then evaluate the selected estimator once."""

    ensure_research_directories()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tuned_model_path = Path(model_output)
    tuned_model_path.parent.mkdir(parents=True, exist_ok=True)
    split = load_research_split()
    baseline_model, encoder = load_baseline_artifacts()
    encoded_classes, decoded_classes = class_positions(baseline_model, encoder)
    baseline_hash_before = sha256_file(BASELINE_MODEL_PATH)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_iter = QUICK_SEARCH_ITERATIONS if quick else FULL_SEARCH_ITERATIONS
    search = RandomizedSearchCV(
        estimator=build_random_forest(),
        param_distributions=parameter_search_space(),
        n_iter=n_iter,
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        refit="macro_f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        return_train_score=False,
        verbose=0,
    )
    search.fit(split.X_train, split.y_train)
    search_table = format_search_results(search.cv_results_)
    search_csv = destination / "rf_hyperparameter_search.csv"
    search_table.to_csv(search_csv, index=False, float_format="%.12f")

    tuned_model = search.best_estimator_
    tuned_probabilities = validate_probability_matrix(
        tuned_model.predict_proba(split.X_test),
        expected_rows=440,
        expected_classes=len(encoded_classes),
    )
    baseline_probabilities = validate_probability_matrix(
        baseline_model.predict_proba(split.X_test),
        expected_rows=440,
        expected_classes=len(encoded_classes),
    )
    baseline_metrics = probability_metrics(
        split.y_test, baseline_probabilities, encoded_classes
    )
    tuned_metrics = probability_metrics(split.y_test, tuned_probabilities, encoded_classes)

    baseline_cv = cross_validate(
        build_random_forest(),
        split.X_train,
        split.y_train,
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        cv=cv,
        n_jobs=1,
    )
    baseline_cv_macro_f1 = float(np.mean(baseline_cv["test_macro_f1"]))
    baseline_cv_accuracy = float(np.mean(baseline_cv["test_accuracy"]))
    tuned_cv_macro_f1 = float(search.best_score_)
    best_index = int(search.best_index_)
    tuned_cv_accuracy = float(search.cv_results_["mean_test_accuracy"][best_index])

    joblib.dump(tuned_model, tuned_model_path)
    tuned_predictions = encoded_classes[np.argmax(tuned_probabilities, axis=1)]
    matrix = confusion_matrix(split.y_test, tuned_predictions, labels=encoded_classes)
    confusion_csv = destination / "tuned_confusion_matrix.csv"
    pd.DataFrame(matrix, index=decoded_classes, columns=decoded_classes).to_csv(confusion_csv)
    confusion_plot = destination / "tuned_confusion_matrix.png"
    _save_confusion_plot(matrix, decoded_classes.tolist(), confusion_plot)

    comparison_rows = []
    for name, metrics, cv_macro_f1, cv_accuracy, parameters, model_path in (
        (
            "Baseline Random Forest",
            baseline_metrics,
            baseline_cv_macro_f1,
            baseline_cv_accuracy,
            baseline_model.get_params(),
            BASELINE_MODEL_PATH,
        ),
        (
            "Tuned Random Forest",
            tuned_metrics,
            tuned_cv_macro_f1,
            tuned_cv_accuracy,
            search.best_params_,
            tuned_model_path,
        ),
    ):
        row = {
            "model": name,
            "training_cv_macro_f1": cv_macro_f1,
            "training_cv_accuracy": cv_accuracy,
            **metrics,
            "model_size_bytes": Path(model_path).stat().st_size,
            "parameters_json": json.dumps(python_value(parameters), sort_keys=True),
        }
        comparison_rows.append(row)
    comparison_csv = destination / "baseline_vs_tuned.csv"
    pd.DataFrame(comparison_rows).to_csv(
        comparison_csv, index=False, float_format="%.12f"
    )

    best_parameters_payload = {
        "best_parameters": python_value(search.best_params_),
        "best_training_cv_macro_f1": tuned_cv_macro_f1,
        "corresponding_training_cv_accuracy": tuned_cv_accuracy,
        "random_seed": RANDOM_STATE,
        "search_iterations": n_iter,
        "quick_mode": quick,
        "cv": {
            "type": "StratifiedKFold",
            "folds": CV_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "optimization_metric": "macro_f1",
        "secondary_metric": "accuracy",
        "held_out_labels_used_in_search": False,
        "search_space": python_value(parameter_search_space()),
    }
    best_json = destination / "rf_best_parameters.json"
    write_json(best_json, best_parameters_payload)

    accuracy_delta = tuned_metrics["accuracy"] - baseline_metrics["accuracy"]
    macro_f1_delta = tuned_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    if macro_f1_delta > 1e-12:
        outcome = "improved"
    elif macro_f1_delta < -1e-12:
        outcome = "worse"
    else:
        outcome = "tied"
    baseline_hash_after = sha256_file(BASELINE_MODEL_PATH)
    if baseline_hash_before != baseline_hash_after:
        raise RuntimeError("Baseline model changed during RF tuning.")
    payload = {
        "experiment": "Training-only RandomizedSearchCV for Random Forest",
        "outcome_by_held_out_macro_f1": outcome,
        "held_out_accuracy_delta": accuracy_delta,
        "held_out_macro_f1_delta": macro_f1_delta,
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "best_parameters": python_value(search.best_params_),
        "methodology": best_parameters_payload,
        "limitations": [
            "The bounded search is not an exhaustive optimization of all RF configurations.",
            "The held-out set was used once after training-only model selection.",
            "Small metric differences are reported exactly and not treated as universal gains.",
        ],
        "artifacts": {
            "all_candidates_csv": str(search_csv),
            "best_parameters_json": str(best_json),
            "tuned_model": str(tuned_model_path),
            "confusion_csv": str(confusion_csv),
            "confusion_plot": str(confusion_plot),
            "comparison_csv": str(comparison_csv),
        },
        "baseline_model_sha256_before": baseline_hash_before,
        "baseline_model_sha256_after": baseline_hash_after,
    }
    write_json(destination / "tuning_summary.json", python_value(payload))
    if verbose:
        print(json.dumps(python_value(payload), indent=2))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_RESULTS_DIR)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=RESEARCH_MODELS_DIR / "random_forest_tuned.joblib",
    )
    args = parser.parse_args(argv)
    tune_random_forest(
        output_dir=args.output_dir,
        model_output=args.model_output,
        quick=args.quick,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
