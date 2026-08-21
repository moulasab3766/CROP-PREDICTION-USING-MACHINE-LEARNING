"""Reliability evaluation for the saved Random Forest crop model.

Held-out testing is kept separate from five-fold cross-validation. Cross-
validation is performed on the 80% training partition only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)

try:  # Supports module and direct-file execution.
    from .preprocessing import (
        DEFAULT_DATASET_PATH,
        DatasetValidationError,
        FEATURE_NAMES,
        preprocess_data,
    )
    from .train import (
        DEFAULT_ENCODER_PATH,
        DEFAULT_MODEL_PATH,
        RANDOM_FOREST_TREES,
        RANDOM_STATE,
        FeatureLeakageError,
        calculate_classification_metrics,
        create_train_test_split,
        require_no_feature_overlap,
    )
except ImportError:  # pragma: no cover - direct `python src/evaluate.py` path.
    from preprocessing import (  # type: ignore
        DEFAULT_DATASET_PATH,
        DatasetValidationError,
        FEATURE_NAMES,
        preprocess_data,
    )
    from train import (  # type: ignore
        DEFAULT_ENCODER_PATH,
        DEFAULT_MODEL_PATH,
        RANDOM_FOREST_TREES,
        RANDOM_STATE,
        FeatureLeakageError,
        calculate_classification_metrics,
        create_train_test_split,
        require_no_feature_overlap,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
CV_SPLITS = 5


def _to_builtin(value: Any) -> Any:
    """Recursively convert numpy values so experiment JSON stays portable."""

    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_and_verify_artifacts(
    model_path: str | Path,
    encoder_path: str | Path,
    dataset_encoder: LabelEncoder,
) -> tuple[RandomForestClassifier, LabelEncoder]:
    model_file = Path(model_path)
    encoder_file = Path(encoder_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Saved Random Forest model not found at {model_file}; run src/train.py first"
        )
    if not encoder_file.is_file():
        raise FileNotFoundError(
            f"Saved label encoder not found at {encoder_file}; run src/train.py first"
        )

    model = joblib.load(model_file)
    saved_encoder = joblib.load(encoder_file)
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(
            f"Expected a RandomForestClassifier at {model_file}, found {type(model).__name__}"
        )
    if not isinstance(saved_encoder, LabelEncoder):
        raise TypeError(
            f"Expected a LabelEncoder at {encoder_file}, found {type(saved_encoder).__name__}"
        )
    check_is_fitted(model)
    check_is_fitted(saved_encoder)

    if model.n_estimators != RANDOM_FOREST_TREES or model.random_state != RANDOM_STATE:
        raise ValueError(
            "Saved model configuration does not match the required 100-tree, "
            "random_state=42 Random Forest"
        )
    if not np.array_equal(saved_encoder.classes_, dataset_encoder.classes_):
        raise ValueError(
            "Saved label encoder classes do not match the validated dataset labels"
        )
    expected_ids = np.arange(len(saved_encoder.classes_))
    if not np.array_equal(np.asarray(model.classes_), expected_ids):
        raise ValueError(
            "Saved model class IDs do not align with the saved LabelEncoder"
        )
    if hasattr(model, "feature_names_in_") and tuple(model.feature_names_in_) != FEATURE_NAMES:
        raise ValueError(
            "Saved model feature order does not match the canonical seven-feature order"
        )
    return model, saved_encoder


def _misclassification_records(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    encoder: LabelEncoder,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    mismatch_positions = np.flatnonzero(np.asarray(y_test) != np.asarray(y_pred))
    for position in mismatch_positions:
        feature_row = X_test.iloc[int(position)]
        records.append(
            {
                "dataset_row_index": _to_builtin(X_test.index[int(position)]),
                "actual_crop": str(encoder.inverse_transform([y_test[position]])[0]),
                "predicted_crop": str(encoder.inverse_transform([y_pred[position]])[0]),
                "features": {
                    feature: _to_builtin(feature_row[feature])
                    for feature in FEATURE_NAMES
                },
            }
        )
    return records


def _save_confusion_matrix_image(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(18, 16))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(class_names),
    )
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set_title("Random Forest Held-Out Test Confusion Matrix", pad=18)
    axis.tick_params(axis="x", labelrotation=90, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def evaluate_random_forest(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    encoder_path: str | Path = DEFAULT_ENCODER_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate the saved model and write all reliability artifacts."""

    X, y, dataset_encoder = preprocess_data(dataset_path)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    leakage = require_no_feature_overlap(X_train, X_test)
    model, encoder = _load_and_verify_artifacts(
        model_path, encoder_path, dataset_encoder
    )

    # The saved model is used only for the untouched held-out test evaluation.
    y_pred = model.predict(X_test)
    class_names = encoder.classes_.tolist()
    class_ids = np.arange(len(class_names))
    metrics = calculate_classification_metrics(y_test, y_pred)
    report_text = classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        digits=6,
        zero_division=0,
    )
    report_structured = classification_report(
        y_test,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=class_ids)
    misclassifications = _misclassification_records(
        X_test, y_test, y_pred, encoder
    )

    # clone() removes fitted state. Each fold therefore fits only its own slice
    # of X_train; the 440 held-out samples never enter cross-validation.
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_scores = cross_val_score(
        clone(model),
        X_train,
        y_train,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
    )
    cv_results = {
        "method": "StratifiedKFold",
        "partition": "training_partition_only",
        "n_splits": CV_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "scoring": "accuracy",
        "training_sample_count": int(len(X_train)),
        "fold_accuracies": [float(score) for score in fold_scores],
        "mean_accuracy": float(np.mean(fold_scores)),
        "standard_deviation": float(np.std(fold_scores)),
        "held_out_minus_cv_mean": float(
            float(metrics["accuracy"]) - float(np.mean(fold_scores))
        ),
    }

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_image_path = output_dir / "confusion_matrix.png"
    confusion_csv_path = output_dir / "confusion_matrix.csv"
    report_path = output_dir / "classification_report.txt"
    evaluation_path = output_dir / "evaluation_results.json"

    confusion_frame = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )
    confusion_frame.index.name = "actual_crop"
    confusion_frame.columns.name = "predicted_crop"
    confusion_frame.to_csv(confusion_csv_path)
    _save_confusion_matrix_image(matrix, class_names, confusion_image_path)
    report_path.write_text(report_text, encoding="utf-8")

    result: dict[str, Any] = {
        "experiment": "Random Forest reliability evaluation",
        "dataset_path": str(dataset_path or DEFAULT_DATASET_PATH),
        "model_path": str(model_path),
        "label_encoder_path": str(encoder_path),
        "feature_order": list(FEATURE_NAMES),
        "class_names": class_names,
        "split": {
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "held_out_test_metrics": metrics,
        "classification_report": report_structured,
        "misclassification_count": len(misclassifications),
        "misclassifications": misclassifications,
        "cross_validation": cv_results,
        "leakage_check": leakage,
        "leakage_prevention": (
            "The held-out test partition was excluded from fitting and from "
            "five-fold cross-validation. Random Forest needs no scaler; model "
            "clones were fitted independently inside training-only CV folds."
        ),
        "limitations": [
            "High accuracy on this public dataset does not guarantee real-world performance.",
            "External validation on independent agricultural data is still required.",
        ],
        "artifacts": {
            "confusion_matrix_image": str(confusion_image_path),
            "confusion_matrix_csv": str(confusion_csv_path),
            "classification_report": str(report_path),
            "evaluation_results": str(evaluation_path),
        },
    }
    evaluation_path.write_text(
        json.dumps(_to_builtin(result), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    if verbose:
        print("Held-out test metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.6f}" if isinstance(value, float) else f"  {name}: {value}")
        print("\nFull 22-class classification report:")
        print(report_text)
        print(
            f"Misclassified held-out samples: {len(misclassifications)} of {len(y_test)}"
        )
        for item in misclassifications:
            print(
                f"  row {item['dataset_row_index']}: "
                f"{item['actual_crop']} -> {item['predicted_crop']}"
            )
        print("Training-partition-only CV fold accuracies:")
        print("  " + ", ".join(f"{score:.6f}" for score in fold_scores))
        print(f"  mean: {cv_results['mean_accuracy']:.6f}")
        print(f"  standard deviation: {cv_results['standard_deviation']:.6f}")
        print(
            "Exact train/test feature-vector overlaps: "
            f"{leakage['overlapping_unique_feature_vectors']}"
        )
        print("External validation is still required; dataset accuracy is not a guarantee of field performance.")
        print(f"Saved evaluation artifacts under: {output_dir}")

    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved Random Forest without retraining it on test data."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        evaluate_random_forest(
            args.dataset,
            model_path=args.model,
            encoder_path=args.encoder,
            results_dir=args.results_dir,
            verbose=True,
        )
    except (
        FileNotFoundError,
        DatasetValidationError,
        FeatureLeakageError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Evaluation stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
