"""Train and persist the required Random Forest crop classifier."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

try:  # Supports both `python -m src.train` and `python src/train.py`.
    from .preprocessing import (
        DEFAULT_DATASET_PATH,
        DatasetValidationError,
        FEATURE_NAMES,
        preprocess_data,
    )
except ImportError:  # pragma: no cover - exercised by direct script invocation.
    from preprocessing import (  # type: ignore
        DEFAULT_DATASET_PATH,
        DatasetValidationError,
        FEATURE_NAMES,
        preprocess_data,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_crop.joblib"
DEFAULT_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"

TEST_SIZE = 0.20
RANDOM_STATE = 42
RANDOM_FOREST_TREES = 100
EXPECTED_TRAINING_SAMPLES = 1760
EXPECTED_TEST_SAMPLES = 440


class FeatureLeakageError(RuntimeError):
    """Raised when an exact seven-feature vector crosses the train/test split."""


def create_train_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Create the one canonical 80/20 stratified split used by all models."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def _feature_row_counts(features: pd.DataFrame | np.ndarray) -> Counter[tuple[Any, ...]]:
    if isinstance(features, pd.DataFrame):
        values = features.loc[:, FEATURE_NAMES].to_numpy()
    else:
        values = np.asarray(features)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected a two-dimensional array with {len(FEATURE_NAMES)} columns"
            )
    return Counter(tuple(row.tolist()) for row in values)


def feature_overlap_report(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
) -> dict[str, int | bool]:
    """Count exact seven-feature-vector overlap between train and test.

    Both unique vectors and cross-partition row-pair counts are reported so the
    meaning of "overlap" remains explicit even if duplicates are ever present.
    """

    train_counts = _feature_row_counts(X_train)
    test_counts = _feature_row_counts(X_test)
    overlap = set(train_counts).intersection(test_counts)
    train_rows = sum(train_counts[row] for row in overlap)
    test_rows = sum(test_counts[row] for row in overlap)
    cross_partition_pairs = sum(
        train_counts[row] * test_counts[row] for row in overlap
    )
    return {
        "overlap_detected": bool(overlap),
        "overlapping_unique_feature_vectors": int(len(overlap)),
        "training_rows_with_overlap": int(train_rows),
        "test_rows_with_overlap": int(test_rows),
        "cross_partition_matching_pairs": int(cross_partition_pairs),
        "compared_feature_count": len(FEATURE_NAMES),
    }


def require_no_feature_overlap(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
) -> dict[str, int | bool]:
    """Return the leakage report, or stop before fitting/reporting if nonzero."""

    report = feature_overlap_report(X_train, X_test)
    if report["overlap_detected"]:
        raise FeatureLeakageError(
            "Exact feature-vector leakage was detected across the train/test "
            f"split ({report['overlapping_unique_feature_vectors']} unique "
            "vectors). No model or evaluation result was saved. Inspect and "
            "resolve the source-data issue without silently deleting rows."
        )
    return report


def build_random_forest() -> RandomForestClassifier:
    """Build the reproducible 100-tree classifier specified for this project."""

    return RandomForestClassifier(
        n_estimators=RANDOM_FOREST_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    """Calculate the complete held-out metric set required by the project."""

    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "correct_predictions": int(np.sum(np.asarray(y_true) == np.asarray(y_pred))),
        "incorrect_predictions": int(
            np.sum(np.asarray(y_true) != np.asarray(y_pred))
        ),
        "sample_count": int(len(y_true)),
    }
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
        metrics[f"{average}_precision"] = float(precision)
        metrics[f"{average}_recall"] = float(recall)
        metrics[f"{average}_f1"] = float(f1)
    return metrics


def _full_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> str:
    class_ids = np.arange(len(class_names))
    return classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=list(class_names),
        digits=6,
        zero_division=0,
    )


def train_random_forest(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
    *,
    model_output_path: str | Path = DEFAULT_MODEL_PATH,
    encoder_output_path: str | Path = DEFAULT_ENCODER_PATH,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train only on the training partition, evaluate once, and save artifacts."""

    X, y, label_encoder = preprocess_data(dataset_path)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    if len(X_train) != EXPECTED_TRAINING_SAMPLES or len(X_test) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            "Unexpected split sizes: expected 1,760 training and 440 test "
            f"samples, found {len(X_train)} and {len(X_test)}"
        )

    leakage = require_no_feature_overlap(X_train, X_test)
    model = build_random_forest()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = calculate_classification_metrics(y_test, y_pred)
    report_text = _full_classification_report(
        y_test, y_pred, label_encoder.classes_.tolist()
    )
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
    )

    model_path = Path(model_output_path)
    encoder_path = Path(encoder_output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    # Save bare sklearn objects so prediction/explanation modules can load them
    # directly, as required by the project structure.
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    if verbose:
        print(f"Training samples: {len(X_train)}")
        print(f"Held-out test samples: {len(X_test)}")
        print(
            "Exact train/test feature-vector overlaps: "
            f"{leakage['overlapping_unique_feature_vectors']}"
        )
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"{name}: {value:.6f}")
            else:
                print(f"{name}: {value}")
        print("\nFull 22-class classification report:")
        print(report_text)
        print(f"Saved Random Forest model: {model_path}")
        print(f"Saved label encoder: {encoder_path}")

    return {
        "model": model,
        "label_encoder": label_encoder,
        "metrics": metrics,
        "classification_report": report_text,
        "confusion_matrix": matrix,
        "leakage_check": leakage,
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "model_path": str(model_path),
        "label_encoder_path": str(encoder_path),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the reproducible 100-tree Random Forest crop model."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--encoder-output", type=Path, default=DEFAULT_ENCODER_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        train_random_forest(
            args.dataset,
            model_output_path=args.model_output,
            encoder_output_path=args.encoder_output,
            verbose=True,
        )
    except (FileNotFoundError, DatasetValidationError, FeatureLeakageError) as exc:
        print(f"Training stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
