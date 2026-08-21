"""Shared deterministic utilities for crop-recommendation research experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support

from src.preprocessing import DEFAULT_DATASET_PATH, FEATURE_NAMES, preprocess_data
from src.train import (
    EXPECTED_TEST_SAMPLES,
    EXPECTED_TRAINING_SAMPLES,
    RANDOM_STATE,
    create_train_test_split,
    require_no_feature_overlap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_RESULTS_DIR = PROJECT_ROOT / "results" / "research"
RESEARCH_MODELS_DIR = PROJECT_ROOT / "models" / "research"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_crop.joblib"
BASELINE_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"
EPSILON = 1e-12


@dataclass(frozen=True)
class ResearchSplit:
    """The canonical baseline split plus its target encoder."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    label_encoder: Any


def ensure_research_directories() -> tuple[Path, Path]:
    """Create and return the experimental output directories."""

    RESEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return RESEARCH_RESULTS_DIR, RESEARCH_MODELS_DIR


def load_research_split(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
) -> ResearchSplit:
    """Reconstruct the exact 80/20 stratified split used by the baseline.

    Feature order, label encoding, split seed, and leakage checks are inherited
    from the production training utilities.  No preprocessing is fit on the full
    feature matrix.
    """

    X, y, label_encoder = preprocess_data(dataset_path)
    if tuple(X.columns) != FEATURE_NAMES:
        raise RuntimeError("Research feature order differs from the baseline contract.")
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    if len(X_train) != EXPECTED_TRAINING_SAMPLES:
        raise RuntimeError("Research training split must contain 1,760 samples.")
    if len(X_test) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError("Research test split must contain 440 samples.")
    require_no_feature_overlap(X_train, X_test)
    return ResearchSplit(X_train, X_test, y_train, y_test, label_encoder)


def load_baseline_artifacts(
    model_path: str | Path = BASELINE_MODEL_PATH,
    encoder_path: str | Path = BASELINE_ENCODER_PATH,
) -> tuple[Any, Any]:
    """Load, but never fit or modify, the protected production artifacts."""

    model_file = Path(model_path)
    encoder_file = Path(encoder_path)
    if not model_file.is_file() or not encoder_file.is_file():
        raise FileNotFoundError("Protected baseline model or label encoder is missing.")
    return joblib.load(model_file), joblib.load(encoder_file)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a deterministic UTF-8 JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    portable_payload = _portable_json_value(payload)
    output.write_text(
        json.dumps(portable_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _portable_json_value(value: Any) -> Any:
    """Replace project-absolute paths with portable repository-relative paths."""

    if isinstance(value, dict):
        return {str(key): _portable_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_json_value(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        try:
            candidate = Path(value)
            if candidate.is_absolute() and candidate.is_relative_to(PROJECT_ROOT):
                return candidate.relative_to(PROJECT_ROOT).as_posix()
        except (OSError, ValueError):
            pass
    return python_value(value)


def python_value(value: Any) -> Any:
    """Convert NumPy scalar/container values to JSON-safe Python values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [python_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [python_value(item) for item in value]
    return value


def validate_probability_matrix(
    probabilities: Any,
    *,
    expected_rows: int | None = None,
    expected_classes: int | None = None,
) -> np.ndarray:
    """Validate a finite two-dimensional class-probability matrix."""

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Probabilities must be a two-dimensional matrix.")
    if expected_rows is not None and matrix.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} probability rows, found {matrix.shape[0]}.")
    if expected_classes is not None and matrix.shape[1] != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} probability columns, found {matrix.shape[1]}."
        )
    if matrix.size == 0:
        raise ValueError("Probability matrix must not be empty.")
    if not np.isfinite(matrix).all():
        raise ValueError("Probability matrix contains non-finite values.")
    if np.any(matrix < -EPSILON) or np.any(matrix > 1.0 + EPSILON):
        raise ValueError("Probabilities must remain between zero and one.")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Each probability row must sum approximately to one.")
    return matrix


def class_positions(model: Any, label_encoder: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return encoded model classes and decoded crop names in model-column order."""

    encoded = np.asarray(model.classes_)
    if encoded.ndim != 1:
        raise ValueError("Model classes must be one-dimensional.")
    try:
        integer_classes = encoded.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model classes do not match encoded crop identifiers.") from exc
    if not np.array_equal(encoded, integer_classes):
        raise ValueError("Model classes do not match integer encoded crop identifiers.")
    decoded = np.asarray(label_encoder.inverse_transform(integer_classes), dtype=str)
    if len(set(decoded.tolist())) != len(decoded):
        raise ValueError("Decoded model classes are not unique.")
    return integer_classes, decoded


def probability_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Any,
    classes: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Calculate classification and descriptive multiclass probability metrics."""

    truth = np.asarray(y_true)
    class_array = np.asarray(classes)
    matrix = validate_probability_matrix(
        probabilities,
        expected_rows=len(truth),
        expected_classes=len(class_array),
    )
    column_by_class = {int(value): position for position, value in enumerate(class_array)}
    try:
        true_columns = np.asarray([column_by_class[int(value)] for value in truth])
    except KeyError as exc:
        raise ValueError("A true label is absent from the probability columns.") from exc
    predicted = class_array[np.argmax(matrix, axis=1)]
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        truth, predicted, average="weighted", zero_division=0
    )
    true_probabilities = matrix[np.arange(len(truth)), true_columns]
    one_hot = np.zeros_like(matrix)
    one_hot[np.arange(len(truth)), true_columns] = 1.0
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "log_loss": float(log_loss(truth, matrix, labels=class_array)),
        "mean_true_class_probability": float(np.mean(true_probabilities)),
        "multiclass_brier_score": float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1))),
        "mean_top_probability": float(np.mean(np.max(matrix, axis=1))),
    }


def top_label_calibration_bins(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Any,
    classes: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Aggregate multiclass predictions into equal-width top-label reliability bins.

    Each sample contributes its maximum class probability and whether that top
    label is correct. Empty bins are retained with a count of zero and null-free
    numeric placeholders so downstream plots can deliberately skip them.
    """

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    truth = np.asarray(y_true)
    class_array = np.asarray(classes)
    matrix = validate_probability_matrix(
        probabilities,
        expected_rows=len(truth),
        expected_classes=len(class_array),
    )
    confidence = np.max(matrix, axis=1)
    predicted = class_array[np.argmax(matrix, axis=1)]
    correctness = (predicted == truth).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.floor(confidence * n_bins).astype(int), n_bins - 1)
    rows: list[dict[str, float | int]] = []
    for index in range(n_bins):
        mask = bin_ids == index
        count = int(np.sum(mask))
        rows.append(
            {
                "bin_index": index,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "count": count,
                "mean_confidence": float(np.mean(confidence[mask])) if count else 0.0,
                "observed_accuracy": float(np.mean(correctness[mask])) if count else 0.0,
            }
        )
    return rows


def expected_calibration_error(
    bins: Iterable[dict[str, float | int]],
) -> float:
    """Calculate count-weighted absolute top-label calibration error."""

    rows = list(bins)
    total = sum(int(row["count"]) for row in rows)
    if total <= 0:
        raise ValueError("Calibration bins must contain at least one sample.")
    error = sum(
        int(row["count"])
        * abs(float(row["observed_accuracy"]) - float(row["mean_confidence"]))
        for row in rows
    )
    return float(error / total)


def mean_or_none(values: Sequence[float]) -> float | None:
    """Return a finite mean, or ``None`` when a group contains no values."""

    if not values:
        return None
    result = float(np.mean(np.asarray(values, dtype=float)))
    return result if math.isfinite(result) else None
