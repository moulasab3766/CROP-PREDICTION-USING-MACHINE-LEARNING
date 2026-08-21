"""Dataset loading, validation, inspection, and target encoding.

The source CSV is treated as immutable.  This module validates it and returns
copies of the model inputs; it never deletes, imputes, or rewrites rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"

# This tuple is the single source of truth for model feature order.
FEATURE_NAMES: tuple[str, ...] = (
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
)
TARGET_NAME = "label"
EXPECTED_COLUMNS: tuple[str, ...] = (*FEATURE_NAMES, TARGET_NAME)
EXPECTED_SHAPE = (2200, 8)
EXPECTED_SAMPLES_PER_CLASS = 100
EXPECTED_CROP_CLASSES: tuple[str, ...] = (
    "rice",
    "maize",
    "chickpea",
    "kidneybeans",
    "pigeonpeas",
    "mothbeans",
    "mungbean",
    "blackgram",
    "lentil",
    "pomegranate",
    "banana",
    "mango",
    "grapes",
    "watermelon",
    "muskmelon",
    "apple",
    "orange",
    "papaya",
    "coconut",
    "cotton",
    "jute",
    "coffee",
)


class DatasetValidationError(ValueError):
    """Raised when the source dataset differs from the required specification."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        details = "\n - ".join(self.issues)
        super().__init__(
            "Dataset validation failed. The CSV was not modified."
            + (f"\n - {details}" if details else "")
        )


def _resolved_dataset_path(dataset_path: str | Path | None) -> Path:
    return DEFAULT_DATASET_PATH if dataset_path is None else Path(dataset_path)


def inspect_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect a loaded dataframe and return facts plus all validation issues.

    The report deliberately records every detected issue instead of silently
    cleaning the data, making unexpected source changes visible to the caller.
    """

    issues: list[str] = []
    actual_columns = dataframe.columns.tolist()

    if dataframe.shape != EXPECTED_SHAPE:
        issues.append(
            f"expected shape {EXPECTED_SHAPE}, found {tuple(dataframe.shape)}"
        )

    if actual_columns != list(EXPECTED_COLUMNS):
        missing = [name for name in EXPECTED_COLUMNS if name not in actual_columns]
        unexpected = [name for name in actual_columns if name not in EXPECTED_COLUMNS]
        issues.append(
            "expected columns in exact order "
            f"{list(EXPECTED_COLUMNS)}, found {actual_columns}"
        )
        if missing:
            issues.append(f"missing required columns: {missing}")
        if unexpected:
            issues.append(f"unexpected columns: {unexpected}")

    for feature in FEATURE_NAMES:
        # A duplicated column name makes dataframe[feature] ambiguous, and the
        # exact-column-order validation above already records that problem.
        if actual_columns.count(feature) != 1:
            continue
        series = dataframe[feature]
        if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
            issues.append(
                f"feature column {feature!r} must be numeric; found {series.dtype}"
            )
        elif not series.isna().any():
            numeric_values = series.to_numpy(dtype=float)
            if not np.isfinite(numeric_values).all():
                issues.append(
                    f"feature column {feature!r} contains non-finite values"
                )

    missing_values = {
        str(column): int(count)
        for column, count in dataframe.isna().sum().items()
    }
    total_missing = int(dataframe.isna().sum().sum())
    if total_missing:
        issues.append(f"expected zero missing values, found {total_missing}")

    duplicate_rows = int(dataframe.duplicated().sum())
    if duplicate_rows:
        issues.append(f"expected zero duplicate rows, found {duplicate_rows}")

    class_counts: dict[str, int] = {}
    unique_crop_count: int | None = None
    is_class_balanced = False

    if actual_columns.count(TARGET_NAME) == 1:
        labels = dataframe[TARGET_NAME]
        non_missing_labels = labels.dropna()
        labels_are_names = bool(
            non_missing_labels.map(
                lambda value: isinstance(value, str) and bool(value.strip())
            ).all()
        )
        if not labels_are_names:
            issues.append("target column 'label' must contain non-empty crop names")

        raw_counts = labels.value_counts(dropna=False)
        for label, count in raw_counts.items():
            key = "<missing>" if pd.isna(label) else str(label)
            class_counts[key] = int(count)

        unique_crop_count = int(labels.nunique(dropna=True))
        if unique_crop_count != len(EXPECTED_CROP_CLASSES):
            issues.append(
                f"expected {len(EXPECTED_CROP_CLASSES)} unique crop labels, "
                f"found {unique_crop_count}"
            )

        actual_class_names = {
            value for value in non_missing_labels.tolist() if isinstance(value, str)
        }
        expected_class_names = set(EXPECTED_CROP_CLASSES)
        missing_classes = sorted(expected_class_names - actual_class_names)
        unexpected_classes = sorted(actual_class_names - expected_class_names)
        if missing_classes:
            issues.append(f"missing expected crop classes: {missing_classes}")
        if unexpected_classes:
            issues.append(f"unexpected crop classes: {unexpected_classes}")

        wrong_counts = {
            crop: int(raw_counts.get(crop, 0))
            for crop in EXPECTED_CROP_CLASSES
            if int(raw_counts.get(crop, 0)) != EXPECTED_SAMPLES_PER_CLASS
        }
        if wrong_counts:
            issues.append(
                "expected 100 samples per crop; mismatched counts: "
                f"{wrong_counts}"
            )

        observed_counts = [
            int(raw_counts.get(crop, 0)) for crop in EXPECTED_CROP_CLASSES
        ]
        is_class_balanced = (
            len(actual_class_names) == len(EXPECTED_CROP_CLASSES)
            and actual_class_names == expected_class_names
            and len(set(observed_counts)) == 1
            and observed_counts[0] == EXPECTED_SAMPLES_PER_CLASS
        )
        if not is_class_balanced:
            issues.append("dataset is not balanced at 100 samples for each crop")

    return {
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "shape": tuple(int(value) for value in dataframe.shape),
        "columns": actual_columns,
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "unique_crop_count": unique_crop_count,
        "class_counts": class_counts,
        "is_class_balanced": is_class_balanced,
        "valid": not issues,
        "issues": issues,
    }


def inspect_dataset(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
) -> dict[str, Any]:
    """Read and inspect the CSV without changing it or training a model."""

    path = _resolved_dataset_path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the unmodified Kaggle "
            "Crop_recommendation.csv file at this path."
        )
    dataframe = pd.read_csv(path)
    return inspect_dataframe(dataframe, dataset_path=path)


def validate_dataset(
    dataframe: pd.DataFrame,
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an already loaded dataframe and return its inspection report."""

    report = inspect_dataframe(dataframe, dataset_path=dataset_path)
    if not report["valid"]:
        raise DatasetValidationError(report["issues"])
    return report


def load_dataset(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
) -> pd.DataFrame:
    """Load and strictly validate the required 2,200-row crop dataset."""

    path = _resolved_dataset_path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the unmodified Kaggle "
            "Crop_recommendation.csv file at this path."
        )
    dataframe = pd.read_csv(path)
    validate_dataset(dataframe, dataset_path=path)
    return dataframe


def preprocess_data(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Return ordered numeric features, encoded labels, and fitted encoder.

    LabelEncoder is fitted only on the target. No feature scaler is fitted here;
    algorithms that require scaling place it inside a training-only pipeline.
    """

    dataframe = load_dataset(dataset_path)
    X = dataframe.loc[:, FEATURE_NAMES].copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataframe[TARGET_NAME].to_numpy())
    return X, y, label_encoder


def load_and_preprocess_data(
    dataset_path: str | Path | None = DEFAULT_DATASET_PATH,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Compatibility-friendly descriptive alias for :func:`preprocess_data`."""

    return preprocess_data(dataset_path)


def _print_inspection(report: dict[str, Any]) -> None:
    print(f"Dataset path: {report['dataset_path']}")
    print(f"Dataset shape: {report['shape']}")
    print(f"Column names: {report['columns']}")
    print("Data types:")
    for column, dtype in report["dtypes"].items():
        print(f"  {column}: {dtype}")
    print("Missing-value counts:")
    for column, count in report["missing_values"].items():
        print(f"  {column}: {count}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Unique crop count: {report['unique_crop_count']}")
    print("Per-crop sample counts:")
    for crop, count in sorted(report["class_counts"].items()):
        print(f"  {crop}: {count}")
    print(f"Class-balanced: {report['is_class_balanced']}")
    print(f"Validation passed: {report['valid']}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and strictly validate the crop recommendation dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"CSV path (default: {DEFAULT_DATASET_PATH})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        report = inspect_dataset(args.dataset)
    except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
        print(f"Dataset inspection failed: {exc}", file=sys.stderr)
        return 1

    _print_inspection(report)
    if not report["valid"]:
        print("Validation issues (the dataset was not modified):", file=sys.stderr)
        for issue in report["issues"]:
            print(f"  - {issue}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
