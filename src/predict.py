"""Reusable inference for the saved crop recommendation model.

This module only loads already-fitted artifacts.  It never trains or mutates a
model.  Probability values are the values returned by ``predict_proba`` and
must not be interpreted as calibrated certainty unless calibration is assessed
separately.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:  # Supports both ``python -m src.predict`` and ``python src/predict.py``.
    from src.preprocessing import FEATURE_NAMES
except ImportError:  # pragma: no cover - exercised only by direct script use
    try:
        from preprocessing import FEATURE_NAMES
    except ImportError:  # Keep the inference module usable during initial setup.
        FEATURE_NAMES = (
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_crop.joblib"
DEFAULT_LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"


class ArtifactCompatibilityError(ValueError):
    """Raised when saved model artifacts cannot be used together safely."""


def _as_artifact_path(path: str | Path | None, default: Path) -> Path:
    return (default if path is None else Path(path)).expanduser().resolve()


@lru_cache(maxsize=8)
def _load_model_cached(model_path: str) -> Any:
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Saved Random Forest model not found: {model_file}. Run the training "
            "module before inference."
        )
    model = joblib.load(model_file)
    _validate_model(model)
    return model


@lru_cache(maxsize=8)
def _load_label_encoder_cached(label_encoder_path: str) -> Any:
    encoder_file = Path(label_encoder_path)
    if not encoder_file.is_file():
        raise FileNotFoundError(
            f"Saved label encoder not found: {encoder_file}. Run the training "
            "module before inference."
        )
    label_encoder = joblib.load(encoder_file)
    _validate_label_encoder(label_encoder)
    return label_encoder


def _validate_model(model: Any) -> None:
    if not callable(getattr(model, "predict", None)):
        raise ArtifactCompatibilityError("The saved model does not provide predict().")
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactCompatibilityError(
            "The saved model does not provide predict_proba(); real Top-K "
            "probabilities cannot be produced."
        )
    if not hasattr(model, "classes_"):
        raise ArtifactCompatibilityError("The saved model has no fitted classes_.")


def _validate_label_encoder(label_encoder: Any) -> None:
    if not callable(getattr(label_encoder, "inverse_transform", None)):
        raise ArtifactCompatibilityError(
            "The saved label encoder does not provide inverse_transform()."
        )


def load_model(model_path: str | Path | None = None) -> Any:
    """Load only the fitted model artifact without retraining."""

    resolved_model = _as_artifact_path(model_path, DEFAULT_MODEL_PATH)
    return _load_model_cached(str(resolved_model))


def load_label_encoder(label_encoder_path: str | Path | None = None) -> Any:
    """Load only the fitted target-label encoder artifact."""

    resolved_encoder = _as_artifact_path(
        label_encoder_path, DEFAULT_LABEL_ENCODER_PATH
    )
    return _load_label_encoder_cached(str(resolved_encoder))


def load_artifacts(
    model_path: str | Path | None = None,
    label_encoder_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load and validate the fitted model and label encoder without retraining."""

    model = load_model(model_path)
    label_encoder = load_label_encoder(label_encoder_path)
    return model, label_encoder


def clear_artifact_cache() -> None:
    """Clear the small artifact cache (mainly useful after replacing model files)."""

    _load_model_cached.cache_clear()
    _load_label_encoder_cached.cache_clear()


def _coerce_feature_mapping(
    values: Mapping[str, Any] | Sequence[Any],
) -> dict[str, Any]:
    if isinstance(values, Mapping):
        expected = set(FEATURE_NAMES)
        received = set(values)
        missing = [name for name in FEATURE_NAMES if name not in received]
        unexpected = sorted(str(name) for name in received - expected)
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise ValueError(
                "Input must contain exactly the seven required features "
                f"({'; '.join(details)})."
            )
        return {name: values[name] for name in FEATURE_NAMES}

    if isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        if len(values) != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected exactly {len(FEATURE_NAMES)} feature values in this "
                f"order: {', '.join(FEATURE_NAMES)}."
            )
        return dict(zip(FEATURE_NAMES, values, strict=True))

    raise TypeError(
        "Features must be a mapping keyed by the seven required names or a "
        "seven-value sequence in canonical order."
    )


def _validated_number(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric value, not {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def validate_features(
    values: Mapping[str, Any] | Sequence[Any],
) -> dict[str, float]:
    """Validate and return the seven inputs in the model's canonical order.

    Only physical input-domain checks are made here; no unverified agronomic
    suitability thresholds are imposed.  Temperature is allowed to be below
    zero, while nutrient quantities and rainfall cannot be negative, humidity
    must be in the percentage range, and pH must be on the standard 0--14 scale.
    """

    raw = _coerce_feature_mapping(values)
    validated = {name: _validated_number(name, raw[name]) for name in FEATURE_NAMES}

    for name in ("N", "P", "K", "rainfall"):
        if validated[name] < 0:
            raise ValueError(f"{name} cannot be negative.")
    if not 0 <= validated["humidity"] <= 100:
        raise ValueError("humidity must be between 0 and 100 percent.")
    if not 0 <= validated["ph"] <= 14:
        raise ValueError("ph must be between 0 and 14.")
    return validated


def _normalise_feature_arguments(
    N: Any,
    P: Any,
    K: Any,
    temperature: Any,
    humidity: Any,
    ph: Any,
    rainfall: Any,
) -> dict[str, float]:
    remaining = (P, K, temperature, humidity, ph, rainfall)
    if isinstance(N, Mapping) or (
        isinstance(N, Sequence) and not isinstance(N, (str, bytes, bytearray))
    ):
        if any(value is not None for value in remaining):
            raise ValueError(
                "Pass either one mapping/sequence or seven individual values, not both."
            )
        return validate_features(N)
    return validate_features((N, P, K, temperature, humidity, ph, rainfall))


def _decode_classes(model: Any, label_encoder: Any) -> np.ndarray:
    encoded_classes = np.asarray(model.classes_)
    try:
        decoded = np.asarray(label_encoder.inverse_transform(encoded_classes))
    except Exception as exc:  # sklearn raises ValueError for incompatible classes.
        raise ArtifactCompatibilityError(
            "Model classes cannot be decoded by the saved label encoder."
        ) from exc
    if decoded.ndim != 1 or len(decoded) != len(encoded_classes):
        raise ArtifactCompatibilityError(
            "Decoded labels do not align with the model probability columns."
        )
    return decoded


def _single_probability_row(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise ArtifactCompatibilityError(
            "predict_proba() must return one probability row for one input row."
        )
    row = probabilities[0]
    if row.size != len(np.asarray(model.classes_)):
        raise ArtifactCompatibilityError(
            "Probability columns do not align with model.classes_."
        )
    if not np.all(np.isfinite(row)) or np.any(row < 0) or np.any(row > 1):
        raise ArtifactCompatibilityError(
            "predict_proba() returned invalid probability values."
        )
    if not np.isclose(float(row.sum()), 1.0, rtol=1e-6, atol=1e-8):
        raise ArtifactCompatibilityError(
            "predict_proba() probabilities do not sum approximately to 1."
        )
    return row


def predict_crop(
    N: Any,
    P: Any = None,
    K: Any = None,
    temperature: Any = None,
    humidity: Any = None,
    ph: Any = None,
    rainfall: Any = None,
    *,
    top_k: int = 3,
    model_path: str | Path | None = None,
    label_encoder_path: str | Path | None = None,
    model: Any | None = None,
    label_encoder: Any | None = None,
) -> dict[str, Any]:
    """Return the decoded prediction and real ``predict_proba`` Top-K values.

    Inputs may be supplied as seven individual arguments, one seven-value
    sequence, or one mapping containing exactly the canonical feature names.
    The return probabilities are fractions in ``[0, 1]``; percentage fields are
    included solely as a display convenience.
    """

    values = _normalise_feature_arguments(
        N, P, K, temperature, humidity, ph, rainfall
    )
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if (model is None) != (label_encoder is None):
        raise ValueError("Provide both model and label_encoder, or neither.")
    if model is None:
        model, label_encoder = load_artifacts(model_path, label_encoder_path)
    else:
        _validate_model(model)
        _validate_label_encoder(label_encoder)

    frame = pd.DataFrame([[values[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    decoded_classes = _decode_classes(model, label_encoder)
    probabilities = _single_probability_row(model, frame)
    if top_k > len(probabilities):
        raise ValueError(
            f"top_k={top_k} exceeds the model's {len(probabilities)} classes."
        )

    encoded_prediction = np.asarray(model.predict(frame))
    if encoded_prediction.shape != (1,):
        raise ArtifactCompatibilityError(
            "predict() must return exactly one class for one input row."
        )
    try:
        predicted_crop = str(
            label_encoder.inverse_transform(encoded_prediction)[0]
        )
    except Exception as exc:
        raise ArtifactCompatibilityError(
            "The predicted model class cannot be decoded by the label encoder."
        ) from exc

    matching_indexes = np.flatnonzero(decoded_classes.astype(str) == predicted_crop)
    if len(matching_indexes) != 1:
        raise ArtifactCompatibilityError(
            "The decoded predicted crop does not uniquely align with probability labels."
        )
    prediction_probability = float(probabilities[int(matching_indexes[0])])

    # Stable sorting makes ties deterministic while preserving model.classes_ order.
    ranked_indexes = np.argsort(-probabilities, kind="stable")[: int(top_k)]
    top_recommendations = [
        {
            "rank": rank,
            "crop": str(decoded_classes[index]),
            "probability": float(probabilities[index]),
            "probability_percent": float(probabilities[index] * 100.0),
        }
        for rank, index in enumerate(ranked_indexes, start=1)
    ]

    return {
        "predicted_crop": predicted_crop,
        "prediction_probability": prediction_probability,
        "top_3" if top_k == 3 else "top_k": top_recommendations,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference using saved crop-model artifacts (no retraining)."
    )
    for feature in FEATURE_NAMES:
        parser.add_argument(feature, type=float)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--label-encoder-path", type=Path, default=DEFAULT_LABEL_ENCODER_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    feature_values = [getattr(args, name) for name in FEATURE_NAMES]
    result = predict_crop(
        feature_values,
        top_k=args.top_k,
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path,
    )
    print(json.dumps(result, indent=2))
    print(
        "Note: reported values are model prediction probabilities, not guaranteed "
        "or automatically calibrated certainty."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
