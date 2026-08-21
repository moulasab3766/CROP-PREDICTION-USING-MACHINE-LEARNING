"""Combined, Streamlit-independent crop recommendation pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # Supports module and direct-script execution.
    from src.explain import get_global_feature_importance
    from src.predict import (
        DEFAULT_LABEL_ENCODER_PATH,
        DEFAULT_MODEL_PATH,
        load_artifacts,
        predict_crop,
        validate_features,
    )
    from src.soil_assessment import ThresholdRule, assess_soil
except ImportError:  # pragma: no cover - direct script use
    from explain import get_global_feature_importance
    from predict import (
        DEFAULT_LABEL_ENCODER_PATH,
        DEFAULT_MODEL_PATH,
        load_artifacts,
        predict_crop,
        validate_features,
    )
    from soil_assessment import ThresholdRule, assess_soil


def run_pipeline(
    N: Any,
    P: Any,
    K: Any,
    temperature: Any,
    humidity: Any,
    ph: Any,
    rainfall: Any,
    *,
    model_path: str | Path | None = None,
    label_encoder_path: str | Path | None = None,
    feature_importance_csv_path: str | Path | None = None,
    feature_importance_chart_path: str | Path | None = None,
    soil_thresholds: Mapping[str, ThresholdRule | Mapping[str, Any]] | None = None,
    soil_threshold_source: str | None = None,
) -> dict[str, Any]:
    """Run prediction, global explanation, and independent soil assessment.

    The returned dictionary always contains the five required keys:
    ``predicted_crop``, ``prediction_probability``, ``top_3``,
    ``feature_importance``, and ``soil_assessment``.  No model is trained here,
    and the soil rules never alter the ML recommendation.
    """

    values = validate_features((N, P, K, temperature, humidity, ph, rainfall))
    model, label_encoder = load_artifacts(model_path, label_encoder_path)

    prediction = predict_crop(
        values,
        top_k=3,
        model=model,
        label_encoder=label_encoder,
    )
    feature_importance = get_global_feature_importance(
        model=model,
        save_csv_path=feature_importance_csv_path,
        save_chart_path=feature_importance_chart_path,
    )
    soil_assessment = assess_soil(
        values["N"],
        values["P"],
        values["K"],
        values["ph"],
        thresholds=soil_thresholds,
        threshold_source=soil_threshold_source,
    )

    return {
        "predicted_crop": prediction["predicted_crop"],
        "prediction_probability": prediction["prediction_probability"],
        "top_3": prediction["top_3"],
        "feature_importance": feature_importance,
        "soil_assessment": soil_assessment,
    }


# Descriptive alias for callers and future UI code.
crop_recommendation_pipeline = run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete saved-model crop recommendation pipeline."
    )
    for feature in ("N", "P", "K", "temperature", "humidity", "ph", "rainfall"):
        parser.add_argument(feature, type=float)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--label-encoder-path", type=Path, default=DEFAULT_LABEL_ENCODER_PATH
    )
    parser.add_argument("--importance-csv", type=Path)
    parser.add_argument("--importance-chart", type=Path)
    parser.add_argument(
        "--soil-thresholds-json",
        type=Path,
        help="Optional sourced and verified soil-threshold mapping.",
    )
    parser.add_argument("--soil-threshold-source")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    thresholds = None
    if args.soil_thresholds_json is not None:
        with args.soil_thresholds_json.open(encoding="utf-8") as handle:
            thresholds = json.load(handle)
    result = run_pipeline(
        args.N,
        args.P,
        args.K,
        args.temperature,
        args.humidity,
        args.ph,
        args.rainfall,
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path,
        feature_importance_csv_path=args.importance_csv,
        feature_importance_chart_path=args.importance_chart,
        soil_thresholds=thresholds,
        soil_threshold_source=args.soil_threshold_source,
    )
    print(json.dumps(result, indent=2))
    print(
        "Note: probability is the model's raw prediction probability, global "
        "importance is not causal, and soil rules are reported separately."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
