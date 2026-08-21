"""Read-only helpers for the final production UI and research dashboard.

This module never trains a model or writes an artifact.  It centralizes the
production-model decision, Top-3 presentation rules, and defensive loading of
the measured Step-1500 research summaries.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.predict import (
    DEFAULT_LABEL_ENCODER_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    load_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODEL_PATH = DEFAULT_MODEL_PATH
PRODUCTION_LABEL_ENCODER_PATH = DEFAULT_LABEL_ENCODER_PATH
TUNED_EXPERIMENTAL_MODEL_PATH = (
    PROJECT_ROOT / "models" / "research" / "random_forest_tuned.joblib"
)
CALIBRATED_EXPERIMENTAL_MODEL_PATH = (
    PROJECT_ROOT / "models" / "research" / "random_forest_calibrated.joblib"
)
EVALUATION_RESULTS_PATH = PROJECT_ROOT / "results" / "evaluation_results.json"
RESEARCH_SUMMARY_PATH = PROJECT_ROOT / "results" / "research" / "research_summary.json"
TOP_K_METRICS_PATH = PROJECT_ROOT / "results" / "research" / "top_k_metrics.json"

# This is a UI messaging rule, not a claim of calibrated uncertainty.  It matches
# the threshold recorded by the Step-1500 Top-K experiment.
CLOSE_RESULT_MARGIN_THRESHOLD = 0.05

RESEARCH_CHARTS: tuple[tuple[str, str], ...] = (
    ("Top-K benchmark accuracy", "top_k_accuracy.png"),
    ("Global SHAP importance", "shap_global_importance.png"),
    ("Feature ablation", "feature_ablation.png"),
    ("Controlled robustness", "robustness_analysis.png"),
    ("Inter-model disagreement", "model_disagreement.png"),
)


class ResearchArtifactError(ValueError):
    """Raised when an optional research artifact exists but is malformed."""


def required_production_artifact_issues() -> list[str]:
    """Return user-presentable issues for missing required production artifacts."""

    issues: list[str] = []
    if not PRODUCTION_MODEL_PATH.is_file():
        issues.append("Baseline Random Forest model is missing.")
    if not PRODUCTION_LABEL_ENCODER_PATH.is_file():
        issues.append("Production label encoder is missing.")
    return issues


def get_production_model_metadata() -> dict[str, Any]:
    """Describe the read-only baseline production model and verified class order."""

    model, encoder = load_artifacts(
        PRODUCTION_MODEL_PATH,
        PRODUCTION_LABEL_ENCODER_PATH,
    )
    encoded_classes = np.asarray(model.classes_)
    try:
        decoded_classes = np.asarray(encoder.inverse_transform(encoded_classes))
    except Exception as exc:
        raise ValueError("Production model and label-encoder classes do not align.") from exc
    encoder_classes = np.asarray(getattr(encoder, "classes_", ()))
    if decoded_classes.ndim != 1 or not np.array_equal(decoded_classes, encoder_classes):
        raise ValueError("Production class ordering does not match the saved encoder.")

    return {
        "model_name": "Baseline Random Forest",
        "model_path": "models/random_forest_crop.joblib",
        "label_encoder_path": "models/label_encoder.joblib",
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "class_names": [str(value) for value in decoded_classes],
        "supported_crop_count": int(len(decoded_classes)),
        "probability_type": "Raw Random Forest predict_proba output",
        "tuned_model_selected": False,
        "calibrated_model_selected": False,
    }


def _probability(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return number


def ranked_top_three(
    recommendations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and stably rank exactly three unmodified model probabilities."""

    if isinstance(recommendations, (str, bytes, bytearray)):
        raise ValueError("Top-3 recommendations must be a sequence of records.")
    items = list(recommendations)
    if len(items) != 3:
        raise ValueError("The prediction pipeline must return exactly three recommendations.")

    validated: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping) or "crop" not in item or "probability" not in item:
            raise ValueError("Each Top-3 record must contain crop and probability values.")
        crop = str(item["crop"]).strip()
        if not crop:
            raise ValueError("Every Top-3 crop label must be non-empty.")
        validated.append(
            {
                "source_index": index,
                "crop": crop,
                "probability": _probability(
                    item["probability"], label=f"Probability for {crop}"
                ),
            }
        )

    validated.sort(key=lambda item: -item["probability"])
    return [
        {
            "rank": rank,
            "crop": item["crop"],
            "probability": item["probability"],
        }
        for rank, item in enumerate(validated, start=1)
    ]


def top_three_display_rows(
    recommendations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return farmer-facing rows while retaining full-precision probabilities."""

    return [
        {
            "Rank": item["rank"],
            "Crop": item["crop"].replace("_", " ").title(),
            "Probability": f"{item['probability'] * 100.0:.2f}%",
            "probability_value": item["probability"],
        }
        for item in ranked_top_three(recommendations)
    ]


def top_prediction_margin(
    recommendations: Sequence[Mapping[str, Any]],
) -> float:
    """Return Top-1 minus Top-2 model probability without calling it uncertainty."""

    ranked = ranked_top_three(recommendations)
    return float(ranked[0]["probability"] - ranked[1]["probability"])


def close_result_message(
    recommendations: Sequence[Mapping[str, Any]],
    *,
    threshold: float = CLOSE_RESULT_MARGIN_THRESHOLD,
) -> str | None:
    """Return deterministic close-ranking wording at or below ``threshold``."""

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("Close-result threshold must be numeric.")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("Close-result threshold must be between 0 and 1.")
    if top_prediction_margin(recommendations) <= threshold_value:
        return (
            "The model has similar support for more than one crop. These are "
            "model-ranked alternatives, not a claim that the crops are agronomically "
            "interchangeable."
        )
    return None


def _read_optional_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        warnings.append(f"Optional research artifact is unavailable: {path.name}.")
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchArtifactError(f"Could not parse {path.name}.") from exc
    if not isinstance(payload, dict):
        raise ResearchArtifactError(f"{path.name} must contain a JSON object.")
    return payload


def _nested_number(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    *,
    fraction: bool = False,
) -> float:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ResearchArtifactError(
                f"Research artifact is missing numeric field: {'.'.join(keys)}."
            )
        value = value[key]
    if isinstance(value, bool):
        raise ResearchArtifactError(f"{'.'.join(keys)} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchArtifactError(f"{'.'.join(keys)} must be numeric.") from exc
    if not math.isfinite(number):
        raise ResearchArtifactError(f"{'.'.join(keys)} must be finite.")
    if fraction and not 0.0 <= number <= 1.0:
        raise ResearchArtifactError(f"{'.'.join(keys)} must be between 0 and 1.")
    return number


def load_research_dashboard_data(
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Load measured summaries for display without executing any experiment."""

    root = Path(project_root).expanduser().resolve()
    warnings: list[str] = []
    evaluation = _read_optional_json(root / "results" / "evaluation_results.json", warnings)
    summary = _read_optional_json(
        root / "results" / "research" / "research_summary.json", warnings
    )
    top_k_payload = _read_optional_json(
        root / "results" / "research" / "top_k_metrics.json", warnings
    )

    baseline: dict[str, Any] | None = None
    if evaluation is not None:
        baseline = {
            "held_out_accuracy": _nested_number(
                evaluation, ("held_out_test_metrics", "accuracy"), fraction=True
            ),
            "macro_f1": _nested_number(
                evaluation, ("held_out_test_metrics", "macro_f1"), fraction=True
            ),
            "held_out_samples": int(
                _nested_number(evaluation, ("split", "test_samples"))
            ),
            "cv_mean_accuracy": _nested_number(
                evaluation, ("cross_validation", "mean_accuracy"), fraction=True
            ),
        }

    top_k: dict[str, Any] | None = None
    if top_k_payload is not None:
        metrics = top_k_payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ResearchArtifactError("top_k_metrics.json is missing metrics.")
        top_k = {
            "top_1_accuracy": _nested_number(
                metrics, ("top_1_accuracy",), fraction=True
            ),
            "top_2_accuracy": _nested_number(
                metrics, ("top_2_accuracy",), fraction=True
            ),
            "top_3_accuracy": _nested_number(
                metrics, ("top_3_accuracy",), fraction=True
            ),
            "sample_count": int(_nested_number(metrics, ("sample_count",))),
            "near_tie_threshold": _nested_number(
                metrics,
                ("top_1_minus_top_2_margin", "near_tie_threshold"),
                fraction=True,
            ),
        }

    if summary is not None and not isinstance(summary.get("models"), list):
        raise ResearchArtifactError("research_summary.json is missing its model list.")

    research_dir = root / "results" / "research"
    charts = [
        {"title": title, "path": research_dir / filename}
        for title, filename in RESEARCH_CHARTS
        if (research_dir / filename).is_file()
    ]
    missing_chart_count = len(RESEARCH_CHARTS) - len(charts)
    if missing_chart_count:
        warnings.append(f"{missing_chart_count} optional research chart(s) are unavailable.")

    return {
        "available": any(item is not None for item in (evaluation, summary, top_k_payload)),
        "baseline": baseline,
        "top_k": top_k,
        "research": summary,
        "charts": charts,
        "warnings": warnings,
    }
