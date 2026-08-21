"""Global Random Forest feature importance reporting.

The impurity-based values exposed by ``feature_importances_`` describe global
model behaviour.  They are not causal effects and are not a causal explanation
of an individual recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

try:  # Supports module and direct-script execution.
    from src.predict import (
        DEFAULT_MODEL_PATH,
        FEATURE_NAMES,
        load_artifacts,
        load_model,
        predict_crop,
        validate_features,
    )
except ImportError:  # pragma: no cover - direct script use
    from predict import (
        DEFAULT_MODEL_PATH,
        FEATURE_NAMES,
        load_artifacts,
        load_model,
        predict_crop,
        validate_features,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPORTANCE_CSV_PATH = PROJECT_ROOT / "results" / "global_feature_importance.csv"
DEFAULT_IMPORTANCE_CHART_PATH = (
    PROJECT_ROOT / "results" / "global_feature_importance.png"
)


def normalize_multiclass_shap(
    raw_values: Any,
    *,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """Normalize supported SHAP versions to samples × features × classes."""

    if isinstance(raw_values, list):
        if len(raw_values) != n_classes:
            raise ValueError("SHAP class-list length does not match model classes.")
        arrays = [np.asarray(values, dtype=float) for values in raw_values]
        if any(values.shape != (n_samples, n_features) for values in arrays):
            raise ValueError("A class-specific SHAP matrix has an unexpected shape.")
        normalized = np.stack(arrays, axis=-1)
    else:
        normalized = np.asarray(raw_values, dtype=float)
        if normalized.shape == (n_classes, n_samples, n_features):
            normalized = np.moveaxis(normalized, 0, -1)
    expected = (n_samples, n_features, n_classes)
    if normalized.shape != expected:
        raise ValueError(
            f"Expected multiclass SHAP shape {expected}, found {normalized.shape}."
        )
    if not np.isfinite(normalized).all():
        raise ValueError("SHAP contributions contain non-finite values.")
    return normalized


def contribution_records(
    feature_values: Sequence[float],
    contributions: Sequence[float],
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> list[dict[str, Any]]:
    """Create absolute-magnitude-sorted local contribution records."""

    values = np.asarray(feature_values, dtype=float)
    shap_values = np.asarray(contributions, dtype=float)
    if values.shape != (len(feature_names),) or shap_values.shape != (
        len(feature_names),
    ):
        raise ValueError("Feature values and SHAP contributions must match feature order.")
    order = np.argsort(-np.abs(shap_values), kind="stable")
    return [
        {
            "feature": str(feature_names[index]),
            "feature_value": float(values[index]),
            "shap_contribution": float(shap_values[index]),
            "direction": "supports" if shap_values[index] >= 0 else "opposes",
        }
        for index in order
    ]


def local_explanation_text(
    predicted_crop: str,
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int = 3,
) -> str:
    """Create deterministic, non-causal wording for a local explanation."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Explanation limit must be a positive integer.")
    strongest = list(records)[:limit]
    details = "; ".join(
        f"{row['feature']} ({row['direction']} the predicted-class model score, "
        f"SHAP {float(row['shap_contribution']):+.6f})"
        for row in strongest
    )
    return (
        f"Predicted crop: {predicted_crop}. Strongest model contributions: {details}. "
        "These values explain model behavior for this input; they are not causal "
        "agronomic effects or proof of crop suitability."
    )


def format_feature_value(feature: str, value: Any) -> str:
    """Format an input value without inventing units for N, P, or K."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature value for {feature} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"Feature value for {feature} must be finite.")
    if feature == "temperature":
        return f"{number:.1f} °C"
    if feature == "humidity":
        return f"{number:.1f}%"
    if feature == "ph":
        return f"{number:.2f} pH"
    if feature == "rainfall":
        return f"{number:.1f} mm"
    if feature in {"N", "P", "K"}:
        return f"{number:g} (dataset-scale value)"
    return f"{number:g}"


@lru_cache(maxsize=4)
def _tree_explainer_for_model(model: Any) -> Any:
    """Cache only the explainer; never cache or retain a user's feature values."""

    import shap

    return shap.TreeExplainer(model)


def clear_local_explainer_cache() -> None:
    """Clear cached SHAP explainers, primarily for artifact-replacement tests."""

    _tree_explainer_for_model.cache_clear()


def generate_local_explanation(
    values: Mapping[str, Any] | Sequence[Any],
    *,
    predicted_crop: str | None = None,
    top_n: int = 5,
    model: Any | None = None,
    label_encoder: Any | None = None,
    explainer: Any | None = None,
) -> dict[str, Any]:
    """Explain the predicted class for one input using the baseline RF and SHAP.

    The explanation is deterministic model attribution, not causal or agronomic
    advice.  The function does not save a plot, mutate artifacts, or train a model.
    """

    started = perf_counter()
    canonical = validate_features(values)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= len(
        FEATURE_NAMES
    ):
        raise ValueError(f"top_n must be between 1 and {len(FEATURE_NAMES)}.")
    if (model is None) != (label_encoder is None):
        raise ValueError("Provide both model and label_encoder, or neither.")
    if model is None:
        model, label_encoder = load_artifacts()

    if predicted_crop is None:
        predicted_crop = str(
            predict_crop(canonical, model=model, label_encoder=label_encoder)[
                "predicted_crop"
            ]
        )
    frame = pd.DataFrame(
        [[canonical[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES
    )
    encoded_classes = np.asarray(model.classes_)
    try:
        decoded_classes = np.asarray(label_encoder.inverse_transform(encoded_classes))
    except Exception as exc:
        raise ValueError("Model classes cannot be decoded for local explanation.") from exc
    matches = np.flatnonzero(decoded_classes.astype(str) == str(predicted_crop))
    if len(matches) != 1:
        raise ValueError("Predicted crop does not map to exactly one SHAP output class.")
    predicted_column = int(matches[0])

    active_explainer = explainer or _tree_explainer_for_model(model)
    raw_values = active_explainer.shap_values(frame)
    normalized = normalize_multiclass_shap(
        raw_values,
        n_samples=1,
        n_features=len(FEATURE_NAMES),
        n_classes=len(encoded_classes),
    )
    records = contribution_records(
        [canonical[name] for name in FEATURE_NAMES],
        normalized[0, :, predicted_column],
    )
    enriched = [
        {
            **record,
            "formatted_value": format_feature_value(
                str(record["feature"]), record["feature_value"]
            ),
            "direction_label": (
                "Supports prediction"
                if record["direction"] == "supports"
                else "Opposes prediction"
            ),
        }
        for record in records
    ]
    title_crop = str(predicted_crop).replace("_", " ").title()
    return {
        "predicted_crop": str(predicted_crop),
        "heading": f"Why did the model recommend {title_crop}?",
        "top_contributions": enriched[:top_n],
        "all_contributions": enriched,
        "text": local_explanation_text(str(predicted_crop), records, limit=top_n),
        "explanation_latency_ms": float((perf_counter() - started) * 1000.0),
        "scope_notice": (
            "SHAP describes how this model scored this input. It does not establish "
            "agricultural causality, soil quality, or guaranteed crop suitability."
        ),
    }


def _importance_records(model: Any) -> list[dict[str, float | str]]:
    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "The saved model has no feature_importances_; a fitted Random Forest "
            "artifact is required for this global explanation."
        )
    importances = np.asarray(model.feature_importances_, dtype=float)
    if importances.ndim != 1 or len(importances) != len(FEATURE_NAMES):
        raise ValueError(
            "Model feature_importances_ does not align with the seven feature names."
        )
    if not np.all(np.isfinite(importances)) or np.any(importances < 0):
        raise ValueError("Model feature_importances_ contains invalid values.")
    total = float(importances.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Model feature_importances_ must have a positive total.")

    normalised = importances / total
    order = np.argsort(-normalised, kind="stable")
    return [
        {
            "feature": str(FEATURE_NAMES[index]),
            "importance": float(normalised[index]),
            "importance_percent": float(normalised[index] * 100.0),
        }
        for index in order
    ]


def save_feature_importance_table(
    records: Sequence[dict[str, float | str]], path: str | Path
) -> Path:
    """Save global importance records as a CSV table and return its path."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    return output


def save_feature_importance_chart(
    records: Sequence[dict[str, float | str]], path: str | Path
) -> Path:
    """Save a readable global model-importance bar chart and return its path."""

    # Import lazily so callers that only need structured data do not initialise a
    # plotting backend.  Agg is suitable for scripts, tests, and headless servers.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    required = {"feature", "importance_percent"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Feature-importance records are empty or malformed.")

    figure_height = max(4.5, 0.55 * len(frame))
    fig, ax = plt.subplots(figsize=(9, figure_height))
    ax.barh(frame["feature"], frame["importance_percent"], color="#2f6f4e")
    ax.invert_yaxis()
    ax.set_xlabel("Importance (%)")
    ax.set_ylabel("Feature")
    ax.set_title("Global Random Forest Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def get_global_feature_importance(
    *,
    model: Any | None = None,
    model_path: str | Path | None = None,
    label_encoder_path: str | Path | None = None,
    save_csv_path: str | Path | None = None,
    save_chart_path: str | Path | None = None,
) -> list[dict[str, float | str]]:
    """Return sorted global model importance and optionally save both artifacts."""

    if model is None:
        # label_encoder_path remains an accepted, ignored compatibility keyword;
        # global model importance does not need or load the target encoder.
        model = load_model(model_path)
    records = _importance_records(model)
    if save_csv_path is not None:
        save_feature_importance_table(records, save_csv_path)
    if save_chart_path is not None:
        save_feature_importance_chart(records, save_chart_path)
    return records


# A concise alias useful to callers that prefer a calculation-oriented name.
calculate_global_feature_importance = get_global_feature_importance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate global Random Forest feature importance (not causal)."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--label-encoder-path", type=Path, default=None)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_IMPORTANCE_CSV_PATH)
    parser.add_argument(
        "--chart-path", type=Path, default=DEFAULT_IMPORTANCE_CHART_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    records = get_global_feature_importance(
        model_path=args.model_path,
        label_encoder_path=args.label_encoder_path,
        save_csv_path=args.csv_path,
        save_chart_path=args.chart_path,
    )
    print(json.dumps(records, indent=2))
    print(f"Saved table: {args.csv_path.resolve()}")
    print(f"Saved chart: {args.chart_path.resolve()}")
    print("Scope: global model importance; these values are not causal effects.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
