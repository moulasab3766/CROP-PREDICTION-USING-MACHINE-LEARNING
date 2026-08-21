"""Global Random Forest feature importance reporting.

The impurity-based values exposed by ``feature_importances_`` describe global
model behaviour.  They are not causal effects and are not a causal explanation
of an individual recommendation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # Supports module and direct-script execution.
    from src.predict import DEFAULT_MODEL_PATH, FEATURE_NAMES, load_model
except ImportError:  # pragma: no cover - direct script use
    from predict import DEFAULT_MODEL_PATH, FEATURE_NAMES, load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPORTANCE_CSV_PATH = PROJECT_ROOT / "results" / "global_feature_importance.csv"
DEFAULT_IMPORTANCE_CHART_PATH = (
    PROJECT_ROOT / "results" / "global_feature_importance.png"
)


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
