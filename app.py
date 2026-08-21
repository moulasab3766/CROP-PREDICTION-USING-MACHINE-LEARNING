"""Streamlit interface for the saved crop-recommendation pipeline.

The frontend intentionally owns no training, prediction, explanation, or soil-rule
logic.  It collects the seven manual inputs, invokes :func:`src.pipeline.run_pipeline`
after form submission, and presents the structured result.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.pipeline import run_pipeline


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_IMPORTANCE_IMAGE = PROJECT_ROOT / "results" / "global_feature_importance.png"


def _apply_accessible_styles() -> None:
    """Use a restrained, high-contrast light presentation."""

    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f8f5;
            color: #14261b;
        }
        .stApp h1, .stApp h2, .stApp h3,
        .stApp p, .stApp label, .stApp [data-testid="stCaptionContainer"] {
            color: #14261b;
        }
        [data-testid="stSidebar"] {
            background: #e7efe8;
            color: #14261b;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #b8c9bb;
            border-radius: 0.65rem;
            padding: 1rem;
        }
        [data-testid="stMetric"] * {
            color: #14261b !important;
        }
        .stButton > button, [data-testid="stFormSubmitButton"] button {
            background: #176b3a;
            border: 2px solid #0e4d29;
            color: #ffffff !important;
            font-weight: 700;
        }
        .stButton > button p, [data-testid="stFormSubmitButton"] button p {
            color: #ffffff !important;
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
            background: #0e4d29;
            border-color: #08371d;
            color: #ffffff !important;
        }
        input {
            background: #ffffff !important;
            color: #14261b !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _probability_as_percent(value: Any) -> float:
    """Convert a model probability in the documented 0..1 range to percent."""

    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model probability must be numeric.") from exc

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Model probability must be between 0 and 1.")
    return probability * 100.0


def _format_probability(value: Any) -> str:
    """Format a genuine pipeline probability without changing its meaning."""

    return f"{_probability_as_percent(value):.2f}%"


def _top_three_rows(top_recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build three descending display rows from pipeline recommendations."""

    if isinstance(top_recommendations, (str, bytes)):
        raise ValueError("Top-3 recommendations must be a sequence of records.")

    recommendations = list(top_recommendations)
    if len(recommendations) != 3:
        raise ValueError("The prediction pipeline must return exactly three recommendations.")

    try:
        recommendations.sort(key=lambda item: float(item["probability"]), reverse=True)
        return [
            {
                "Rank": rank,
                "Crop": str(item["crop"]).replace("_", " ").title(),
                "Probability": _format_probability(item["probability"]),
            }
            for rank, item in enumerate(recommendations, start=1)
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("Each Top-3 record must contain crop and probability values.") from exc


def _importance_rows(feature_importance: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Prepare global feature-importance values for a readable table."""

    rows: list[dict[str, Any]] = []
    for item in feature_importance:
        try:
            feature = str(item["feature"])
            if "importance_percent" in item:
                percentage = float(item["importance_percent"])
            else:
                percentage = float(item["importance"]) * 100.0
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Feature-importance records are incomplete or invalid.") from exc
        rows.append({"Feature": feature, "Global importance": f"{percentage:.2f}%"})
    return rows


def _run_if_submitted(
    submitted: bool,
    inputs: Mapping[str, float | int],
    pipeline_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any] | None:
    """Invoke the combined pipeline only in response to form submission."""

    if not submitted:
        return None
    active_pipeline = pipeline_fn or run_pipeline
    return active_pipeline(**dict(inputs))


def _soil_status_value(assessment: Mapping[str, Any], key: str) -> str:
    value = assessment.get(key)
    return str(value) if value not in (None, "") else "Unavailable"


def _render_results(
    result: Mapping[str, Any],
    *,
    importance_image: Path = FEATURE_IMPORTANCE_IMAGE,
) -> None:
    """Render a validated pipeline response using Streamlit."""

    required = {
        "predicted_crop",
        "prediction_probability",
        "top_3",
        "feature_importance",
        "soil_assessment",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"Prediction result is missing: {', '.join(missing)}.")

    st.divider()
    st.subheader("Recommendation")
    crop_column, probability_column = st.columns(2)
    crop_name = str(result["predicted_crop"]).replace("_", " ").title()
    with crop_column:
        st.metric("Recommended crop", crop_name)
    with probability_column:
        st.metric("Prediction probability", _format_probability(result["prediction_probability"]))
    st.caption(
        "Prediction probability is the model's raw class probability; it is not a "
        "calibrated guarantee of agronomic success."
    )

    st.subheader("Top-3 Crop Recommendations")
    top_rows = _top_three_rows(result["top_3"])
    st.dataframe(
        pd.DataFrame(top_rows, columns=["Rank", "Crop", "Probability"]),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Global Model Feature Importance")
    st.caption(
        "These values summarize the fitted Random Forest globally. They are not causal "
        "explanations and do not explain this individual recommendation on their own."
    )
    importance = result["feature_importance"]
    if importance:
        st.dataframe(
            pd.DataFrame(
                _importance_rows(importance),
                columns=["Feature", "Global importance"],
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Global feature-importance data is not available yet.")
    if importance_image.is_file():
        st.image(str(importance_image), caption="Global Random Forest Feature Importance")

    st.subheader("Soil Nutrient Assessment")
    assessment = result["soil_assessment"]
    if not isinstance(assessment, Mapping):
        raise ValueError("Soil assessment must be a structured record.")

    status_fields = (
        ("Nitrogen (N)", "nitrogen_status"),
        ("Phosphorus (P)", "phosphorus_status"),
        ("Potassium (K)", "potassium_status"),
        ("Soil pH", "ph_status"),
    )
    for row_start in range(0, len(status_fields), 2):
        status_columns = st.columns(2)
        for column, (label, key) in zip(
            status_columns,
            status_fields[row_start : row_start + 2],
        ):
            with column:
                st.metric(label, _soil_status_value(assessment, key))

    st.markdown(f"**Overall soil assessment:** {_soil_status_value(assessment, 'overall_assessment')}")
    if assessment.get("thresholds_verified") is not True:
        st.warning(
            "Soil-status thresholds still require verification against a credible, "
            "region-appropriate agricultural source. Treat these statuses as provisional."
        )
    threshold_source = assessment.get("threshold_source")
    if threshold_source:
        st.caption(f"Threshold source/status: {threshold_source}")
    st.caption(
        "The rule-based soil assessment is separate from the ML prediction and does not "
        "alter the crop ranking."
    )


def main() -> None:
    st.set_page_config(
        page_title="Smart Crop Recommendation System",
        page_icon="🌱",
        layout="wide",
    )
    _apply_accessible_styles()

    st.title("Smart Crop Recommendation System")
    st.write(
        "An explainable machine-learning crop recommendation system using tabular soil "
        "and environmental measurements. Enter all seven values to receive a ranked result."
    )
    st.info(
        "Decision-support prototype only: recommendations require local agronomic review "
        "and external field validation."
    )

    st.subheader("Manual Input")
    with st.form("crop_input_form"):
        nutrient_columns = st.columns(3)
        with nutrient_columns[0]:
            nitrogen = st.number_input("Nitrogen (N)", value=90, step=1)
        with nutrient_columns[1]:
            phosphorus = st.number_input("Phosphorus (P)", value=42, step=1)
        with nutrient_columns[2]:
            potassium = st.number_input("Potassium (K)", value=43, step=1)

        environment_columns = st.columns(4)
        with environment_columns[0]:
            temperature = st.number_input("Temperature (°C)", value=25.0, step=0.1)
        with environment_columns[1]:
            humidity = st.number_input("Humidity (%)", value=80.0, step=0.1)
        with environment_columns[2]:
            ph = st.number_input("Soil pH", value=6.5, step=0.1)
        with environment_columns[3]:
            rainfall = st.number_input("Rainfall (mm)", value=200.0, step=0.1)

        submitted = st.form_submit_button(
            "Predict Crop",
            type="primary",
            use_container_width=True,
        )

    inputs: dict[str, float | int] = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }

    if not submitted:
        st.caption("The saved model is used only after you select Predict Crop; no training runs here.")
        return

    try:
        with st.spinner("Generating recommendation..."):
            result = _run_if_submitted(submitted, inputs)
        if result is not None:
            _render_results(result)
    except (TypeError, ValueError) as exc:
        st.error(f"Please check the entered values: {exc}")
    except FileNotFoundError as exc:
        LOGGER.warning("A required prediction artifact was not found: %s", exc)
        st.error(
            "A required trained-model artifact is missing. Run the documented training "
            "and explanation commands, then try again."
        )
    except Exception:
        LOGGER.exception("Crop recommendation failed")
        st.error(
            "The recommendation could not be completed. Verify the saved artifacts and "
            "input values, then try again."
        )


if __name__ == "__main__":
    main()
