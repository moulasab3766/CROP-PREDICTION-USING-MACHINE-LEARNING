"""Streamlit interface for manual and Open-Meteo-assisted model input.

The frontend intentionally owns no training, prediction, explanation, or soil-rule
logic. Both input modes call the same saved-model :func:`src.pipeline.run_pipeline`.
Location assistance is an application enhancement and remains separate from the
reproducible ML experiments.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.pipeline import run_pipeline
from src.weather import (
    RAINFALL_COMPATIBILITY_WARNING,
    CurrentWeather,
    LocationCandidate,
    WeatherIntegrationError,
    get_current_weather,
    map_weather_to_model_fields,
    search_locations,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_IMPORTANCE_IMAGE = PROJECT_ROOT / "results" / "global_feature_importance.png"
MANUAL_MODE = "Manual Input"
LOCATION_MODE = "Location-Assisted Input"
WEATHER_STATE_DEFAULTS: dict[str, Any] = {
    "weather_candidates": [],
    "weather_selected_identity": None,
    "weather_current": None,
    "weather_refresh_token": 0,
}


def _cache_data(*, ttl: int):
    """Use Streamlit caching while keeping dependency-light import tests possible."""

    cache_function = getattr(st, "cache_data", None)
    if cache_function is None:
        return lambda function: function
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is None:
            return lambda function: function
    except (ImportError, ModuleNotFoundError, TypeError):
        # Lightweight test doubles do not expose Streamlit runtime internals.
        return lambda function: function
    return cache_function(ttl=ttl, show_spinner=False)


@_cache_data(ttl=900)
def _cached_search_locations(query: str) -> list[LocationCandidate]:
    """Cache identical place searches briefly to avoid unnecessary API calls."""

    return search_locations(query)


@_cache_data(ttl=300)
def _cached_current_weather(
    latitude: float,
    longitude: float,
    refresh_token: int,
) -> CurrentWeather:
    """Cache current conditions briefly; refresh_token permits explicit refresh."""

    del refresh_token  # It participates in the cache key but not the provider request.
    return get_current_weather(latitude, longitude)


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
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] span {
            background: #ffffff;
            color: #14261b !important;
        }
        .location-card {
            background: #ffffff;
            border: 1px solid #b8c9bb;
            border-left: 0.35rem solid #176b3a;
            border-radius: 0.65rem;
            color: #14261b;
            padding: 0.9rem 1rem;
            margin: 0.35rem 0 0.8rem;
        }
        .location-card strong, .location-card span {
            color: #14261b !important;
        }
        [data-testid="stRadio"] {
            background: #e7efe8;
            border: 1px solid #b8c9bb;
            border-radius: 0.65rem;
            padding: 0.65rem 0.9rem;
        }
        button:disabled, button:disabled p {
            background: #d9e1da !important;
            border-color: #a6b2a8 !important;
            color: #4c5d50 !important;
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


def _ensure_weather_state() -> None:
    """Initialize only the session values owned by location-assisted mode."""

    for key, default in WEATHER_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = list(default) if isinstance(default, list) else default


def _candidate_identity(candidate: LocationCandidate) -> str:
    """Return a stable identity for selection-change detection."""

    if candidate.provider_id is not None:
        return f"open-meteo:{candidate.provider_id}"
    return f"coordinates:{candidate.latitude:.6f},{candidate.longitude:.6f}"


def _candidate_label(candidate: LocationCandidate) -> str:
    """Include coordinates so similarly named geocoding matches remain distinguishable."""

    return (
        f"{candidate.display_name} "
        f"({candidate.latitude:.4f}, {candidate.longitude:.4f})"
    )


def _measurement(value: float | None, unit: str) -> str:
    """Format provider measurements without replacing missing data with a default."""

    return "Unavailable" if value is None else f"{value:g} {unit}"


def _build_location_model_inputs(
    weather: CurrentWeather,
    *,
    nitrogen: int | float,
    phosphorus: int | float,
    potassium: int | float,
    ph: float,
    rainfall: float,
) -> dict[str, float | int]:
    """Combine explicit soil inputs with only the compatible weather fields."""

    mapped_weather = map_weather_to_model_fields(weather)
    return {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": mapped_weather["temperature"],
        "humidity": mapped_weather["humidity"],
        "ph": ph,
        "rainfall": rainfall,
    }


def _render_manual_input() -> tuple[dict[str, float | int], bool]:
    """Render the original seven-field workflow unchanged in meaning."""

    st.subheader("Manual Input")
    st.caption("Enter all seven model features yourself, using the units shown.")
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

    return (
        {
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        },
        submitted,
    )


def _render_location_input() -> tuple[dict[str, float | int] | None, bool]:
    """Render explicit-search location assistance without background API calls."""

    _ensure_weather_state()
    st.subheader("Location & Weather")
    st.write(
        "Search for a place, choose the exact match, then retrieve current conditions "
        "from Open-Meteo. A request is made only when you select Search or Get/Refresh Weather."
    )
    st.caption(
        "Privacy note: the place text and selected coordinates are sent to Open-Meteo; "
        "this app does not request your device location."
    )

    search_column, search_button_column = st.columns([4, 1])
    with search_column:
        query = st.text_input(
            "City, district, or place",
            placeholder="For example: Bengaluru",
            key="weather_location_query",
        )
    with search_button_column:
        st.write("")
        search_clicked = st.button(
            "Search Location",
            type="primary",
            use_container_width=True,
        )

    if search_clicked:
        st.session_state["weather_candidates"] = []
        st.session_state["weather_selected_identity"] = None
        st.session_state["weather_current"] = None
        st.session_state.pop("weather_candidate_index", None)
        try:
            with st.spinner("Searching Open-Meteo locations..."):
                st.session_state["weather_candidates"] = _cached_search_locations(query)
        except WeatherIntegrationError as exc:
            st.error(str(exc))

    candidates = st.session_state["weather_candidates"]
    if not candidates:
        st.info("Search for a location to choose an unambiguous Open-Meteo match.")
        return None, False

    selected_index = st.selectbox(
        "Select location match",
        options=range(len(candidates)),
        format_func=lambda index: _candidate_label(candidates[index]),
        key="weather_candidate_index",
    )
    candidate = candidates[selected_index]
    identity = _candidate_identity(candidate)
    if identity != st.session_state["weather_selected_identity"]:
        st.session_state["weather_selected_identity"] = identity
        st.session_state["weather_current"] = None

    region = candidate.administrative_region or "Region unavailable"
    country = candidate.country or "Country unavailable"
    timezone = candidate.timezone or "Timezone unavailable"
    st.markdown(
        (
            '<div class="location-card"><strong>Selected location:</strong> '
            f"{escape(candidate.name)}<br>"
            f"<span>{escape(region)}, {escape(country)} · "
            f"{candidate.latitude:.5f}, {candidate.longitude:.5f} · "
            f"{escape(timezone)} · Source: {escape(candidate.source)}</span></div>"
        ),
        unsafe_allow_html=True,
    )

    persisted_weather = st.session_state["weather_current"]
    if persisted_weather is not None and not isinstance(persisted_weather, CurrentWeather):
        # A Streamlit hot reload can leave an instance of the previous class
        # definition in session state. Require a fresh provider response.
        st.session_state["weather_current"] = None
        st.info("The app was updated; retrieve current weather again before prediction.")

    get_column, refresh_column, spacer_column = st.columns([1, 1, 3])
    with get_column:
        get_clicked = st.button(
            "Get Weather",
            type="primary",
            use_container_width=True,
        )
    with refresh_column:
        refresh_clicked = st.button(
            "Refresh Weather",
            disabled=st.session_state["weather_current"] is None,
            use_container_width=True,
        )
    del spacer_column

    if get_clicked or refresh_clicked:
        if refresh_clicked:
            st.session_state["weather_refresh_token"] += 1
        try:
            with st.spinner("Retrieving current Open-Meteo conditions..."):
                st.session_state["weather_current"] = _cached_current_weather(
                    candidate.latitude,
                    candidate.longitude,
                    st.session_state["weather_refresh_token"],
                )
            st.rerun()
        except WeatherIntegrationError as exc:
            st.session_state["weather_current"] = None
            st.error(str(exc))

    weather = st.session_state["weather_current"]
    ready_for_prediction = False
    if weather is None:
        st.info("Get current weather before making a location-assisted prediction.")
    else:
        weather_columns = st.columns(3)
        with weather_columns[0]:
            st.metric(
                "Current temperature",
                _measurement(weather.temperature, weather.temperature_unit),
            )
        with weather_columns[1]:
            st.metric(
                "Current relative humidity",
                _measurement(weather.humidity, weather.humidity_unit),
            )
        with weather_columns[2]:
            st.metric(
                "Current precipitation (context only)",
                _measurement(weather.precipitation, weather.precipitation_unit),
            )
        observed_at = weather.timestamp or "Unavailable"
        observed_timezone = weather.timezone or "Timezone unavailable"
        st.caption(
            f"Provider: {weather.source} · Provider timestamp: {observed_at} "
            f"({observed_timezone}) · Coordinates requested: "
            f"{weather.latitude:.5f}, {weather.longitude:.5f}"
        )
        if weather.missing_fields:
            st.warning(
                "Open-Meteo did not supply: " + ", ".join(weather.missing_fields) + "."
            )
        try:
            mapped_weather = map_weather_to_model_fields(weather)
            ready_for_prediction = True
        except WeatherIntegrationError as exc:
            mapped_weather = {}
            st.error(str(exc))

        st.warning(RAINFALL_COMPATIBILITY_WARNING)
        st.caption(weather.precipitation_description)

    st.subheader("Soil & Model Inputs")
    st.caption(
        "Supply N, P, K, and pH from compatible soil measurements or a soil-test report; "
        "model rainfall also remains manual. Only current temperature and relative "
        "humidity are mapped from Open-Meteo."
    )
    st.session_state["location_temperature"] = (
        float(mapped_weather["temperature"]) if ready_for_prediction else 0.0
    )
    st.session_state["location_humidity"] = (
        float(mapped_weather["humidity"]) if ready_for_prediction else 0.0
    )
    with st.form("location_crop_input_form"):
        nutrient_columns = st.columns(3)
        with nutrient_columns[0]:
            nitrogen = st.number_input("Nitrogen (N)", value=90, step=1, key="location_N")
        with nutrient_columns[1]:
            phosphorus = st.number_input(
                "Phosphorus (P)", value=42, step=1, key="location_P"
            )
        with nutrient_columns[2]:
            potassium = st.number_input("Potassium (K)", value=43, step=1, key="location_K")

        environment_columns = st.columns(4)
        with environment_columns[0]:
            st.number_input(
                "Temperature (°C) · Open-Meteo",
                step=0.1,
                disabled=True,
                key="location_temperature",
            )
        with environment_columns[1]:
            st.number_input(
                "Humidity (%) · Open-Meteo",
                step=0.1,
                disabled=True,
                key="location_humidity",
            )
        with environment_columns[2]:
            ph = st.number_input("Soil pH", value=6.5, step=0.1, key="location_ph")
        with environment_columns[3]:
            rainfall = st.number_input(
                "Rainfall (mm) · manual",
                value=200.0,
                step=0.1,
                key="location_rainfall",
            )

        submitted = st.form_submit_button(
            "Predict Crop with These Inputs",
            type="primary",
            use_container_width=True,
            disabled=not ready_for_prediction,
        )

    if not ready_for_prediction:
        st.caption(
            "Prediction is disabled until compatible current temperature and humidity "
            "have been retrieved. No missing weather value is silently replaced."
        )
        return None, False

    return (
        _build_location_model_inputs(
            weather,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            ph=ph,
            rainfall=rainfall,
        ),
        submitted,
    )


def _soil_status_value(assessment: Mapping[str, Any], key: str) -> str:
    value = assessment.get(key)
    return str(value) if value not in (None, "") else "Unavailable"


def _render_results(
    result: Mapping[str, Any],
    *,
    importance_image: Path = FEATURE_IMPORTANCE_IMAGE,
    input_mode: str | None = None,
    model_inputs: Mapping[str, float | int] | None = None,
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
    if input_mode is not None and model_inputs is not None:
        st.subheader("Prediction Summary")
        st.markdown(f"**Input mode:** {input_mode}")
        ordered_fields = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
        st.dataframe(
            pd.DataFrame(
                [{field: model_inputs[field] for field in ordered_fields}],
                columns=ordered_fields,
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("These are the exact seven values passed to the saved prediction pipeline.")

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
        "and environmental measurements. Enter values manually or use location-assisted "
        "current temperature and humidity."
    )
    st.info(
        "Decision-support prototype only: recommendations require local agronomic review "
        "and external field validation."
    )

    input_mode = st.radio(
        "Choose input mode",
        options=(MANUAL_MODE, LOCATION_MODE),
        horizontal=True,
        index=0,
        help="Manual Input is the default and does not contact a weather provider.",
    )

    if input_mode == MANUAL_MODE:
        inputs, submitted = _render_manual_input()
    else:
        inputs, submitted = _render_location_input()

    if not submitted:
        st.caption(
            "The saved model is used only after you select Predict Crop; no training runs here."
        )
        return

    if inputs is None:
        st.error("Complete the required inputs before requesting a prediction.")
        return

    try:
        with st.spinner("Generating recommendation..."):
            result = _run_if_submitted(submitted, inputs)
        if result is not None:
            _render_results(result, input_mode=input_mode, model_inputs=inputs)
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
