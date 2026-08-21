"""Streamlit interface for manual and Open-Meteo-assisted model input.

The frontend intentionally owns no training, prediction, explanation, or soil-rule
logic. Both input modes call the same saved-model :func:`src.pipeline.run_pipeline`.
Location assistance is an application enhancement and remains separate from the
reproducible ML experiments.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.app_support import (
    CLOSE_RESULT_MARGIN_THRESHOLD,
    ResearchArtifactError,
    close_result_message,
    get_production_model_metadata,
    load_research_dashboard_data,
    required_production_artifact_issues,
    top_prediction_margin,
    top_three_display_rows,
)
from src.explain import generate_local_explanation
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
PREDICTION_PAGE = "Crop Recommendation"
RESEARCH_PAGE = "Model & Research Evaluation"
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
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stColumn"] {
            width: auto !important;
            max-width: 100% !important;
            min-width: min(100%, 14rem) !important;
            flex: 1 1 14rem !important;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                flex-wrap: nowrap !important;
                gap: 0.55rem;
            }
            [data-testid="stColumn"] {
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }
            [role="radiogroup"] {
                flex-wrap: wrap !important;
            }
            [data-testid="stDataFrame"] {
                max-width: 100%;
            }
            h1 {
                font-size: 1.75rem !important;
                line-height: 1.15 !important;
                overflow-wrap: anywhere;
            }
            .stApp p, .stApp label {
                overflow-wrap: anywhere;
            }
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

    return [
        {key: row[key] for key in ("Rank", "Crop", "Probability")}
        for row in top_three_display_rows(top_recommendations)
    ]


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


def _attach_local_explanation(
    result: Mapping[str, Any],
    inputs: Mapping[str, float | int],
    explanation_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach deterministic local SHAP output without suppressing prediction errors."""

    enriched = dict(result)
    active_explanation = explanation_fn or generate_local_explanation
    try:
        enriched["local_explanation"] = active_explanation(
            inputs,
            predicted_crop=str(result["predicted_crop"]),
            top_n=5,
        )
        enriched["local_explanation_error"] = None
    except Exception:
        LOGGER.exception("Local SHAP explanation could not be produced")
        enriched["local_explanation"] = None
        enriched["local_explanation_error"] = (
            "Detailed explanation temporarily unavailable. The crop prediction and "
            "model probabilities are still available."
        )
    return enriched


def _primary_contributions(
    records: Sequence[Mapping[str, Any]], *, limit: int = 5
) -> list[Mapping[str, Any]]:
    """Keep the strongest records while retaining both directions when present."""

    ordered = list(records)
    selected = ordered[:limit]
    for direction in ("supports", "opposes"):
        if any(row.get("direction") == direction for row in ordered) and not any(
            row.get("direction") == direction for row in selected
        ):
            replacement = next(row for row in ordered if row.get("direction") == direction)
            if selected:
                selected[-1] = replacement
            else:
                selected.append(replacement)
    return selected


def _local_contribution_figure(records: Sequence[Mapping[str, Any]]):
    """Create an in-memory accessible contribution chart; never save user inputs."""

    import matplotlib.pyplot as plt

    ordered = list(reversed(list(records)))
    figure, axis = plt.subplots(figsize=(8, max(3.2, 0.52 * len(ordered))))
    values = [float(row["shap_contribution"]) for row in ordered]
    colors = ["#25704a" if value >= 0 else "#a64b38" for value in values]
    hatches = ["" if value >= 0 else "//" for value in values]
    bars = axis.barh(
        [str(row["feature"]) for row in ordered],
        values,
        color=colors,
        edgecolor="#24352a",
    )
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    axis.axvline(0.0, color="#24352a", linewidth=0.9)
    axis.set_xlabel("SHAP contribution to the predicted-class model output")
    axis.set_title("Explanation for This Prediction")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    return figure


def _technical_expander(label: str):
    """Use an expander in Streamlit and a no-op context in lightweight tests."""

    expander = getattr(st, "expander", None)
    return expander(label) if callable(expander) else nullcontext()


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


def _current_location_context() -> dict[str, Any] | None:
    """Return provider metadata for the current location result, if available."""

    weather = st.session_state.get("weather_current")
    candidates = st.session_state.get("weather_candidates", [])
    selected_index = st.session_state.get("weather_candidate_index")
    if not isinstance(weather, CurrentWeather) or not candidates:
        return None
    try:
        candidate = candidates[int(selected_index)]
    except (IndexError, TypeError, ValueError):
        candidate = candidates[0]
    return {
        "location": candidate.display_name,
        "coordinates": f"{candidate.latitude:.5f}, {candidate.longitude:.5f}",
        "weather_source": weather.source,
        "weather_timestamp": weather.timestamp or "Unavailable",
        "weather_timezone": weather.timezone or "Timezone unavailable",
        "temperature": weather.temperature,
        "temperature_unit": weather.temperature_unit,
        "humidity": weather.humidity,
        "humidity_unit": weather.humidity_unit,
        "precipitation": weather.precipitation,
        "precipitation_unit": weather.precipitation_unit,
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
    weather_context: Mapping[str, Any] | None = None,
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
    st.subheader("Recommended Crop")
    crop_column, probability_column = st.columns(2)
    crop_name = str(result["predicted_crop"]).replace("_", " ").title()
    with crop_column:
        st.metric("Recommended crop", crop_name)
    with probability_column:
        st.metric("Prediction probability", _format_probability(result["prediction_probability"]))
    st.caption(
        "Prediction Probability is the raw value produced by the trained Random Forest. "
        "It is not calibrated real-world certainty or a guarantee of crop success."
    )

    st.subheader("Top-3 Crop Recommendations")
    top_rows = _top_three_rows(result["top_3"])
    st.dataframe(
        pd.DataFrame(top_rows, columns=["Rank", "Crop", "Probability"]),
        hide_index=True,
        use_container_width=True,
    )
    progress = getattr(st, "progress", None)
    if callable(progress):
        for row in top_three_display_rows(result["top_3"]):
            progress(
                float(row["probability_value"]),
                text=f"#{row['Rank']} {row['Crop']} — {row['Probability']}",
            )
    st.caption(
        "Top-3 entries are alternatives ranked by the model's unchanged probabilities; "
        "they are not guaranteed to be agronomically interchangeable."
    )
    similar_message = close_result_message(result["top_3"])
    if similar_message:
        st.info(similar_message)

    st.subheader("Why this crop?")
    explanation = result.get("local_explanation")
    if isinstance(explanation, Mapping):
        st.markdown(f"**{explanation.get('heading', 'Explanation for This Prediction')}**")
        all_contributions = explanation.get("all_contributions", [])
        if not isinstance(all_contributions, Sequence) or isinstance(
            all_contributions, (str, bytes)
        ):
            raise ValueError("Local explanation contributions are malformed.")
        primary = _primary_contributions(all_contributions, limit=5)
        primary_rows = [
            {
                "Feature": str(row["feature"]),
                "Input value": str(row.get("formatted_value", row["feature_value"])),
                "Model direction": str(row.get("direction_label", row["direction"])),
            }
            for row in primary
        ]
        for row in primary_rows:
            st.markdown(
                f"- **{row['Feature']}** · {row['Input value']} — "
                f"**{row['Model direction']}**"
            )
        st.caption(str(explanation.get("scope_notice", "SHAP explains model behavior, not causality.")))
        with _technical_expander("Technical explanation — local SHAP values"):
            technical_rows = [
                {
                    "Feature": str(row["feature"]),
                    "Input value": str(row.get("formatted_value", row["feature_value"])),
                    "Direction": str(row.get("direction_label", row["direction"])),
                    "SHAP contribution": f"{float(row['shap_contribution']):+.6f}",
                }
                for row in all_contributions
            ]
            st.dataframe(
                pd.DataFrame(
                    technical_rows,
                    columns=["Feature", "Input value", "Direction", "SHAP contribution"],
                ),
                hide_index=True,
                use_container_width=True,
            )
            pyplot = getattr(st, "pyplot", None)
            if callable(pyplot):
                figure = _local_contribution_figure(all_contributions)
                pyplot(figure, use_container_width=True)
                import matplotlib.pyplot as plt

                plt.close(figure)
            latency = explanation.get("explanation_latency_ms")
            if latency is not None:
                st.caption(f"Local explanation latency: {float(latency):.1f} ms")
    else:
        st.info(
            str(
                result.get("local_explanation_error")
                or "Detailed explanation temporarily unavailable. The crop prediction remains valid."
            )
        )

    if weather_context is not None:
        st.subheader("Weather Context")
        weather_rows = [
            {"Item": "Selected location", "Value": weather_context.get("location", "Unavailable")},
            {"Item": "Coordinates", "Value": weather_context.get("coordinates", "Unavailable")},
            {
                "Item": "Current temperature",
                "Value": _measurement(
                    weather_context.get("temperature"),
                    str(weather_context.get("temperature_unit", "°C")),
                ),
            },
            {
                "Item": "Current humidity",
                "Value": _measurement(
                    weather_context.get("humidity"),
                    str(weather_context.get("humidity_unit", "%")),
                ),
            },
            {
                "Item": "Current precipitation (context only)",
                "Value": _measurement(
                    weather_context.get("precipitation"),
                    str(weather_context.get("precipitation_unit", "mm")),
                ),
            },
            {"Item": "Weather source", "Value": weather_context.get("weather_source", "Unavailable")},
            {
                "Item": "Provider timestamp",
                "Value": (
                    f"{weather_context.get('weather_timestamp', 'Unavailable')} "
                    f"({weather_context.get('weather_timezone', 'Timezone unavailable')})"
                ),
            },
        ]
        st.dataframe(
            pd.DataFrame(weather_rows, columns=["Item", "Value"]),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Current precipitation is shown for context and is not automatically used "
            "as the model rainfall input."
        )

    if input_mode is not None and model_inputs is not None:
        with _technical_expander("Prediction Details"):
            st.markdown(f"**Input mode:** {input_mode}")
            ordered_fields = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
            details = []
            for field in ordered_fields:
                api_derived = input_mode == LOCATION_MODE and field in {
                    "temperature",
                    "humidity",
                }
                details.append(
                    {
                        "Feature": field,
                        "Value used": model_inputs[field],
                        "Source": "Open-Meteo" if api_derived else "Manual input",
                    }
                )
            st.dataframe(
                pd.DataFrame(details, columns=["Feature", "Value used", "Source"]),
                hide_index=True,
                use_container_width=True,
            )
            if input_mode == LOCATION_MODE:
                st.caption(
                    "Temperature and humidity came from Open-Meteo. N, P, K, pH, and "
                    "model rainfall remained manual inputs."
                )
            else:
                st.caption("All seven model features were entered manually.")
            st.caption(
                "These are the exact seven values passed to the saved baseline Random Forest."
            )
            margin = top_prediction_margin(result["top_3"])
            st.markdown(f"**Top prediction margin:** {margin * 100.0:.2f} percentage points")
            st.caption(
                f"A margin at or below {CLOSE_RESULT_MARGIN_THRESHOLD * 100:.0f} percentage "
                "points triggers the similar-support message. This is not formal uncertainty."
            )

    with _technical_expander("Overall Model Feature Importance"):
        st.subheader("Overall Model Feature Importance")
        st.caption(
            "These values summarize the fitted Random Forest globally. They are not causal "
            "and are distinct from the local explanation for this prediction."
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
            st.image(str(importance_image), caption="Overall Model Feature Importance")

    st.subheader("Indicative Soil Parameter Assessment")
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
        st.metric("Verification status", "Provisional")
        st.warning(
            "These categories are informational and require verification against a "
            "credible, region-appropriate agronomic source, testing method, and unit basis."
        )
    threshold_source = assessment.get("threshold_source")
    if threshold_source:
        st.caption(f"Threshold source/status: {threshold_source}")
    st.caption(
        "The rule-based soil assessment is separate from the ML prediction and does not "
        "alter the crop ranking. It does not prescribe fertilizer products or quantities."
    )


def _research_model(
    models: Sequence[Mapping[str, Any]], name: str
) -> Mapping[str, Any] | None:
    return next((row for row in models if str(row.get("Model")) == name), None)


def _render_research_dashboard() -> None:
    """Render stored measurements only; never execute a research experiment."""

    st.header(RESEARCH_PAGE)
    st.warning(
        "These results describe benchmark evaluation and do not establish field-level "
        "agricultural performance."
    )
    try:
        dashboard = load_research_dashboard_data()
    except ResearchArtifactError as exc:
        LOGGER.warning("Optional research summary is malformed: %s", exc)
        st.error(f"Stored research information could not be read: {exc}")
        return

    for warning in dashboard["warnings"]:
        st.info(warning)
    if not dashboard["available"]:
        st.info("Optional research artifacts are not available; prediction remains usable.")
        return

    try:
        metadata = get_production_model_metadata()
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Production metadata unavailable: %s", exc)
        metadata = None

    st.subheader("Production Model")
    if metadata is None:
        st.error("Production-model metadata is temporarily unavailable.")
    else:
        model_columns = st.columns(3)
        with model_columns[0]:
            st.metric("Model", metadata["model_name"])
        with model_columns[1]:
            st.metric("Input features", metadata["feature_count"])
        with model_columns[2]:
            st.metric("Crop classes", metadata["supported_crop_count"])
        st.caption(
            "Production uses models/random_forest_crop.joblib. Tuned and calibrated "
            "artifacts remain isolated experiments and are not used for predictions."
        )

    baseline = dashboard["baseline"]
    if isinstance(baseline, Mapping):
        st.subheader("Baseline Evaluation")
        metric_columns = st.columns(4)
        values = (
            ("Held-out samples", f"{int(baseline['held_out_samples'])}"),
            ("Held-out accuracy", f"{float(baseline['held_out_accuracy']) * 100:.2f}%"),
            ("Macro F1", f"{float(baseline['macro_f1']):.6f}"),
            ("5-fold CV mean", f"{float(baseline['cv_mean_accuracy']) * 100:.2f}%"),
        )
        for column, (label, value) in zip(metric_columns, values, strict=True):
            with column:
                st.metric(label, value)

    top_k = dashboard["top_k"]
    if isinstance(top_k, Mapping):
        st.subheader("Top-K Benchmark Decision Support")
        columns = st.columns(3)
        for column, key, label in zip(
            columns,
            ("top_1_accuracy", "top_2_accuracy", "top_3_accuracy"),
            ("Top-1", "Top-2", "Top-3"),
            strict=True,
        ):
            with column:
                st.metric(label, f"{float(top_k[key]) * 100:.2f}%")
        st.caption(
            "Stored measurements from the fixed held-out split. Top-2 recovered the two "
            "observed baseline errors on this benchmark; this does not guarantee future results."
        )

    research = dashboard["research"]
    if isinstance(research, Mapping):
        models = research.get("models", [])
        if not isinstance(models, Sequence):
            models = []
        baseline_model = _research_model(models, "Baseline Random Forest")
        tuned_model = _research_model(models, "Tuned Random Forest")
        calibrated_model = _research_model(models, "Sigmoid-Calibrated Random Forest")

        st.subheader("Final Model Decision")
        if baseline_model and tuned_model:
            st.markdown(
                "**Tuned RF not selected:** held-out accuracy and macro F1 were unchanged "
                f"({float(tuned_model['Accuracy']):.6f} and "
                f"{float(tuned_model['Macro F1']):.6f}), while tuned log loss "
                f"{float(tuned_model['Log Loss']):.6f} was worse than baseline "
                f"{float(baseline_model['Log Loss']):.6f}."
            )
        calibration = research.get("calibration", {})
        if baseline_model and calibrated_model and isinstance(calibration, Mapping):
            improved = calibration.get("improved") is True
            st.markdown(
                "**Calibrated RF not selected:** "
                + (
                    "the stored decision rule found an overall improvement."
                    if improved
                    else "the stored decision rule found no overall probability-quality improvement."
                )
                + f" Calibrated log loss was {float(calibrated_model['Log Loss']):.6f}."
            )
        st.caption(
            "The application therefore displays raw Random Forest Prediction Probability, "
            "not guaranteed confidence, yield probability, or crop-success probability."
        )

        ablation = research.get("ablation", {})
        if isinstance(ablation, Mapping):
            largest = ablation.get("largest_macro_f1_degradation")
            if isinstance(largest, Mapping):
                st.subheader("Feature Ablation")
                st.markdown(
                    f"Largest measured degradation: **{largest.get('configuration', 'Unavailable')}**, "
                    f"macro-F1 change {float(largest['macro_f1_delta']):+.6f}."
                )
                st.caption("This is benchmark-model dependence, not agricultural causality.")

        shap_summary = research.get("shap", {})
        if isinstance(shap_summary, Mapping):
            features = shap_summary.get("top_global_features", [])
            if isinstance(features, Sequence) and features:
                st.subheader("Global SHAP Ranking")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Rank": index,
                                "Feature": row["feature"],
                                "Mean |SHAP|": f"{float(row['mean_absolute_shap']):.6f}",
                            }
                            for index, row in enumerate(features, start=1)
                        ],
                        columns=["Rank", "Feature", "Mean |SHAP|"],
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        robustness = research.get("robustness", {})
        if isinstance(robustness, Mapping):
            sensitive = robustness.get("most_sensitivity_causing_feature")
            if isinstance(sensitive, Mapping):
                st.subheader("Controlled Robustness")
                st.markdown(
                    f"Most sensitivity-causing feature under the declared ordering: "
                    f"**{str(sensitive['feature']).title()}**, with "
                    f"{int(sensitive['prediction_flips'])} flips across "
                    f"{int(sensitive['evaluated_perturbations'])} non-zero perturbations."
                )
                st.caption("These are synthetic numerical perturbations, not field validation.")

        disagreement = research.get("inter_model_disagreement", {})
        if isinstance(disagreement, Mapping):
            st.subheader("Inter-model Disagreement")
            st.markdown(
                f"Low modal agreement (4/6 or fewer): **{int(disagreement.get('low_agreement_count', 0))} "
                "of 440 held-out samples**."
            )
            st.caption("Inter-model disagreement is not formal uncertainty quantification.")

        errors = research.get("error_analysis", {})
        if isinstance(errors, Mapping):
            st.subheader("Error Analysis")
            st.metric("Held-out baseline errors", int(errors.get("error_count", 0)))

    if dashboard["charts"]:
        st.subheader("Selected Stored Research Charts")
        for chart in dashboard["charts"]:
            with _technical_expander(str(chart["title"])):
                st.image(str(chart["path"]), caption=str(chart["title"]))

    st.info(
        "Research artifacts are read from results/research/. Opening this page never "
        "runs tuning, calibration, ablation, robustness, SHAP research, or model training."
    )


def main() -> None:
    st.set_page_config(
        page_title="Explainable Crop Recommendation System",
        page_icon="🌱",
        layout="wide",
    )
    _apply_accessible_styles()

    st.title("Explainable Crop Recommendation System")
    st.write(
        "ML-based decision support using tabular soil and environmental measurements, "
        "with optional Open-Meteo assistance and model-behavior explanations."
    )
    st.info(
        "Decision-support prototype only: recommendations require local agronomic review "
        "and external field validation."
    )

    application_page = st.radio(
        "Application section",
        options=(PREDICTION_PAGE, RESEARCH_PAGE),
        horizontal=True,
        index=0,
        help="Research information is loaded from stored artifacts and never recalculated here.",
    )
    if application_page == RESEARCH_PAGE:
        _render_research_dashboard()
        return

    for artifact_issue in required_production_artifact_issues():
        st.error(
            f"{artifact_issue} Restore the required committed artifact; the application "
            "will not download or retrain it automatically."
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
            with st.spinner("Generating model explanation..."):
                result = _attach_local_explanation(result, inputs)
            _render_results(
                result,
                input_mode=input_mode,
                model_inputs=inputs,
                weather_context=(
                    _current_location_context() if input_mode == LOCATION_MODE else None
                ),
            )
    except (TypeError, ValueError) as exc:
        st.error(f"Please check the entered values: {exc}")
    except FileNotFoundError as exc:
        LOGGER.warning("A required prediction artifact was not found: %s", exc)
        st.error(
            "A required committed production artifact is missing. Restore the documented "
            "model and encoder files, then try again; the app does not retrain automatically."
        )
    except Exception:
        LOGGER.exception("Crop recommendation failed")
        st.error(
            "The recommendation could not be completed. Verify the saved artifacts and "
            "input values, then try again."
        )


if __name__ == "__main__":
    main()
