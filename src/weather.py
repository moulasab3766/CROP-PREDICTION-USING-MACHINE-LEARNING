"""Open-Meteo location search and current-weather integration.

This module is an application-level enhancement.  It is deliberately isolated
from model training, preprocessing, evaluation, and the saved ML artifacts.
Only temperature and relative humidity have a verified unit-level mapping to
the model inputs.  Open-Meteo current precipitation is contextual information;
it is not automatically mapped to the dataset's underspecified ``rainfall``
feature.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

PROVIDER_NAME = "Open-Meteo"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_LOCATION_RESULTS = 5
MAX_LOCATION_RESULTS = 10
MAX_LOCATION_QUERY_LENGTH = 200

CURRENT_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
)
PRECIPITATION_DESCRIPTION = (
    "Current precipitation from Open-Meteo in millimetres. Open-Meteo current "
    "conditions are based on recent model time steps; this value is not assumed "
    "equivalent to the crop dataset's rainfall feature."
)
RAINFALL_COMPATIBILITY_WARNING = (
    "The Kaggle data card describes rainfall only as 'rainfall in mm' and does "
    "not state its measurement period. Current Open-Meteo precipitation is "
    "therefore not automatically used as the model rainfall input."
)


class WeatherIntegrationError(RuntimeError):
    """Base class for user-presentable weather integration failures."""


class LocationQueryError(WeatherIntegrationError):
    """Raised when a place-search query is invalid."""


class LocationNotFoundError(WeatherIntegrationError):
    """Raised when Open-Meteo returns no match for a place query."""


class WeatherServiceError(WeatherIntegrationError):
    """Raised when a provider request cannot be completed."""


class WeatherResponseError(WeatherIntegrationError):
    """Raised when the provider response is malformed or incompatible."""


class WeatherCompatibilityError(WeatherIntegrationError):
    """Raised when weather data cannot safely populate required model fields."""


@dataclass(frozen=True)
class LocationCandidate:
    """A compact, deterministic representation of one geocoding match."""

    name: str
    latitude: float
    longitude: float
    country: str | None = None
    administrative_region: str | None = None
    timezone: str | None = None
    country_code: str | None = None
    provider_id: int | str | None = None
    source: str = PROVIDER_NAME

    @property
    def display_name(self) -> str:
        """Return a readable label suitable for an ambiguity-selection widget."""

        parts: list[str] = []
        for value in (self.name, self.administrative_region, self.country):
            if value and value not in parts:
                parts.append(value)
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary without exposing irrelevant API fields."""

        result = asdict(self)
        result["display_name"] = self.display_name
        return result


@dataclass(frozen=True)
class CurrentWeather:
    """Validated current conditions for one selected coordinate pair."""

    latitude: float
    longitude: float
    temperature: float | None
    humidity: float | None
    precipitation: float | None
    timestamp: str | None
    timezone: str | None
    temperature_unit: str
    humidity_unit: str
    precipitation_unit: str
    source: str
    source_url: str
    missing_fields: tuple[str, ...]
    precipitation_description: str
    requested_fields: tuple[str, ...]
    grid_latitude: float | None = None
    grid_longitude: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary for UI/session-state use."""

        result = asdict(self)
        result["missing_fields"] = list(self.missing_fields)
        result["requested_fields"] = list(self.requested_fields)
        return result


def validate_location_query(query: Any) -> str:
    """Trim and validate a user-entered location search string."""

    if not isinstance(query, str):
        raise LocationQueryError("Location must be entered as text.")
    cleaned = query.strip()
    if not cleaned:
        raise LocationQueryError("Enter a location before searching.")
    if len(cleaned) < 2:
        raise LocationQueryError("Location searches require at least two characters.")
    if len(cleaned) > MAX_LOCATION_QUERY_LENGTH:
        raise LocationQueryError(
            f"Location text is too long; use at most {MAX_LOCATION_QUERY_LENGTH} characters."
        )
    return cleaned


def _validate_timeout(timeout: Any) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, Real):
        raise TypeError("HTTP timeout must be numeric.")
    value = float(timeout)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("HTTP timeout must be a positive finite number.")
    return value


def _validate_result_count(count: Any) -> int:
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError("Location result count must be an integer.")
    value = int(count)
    if not 1 <= value <= MAX_LOCATION_RESULTS:
        raise ValueError(
            f"Location result count must be between 1 and {MAX_LOCATION_RESULTS}."
        )
    return value


def _request_json(
    url: str,
    params: dict[str, Any],
    *,
    timeout: float,
    session: Any | None,
) -> dict[str, Any]:
    """Issue one bounded provider request and return a validated JSON object."""

    client = session if session is not None else requests
    try:
        response = client.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout as exc:
        LOGGER.warning("Open-Meteo request timed out for %s: %s", url, exc)
        LOGGER.debug("Open-Meteo timeout details", exc_info=True)
        raise WeatherServiceError(
            "Open-Meteo did not respond before the request timeout. Try again later."
        ) from exc
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is None:
            status = getattr(locals().get("response"), "status_code", "unknown")
        LOGGER.warning("Open-Meteo HTTP failure %s for %s", status, url)
        LOGGER.debug("Open-Meteo HTTP failure details", exc_info=True)
        raise WeatherServiceError(
            f"Open-Meteo returned an HTTP error ({status}). Try again later."
        ) from exc
    except (requests.RequestException, OSError) as exc:
        LOGGER.warning("Open-Meteo network failure for %s: %s", url, exc)
        LOGGER.debug("Open-Meteo network failure details", exc_info=True)
        raise WeatherServiceError(
            "Open-Meteo could not be reached. Check the network connection and try again."
        ) from exc

    try:
        payload = response.json()
    except (ValueError, requests.JSONDecodeError) as exc:
        LOGGER.warning("Open-Meteo returned malformed JSON for %s", url)
        LOGGER.debug("Open-Meteo malformed JSON details", exc_info=True)
        raise WeatherResponseError(
            "Open-Meteo returned an unreadable response. Try again later."
        ) from exc

    if not isinstance(payload, dict):
        raise WeatherResponseError("Open-Meteo returned an unexpected response format.")
    if payload.get("error") is True:
        reason = payload.get("reason")
        detail = str(reason).strip() if reason else "request rejected"
        raise WeatherServiceError(f"Open-Meteo rejected the request: {detail}.")
    return payload


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _coordinate(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise WeatherResponseError(f"{name} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise WeatherResponseError(
            f"{name} must be between {minimum:g} and {maximum:g}."
        )
    return number


def _parse_location(raw: Any) -> LocationCandidate:
    if not isinstance(raw, dict):
        raise WeatherResponseError("A location result was not a JSON object.")
    name = _optional_text(raw.get("name"))
    if name is None:
        raise WeatherResponseError("A location result is missing its place name.")
    return LocationCandidate(
        name=name,
        latitude=_coordinate("latitude", raw.get("latitude"), -90.0, 90.0),
        longitude=_coordinate("longitude", raw.get("longitude"), -180.0, 180.0),
        country=_optional_text(raw.get("country")),
        administrative_region=_optional_text(raw.get("admin1")),
        timezone=_optional_text(raw.get("timezone")),
        country_code=_optional_text(raw.get("country_code")),
        provider_id=raw.get("id") if isinstance(raw.get("id"), (int, str)) else None,
    )


def search_locations(
    query: Any,
    *,
    count: int = DEFAULT_LOCATION_RESULTS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: Any | None = None,
) -> list[LocationCandidate]:
    """Search Open-Meteo geocoding and return clean location candidates.

    User input is passed through ``requests`` query parameters, which performs
    URL encoding without concatenating raw text into the endpoint URL.  Provider
    ordering is preserved so a selected candidate remains deterministic.
    """

    cleaned_query = validate_location_query(query)
    result_count = _validate_result_count(count)
    request_timeout = _validate_timeout(timeout)
    payload = _request_json(
        GEOCODING_API_URL,
        {
            "name": cleaned_query,
            "count": result_count,
            "language": "en",
            "format": "json",
        },
        timeout=request_timeout,
        session=session,
    )
    raw_results = payload.get("results")
    if raw_results in (None, []):
        raise LocationNotFoundError(
            f"No Open-Meteo location matches were found for {cleaned_query!r}."
        )
    if not isinstance(raw_results, list):
        raise WeatherResponseError("Open-Meteo geocoding results were malformed.")

    candidates: list[LocationCandidate] = []
    for position, raw in enumerate(raw_results):
        try:
            candidates.append(_parse_location(raw))
        except WeatherResponseError as exc:
            LOGGER.warning(
                "Skipping malformed Open-Meteo location result at position %s: %s",
                position,
                exc,
            )
            LOGGER.debug("Malformed Open-Meteo location details", exc_info=True)
    if not candidates:
        raise WeatherResponseError(
            "Open-Meteo returned location matches, but none contained usable coordinates."
        )
    return candidates


def _optional_measurement(
    container: dict[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if key not in container or container[key] is None:
        return None
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise WeatherResponseError(f"Open-Meteo field {key!r} was not numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise WeatherResponseError(f"Open-Meteo field {key!r} was not finite.")
    if minimum is not None and number < minimum:
        raise WeatherResponseError(f"Open-Meteo field {key!r} was below its valid range.")
    if maximum is not None and number > maximum:
        raise WeatherResponseError(f"Open-Meteo field {key!r} exceeded its valid range.")
    return number


def _unit(units: dict[str, Any], key: str, requested: str) -> str:
    value = units.get(key)
    if value is None:
        return requested
    if not isinstance(value, str) or not value.strip():
        raise WeatherResponseError(f"Open-Meteo unit for {key!r} was malformed.")
    return value.strip()


def get_current_weather(
    latitude: Any,
    longitude: Any,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: Any | None = None,
) -> CurrentWeather:
    """Retrieve validated current conditions for selected WGS84 coordinates.

    Missing current measurements are returned as ``None`` and named in
    ``missing_fields``.  No arbitrary defaults are substituted.  Temperature is
    requested in Celsius, relative humidity in percent, and precipitation in
    millimetres.
    """

    requested_latitude = _coordinate("latitude", latitude, -90.0, 90.0)
    requested_longitude = _coordinate("longitude", longitude, -180.0, 180.0)
    request_timeout = _validate_timeout(timeout)
    payload = _request_json(
        FORECAST_API_URL,
        {
            "latitude": requested_latitude,
            "longitude": requested_longitude,
            "current": ",".join(CURRENT_FIELDS),
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "timezone": "auto",
        },
        timeout=request_timeout,
        session=session,
    )
    current = payload.get("current")
    if not isinstance(current, dict):
        raise WeatherResponseError("Open-Meteo response is missing current conditions.")
    current_units = payload.get("current_units", {})
    if not isinstance(current_units, dict):
        raise WeatherResponseError("Open-Meteo current-condition units were malformed.")

    temperature = _optional_measurement(
        current, "temperature_2m", minimum=-100.0, maximum=70.0
    )
    humidity = _optional_measurement(
        current, "relative_humidity_2m", minimum=0.0, maximum=100.0
    )
    precipitation = _optional_measurement(current, "precipitation", minimum=0.0)
    timestamp = _optional_text(current.get("time"))
    timezone = _optional_text(payload.get("timezone"))

    temperature_unit = _unit(current_units, "temperature_2m", "°C")
    humidity_unit = _unit(current_units, "relative_humidity_2m", "%")
    precipitation_unit = _unit(current_units, "precipitation", "mm")
    if temperature_unit not in {"°C", "C", "celsius"}:
        raise WeatherResponseError(
            f"Open-Meteo returned unexpected temperature unit {temperature_unit!r}."
        )
    if humidity_unit != "%":
        raise WeatherResponseError(
            f"Open-Meteo returned unexpected humidity unit {humidity_unit!r}."
        )
    if precipitation_unit != "mm":
        raise WeatherResponseError(
            f"Open-Meteo returned unexpected precipitation unit {precipitation_unit!r}."
        )

    missing: list[str] = []
    for name, value in (
        ("temperature", temperature),
        ("humidity", humidity),
        ("precipitation", precipitation),
        ("timestamp", timestamp),
        ("timezone", timezone),
    ):
        if value is None:
            missing.append(name)

    grid_latitude = None
    grid_longitude = None
    if payload.get("latitude") is not None:
        grid_latitude = _coordinate("grid latitude", payload["latitude"], -90.0, 90.0)
    if payload.get("longitude") is not None:
        grid_longitude = _coordinate(
            "grid longitude", payload["longitude"], -180.0, 180.0
        )

    return CurrentWeather(
        latitude=requested_latitude,
        longitude=requested_longitude,
        temperature=temperature,
        humidity=humidity,
        precipitation=precipitation,
        timestamp=timestamp,
        timezone=timezone,
        temperature_unit=temperature_unit,
        humidity_unit=humidity_unit,
        precipitation_unit=precipitation_unit,
        source=PROVIDER_NAME,
        source_url=FORECAST_API_URL,
        missing_fields=tuple(missing),
        precipitation_description=PRECIPITATION_DESCRIPTION,
        requested_fields=CURRENT_FIELDS,
        grid_latitude=grid_latitude,
        grid_longitude=grid_longitude,
    )


def map_weather_to_model_fields(weather: CurrentWeather) -> dict[str, float]:
    """Map only unit-compatible API fields to model temperature and humidity.

    Rainfall is intentionally absent from the return value because the Kaggle
    data card states only ``rainfall in mm`` and does not document a temporal
    measurement period compatible with Open-Meteo current precipitation.
    """

    if not isinstance(weather, CurrentWeather):
        raise TypeError("weather must be a CurrentWeather instance.")
    if weather.temperature is None:
        raise WeatherCompatibilityError(
            "Current temperature is missing; prediction requires a temperature value."
        )
    if weather.humidity is None:
        raise WeatherCompatibilityError(
            "Current humidity is missing; prediction requires a humidity value."
        )
    if weather.temperature_unit not in {"°C", "C", "celsius"}:
        raise WeatherCompatibilityError(
            "Temperature cannot be mapped because it is not in degrees Celsius."
        )
    if weather.humidity_unit != "%":
        raise WeatherCompatibilityError(
            "Humidity cannot be mapped because it is not a percentage."
        )
    return {
        "temperature": float(weather.temperature),
        "humidity": float(weather.humidity),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Open-Meteo locations and optionally retrieve current weather."
    )
    parser.add_argument("query", help="Place name, optionally qualified by region/country")
    parser.add_argument("--count", type=int, default=DEFAULT_LOCATION_RESULTS)
    parser.add_argument(
        "--weather",
        action="store_true",
        help="Retrieve current weather for the selected result index.",
    )
    parser.add_argument("--result-index", type=int, default=0)
    return parser


def main() -> int:
    """Run a small standalone Open-Meteo development check."""

    args = _build_parser().parse_args()
    try:
        candidates = search_locations(args.query, count=args.count)
        print(json.dumps([candidate.to_dict() for candidate in candidates], indent=2))
        if args.weather:
            if not 0 <= args.result_index < len(candidates):
                raise LocationQueryError("Selected result index is out of range.")
            selected = candidates[args.result_index]
            weather = get_current_weather(selected.latitude, selected.longitude)
            print(json.dumps(weather.to_dict(), indent=2))
            print(RAINFALL_COMPATIBILITY_WARNING)
    except WeatherIntegrationError as exc:
        print(f"Weather integration error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
