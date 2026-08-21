"""Network-independent unit tests for the Open-Meteo weather module."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

import requests

from src.weather import (
    FORECAST_API_URL,
    GEOCODING_API_URL,
    CurrentWeather,
    LocationNotFoundError,
    LocationQueryError,
    WeatherCompatibilityError,
    WeatherResponseError,
    WeatherServiceError,
    get_current_weather,
    map_weather_to_model_fields,
    search_locations,
    validate_location_query,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def fake_session(payload, status_code=200):
    session = Mock()
    session.get.return_value = FakeResponse(payload, status_code=status_code)
    return session


def sample_location_payload():
    return {
        "results": [
            {
                "id": 1277333,
                "name": "Bengaluru",
                "latitude": 12.97194,
                "longitude": 77.59369,
                "country": "India",
                "country_code": "IN",
                "admin1": "Karnataka",
                "timezone": "Asia/Kolkata",
            },
            {
                "id": 123,
                "name": "Bengaluru Rural",
                "latitude": 13.2,
                "longitude": 77.5,
                "country": "India",
                "admin1": "Karnataka",
                "timezone": "Asia/Kolkata",
            },
        ]
    }


def sample_weather_payload(*, precipitation=0.0):
    return {
        "latitude": 12.97,
        "longitude": 77.59,
        "timezone": "Asia/Kolkata",
        "current_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
        },
        "current": {
            "time": "2026-08-21T12:30",
            "temperature_2m": 24.6,
            "relative_humidity_2m": 74,
            "precipitation": precipitation,
        },
    }


class LocationValidationTests(unittest.TestCase):
    def test_query_is_trimmed_and_unicode_is_preserved(self):
        self.assertEqual(validate_location_query("  São Paulo  "), "São Paulo")

    def test_empty_short_and_extremely_long_queries_are_rejected(self):
        for query in ("", "   ", "x", "a" * 201):
            with self.subTest(query_length=len(query)):
                with self.assertRaises(LocationQueryError):
                    validate_location_query(query)

    def test_search_uses_query_params_and_returns_multiple_clean_candidates(self):
        session = fake_session(sample_location_payload())
        candidates = search_locations("  Bengaluru, India  ", session=session)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].name, "Bengaluru")
        self.assertEqual(candidates[0].display_name, "Bengaluru, Karnataka, India")
        self.assertEqual(candidates[0].timezone, "Asia/Kolkata")
        called_url = session.get.call_args.args[0]
        called_kwargs = session.get.call_args.kwargs
        self.assertEqual(called_url, GEOCODING_API_URL)
        self.assertNotIn("Bengaluru", called_url)
        self.assertEqual(called_kwargs["params"]["name"], "Bengaluru, India")
        self.assertEqual(called_kwargs["timeout"], 10.0)

    def test_special_characters_and_lowercase_are_passed_as_data(self):
        session = fake_session(sample_location_payload())
        search_locations("mysuru & district?", session=session)
        params = session.get.call_args.kwargs["params"]
        self.assertEqual(params["name"], "mysuru & district?")

    def test_unknown_location_is_informative(self):
        with self.assertRaisesRegex(LocationNotFoundError, "No Open-Meteo"):
            search_locations("zzzznonsenseplace", session=fake_session({}))

    def test_malformed_location_results_do_not_escape_to_the_ui(self):
        session = fake_session({"results": [{"name": "Broken"}]})
        with self.assertRaisesRegex(WeatherResponseError, "usable coordinates"):
            search_locations("Broken", session=session)


class ProviderFailureTests(unittest.TestCase):
    def test_timeout_is_wrapped_in_user_presentable_error(self):
        session = Mock()
        session.get.side_effect = requests.Timeout("slow")
        with self.assertRaisesRegex(WeatherServiceError, "timeout"):
            search_locations("Bengaluru", session=session)

    def test_network_failure_is_wrapped(self):
        session = Mock()
        session.get.side_effect = requests.ConnectionError("offline")
        with self.assertRaisesRegex(WeatherServiceError, "could not be reached"):
            search_locations("Bengaluru", session=session)

    def test_http_error_is_wrapped(self):
        with self.assertRaisesRegex(WeatherServiceError, r"HTTP error \(500\)"):
            get_current_weather(12.97, 77.59, session=fake_session({}, 500))

    def test_malformed_json_is_wrapped(self):
        session = fake_session(ValueError("not json"))
        with self.assertRaisesRegex(WeatherResponseError, "unreadable response"):
            search_locations("Bengaluru", session=session)


class CurrentWeatherTests(unittest.TestCase):
    def test_successful_weather_preserves_units_source_time_and_coordinates(self):
        session = fake_session(sample_weather_payload(precipitation=1.2))
        weather = get_current_weather(12.97194, 77.59369, session=session)

        self.assertEqual(weather.temperature, 24.6)
        self.assertEqual(weather.humidity, 74.0)
        self.assertEqual(weather.precipitation, 1.2)
        self.assertEqual(weather.timestamp, "2026-08-21T12:30")
        self.assertEqual(weather.timezone, "Asia/Kolkata")
        self.assertEqual(weather.temperature_unit, "°C")
        self.assertEqual(weather.humidity_unit, "%")
        self.assertEqual(weather.precipitation_unit, "mm")
        self.assertEqual(weather.source, "Open-Meteo")
        self.assertEqual(weather.latitude, 12.97194)
        self.assertEqual(weather.longitude, 77.59369)
        self.assertEqual(weather.missing_fields, ())
        params = session.get.call_args.kwargs["params"]
        self.assertEqual(session.get.call_args.args[0], FORECAST_API_URL)
        self.assertEqual(params["temperature_unit"], "celsius")
        self.assertEqual(params["precipitation_unit"], "mm")
        self.assertIn("relative_humidity_2m", params["current"])

    def test_zero_precipitation_is_valid(self):
        weather = get_current_weather(
            12.97, 77.59, session=fake_session(sample_weather_payload(precipitation=0))
        )
        self.assertEqual(weather.precipitation, 0.0)
        self.assertNotIn("precipitation", weather.missing_fields)

    def test_missing_measurements_are_reported_without_defaults(self):
        payload = sample_weather_payload()
        del payload["current"]["temperature_2m"]
        del payload["current"]["precipitation"]
        weather = get_current_weather(12.97, 77.59, session=fake_session(payload))
        self.assertIsNone(weather.temperature)
        self.assertIsNone(weather.precipitation)
        self.assertIn("temperature", weather.missing_fields)
        self.assertIn("precipitation", weather.missing_fields)

    def test_missing_current_object_is_rejected(self):
        with self.assertRaisesRegex(WeatherResponseError, "current conditions"):
            get_current_weather(12.97, 77.59, session=fake_session({}))

    def test_impossible_or_malformed_values_are_rejected(self):
        cases = (
            ("temperature_2m", 200),
            ("relative_humidity_2m", 101),
            ("precipitation", -0.1),
            ("temperature_2m", "hot"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                payload = sample_weather_payload()
                payload["current"][field] = value
                with self.assertRaises(WeatherResponseError):
                    get_current_weather(12.97, 77.59, session=fake_session(payload))

    def test_invalid_coordinates_are_rejected_before_request(self):
        for latitude, longitude in ((91, 0), (-91, 0), (0, 181), (0, -181), ("x", 0)):
            with self.subTest(latitude=latitude, longitude=longitude):
                with self.assertRaises(WeatherResponseError):
                    get_current_weather(latitude, longitude, session=Mock())


class CompatibilityMappingTests(unittest.TestCase):
    def test_only_temperature_and_humidity_are_mapped(self):
        weather = get_current_weather(
            12.97, 77.59, session=fake_session(sample_weather_payload(precipitation=8.5))
        )
        mapped = map_weather_to_model_fields(weather)
        self.assertEqual(mapped, {"temperature": 24.6, "humidity": 74.0})
        self.assertNotIn("rainfall", mapped)
        self.assertNotIn("precipitation", mapped)

    def test_mapping_rejects_missing_required_api_fields(self):
        payload = sample_weather_payload()
        payload["current"]["temperature_2m"] = None
        weather = get_current_weather(12.97, 77.59, session=fake_session(payload))
        with self.assertRaisesRegex(WeatherCompatibilityError, "temperature is missing"):
            map_weather_to_model_fields(weather)

    def test_mapping_requires_current_weather_instance(self):
        with self.assertRaises(TypeError):
            map_weather_to_model_fields({"temperature": 25, "humidity": 70})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
