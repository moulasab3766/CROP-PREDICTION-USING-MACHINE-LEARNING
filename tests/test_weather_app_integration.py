"""Focused tests for the UI-to-model weather mapping boundary."""

from __future__ import annotations

import unittest

import app
from src.weather import CURRENT_FIELDS, CurrentWeather, LocationCandidate


def _weather(*, precipitation: float = 999.0) -> CurrentWeather:
    return CurrentWeather(
        latitude=12.97194,
        longitude=77.59369,
        temperature=28.0,
        humidity=54.0,
        precipitation=precipitation,
        timestamp="2026-08-21T13:30",
        timezone="Asia/Kolkata",
        temperature_unit="°C",
        humidity_unit="%",
        precipitation_unit="mm",
        source="Open-Meteo",
        source_url="https://api.open-meteo.com/v1/forecast",
        missing_fields=(),
        precipitation_description="Context only.",
        requested_fields=CURRENT_FIELDS,
    )


class WeatherAppMappingTests(unittest.TestCase):
    def test_location_mode_builds_exact_seven_model_fields(self) -> None:
        inputs = app._build_location_model_inputs(
            _weather(),
            nitrogen=90,
            phosphorus=42,
            potassium=43,
            ph=6.5,
            rainfall=200.0,
        )

        self.assertEqual(
            list(inputs),
            ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        )
        self.assertEqual(inputs["temperature"], 28.0)
        self.assertEqual(inputs["humidity"], 54.0)

    def test_provider_precipitation_never_overwrites_manual_rainfall(self) -> None:
        inputs = app._build_location_model_inputs(
            _weather(precipitation=999.0),
            nitrogen=90,
            phosphorus=42,
            potassium=43,
            ph=6.5,
            rainfall=123.4,
        )

        self.assertEqual(inputs["rainfall"], 123.4)
        self.assertNotIn("precipitation", inputs)

    def test_ambiguous_location_labels_include_context_and_coordinates(self) -> None:
        candidate = LocationCandidate(
            name="Paris",
            administrative_region="Île-de-France",
            country="France",
            latitude=48.85341,
            longitude=2.3488,
            provider_id=2988507,
        )

        label = app._candidate_label(candidate)
        self.assertIn("Paris, Île-de-France, France", label)
        self.assertIn("48.8534, 2.3488", label)
        self.assertEqual(app._candidate_identity(candidate), "open-meteo:2988507")


if __name__ == "__main__":
    unittest.main()
