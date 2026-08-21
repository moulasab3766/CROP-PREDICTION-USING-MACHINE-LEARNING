"""Boundary, provenance, and failure-isolation tests for soil assessment."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.pipeline import run_pipeline
from src.soil_assessment import NEEDS_VERIFICATION, assess_soil


VERIFIED_TEST_RULES = {
    feature: {
        "lower": lower,
        "upper": upper,
        "source": "Synthetic test-only boundary fixture; not agronomic guidance",
        "verified": True,
    }
    for feature, lower, upper in (
        ("N", 80.0, 100.0),
        ("P", 40.0, 50.0),
        ("K", 40.0, 50.0),
        ("ph", 6.0, 7.0),
    )
}


class FinalSoilSafetyTests(unittest.TestCase):
    def test_exact_just_below_and_just_above_boundaries(self) -> None:
        at_lower = assess_soil(80.0, 40.0, 40.0, 6.0, thresholds=VERIFIED_TEST_RULES)
        at_upper = assess_soil(100.0, 50.0, 50.0, 7.0, thresholds=VERIFIED_TEST_RULES)
        below = assess_soil(79.99, 39.99, 39.99, 5.99, thresholds=VERIFIED_TEST_RULES)
        above = assess_soil(100.01, 50.01, 50.01, 7.01, thresholds=VERIFIED_TEST_RULES)

        for result in (at_lower, at_upper):
            self.assertEqual(result["nitrogen_status"], "Adequate")
            self.assertEqual(result["ph_status"], "Suitable")
        self.assertEqual(below["nitrogen_status"], "Low")
        self.assertEqual(below["ph_status"], "Low")
        self.assertEqual(above["nitrogen_status"], "High")
        self.assertEqual(above["ph_status"], "High")

    def test_unverified_rules_never_become_scientific_categories(self) -> None:
        provisional = {
            key: {**rule, "verified": False} for key, rule in VERIFIED_TEST_RULES.items()
        }
        result = assess_soil(90.5, 42.25, 43.75, 6.5, thresholds=provisional)
        self.assertFalse(result["thresholds_verified"])
        self.assertEqual(result["overall_assessment"], NEEDS_VERIFICATION)
        self.assertTrue(
            all(
                result[key] == NEEDS_VERIFICATION
                for key in (
                    "nitrogen_status",
                    "phosphorus_status",
                    "potassium_status",
                    "ph_status",
                )
            )
        )

    def test_negative_malformed_and_invalid_ph_inputs(self) -> None:
        for values in ((-0.01, 42, 43, 6.5), (90, -0.01, 43, 6.5), (90, 42, -0.01, 6.5)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                assess_soil(*values)
        for ph in (-0.01, 14.01):
            with self.subTest(ph=ph), self.assertRaises(ValueError):
                assess_soil(90, 42, 43, ph)
        with self.assertRaises(TypeError):
            assess_soil(90, 42, "malformed", 6.5)

    def test_assessment_failure_does_not_remove_model_prediction(self) -> None:
        with patch("src.pipeline.assess_soil", side_effect=RuntimeError("test-only failure")):
            result = run_pipeline(90, 42, 43, 25, 80, 6.5, 200)
        self.assertEqual(result["predicted_crop"], "rice")
        self.assertEqual(result["prediction_probability"], 0.54)
        self.assertTrue(result["soil_assessment"]["assessment_error"])
        self.assertFalse(result["soil_assessment"]["thresholds_verified"])


if __name__ == "__main__":
    unittest.main()
