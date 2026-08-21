"""Final invalid, missing, extreme-input, and weather-independence checks."""

from __future__ import annotations

import unittest

from src.pipeline import run_pipeline
from src.predict import validate_features


class FinalInputErrorMatrixTests(unittest.TestCase):
    def test_negative_nutrients_invalid_ph_and_missing_rainfall_are_rejected(self) -> None:
        valid = [90, 42, 43, 25, 80, 6.5, 200]
        for index, name in enumerate(("N", "P", "K")):
            values = list(valid)
            values[index] = -0.01
            with self.subTest(feature=name), self.assertRaisesRegex(
                ValueError, f"{name} cannot be negative"
            ):
                validate_features(values)
        for ph in (-0.01, 14.01):
            values = list(valid)
            values[5] = ph
            with self.subTest(ph=ph), self.assertRaisesRegex(ValueError, "ph must be between"):
                validate_features(values)
        missing_rainfall = list(valid)
        missing_rainfall[6] = None
        with self.assertRaisesRegex(TypeError, "rainfall must be a numeric"):
            validate_features(missing_rainfall)

    def test_finite_very_low_and_high_values_do_not_crash_inference(self) -> None:
        low = run_pipeline(0, 0, 0, -100, 0, 0, 0)
        high = run_pipeline(1_000_000, 1_000_000, 1_000_000, 100, 100, 14, 1_000_000)
        for result in (low, high):
            self.assertIsInstance(result["predicted_crop"], str)
            self.assertEqual(len(result["top_3"]), 3)
            self.assertGreaterEqual(result["prediction_probability"], 0.0)
            self.assertLessEqual(result["prediction_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
