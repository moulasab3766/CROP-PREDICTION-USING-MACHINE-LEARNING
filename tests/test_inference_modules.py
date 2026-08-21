"""Focused tests for saved-artifact inference and assessment modules."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.explain import get_global_feature_importance
from src.pipeline import run_pipeline
from src.predict import FEATURE_NAMES, clear_artifact_cache, predict_crop, validate_features
from src.soil_assessment import NEEDS_VERIFICATION, assess_soil


class FakeRandomForest:
    """Small fitted-artifact stand-in; production code never trains it."""

    classes_ = np.asarray([0, 1, 2, 3])
    feature_importances_ = np.asarray([0.10, 0.20, 0.05, 0.15, 0.25, 0.05, 0.20])

    def _check_frame(self, frame):
        if list(frame.columns) != list(FEATURE_NAMES):
            raise AssertionError("Feature order was not preserved")

    def predict(self, frame):
        self._check_frame(frame)
        return np.asarray([1])

    def predict_proba(self, frame):
        self._check_frame(frame)
        return np.asarray([[0.05, 0.65, 0.20, 0.10]])


class FakeLabelEncoder:
    labels = np.asarray(["apple", "banana", "coffee", "rice"])

    def inverse_transform(self, values):
        values = np.asarray(values)
        if np.any(values < 0) or np.any(values >= len(self.labels)):
            raise ValueError("unseen label")
        return self.labels[values.astype(int)]


VALID_VALUES = (90, 42, 43, 25, 80, 6.5, 200)


class PredictionTests(unittest.TestCase):
    def tearDown(self):
        clear_artifact_cache()

    def test_real_probability_columns_align_with_decoded_top_three(self):
        shuffled = {
            "rainfall": 200,
            "ph": 6.5,
            "humidity": 80,
            "temperature": 25,
            "K": 43,
            "P": 42,
            "N": 90,
        }
        result = predict_crop(
            shuffled, model=FakeRandomForest(), label_encoder=FakeLabelEncoder()
        )

        self.assertEqual(result["predicted_crop"], "banana")
        self.assertEqual(result["prediction_probability"], 0.65)
        self.assertEqual(
            [entry["crop"] for entry in result["top_3"]],
            ["banana", "coffee", "rice"],
        )
        self.assertEqual(
            [entry["probability"] for entry in result["top_3"]],
            [0.65, 0.20, 0.10],
        )

    def test_saved_artifacts_are_loaded_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            encoder_path = Path(directory) / "encoder.joblib"
            joblib.dump(FakeRandomForest(), model_path)
            joblib.dump(FakeLabelEncoder(), encoder_path)

            result = predict_crop(
                VALID_VALUES,
                model_path=model_path,
                label_encoder_path=encoder_path,
            )
            self.assertEqual(result["predicted_crop"], "banana")

    def test_strict_input_shape_names_types_and_domains(self):
        self.assertEqual(list(validate_features(VALID_VALUES)), list(FEATURE_NAMES))
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_features({name: 1 for name in FEATURE_NAMES if name != "N"})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_features(
                {**dict(zip(FEATURE_NAMES, VALID_VALUES, strict=True)), "extra": 1}
            )
        with self.assertRaisesRegex(TypeError, "N must be a numeric"):
            validate_features(("90", *VALID_VALUES[1:]))
        with self.assertRaisesRegex(ValueError, "N cannot be negative"):
            validate_features((-1, *VALID_VALUES[1:]))
        with self.assertRaisesRegex(ValueError, "ph must be between"):
            validate_features((*VALID_VALUES[:5], 15, VALID_VALUES[6]))


class ExplanationTests(unittest.TestCase):
    def test_global_importance_is_sorted_normalised_and_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "importance.csv"
            chart_path = Path(directory) / "importance.png"
            records = get_global_feature_importance(
                model=FakeRandomForest(),
                save_csv_path=csv_path,
                save_chart_path=chart_path,
            )

            self.assertEqual(records[0]["feature"], "humidity")
            self.assertAlmostEqual(
                sum(float(record["importance_percent"]) for record in records),
                100.0,
            )
            self.assertTrue(csv_path.is_file())
            self.assertTrue(chart_path.is_file())


class SoilAssessmentTests(unittest.TestCase):
    def test_safe_default_does_not_invent_threshold_classifications(self):
        result = assess_soil(90, 42, 43, 6.5)
        for key in (
            "nitrogen_status",
            "phosphorus_status",
            "potassium_status",
            "ph_status",
            "overall_assessment",
        ):
            self.assertEqual(result[key], NEEDS_VERIFICATION)
        self.assertFalse(result["thresholds_verified"])

    def test_only_explicitly_verified_sourced_rules_enable_categories(self):
        rules = {
            feature: {
                "lower": lower,
                "upper": upper,
                "source": "Test-only fixture; not an agronomic recommendation",
                "verified": True,
            }
            for feature, lower, upper in (
                ("N", 80, 100),
                ("P", 40, 50),
                ("K", 40, 50),
                ("ph", 6, 7),
            )
        }
        result = assess_soil(90, 42, 43, 6.5, thresholds=rules)
        self.assertEqual(result["nitrogen_status"], "Adequate")
        self.assertEqual(result["ph_status"], "Suitable")
        self.assertTrue(result["thresholds_verified"])

    def test_invalid_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "P cannot be negative"):
            assess_soil(90, -1, 43, 6.5)
        with self.assertRaisesRegex(ValueError, "ph must be between"):
            assess_soil(90, 42, 43, 14.1)
        with self.assertRaisesRegex(TypeError, "K must be a numeric"):
            assess_soil(90, 42, "43", 6.5)


class CombinedPipelineTests(unittest.TestCase):
    def tearDown(self):
        clear_artifact_cache()

    def test_pipeline_returns_all_required_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            encoder_path = Path(directory) / "encoder.joblib"
            joblib.dump(FakeRandomForest(), model_path)
            joblib.dump(FakeLabelEncoder(), encoder_path)

            result = run_pipeline(
                *VALID_VALUES,
                model_path=model_path,
                label_encoder_path=encoder_path,
            )
            self.assertEqual(
                set(result),
                {
                    "predicted_crop",
                    "prediction_probability",
                    "top_3",
                    "feature_importance",
                    "soil_assessment",
                },
            )
            self.assertEqual(len(result["top_3"]), 3)
            self.assertEqual(result["soil_assessment"]["overall_assessment"], NEEDS_VERIFICATION)

    def test_pipeline_rejects_invalid_nutrients_ph_and_non_numeric_input(self):
        for index, name in enumerate(("N", "P", "K")):
            values = list(VALID_VALUES)
            values[index] = -1
            with self.subTest(feature=name), self.assertRaisesRegex(
                ValueError, f"{name} cannot be negative"
            ):
                run_pipeline(*values)

        invalid_ph = list(VALID_VALUES)
        invalid_ph[5] = 15
        with self.assertRaisesRegex(ValueError, "ph must be between"):
            run_pipeline(*invalid_ph)

        non_numeric = list(VALID_VALUES)
        non_numeric[3] = "warm"
        with self.assertRaisesRegex(TypeError, "temperature must be a numeric"):
            run_pipeline(*non_numeric)


if __name__ == "__main__":
    unittest.main()
