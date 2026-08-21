"""Final production-model selection and local-SHAP integration tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import app
from src.app_support import (
    CALIBRATED_EXPERIMENTAL_MODEL_PATH,
    PRODUCTION_MODEL_PATH,
    TUNED_EXPERIMENTAL_MODEL_PATH,
    get_production_model_metadata,
    required_production_artifact_issues,
)
from src.explain import (
    clear_local_explainer_cache,
    format_feature_value,
    generate_local_explanation,
)
from src.predict import FEATURE_NAMES, load_artifacts, predict_crop


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"


class ProductionModelDecisionTests(unittest.TestCase):
    def test_production_path_is_baseline_not_experimental(self) -> None:
        self.assertEqual(
            PRODUCTION_MODEL_PATH.resolve(),
            (PROJECT_ROOT / "models" / "random_forest_crop.joblib").resolve(),
        )
        self.assertNotEqual(PRODUCTION_MODEL_PATH.resolve(), TUNED_EXPERIMENTAL_MODEL_PATH.resolve())
        self.assertNotEqual(
            PRODUCTION_MODEL_PATH.resolve(), CALIBRATED_EXPERIMENTAL_MODEL_PATH.resolve()
        )
        self.assertEqual(PRODUCTION_MODEL_PATH.parent.name, "models")

    def test_metadata_verifies_saved_class_order(self) -> None:
        metadata = get_production_model_metadata()
        model, encoder = load_artifacts()
        decoded = [str(value) for value in encoder.inverse_transform(model.classes_)]

        self.assertEqual(metadata["model_name"], "Baseline Random Forest")
        self.assertEqual(metadata["class_names"], decoded)
        self.assertEqual(decoded, [str(value) for value in encoder.classes_])
        self.assertEqual(metadata["feature_names"], list(FEATURE_NAMES))
        self.assertFalse(metadata["tuned_model_selected"])
        self.assertFalse(metadata["calibrated_model_selected"])

    def test_missing_required_artifacts_are_detected_without_training(self) -> None:
        missing_root = PROJECT_ROOT / "does-not-exist"
        with patch("src.app_support.PRODUCTION_MODEL_PATH", missing_root / "model.joblib"), patch(
            "src.app_support.PRODUCTION_LABEL_ENCODER_PATH", missing_root / "encoder.joblib"
        ):
            issues = required_production_artifact_issues()
        self.assertEqual(len(issues), 2)
        self.assertIn("Baseline Random Forest", issues[0])
        self.assertIn("label encoder", issues[1])


class LocalShapProductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = pd.read_csv(DATASET_PATH)
        cls.model, cls.encoder = load_artifacts()

    @classmethod
    def tearDownClass(cls) -> None:
        clear_local_explainer_cache()

    def _explain_row(self, row_index: int):
        row = self.dataset.loc[row_index]
        inputs = {feature: float(row[feature]) for feature in FEATURE_NAMES}
        prediction = predict_crop(
            inputs,
            model=self.model,
            label_encoder=self.encoder,
        )
        explanation = generate_local_explanation(
            inputs,
            predicted_crop=prediction["predicted_crop"],
            model=self.model,
            label_encoder=self.encoder,
        )
        return inputs, prediction, explanation

    def test_correct_another_crop_low_margin_and_misclassified_samples(self) -> None:
        # 1609: correct orange; 838: lowest-margin stored example; 789:
        # blackgram benchmark row misclassified as maize.  The standard input is
        # the familiar correct Rice functional check.
        cases = (
            ({"N": 90, "P": 42, "K": 43, "temperature": 25, "humidity": 80, "ph": 6.5, "rainfall": 200}, "rice"),
            (1609, "orange"),
            (838, "lentil"),
            (789, "maize"),
        )
        for source, expected_crop in cases:
            with self.subTest(source=source):
                if isinstance(source, int):
                    inputs, prediction, explanation = self._explain_row(source)
                else:
                    inputs = source
                    prediction = predict_crop(
                        inputs, model=self.model, label_encoder=self.encoder
                    )
                    explanation = generate_local_explanation(
                        inputs,
                        predicted_crop=prediction["predicted_crop"],
                        model=self.model,
                        label_encoder=self.encoder,
                    )
                self.assertEqual(prediction["predicted_crop"], expected_crop)
                self.assertEqual(explanation["predicted_crop"], expected_crop)
                self.assertEqual(len(explanation["all_contributions"]), 7)
                self.assertEqual(
                    {row["feature"] for row in explanation["all_contributions"]},
                    set(FEATURE_NAMES),
                )
                for row in explanation["all_contributions"]:
                    self.assertAlmostEqual(row["feature_value"], inputs[row["feature"]])
                    self.assertIn(row["direction"], {"supports", "opposes"})
                self.assertGreaterEqual(explanation["explanation_latency_ms"], 0.0)
                self.assertIn("not establish agricultural causality", explanation["scope_notice"])

    def test_explanation_failure_does_not_remove_prediction(self) -> None:
        prediction = {
            "predicted_crop": "rice",
            "prediction_probability": 0.54,
            "top_3": [],
            "feature_importance": [],
            "soil_assessment": {},
        }

        def fail_explanation(*_args, **_kwargs):
            raise RuntimeError("test-only SHAP failure")

        result = app._attach_local_explanation(
            prediction,
            {feature: 1.0 for feature in FEATURE_NAMES},
            fail_explanation,
        )
        self.assertEqual(result["predicted_crop"], "rice")
        self.assertEqual(result["prediction_probability"], 0.54)
        self.assertIsNone(result["local_explanation"])
        self.assertIn("temporarily unavailable", result["local_explanation_error"])

    def test_each_attachment_uses_current_prediction_and_inputs(self) -> None:
        seen: list[tuple[str, float]] = []

        def record_explanation(values, *, predicted_crop, top_n):
            seen.append((predicted_crop, float(values["N"])))
            return {"predicted_crop": predicted_crop, "all_contributions": [], "top_n": top_n}

        first = app._attach_local_explanation(
            {"predicted_crop": "rice"},
            {feature: 1.0 for feature in FEATURE_NAMES},
            record_explanation,
        )
        second_inputs = {feature: 2.0 for feature in FEATURE_NAMES}
        second = app._attach_local_explanation(
            {"predicted_crop": "orange"}, second_inputs, record_explanation
        )

        self.assertEqual(seen, [("rice", 1.0), ("orange", 2.0)])
        self.assertEqual(first["local_explanation"]["predicted_crop"], "rice")
        self.assertEqual(second["local_explanation"]["predicted_crop"], "orange")

    def test_feature_value_units_are_conservative(self) -> None:
        self.assertEqual(format_feature_value("temperature", 25), "25.0 °C")
        self.assertEqual(format_feature_value("humidity", 80), "80.0%")
        self.assertEqual(format_feature_value("ph", 6.5), "6.50 pH")
        self.assertEqual(format_feature_value("rainfall", 200), "200.0 mm")
        self.assertNotIn("kg", format_feature_value("N", 90).lower())


if __name__ == "__main__":
    unittest.main()
