"""Tests for calibration calculations and SHAP post-processing."""

from __future__ import annotations

import unittest

import numpy as np
import shap

from src.preprocessing import FEATURE_NAMES
from src.research.calibrate_probabilities import (
    calibration_improved,
    evaluate_probability_model,
)
from src.research.common import load_baseline_artifacts, load_research_split
from src.research.shap_explain import (
    contribution_records,
    explain_one_sample,
    local_explanation_text,
    normalize_multiclass_shap,
)


class CalibrationUtilityTests(unittest.TestCase):
    def test_evaluation_returns_valid_ece_with_zero_and_one_probabilities(self) -> None:
        metrics, bins = evaluate_probability_model(
            "synthetic",
            np.array([0, 1]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0, 1]),
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["top_label_ece"], 0.0)
        self.assertEqual(sum(int(row["count"]) for row in bins), 2)

    def test_calibration_decision_rule_requires_both_reliability_improvements(self) -> None:
        baseline = {"log_loss": 0.2, "top_label_ece": 0.1, "macro_f1": 0.9}
        improved = {"log_loss": 0.1, "top_label_ece": 0.05, "macro_f1": 0.895}
        worse_ece = {"log_loss": 0.1, "top_label_ece": 0.2, "macro_f1": 0.9}
        self.assertTrue(calibration_improved(baseline, improved))
        self.assertFalse(calibration_improved(baseline, worse_ece))


class ShapPostProcessingTests(unittest.TestCase):
    def test_new_array_and_legacy_list_shapes_normalize_identically(self) -> None:
        array = np.arange(24, dtype=float).reshape(2, 3, 4)
        legacy = [array[:, :, index] for index in range(4)]
        normalized_array = normalize_multiclass_shap(
            array, n_samples=2, n_features=3, n_classes=4
        )
        normalized_list = normalize_multiclass_shap(
            legacy, n_samples=2, n_features=3, n_classes=4
        )
        np.testing.assert_array_equal(normalized_array, normalized_list)

    def test_contributions_keep_feature_order_mapping_then_sort_by_magnitude(self) -> None:
        records = contribution_records(
            [1, 2, 3, 4, 5, 6, 7],
            [0.1, -0.8, 0.2, 0.3, -0.4, 0.0, 0.5],
        )
        self.assertEqual(records[0]["feature"], "P")
        self.assertEqual(records[0]["feature_value"], 2.0)
        self.assertEqual(records[0]["direction"], "opposes")
        self.assertEqual(records[-1]["feature"], "ph")
        self.assertEqual(records[-1]["direction"], "supports")

    def test_text_is_deterministic_and_non_causal(self) -> None:
        records = contribution_records(
            [1, 2, 3, 4, 5, 6, 7], [0.1, -0.8, 0.2, 0.3, -0.4, 0.0, 0.5]
        )
        first = local_explanation_text("rice", records)
        second = local_explanation_text("rice", records)
        self.assertEqual(first, second)
        self.assertIn("not causal", first)
        self.assertNotIn("caused the crop", first)

    def test_actual_one_sample_explanation_maps_predicted_class_and_features(self) -> None:
        split = load_research_split()
        model, encoder = load_baseline_artifacts()
        raw = shap.TreeExplainer(model).shap_values(split.X_test.iloc[:1])
        values = normalize_multiclass_shap(
            raw,
            n_samples=1,
            n_features=len(FEATURE_NAMES),
            n_classes=len(model.classes_),
        )
        explanation = explain_one_sample(
            model, encoder, values, 0, split.X_test.iloc[0]
        )
        self.assertIn(explanation["predicted_crop"], encoder.classes_)
        self.assertEqual(len(explanation["contributions"]), 7)
        self.assertGreaterEqual(explanation["model_probability"], 0.0)
        self.assertLessEqual(explanation["model_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
